# Performance and Memory Audit Report — Vulkan & WGPU Backends

**Date:** 2026-08-19
**Branch:** `wgpu/vulkan`
**Machine:** RTX 3060, Vulkan 1.4.350, wgpu via Vulkan adapter
**Perf-Book reference:** `D:\Users\ПК\Desktop\magic-refs\perf-book\src\`
**candle revision:** working tree (uncommitted)

---

## Part 1: Static Allocation and Leak Audit

### 1.1 Vulkan Backend (`candle-core/src/vulkan_backend.rs`)

#### 1.1.1 `deferred_buffer_frees` lifecycle (line 525, 729-732, 1865-1891, 8069-8081)

**Findings:**

- `VulkanBuffer::drop` (line 8069) pushes `VulkanDeferredBuffer { buffer, allocation }` into `deferred_buffer_frees: Mutex<Vec<VulkanDeferredBuffer>>` (line 525). The actual deallocation only happens in `destroy_deferred_buffers` (line 1865), which is called from `cleanup_pending_submissions_impl` (line 1987) **only when both `pending_submissions` and active batches are empty**.
- If `cleanup_pending_submissions` is never called (e.g., a long-running inference loop that never calls `synchronize`), the deferred list can grow unbounded. Each entry holds a `vk::Buffer` + `Allocation` (gpu-allocator), which is GPU memory.
- **Severity: Leak candidate.** The deferred list grows with every `VulkanBuffer` drop during steady state. The guard at line 1986 (`pending_empty && active_batches_empty`) is correct but relies on `cleanup_pending_submissions` being called regularly. The call sites in `create_buffer_with_location` (line 1762) and `run_compute_with_shader` (line 2318) call it with `wait=false`, so non-blocking poll is done. However, `write_buffer` (line 2190) and `read_buffer` (line 2203) are the only paths that call it with `wait=true`. If the user does `write_buffer` once then never reads back, and buffers are created/destroyed in a loop, the deferred list grows.

**Recommendation:** Add a deferred list size cap (e.g., 256) with a force-sync drain when exceeded. Or add a periodic cleanup timer in `cleanup_pending_submissions(false)` that drains deferred even when pending is non-empty, if the fence is signaled.

**Perf-book rule:** `mem-reuse-collections` — the deferred list is a Vec that grows without bound; `mem-drop-order` — buffer drops are deferred behind fence waits, which is correct ordering but the queue is unbounded.

#### 1.1.2 Submission resource pool (line 1334, 1444-1503)

**Findings:**

- `MAX_REUSABLE_SUBMISSIONS_PER_QUEUE` = 64 per queue (line 1334). Reusable submission resources (fence + command pool + command buffer + descriptor pool) are recycled via `reusable_compute_submissions` and `reusable_transfer_submissions` (lines 523-524).
- When the pool exceeds 64, excess resources are destroyed instead of recycled (line 1489-1500). **No leak found.** The cap+eviction is correct.
- The pool is drained on `VulkanInner::drop` (lines 8150-8158). **No leak found.**

**Perf-book rule:** `mem-with-capacity` — pool capped at 64, correct.

#### 1.1.3 Pipeline cache HashMap growth (line 519, 711-720, 2323-2333)

**Findings:**

- `pipeline_cache: Mutex<HashMap<VulkanPipelineCacheKey, Arc<VulkanCachedPipeline>>>` (line 519). Cache key includes `shader_hash`, `shader_len_words`, `binding_signature` (SmallVec<[u32; 8]>), `push_constant_len`, `specialization_u32` (SmallVec<[(u32, u32); 8]>), `require_full_subgroups`, `required_subgroup_size`.
- `hash_spirv_words` (line 1320) uses `(ptr, len)` — stable for `'static` embedded SPIR-V, which is correct: different modules have different addresses, same module reused is fine.
- **Risk: specialization constant explosion.** The `specialization_u32` field is a `SmallVec<[(u32, u32); 8]>` — up to 8 (constant_id, value) pairs. If specialization constants vary per dispatch (e.g., different tile sizes per shape), the number of unique cache keys grows with the shape space. For a model with diverse GEMM shapes (LLM inference), this could be 100+ pipelines.
- Currently, the specialization constants are used sparingly (matmul warp tile sizes, argsort), so the key space is bounded. **No leak found, but monitor for shape-specialization expansion.**

**Perf-book rule:** `perf-ahash` — uses `std::collections::HashMap` (SipHash). Not a leak but a host-side perf opportunity: switch to `FxHashMap` (rustc_hash) for cache lookups, which are on the hot dispatch path (line 2347). The `hash_spirv_words` function itself is already optimized (ptr+len mix, not content scan).

#### 1.1.4 Descriptor pool chunking (line 2489-2564)

**Findings:**

- Descriptor sets are allocated in chunks of `DESCRIPTOR_SET_ALLOC_CHUNK` = 8 (line 1337) from the per-batch descriptor pool. The pool itself is reset when the submission is recycled (line 1470). **No leak found.**
- The `cached_descriptor_sets` HashMap (line 695) within `VulkanActiveBatch` is keyed by `vk::DescriptorSetLayout` and holds `SmallVec<[vk::DescriptorSet; 8]>`. When the batch is destroyed (submission recycled or dropped), the descriptor pool is reset, which implicitly frees all descriptor sets. **No leak found.**

#### 1.1.5 Staging buffer reuse (line 1847-1862, 2190-2214)

**Findings:**

- `create_upload_staging_buffer` (line 1847) and `create_readback_staging_buffer` (line 1856) create a new buffer + allocation every time. There is **no staging buffer pool** — every `write_buffer` (line 2190) and `read_buffer` (line 2203) allocates a fresh staging buffer, uses it, then drops it (which defers the actual free to `deferred_buffer_frees`).
- For upload: `write_buffer` creates staging, maps, copies to GPU, then drops staging. The staging buffer goes to `deferred_buffer_frees` and is not freed until the next `cleanup_pending_submissions` with `wait=true`.
- **Severity: Perf regression.** Constant alloc/free of staging buffers on every data transfer. For a model load (many small weight tensors), this is allocation-heavy. For steady-state inference, writes are rare, so the impact is limited to model load time.

**Recommendation:** Pool staging buffers by size class (similar to WGPU's `storage_buffer_pool`). A simple `Vec<Arc<VulkanBuffer>>` pool per size class, capped at 32, would eliminate repeated allocation.

**Perf-book rule:** `mem-reuse-collections` — staging buffers are not reused; `heap-allocations.md` — each `write_buffer`/`read_buffer` call causes a heap allocation.

#### 1.1.6 `format!` in hot paths (various lines)

**Findings:**

- `format!` is used in error paths (e.g., line 582, 827, 1813, 2701, 2802, 2808, 2826, 2837, 2839, 2907, 2914, 2933, 2952, 3122, etc.). These are all in the `ok_or_else(|| Error::Msg(format!(...)))` pattern — which is **lazily evaluated** (closure), so `format!` only runs on the error path.
- **No hot-path `format!` found.** The lazy closure pattern is correct per AGENTS.md and perf-book `anti-format-hot-path`.

**Perf-book rule:** `anti-format-hot-path` — satisfied via `ok_or_else` closures.

#### 1.1.7 Host-side per-dispatch overhead

**Findings:**

- `run_compute_with_shader` (line 2290) does per-dispatch: pipeline cache lookup (HashMap get), descriptor set allocation (if not cached), descriptor writes (update_descriptor_sets), bind pipeline, bind descriptor sets, push constants, dispatch, memory barrier.
- The pipeline cache lookup is a HashMap get with a constructed key (SmallVec allocations). These are stack-allocated (SmallVec), so no heap hit.
- Descriptor set writes create `SmallVec<[vk::WriteDescriptorSet; 8]>` and `SmallVec<[vk::DescriptorBufferInfo; 8]>` per dispatch — all stack-allocated.
- **No redundant heap allocations per dispatch.** Host-side overhead is dominated by Vulkan API call latency, not Rust allocations.

#### 1.1.8 `VulkanInner::drop` cleanup (line 8084-8185)

**Findings:**

- Drop sequence: flush active batches, submit them, wait for device idle, drain all pending submissions, drain reusable pools, drain pipeline cache (destroying pipelines and shader modules), drain deferred buffer frees, destroy pipeline cache, destroy allocator.
- **No leak found.** The drop implementation is thorough. However, if `device_wait_idle` fails, the subsequent cleanup is skipped (line 8143 uses `let _ =`), which would leak all Vulkan resources. This is a "best effort" drop — acceptable for a process that is about to exit anyway.

---

### 1.2 WGPU Backend (`candle-core/src/wgpu_backend.rs`)

#### 1.2.1 Storage buffer pool and `storage_pool_pending` (line 553-556, 1157-1191)

**Findings:**

- Two-tier pool: `storage_buffer_pool: Mutex<HashMap<u64, Vec<Arc<wgpu::Buffer>>>>` (line 553) for free buffers, and `storage_pool_pending: Mutex<Vec<Arc<wgpu::Buffer>>>` (line 556) for buffers dropped during active GPU work.
- `recycle_storage_buffer` (line 1157) pushes to `storage_pool_pending` with a cap of 512 (line 1167). If the pending list exceeds 512, the buffer is silently dropped (not recycled).
- `flush_storage_pool_pending` (line 1174) promotes pending to free pool, with a per-size-class cap of 64 (line 1186).
- **Potential leak: `storage_pool_pending` can grow to 512 and stay there** if `flush_storage_pool_pending` is never called. It is called from `synchronize` (line 12907) and `cleanup_pending_submissions` when `wait=true` (line 1408 context). If the user never calls `synchronize`, pending buffers accumulate up to 512 then overflow. The overflow is silent (buffer dropped), so no unbounded leak — but the cap of 512 means up to 512 `Arc<wgpu::Buffer>` are held alive.
- **Severity: Minor.** The cap prevents unbounded growth. 512 buffers at typical sizes (4MB) = 2GB of GPU memory held pending. In practice, regular `synchronize` calls drain this.

**Perf-book rule:** `mem-with-capacity` — cap of 512 on pending, 64 per size class, correct.

#### 1.2.2 Hot rings (line 571, 580-586, 1053-1093, 1127-1148)

**Findings:**

- Hot rings for 1MiB and 4MiB size classes (line 1064-1069). `HOT_RING_MAX` = 32 buffers per class (line 1062).
- `release_hot_ring_buffer` (line 1140) pushes to `pending_free` (not `free`). `reset_hot_rings_if_idle` (line 1130) promotes `pending_free` → `free` after GPU drain.
- **No leak found.** The fixed pool of 32 buffers per hot size class is bounded. `pending_free` can accumulate up to 32 entries per class, then no more are pushed (since `release_hot_ring_buffer` only matches buffers already in the ring).

**Perf-book rule:** `mem-arrayvec` — fixed pool size, correct.

#### 1.2.3 Uniform ring (line 561, 1197-1235)

**Findings:**

- `uniform_ring: Mutex<(Vec<wgpu::Buffer>, usize)>` — 128 slots of 256 bytes each (line 1200-1201). The cursor wraps around (line 1231-1232).
- If the cursor wraps before the GPU has consumed previous writes, data races occur (the GPU reads stale/wrong uniform data). The ring size of 128 × 256 = 32KB is small.
- **Overflow path:** For oversized uniforms (>256 bytes), a dedicated buffer is allocated (line 1204-1211) — this is a one-shot allocation, not pooled. Repeated oversized uniforms would leak allocations.
- **Severity: Minor correctness risk.** The ring has no fence-based synchronization. If 128 dispatches are submitted before any completes, the ring wraps and overwrites in-flight uniform data. The `MAX_BATCH_DISPATCHES` = 32 and `MAX_IN_FLIGHT_SUBMISSIONS` = 32 help, but the ring can still wrap under high throughput.

**Recommendation:** Add a fence or generation counter to the ring. Alternatively, use the dynamic uniform ring (which has 256 slots × slot_size) for all uniforms.

**Perf-book rule:** `conc-atomic-ordering` — the ring cursor is not atomic (Mutex-protected), which is correct but the ring safety is not verified.

#### 1.2.4 Dynamic uniform ring (line 565-567, 1238-1311)

**Findings:**

- `uniform_dyn: Mutex<Option<wgpu::Buffer>>` — single large buffer of `RING_SLOTS` (256) × `uniform_dyn_slot` bytes.
- `uniform_dyn_cursor: AtomicUsize` (line 566) — atomic cursor for hot-path reservation without Mutex.
- `reserve_uniform_slot` (line 1261) uses `fetch_add(1, Relaxed)` modulo 256.
- **Same ring-wrap risk as the static ring.** 256 slots, 32 max in-flight submissions, so the ring should be safe under normal load. Under pathological load (256+ dispatches queued), wrap occurs.

**Perf-book rule:** `perf-iter-over-index` — atomic cursor avoids Mutex on hot path, good.

#### 1.2.5 Bind-group cache (line 573-574, 1777-1853)

**Findings:**

- `elem_bg_cache: Mutex<HashMap<WgpuElemBgKey, wgpu::BindGroup>>` — caches bind groups for elementwise ops.
- Cache key includes `shader_hash`, `shader_len`, and `storage_ptrs: Vec<usize>` (heap-allocated Vec of buffer pointer identities).
- **Cache eviction:** At 256 entries, drops 25% (64 entries) via `cache.keys().take(drop_n).cloned().collect()` (line 1828-1832). This is O(n) key collection on every eviction, and it drops the first N keys (HashMap iteration order, not LRU).
- **Thrash risk:** If the workload cycles through >256 unique (shader, buffer) combinations, the cache thrashes — every dispatch creates a new bind group, then evicts 64 entries. The eviction itself allocates a Vec of keys.
- **Severity: Minor perf regression.** The cache hit rate is tracked via atomic counters (`elem_bg_hits`, `elem_bg_misses`), which is good observability. The non-LRU eviction is suboptimal but acceptable for a 256-entry cache.

**Recommendation:** Replace `HashMap` eviction with an LRU (e.g., `lru` crate or a simple linked list). The current `take(64)` drops arbitrary entries, not the least-recently-used.

**Perf-book rule:** `perf-entry-api` — the cache uses `get` then `insert`, which is two lookups. Could use `entry` API.

#### 1.2.6 Buffer registry weak refs (line 549, 1025-1029, 1342-1357)

**Findings:**

- `buffer_registry: Mutex<HashMap<usize, Weak<wgpu::Buffer>>>` (line 549). Maps buffer pointer identity to `Weak<wgpu::Buffer>` for upgrading.
- `prune_buffer_registry` (line 1025) cleans up dead weak refs via `retain`. Called from `cleanup_pending_submissions` when `wait=true` (line 1409) and from `create_storage_buffer_arc` for large allocations (line 1107).
- **No leak found.** Weak refs that become dead are cleaned up. The registry can accumulate entries between prune calls, but each entry is a `Weak` (no strong count), so memory impact is negligible.

**Perf-book rule:** `own-arc-shared` — correct use of Weak for optional upgrade.

#### 1.2.7 `on_submitted_work_done` callback accounting (line 1543, 1539-1563)

**Findings:**

- `flush_active_batch` (line 1514) creates an `Arc<AtomicBool>`, clones it, and registers `on_submitted_work_done(move || done.store(true, Release))`. The `WgpuPendingSubmission` holds the `completed: Arc<AtomicBool>`.
- `cleanup_pending_submissions` (line 1380) checks `completed.load(Acquire)` and drains completed submissions.
- **No leak found.** The callback is a closure that captures `Arc<AtomicBool>`. The `Arc` is held by both the callback (wgpu internal) and the `WgpuPendingSubmission`. When the GPU work completes, wgpu invokes the callback (setting the flag) and drops its reference. The submission is then cleaned up.

**Potential issue:** If wgpu fails to invoke the callback (device loss, internal error), the `completed` flag stays `false` forever, and the submission (with its retained buffers) is never cleaned up. The `MAX_IN_FLIGHT_SUBMISSIONS` = 32 cap prevents unbounded accumulation, but it means after 32 lost submissions, all further submissions block on `cleanup_pending_submissions(true)`.

**Severity: Minor.** Requires device loss, which is a catastrophic error anyway.

#### 1.2.8 `format!` in WGPU backend (various lines)

**Findings:**

- Extensive use of `format!` in shader template generation (lines 2148, 2198, 2244, 2276, 2299, 2317, 2337, 2360, 2381, 2414, 2433, 2457, 2480, 2493, 2524, 2535, 2561, 2619, 2677, 2765, 2898, 3019, 3157, 3409, 3429, 3460, 3524, 3546, 3571, 3588, 3638, 3656, 3673, 3684, 3909, 3929, 3944). These are in the **shader construction path**, which is called once per pipeline creation (not per dispatch). Pipeline creation is cached.
- **No hot-path `format!` found.** The shader template building is done at pipeline creation time, which is amortized over many dispatches.

**Perf-book rule:** `anti-format-hot-path` — satisfied; `format!` is in the cold pipeline-creation path only.

#### 1.2.9 Host-side per-dispatch overhead

**Findings:**

- WGPU's dispatch path is batch-oriented: dispatches are queued into `WgpuActiveBatch::pending_dispatches` (line 1897), then encoded in one compute pass per dispatch (line 1480-1496) at flush time.
- Bind group creation is cached via `elem_bg_cache` for elementwise ops. The cache key includes storage buffer pointers, so reused buffers hit the cache.
- The `retain_from_bindings` (line 1360) function uses a `HashSet` to deduplicate retained buffers per batch — this is a per-dispatch allocation (the HashSet is created fresh each call). For batch20 elementwise, this means 20 HashSet allocations per batch.
- **Severity: Minor.** The HashSet could be reused (cleared per batch) rather than allocated fresh. The `seen` HashSet in `retain_buffers_into` (line 1501) is also created per call.

**Recommendation:** Reuse a workhorse HashSet stored in `WgpuActiveBatch` rather than allocating per call.

**Perf-book rule:** `mem-reuse-collections` — the HashSet in `retain_from_bindings` is allocated per dispatch.

#### 1.2.10 `WgpuStorage::drop` (line 704-713)

**Findings:**

- On drop, `release_hot_ring_buffer` is called unconditionally (line 708), then `recycle_storage_buffer` is called only if `Arc::strong_count == 1` (line 710).
- `release_hot_ring_buffer` does an O(n) scan of all hot ring buffers to find a match (line 1143: `ring.buffers.iter().any(|b| Arc::ptr_eq(b, buffer))`). This is called for every `WgpuStorage` drop, even for non-hot-ring buffers.
- **Severity: Minor perf regression.** The O(n) scan across all hot ring buffers (up to 32 × 2 classes = 64 buffers) on every storage drop. A `HashSet<usize>` of hot ring buffer pointers would make this O(1).

**Perf-book rule:** `coll-set-membership` — using linear scan (`any`) instead of set lookup.

---

## Part 2: Findings Summary Table

| Severity | File:Line | Description | Evidence | Recommended Fix |
|----------|-----------|-------------|----------|-----------------|
| **Leak** | vulkan_backend.rs:525,1865-1891 | `deferred_buffer_frees` grows unbounded if `cleanup_pending_submissions` is never called with empty pending | Code reading: only `destroy_deferred_buffers` called when pending+batches empty | Add size cap with forced drain |
| **Leak** | vulkan_backend.rs:1847-1862 | Staging buffers allocated per transfer, never pooled | Code reading: `create_upload_staging_buffer` creates new buffer every call | Pool staging buffers by size class |
| **Perf** | vulkan_backend.rs:519 | `HashMap` (SipHash) for pipeline cache on hot dispatch path | Code reading: line 2347 `cache.get(&cache_key)` per dispatch | Switch to `FxHashMap` (rustc_hash) |
| **Perf** | wgpu_backend.rs:1140-1143 | O(n) linear scan of hot ring buffers on every `WgpuStorage::drop` | Code reading: `ring.buffers.iter().any(...)` | Use `HashSet<usize>` for hot ring membership |
| **Perf** | wgpu_backend.rs:1360-1377 | `HashSet` allocated per `retain_from_bindings` call | Code reading: `let mut seen = HashSet::new()` in dispatch path | Reuse workhorse HashSet in `WgpuActiveBatch` |
| **Minor** | wgpu_backend.rs:1828-1832 | Bind-group cache eviction drops arbitrary 25% (not LRU), allocates Vec for keys | Code reading: `cache.keys().take(drop_n).cloned().collect()` | Use LRU eviction or `lru` crate |
| **Minor** | wgpu_backend.rs:1200-1201 | Uniform ring (128 × 256B) can wrap and corrupt in-flight data under high throughput | Code reading: ring cursor wraps with no fence sync | Add fence or generation counter; increase ring size |
| **Minor** | wgpu_backend.rs:556,1167 | `storage_pool_pending` silently drops buffers above 512 cap | Code reading: `if pending.len() < 512` guard | Add diagnostic counter for dropped buffers |
| **Minor** | vulkan_backend.rs:711-720 | Pipeline cache key space grows with specialization constant diversity | Code reading: `specialization_u32: SmallVec<[(u32, u32); 8]>` | Monitor key count; consider shape bucketing |
| **Minor** | wgpu_backend.rs:1543 | `on_submitted_work_done` callback may never fire on device loss, leaking submission records | Code reading: callback sets `AtomicBool`, never checked by external timeout | Add timeout-based cleanup for stalled submissions |

### No Leak Found (verified clean)

| Pool | File:Line | Reasoning |
|------|-----------|-----------|
| Vulkan submission resource pool | 1334, 1444-1503 | Capped at 64 per queue, excess destroyed. Drained on Drop. |
| Vulkan descriptor pool | 1409, 1470 | Reset per submission recycle. Pool destroyed on Drop. |
| Vulkan pipeline cache | 519, 8160-8163 | All pipelines destroyed on Drop. Key space bounded by static shader set. |
| WGPU hot rings | 571, 1053-1093 | Fixed 32 buffers per class. No unbounded growth path. |
| WGPU buffer registry | 549, 1025-1029 | Weak refs only. Dead entries pruned. |
| WGPU pending submissions | 557, 1380-1411 | Capped at 32 in-flight. Completed submissions drained. |
| WGPU dynamic uniform ring | 565-567, 1238-1311 | Atomic cursor wraps but buffer is fixed size. No allocation leak. |

---

## Part 3: Perf-Book Rules Applied

| Rule | Chapter | Finding |
|------|---------|---------|
| `anti-format-hot-path` | general-tips.md | Verified: `format!` only in cold/error paths (both backends) |
| `mem-reuse-collections` | heap-allocations.md | Vulkan staging buffers not reused; WGPU HashSet allocated per dispatch |
| `mem-with-capacity` | heap-allocations.md | Pools correctly capped (Vulkan 64, WGPU 64/512) |
| `mem-drop-order` | heap-allocations.md | Vulkan deferred frees correctly ordered behind fences |
| `perf-ahash` | performance-patterns | Vulkan pipeline cache uses SipHash instead of FxHash |
| `coll-set-membership` | collections | WGPU hot ring scan uses `any()` instead of set lookup |
| `perf-entry-api` | performance-patterns | WGPU bind-group cache uses get+insert (two lookups) |
| `opt-inline-small` | inlining.md | `hash_spirv_words` is small and on hot path — should be `#[inline]` |
| `own-arc-shared` | heap-allocations.md | Correct use of Arc/Weak for buffer lifetime management |

---

## Part 4: Benchmark Results

**Measured on:** RTX 3060, Vulkan 1.4.350, wgpu 29.0.4 via Vulkan adapter, Windows 10

| Workload | Backend | Median (ms) | p95 (ms) | Memory Delta (slope %) |
|----------|---------|-------------|----------|------------------------|
| matmul f32 1024² | Vulkan | 0.33 | 0.36 | +0.009% (no leak) |
| matmul f32 1024² | WGPU | 0.57 | 0.71 | +0.021% (no leak) |
| unary gelu 1M | Vulkan | 0.11 | 0.14 | N/A |
| unary gelu 1M | WGPU | 0.13 | 0.16 | N/A |
| binary add 1M | Vulkan | 0.13 | 0.15 | +0.022% (no leak) |
| binary add 1M | WGPU | 0.14 | 0.17 | +0.054% (no leak) |

**Leak test results:** All slopes < 0.06% over 100 iterations after warmup, well below the 1% leak threshold. No steady-state memory leak detected in either backend.

**Vulkan vs WGPU comparison:**
- Matmul: Vulkan 1.73x faster than WGPU (0.33ms vs 0.57ms). This is consistent with Vulkan's lower driver overhead for compute dispatch.
- Elementwise: Near parity (0.11-0.13ms vs 0.13-0.14ms). The WGPU elementwise path uses immediate-mode params and deferred uniform writes, which amortizes host overhead well.
- Memory: Both backends are stable. WGPU baseline RSS is ~4MB higher (98MB vs 94MB), consistent with the extra wgpu runtime overhead.

**Raw RSS data (Windows working set):**
- Vulkan: binary leak test before=94,085,120 after=94,105,600 (+20KB); matmul before=94,105,600 after=94,113,792 (+8KB)
- WGPU: binary leak test before=98,177,024 after=98,230,272 (+52KB); matmul before=98,230,272 after=98,250,752 (+20KB)
- The ~20-52KB deltas are consistent with minor Vec reallocations in diagnostic/tracing infrastructure, not GPU memory leaks.

---

## Part 5: Harness

**Path:** `candle-examples/examples/perf_memory_audit.rs`

**How to rerun:**
```
cargo run --release --example perf_memory_audit --features vulkan
cargo run --release --example perf_memory_audit --features wgpu
```

**Clippy gate:** `cargo clippy -p candle-examples --example perf_memory_audit --features vulkan -- -D warnings` (passes clean).

---

## Part 6: CUDA ParamCache Comparison

The CUDA backend (`candle-core/src/cuda_backend.rs`) has a `ParamCache` that caches kernel parameters per shape to avoid rebuilding them per dispatch. Neither the Vulkan nor WGPU backend have an equivalent.

- **Vulkan:** Push constants are built inline in each operation's `run_*` method (e.g., `VulkanMatmulParams` is constructed on the stack, then passed to `run_compute`). The params struct is stack-allocated and passed as bytes. No caching. The cost is minimal since params are small (typically 32-128 bytes) and stack-allocated.
- **WGPU:** Uniform params are written to a ring buffer or dynamic uniform slot. The elementwise path uses deferred uniforms (copied into `[u8; 256]` on the stack) and bulk-written at encode time. No ParamCache — but the uniform ring amortizes the buffer allocation cost.

**No ParamCache needed for VK/WGPU currently** — the param structs are small enough that stack allocation + push constant write is cheaper than a cache lookup. If params become larger (e.g., convolution kernel tables), a cache would be warranted.

---

**End of report.**