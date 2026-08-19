# Ревью и полное сравнение бекендов: CUDA vs Vulkan vs WGPU

Дата: 2026-08-19 · Baseline: commit `54451acc` (branch с wgpu/vulkan) · Метод: прямой аудит кода (4 параллельных субагента + выборочная верификация по исходникам). Существовавшие `docs/backend-parity-*.md|json` **не использовались** как источник — признаны устаревшими; этот документ их заменяет.

---

## 1. TL;DR

Паритет по контракту `BackendStorage` **функционально почти полный**: все 33 метода трейта имеют match-arm во всех 5 бекендах (`storage.rs`), и скрытых CPU-fallback в compute-путях Vulkan/wgpu нет (есть телеметрия `CANDLE_STRICT_NO_CPU_FALLBACK`, счётчики нулевые в штатных путях). Однако:

- **Главный функциональный гэп VK/wgpu — dtype `F8E4M3`**: на CUDA весь конвейер (unary/binary/cmp/reduce/indexing/conv/affine), на Vulkan/wgpu — только хранение/копирование/const_set.
- **Гэпы wgpu против CUDA**: i16/i32 unary/binary/cmp, i32/F64 reduce.
- **Гэпы Vulkan против CUDA**: scatter_add только F32/F16; F64 unary/binary/cmp через F32-хаб (числовая деградация).
- **F16/BF16 conv/pool/upsample** на VK/wgpu идут через «F32-хаб» (GPU-resident, но не нативные kernels) — на CUDA custom kernels на все dtypes.
- **CUDA-эксклюзив**: candle-ug (runtime NVRTC), candle-flash-attn v2/v3, cuDNN conv, cuBLAS TF32, CUDA Graphs (HtD cache), MoE WMMA/prefill, on-GPU quantize. **VK/wgpu-эксклюзив (инверсии)**: `clamp`, `cumsum_last_dim`, `upsample_nearest1d` (на CUDA — `bail!`), полная to_dtype-матрица (вкл. I16/I32 targets), unified flash_attn/rms_norm/rope/softmax как методы storage, coopmat/int-dot тенсорные пути, wgpu: mxfp4/nvfp4 dequant API.

---

## 2. Методология

1. Извлечён канонический контракт: `candle-core/src/op.rs` (18 UnaryOp, 6 BinaryOp, 6 CmpOp, 5 ReduceOp, 36 Op variants), `backend.rs` (33 метода BackendStorage + 14 BackendDevice).
2. 4 субагента параллельно аудировали: CUDA (`cuda_backend/` + `candle-kernels/`), Vulkan (`vulkan_backend.rs` 9.9k строк + `candle-vulkan-kernels/`), WGPU (`wgpu_backend.rs` 12k строк + `candle-wgpu-kernels/`), интеграционный слой (`storage.rs`, `device.rs`, `quantized/`, `candle-nn`, `candle-ug`, examples).
3. Ключевые утверждения верифицированы прямыми grep/чтением (см. §9).
4. Референсы `candle_refs/`: llama.cpp b10455 (источник 153/181 Vulkan-шейдеров: 111 идентичны, 42 модифицированы, 28 — собственные), cudarc 0.19.8, wgpu 29.0.4, ash 0.38.

---

## 3. Матрица операций BackendStorage (CUDA → Vulkan / WGPU)

| Метод | CUDA | Vulkan | WGPU | Вердикт |
|---|---|---|---|---|
| try_clone | native (dtoh-free, `Clone` kernel) | native | native (b2b copy) | ✅ паритет |
| affine / powf / elu | native, все 10 dtypes | native F32; F16/BF16 через F32-хаб | native F32/F16(SHADER_F16); BF16/F16 через хаб | ⚠️ F8E4M3 нет на VK/wgpu |
| reduce_op (sum/min/max/argmin/argmax) | native `fast_*`, все dtypes | native F32/F16(→F32)/int U8/U32/I32/I64 | native F32, int U8/U32/I64; **I32/F64 — ошибка** | ⚠️ wgpu: нет I32/F64; F8E4M3 нет на VK/wgpu |
| cumsum_last_dim | **нет (default trait error)** | native F32 (ggml cumsum) | native F32 (`cumsum.wgsl`) | 🔁 инверсия: CUDA хуже |
| clamp | **нет (default trait error)** | native F32 | native F32 | 🔁 инверсия |
| cmp (6 ops) | native, все dtypes, вых. u8 | native F32/F16/U8/U32/I64 | native F32/F16/U8/U32/I32/I64; **I16 — ошибка** | ⚠️ wgpu: нет I16; F8E4M3 нет |
| to_dtype | cast.cu; **target I16/I32 — ошибка** | полная матрица 50+ convert | cpy.wgsl + emulated полная матрица | 🔁 инверсия: CUDA хуже (I16/I32) |
| unary_impl (18 ops) | native все dtypes (`u*_` kernels) | native F32/F16; BF16/F64 через F32-хаб; int — нет (не требуется semантикой, кроме Sign/Abs…) | native F32/F16; **int dtypes и F8E4M3 — ошибка** | ⚠️ F8E4M3 нет на VK/wgpu; wgpu: int нет; VK: F64 через F32 (точность!) |
| binary_impl (6 ops) | native все dtypes | native F32/F16 + int U8/U32/I32/I64; F64/BF16 через хаб; max/min F32 — композицией | native F32/F16/BF16 + int U8/U32/I32/I64; **I16 — ошибка** | ⚠️ см. выше |
| where_cond | native (ternary.cu), cond u8/u32/i64 | native (where_u8.comp) | native (generated WGSL) | ✅ (dtype-нюансы как у binary) |
| conv1d | cuDNN **или** custom kernel, все dtypes | im2col+matmul F32; F16/BF16 хаб | conv2d-reshape F32; F16/BF16 хаб | ⚠️ VK/wgpu: только F32 нативно |
| conv_transpose1d | native все dtypes | native F32 | inline WGSL F32 | ⚠️ F32-only VK/wgpu |
| conv2d | cuDNN или custom, все dtypes | im2col+matmul F32 | `conv2d.wgsl` F32 | ⚠️ как conv1d |
| conv_transpose2d | native все dtypes | native F32 | inline WGSL F32 | ⚠️ F32-only |
| avg_pool2d / max_pool2d | native все dtypes | `pool2d_f32` | im2col+reduce F32 | ⚠️ F32-only VK/wgpu |
| upsample_nearest1d | **`bail!` не поддержано** (mod.rs:2154) | native F32 (matmul-weights) | native F32 | 🔁 инверсия |
| upsample_nearest2d / bilinear2d | native все dtypes | native F32 (2×matmul) | native F32 | ⚠️ F32-only VK/wgpu |
| gather / index_select | native (indexing.cu), все dtypes | native `get_rows_*` все dtypes | native `get_rows.wgsl` | ✅ |
| scatter_set | native | native | native | ✅ |
| scatter_add_set | native все dtypes | **только F32 (atomic) + F16 (packed-half CAS)** | F32 (+F16 через хаб) | ⚠️ гэп VK: U8/U32/I64/F64 |
| index_add | native | permute+scatter-add F32/F16 | clone+scatter_add F32 | ✅ с dtype-нюансами |
| matmul | cuBLAS f16/bf16/f32(+TF32)/f64 | tiled GEMM F32, coopmat CM1/virtual-BT, bf16, f16→fp32, f64 | reg_tile/warptile/coop(64)/basic, bf16, f64, matvec | ✅ функционально; ⚠️ TF32 только CUDA |
| copy_strided_src / copy2d / const_set | native | native | native | ✅ |
| rand_uniform / rand_normal | curand **F32/F64 only** (f16/bf16 TODO) | native F32/F64; F16/BF16 через F32 | native F32 (+F16 при SHADER_F16) | ✅/🔁 CUDA сам ограничен |
| sort (argsort) | asort kernels все dtypes | argsort + argsort_large (нужен robust_buffer/VMM) | argsort + merge | ✅ |
| CustomOp1/2/3 | cuda_fwd | vulkan_fwd | wgpu_fwd | ✅ контракт есть везде |

---

## 4. Что есть на CUDA, чего НЕТ на Vulkan и WGPU (главный вопрос)

### 4.1 Функциональные гэпы (блокируют запуск моделей/операций)

1. **F8E4M3 compute** — весь перечень: unary, binary, cmp, reduce, affine/powf/elu, indexing, conv, pool, upsample. На VK/wgpu тип можно только хранить, копировать и `const_set`. Кернелы CUDA: `unary.cu/binary.cu/cast.cu/...` с инстанциацией f8e4m3.
2. **wgpu: i16/i32 unary+binary+cmp** (`UnsupportedDTypeForOp`), **i32/F64 reduce**. CUDA поддерживает всё.
3. **Vulkan: scatter_add для U8/U32/I64/F64** (только F32/F16). CUDA — все dtypes.
4. **Vulkan/wgpu: candle-ug** — runtime-компиляция SSA→PTX (NVRTC). Только CUDA/Metal (`candle-ug/Cargo.toml`).
5. **candle-flash-attn v2/v3** (CustomOp3, Ampere/Hopper, GQA-packing, paged varlen). У VK/wgpu свои `flash_attn`/`flash_attn_ext` через `Sdpa` — функциональный аналог есть, но нет varlen/paged и Hopper-специфики.

### 4.2 Качественные/числовые гэпы (работает, но деградирует)

6. **F64 unary/binary/cmp/reduce на Vulkan** — через F32-хаб: GPU-resident, но **теряется двойная точность** (для f64-нагрузок это неправильно). На wgpu unary F64 аналогично. F64 matmul при этом нативный на обоих.
7. **F16/BF16 conv/pool/upsample/upsample_bilinear** на VK/wgpu — материализация в F32 и обратно (2 доп. dispatch + память на вызов). CUDA исполняет нативно в исходном dtype.
8. **quantize (f32→GGUF) на GPU**: CUDA — на устройстве; VK/wgpu — **CPU round-trip** (`quantized/mod.rs`, QWgpuStorage::quantize / QVulkanStorage::quantize). Единственный настоящий CPU-fallback в quant-стеке (dequant — GPU везде).

### 4.3 Перф-фичи без аналога

9. **cuDNN** conv1d/conv2d (auto-algo, Winograd/FFT) — у VK/wgpu свой im2col+GEMM.
10. **cuBLAS TF32** (`set_gemm_reduced_precision_f32`), FAST_16BF/16F compute modes.
11. **CUDA Graphs** + HtD param-cache (запись графов, быстрый re-play).
12. **MoE-семейство**: `moe_gemm_wmma` (f16/bf16, WMMA), `moe_gemm_gguf(_prefill)` — матричный (prefill) MoE. У wgpu есть `mul_mat_id` (матричный), у Vulkan — только `mul_mat_vec_id` (декодовый, M-маленький). → **гэп Vulkan: нет матричного MoE**.
13. **MMVQ/MMQ fast-пути** CUDA (bf16/f16/f32-вых., scale-layouts D4/DS4/D2S6, мультитредовые). У VK/wgpu есть свои (int-dot Q8_1 у VK; quantize_q8_1+reg_tile у wgpu) — покрытие частичное, perf-класс другой.

### 4.4 Чего нет нигде (паритет отсутствия — не является гэпом VK/wgpu)

- `top_k` как тензорный метод — нет ни в одном бекенде (argsort+gather).
- IQ-форматы (IQ1_S…IQ4_XS), mxfp4/nvfp4 в `GgmlDType` — enum их не содержит. wgpu имеет публичные `dequantize_mxfp4/nvfp4` (экстра).
- Q8K matmul fast-path: CUDA нет, wgpu — dense-fallback, VK — только dequant. Равно.
- rand F8E4M3 — нет нигде.
- `data_ptr()` для QStorage — только Cuda.

---

## 5. Инверсии: что есть на Vulkan/WGPU, чего НЕТ на CUDA

1. `clamp` — CUDA не реализует (default error), VK/wgpu native.
2. `cumsum_last_dim` — то же.
3. `upsample_nearest1d` — на CUDA явный `bail!` (mod.rs:2154).
4. `to_dtype` в I16/I32 — CUDA отвергает target, VK/wgpu конвертируют.
5. Unified fused-методы storage: `softmax_last_dim`, `rms_norm`, `layer_norm`, `ggml_rope` (neox/norm/vision/mrope + yarn у wgpu), `flash_attn(_ext)` — на CUDA то же собирается композицией ops/внешними крейтами.
6. Vulkan: coopmat GEMM (`aligned_cm1`, virtual-BT), integer-dot Q8_1 rhs, packed-half CAS scatter-add F16.
7. wgpu: coop-matrix matmul (128×64 / 64×64), IMMEDIATES (push-constants), mxfp4/nvfp4 dequant API, YaRN-rope.
8. Полный intake: batched command-буферы, pipeline cache, buffer pools и на VK, и на wgpu — по архитектуре сопоставимы со stream-моделью CUDA.

---

## 6. Квантование (GGUF)

| Возможность | CUDA | Vulkan | WGPU |
|---|---|---|---|
| Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2K/Q3K/Q4K/Q5K/Q6K | dequant+get_rows+matmul+matvec | то же (стем-таблица `vulkan_quantized_stem`) | то же (`mul_mat*` WGSL) |
| Q8K | только dequant | только dequant | dense-fallback (dequant+GEMM) |
| quantize на GPU | ✅ | ❌ CPU round-trip | ❌ CPU round-trip |
| indexed MoE | fused q8_1-input, Q2K–Q6K+Q8_0 | `mul_mat_vec_id_*` (все 11) | `mul_mat_id` + gather |
| активации q8_1 | mmvq/mmq quantize | `quantize_q8_1` (int-dot) | `quantize_q8_1.wgsl` |

Шейдеры Vulkan: 153/181 из llama.cpp b10455 (111 идентичны, 42 адаптированы), 28 собственных (unary-активации), +32 в `candle-shaders/` (конвертации, cmp, int-reduce, flash_attn, argsort_large, matmul_f64, rand).

---

## 7. Интеграция

- `Storage`/`DeviceLocation`: 5 вариантов везде, отсутствующих match-arm нет.
- `supports_bf16()`: Cuda/Metal=true, **Wgpu/Vulkan=false** (device.rs:367) — верхние слои считают bf16 неподдержанным, хотя ops работают через F32-хаб.
- Телеметрия fallback: `record_wgpu/vulkan_cpu_fallback`, `CANDLE_STRICT_NO_CPU_FALLBACK`, `CANDLE_DEBUG_GPU_FALLBACK` — в compute-путях срабатываний нет; единственный постоянный CPU-путь — `quantize` (см. §4.2.8).
- candle-nn: `Sigmoid/RmsNorm/LayerNorm/Sdpa` имеют `wgpu_fwd`/`vulkan_fwd`; rotary_emb на VK/wgpu идёт rank-4 декомпозицией (обход rank-5 лимита RotaryEmbI).
- Примеры: фичи `wgpu`/`vulkan` в candle-examples; тесты паритета в `quantized_qwen3.rs`, `bert.rs`.

---

## 8. План до полного паритета (приоритет)

**P0 — функциональные гэпы**
1. F8E4M3 на Vulkan: расширить `convert.comp`/`binary_int`-паттерн → unary/binary/cmp/reduce/affine (по образцу F32-хаба с упаковкой 2×f8 в u16). Затем wgpu (generated WGSL уже умеет packed-подходы для u8/i64 — переиспользовать).
2. wgpu: i16/i32 unary/binary/cmp + i32/F64 reduce (обобщить `custom_int_binary_wgsl`/`int_reduce`).
3. Vulkan: scatter_add U8/U32/I64 (atomicCAS по образцу F16 packed-half).
4. Vulkan: матричный MoE (`mul_mat_id`-аналог) — порт с wgpu WGSL.

**P1 — числовая корректность**
5. F64 unary/binary/cmp/reduce на Vulkan и wgpu — нативные double-шейдеры (Vulkan GLSL double при shaderFloat64; wgpu SHADER_F64 уже обязателен).
6. F16/BF16 нативные conv/pool/upsample (сейчас F32-хаб): прямые f16-варианты im2col/pool2d/conv2d по образцу matmul_f16.

**P2 — перф/экосистема**
7. quantize на GPU (порт `quantize_q8_1`-подхода на остальные форматы) — устранить последний CPU round-trip.
8. candle-ug для Vulkan (SPIR-V бекенд компиляции SSA) / wgpu (WGSL/Naga IR).
9. flash-attn varlen/paged-варианты на VK/wgpu.
10. Аналог CUDA Graphs: Vulkan command-buffer replay; wgpu — bundle/cached encoders.
11. TF32-класс точности: coopmat f16→f32 уже близко; задокументировать соответствие режимов.

**Обслуживание документации**
12. Устаревшие `docs/backend-parity*.md|json` либо регенерировать из кода, либо удалить; данный файл — актуальный срез на `54451acc`.

---

## 9. Верификационные ссылки (выборочно проверено)

- CUDA `upsample_nearest1d` bail: `candle-core/src/cuda_backend/mod.rs:2154`.
- Отсутствие `fn clamp`/`fn cumsum_last_dim` в `cuda_backend/mod.rs` (grep пуст) → default trait error.
- VK `clamp`: `vulkan_backend.rs:7980`, `cumsum_last_dim`: `:7952`, `argsort_last_dim`: `:4581`, `softmax_last_dim`: `:4866`, `ggml_rope`: `:4965`, `rms_norm`: `:5059`, `flash_attn`: `:5308`.
- wGPU `cumsum_last_dim`: `wgpu_backend.rs:10829`, `clamp`: `:10857`, `argsort_last_dim`: `:6685`, `flash_attn`: `:7206`, `rms_norm`: `:7472`.
- `supports_bf16`: `device.rs:367-372`.
- `GgmlDType` (15 значений, без IQ/mxfp4/nvfp4): `quantized/mod.rs:884-900`.
- wgpu mxfp4/nvfp4 dequant API: `wgpu_backend.rs:9760/9818`.
- wgpu Q8K dense-fallback: `quantized/mod.rs:370-371`.
- VK scatter_add только F32/F16: `vulkan_backend.rs:8394` (set_rows_add).
- VK quant-stem 11 форматов: `vulkan_backend.rs:838-852`.
- sort-диспетчер: `sort.rs:253-268` (wgpu_fwd/vulkan_fwd → `argsort_last_dim`).
- Шейдеры vs llama.cpp: comm/diff по `candle_refs/llama.cpp-b10455/ggml/src/ggml-vulkan/vulkan-shaders` (111 identical / 42 modified / 28 fork-only).
- Референс-версии: cudarc 0.19.8, wgpu 29.0.4, ash 0.38 (`candle_refs/manifest.toml`).
