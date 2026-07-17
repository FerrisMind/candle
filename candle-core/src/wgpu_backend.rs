use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, Mul, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, WithDType};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

const WG_SIZE: u32 = 256;
/// wgpu validates dispatch dims <= 65535 even when the adapter reports 65536.
const WGPU_DISPATCH_WG_CAP: u32 = 65535;

fn wgpu_dispatch_wg_cap(device: &WgpuDevice) -> u32 {
    device
        .inner
        .limits
        .max_compute_workgroups_per_dimension
        .clamp(1, WGPU_DISPATCH_WG_CAP)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UnaryParams {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FillParams {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    fill_val: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawFillParams {
    ne: u32,
    offset_dst: u32,
    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32,
    value0: u32,
    value1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClampParams {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    clamp_min: f32,
    clamp_max: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BinaryParams {
    ne: u32,
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,
    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,
    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WhereParams {
    ne: u32,
    offset_cond: u32,
    offset_true: u32,
    offset_false: u32,
    offset_dst: u32,
    stride_cond0: u32,
    stride_cond1: u32,
    stride_cond2: u32,
    stride_cond3: u32,
    stride_true0: u32,
    stride_true1: u32,
    stride_true2: u32,
    stride_true3: u32,
    stride_false0: u32,
    stride_false1: u32,
    stride_false2: u32,
    stride_false3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    base: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SumRowsParams {
    offset_src: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArgMaxParams {
    offset_src: u32,
    offset_dst: u32,
    ne0: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArgsortParams {
    offset_src: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    src_ne0: u32,
    ne1: u32,
    ne2: u32,
    ne0: u32,
    top_k: u32,
    npr: u32,
    nrows: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ArgsortMergeParams {
    offset_src: u32,
    offset_in: u32,
    offset_out: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_idx1: u32,
    stride_idx2: u32,
    stride_idx3: u32,
    stride_out1: u32,
    stride_out2: u32,
    stride_out3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    top_k: u32,
    len: u32,
    nm: u32,
    nrows: u32,
    _pad0: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WgpuArgsortDType {
    F32,
    U32,
    I64,
    F64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CumsumParams {
    offset_src: u32,
    offset_dst: u32,
    ne0: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SoftmaxParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_sinks: u32,
    offset_dst: u32,
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,
    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne12: u32,
    ne13: u32,
    scale: f32,
    max_bias: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
    row_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WgpuRopeParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_src2: u32,
    offset_dst: u32,
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    n_threads: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    n_dims: u32,
    mode: u32,
    theta_scale: f32,
    attn_factor: f32,
    freq_scale: f32,
    ext_factor: f32,
    corr_dim0: f32,
    corr_dim1: f32,
    sections0: u32,
    sections1: u32,
    sections2: u32,
    sections3: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormMulParams {
    offset_rn_src: u32,
    offset_mul_src: u32,
    offset_dst: u32,
    stride_rn_src1: u32,
    stride_rn_src2: u32,
    stride_rn_src3: u32,
    stride_mul_src1: u32,
    stride_mul_src2: u32,
    stride_mul_src3: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    mul_src_ne0: u32,
    mul_src_ne1: u32,
    mul_src_ne2: u32,
    mul_src_ne3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GetRowsParams {
    offset_src: u32,
    offset_idx: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,
    idx1: u32,
    idx2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleParams {
    offset_src: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    scale: f32,
    bias: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RandParams {
    seed_lo: u32,
    seed_hi: u32,
    min_val: f32,
    max_val: f32,
    ne: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RandNormalParams {
    seed_lo: u32,
    seed_hi: u32,
    mean: f32,
    std: f32,
    ne: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CopyParams {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,
    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32,
    elem_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmulatedStridedCopyParams {
    ne: u32,
    ne_total: u32,
    offset_src: u32,
    offset_dst: u32,
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    elem_base: u32,
    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct F64CastParams {
    ne: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32,
    /// Inner K stride for src0 (1 = contiguous K). Used by reg-tile GEMM.
    stride_0k: u32,
    /// Inner K stride for src1 (1 = contiguous K). Used by reg-tile GEMM.
    stride_1k: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Conv2dParams {
    offset_w: u32,
    offset_i: u32,
    offset_o: u32,
    sw0: u32,
    sw1: u32,
    sw2: u32,
    sw3: u32,
    si0: u32,
    si1: u32,
    si2: u32,
    si3: u32,
    so0: u32,
    so1: u32,
    so2: u32,
    so3: u32,
    kw: u32,
    kh: u32,
    ic: u32,
    iw: u32,
    ih: u32,
    ow: u32,
    oh: u32,
    oc_out: u32,
    n_out: u32,
    s0: u32,
    s1: u32,
    p0: u32,
    p1: u32,
    d0: u32,
    d1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Im2ColParams {
    offset_i: u32,
    offset_o: u32,
    si0: u32,
    si1: u32,
    si2: u32,
    si3: u32,
    so0: u32,
    so1: u32,
    so2: u32,
    so3: u32,
    kw: u32,
    kh: u32,
    ic: u32,
    iw: u32,
    ih: u32,
    n: u32,
    ow: u32,
    oh: u32,
    s0: u32,
    s1: u32,
    p0: u32,
    p1: u32,
    d0: u32,
    d1: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for WgpuError {
    fn from(e: String) -> Self {
        Self::Message(e)
    }
}

#[derive(Debug, Clone)]
pub struct WgpuDevice {
    inner: Arc<WgpuInner>,
}

#[derive(Debug)]
struct WgpuInner {
    ordinal: usize,
    adapter_name: String,
    adapter_backend: String,
    adapter_driver: String,
    adapter_driver_info: String,
    adapter_pci_bus_id: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    features: wgpu::Features,
    limits: wgpu::Limits,
    /// Cached `CANDLE_WGPU_COOP_MATMUL` (default true when hardware allows).
    coop_matmul_enabled: bool,
    seed_value: RwLock<u64>,
    pipeline_cache: Mutex<HashMap<WgpuPipelineCacheKey, Arc<WgpuCachedPipeline>>>,
    buffer_registry: Mutex<HashMap<usize, Weak<wgpu::Buffer>>>,
    /// Free storage buffers by size (wgpu_copy_size key). Recycled when the
    /// last Arc drops via `recycle_storage_buffer` after GPU work completed.
    storage_buffer_pool: Mutex<HashMap<u64, Vec<Arc<wgpu::Buffer>>>>,
    pending_submissions: Mutex<Vec<WgpuPendingSubmission>>,
    active_batch: Mutex<Option<WgpuActiveBatch>>,
    /// Ring of reusable uniform buffers (avoids per-dispatch create_buffer).
    /// Used by non-dynamic uniform bindings (`as_entire_binding`).
    uniform_ring: Mutex<(Vec<wgpu::Buffer>, usize)>,
    /// Single large uniform buffer for dynamic-offset slots (elementwise host path).
    /// Slot size is `uniform_dyn_slot` (aligned to min_uniform_buffer_offset_alignment).
    uniform_dyn: Mutex<Option<(wgpu::Buffer, usize)>>,
    uniform_dyn_slot: u64,
    /// Permanent size-class rings for latency-sensitive elementwise outputs.
    /// Buffers are never destroyed; cursor resets on synchronize when safe so
    /// bind-group cache hits across microbench samples.
    hot_rings: Mutex<HashMap<u64, WgpuHotRing>>,
    /// Bind groups for elementwise dyn-uniform path (src+dst+params permanent).
    elem_bg_cache: Mutex<HashMap<WgpuElemBgKey, wgpu::BindGroup>>,
    elem_bg_hits: std::sync::atomic::AtomicU64,
    elem_bg_misses: std::sync::atomic::AtomicU64,
}

/// Permanent buffer ring for one size class (e.g. 4MiB for 1024² f32).
#[derive(Debug)]
struct WgpuHotRing {
    buffers: Vec<Arc<wgpu::Buffer>>,
    /// Ready to hand out (GPU-idle, may still be pinned by bind-group cache).
    free: Vec<Arc<wgpu::Buffer>>,
    /// Dropped since last synchronize; moved to `free` after GPU drain.
    pending_free: Vec<Arc<wgpu::Buffer>>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct WgpuElemBgKey {
    shader_hash: u64,
    shader_len: usize,
    /// All storage buffer identities (excludes trailing uniform).
    storage_ptrs: Vec<usize>,
}

struct WgpuPendingDispatch {
    pipeline: Arc<WgpuCachedPipeline>,
    bind_group: wgpu::BindGroup,
    workgroups: (u32, u32, u32),
    dynamic_offsets: Vec<u32>,
}

struct WgpuActiveBatch {
    encoder: wgpu::CommandEncoder,
    retained_buffers: Vec<Arc<wgpu::Buffer>>,
    /// Compute dispatches held until encode_pending_dispatches (one pass).
    pending_dispatches: Vec<WgpuPendingDispatch>,
    dispatch_count: u32,
}

impl std::fmt::Debug for WgpuActiveBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuActiveBatch")
            .field("retained_buffers", &self.retained_buffers.len())
            .field("pending_dispatches", &self.pending_dispatches.len())
            .field("dispatch_count", &self.dispatch_count)
            .finish_non_exhaustive()
    }
}

struct WgpuPendingSubmission {
    retained_buffers: Vec<Arc<wgpu::Buffer>>,
    completed: Arc<AtomicBool>,
}

impl std::fmt::Debug for WgpuPendingSubmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuPendingSubmission")
            .field("retained_buffers", &self.retained_buffers.len())
            .field("completed", &self.completed.load(Ordering::Relaxed))
            .finish()
    }
}

fn wgpu_buffer_key(buffer: &wgpu::Buffer) -> usize {
    std::ptr::from_ref(buffer) as usize
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum WgpuBindingKindKey {
    Storage { read_only: bool },
    Uniform,
    /// Uniform with `has_dynamic_offset` (offset applied at set_bind_group).
    UniformDynamic,
}

/// Pipeline cache key — hash the WGSL source instead of cloning it on every
/// dispatch (coopmat shaders are multi-KB; string clone dominated host path).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct WgpuPipelineCacheKey {
    shader_hash: u64,
    shader_len: usize,
    entries: Vec<(u32, WgpuBindingKindKey)>,
}

fn wgpu_shader_cache_key(shader: &str) -> (u64, usize) {
    use std::hash::{Hash, Hasher};
    // Avoid full multi-KB DefaultHasher on every dispatch (coopmat WGSL), but
    // never key only on (ptr,len): heap-allocated template strings can reuse
    // the same address with different content after free (observed as wrong
    // pipelines on int binary / unary tests). Content-sample head+mid+tail.
    let len = shader.len();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    len.hash(&mut hasher);
    if len <= 512 {
        shader.hash(&mut hasher);
    } else {
        let head = 128.min(len);
        let tail = 128.min(len);
        let mid = len / 2;
        let mid0 = mid.saturating_sub(64);
        let mid1 = (mid + 64).min(len);
        shader[..head].hash(&mut hasher);
        shader[mid0..mid1].hash(&mut hasher);
        shader[len - tail..].hash(&mut hasher);
    }
    (hasher.finish(), len)
}

#[derive(Debug)]
struct WgpuCachedPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

#[derive(Debug, Clone)]
pub struct WgpuStorage {
    buffer: Arc<wgpu::Buffer>,
    device: WgpuDevice,
    count: usize,
    dtype: DType,
}

impl Drop for WgpuStorage {
    fn drop(&mut self) {
        // Hot-ring: schedule for reuse after next synchronize (GPU may still
        // hold the buffer via retained batch / bind-group cache).
        self.device.release_hot_ring_buffer(&self.buffer);
        // Recycle only when this is the last owner (not retained by a batch).
        if Arc::strong_count(&self.buffer) == 1 {
            self.device.recycle_storage_buffer(&self.buffer);
        }
    }
}

impl WgpuDevice {
    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }

    pub fn adapter_backend(&self) -> &str {
        &self.inner.adapter_backend
    }

    pub fn adapter_driver(&self) -> &str {
        &self.inner.adapter_driver
    }

    pub fn adapter_driver_info(&self) -> &str {
        &self.inner.adapter_driver_info
    }

    pub fn adapter_pci_bus_id(&self) -> &str {
        &self.inner.adapter_pci_bus_id
    }

    pub fn shader_f64_enabled(&self) -> bool {
        wgpu_shader_f64_enabled(self)
    }

    fn advance_rand_seed(&self, count: usize) -> Result<u64> {
        let mut guard = self
            .inner
            .seed_value
            .write()
            .map_err(|_| Error::msg("wgpu seed lock poisoned"))?;
        let seed_at_call = *guard;
        *guard = seed_at_call.wrapping_add(count as u64);
        Ok(seed_at_call)
    }

    fn run_rand_kernel(
        &self,
        shape: &Shape,
        kernel_dtype: DType,
        shader: String,
        params_bytes: &[u8],
        label: &'static str,
    ) -> Result<WgpuStorage> {
        let count = shape.elem_count();
        let storage = unsafe { self.alloc_uninit(shape, kernel_dtype)? };
        let param_buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle-wgpu-rand-params"),
            size: params_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.inner
            .queue
            .write_buffer(&param_buffer, 0, params_bytes);
        let entries = [storage_entry(0, false), uniform_entry(1)];
        let bindings = [
            buffer_binding(0, &storage.buffer),
            buffer_binding(1, &param_buffer),
        ];
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        self.run_compute(&shader, &entries, &bindings, workgroups, label)?;
        Ok(storage)
    }

    fn rand_uniform_gpu(
        &self,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<WgpuStorage> {
        let count = shape.elem_count();
        if count == 0 {
            return self.zeros_impl(shape, dtype);
        }
        let seed = self.advance_rand_seed(count)?;
        let kernel_dtype = match dtype {
            DType::F32 => DType::F32,
            DType::F16 if self.inner.features.contains(wgpu::Features::SHADER_F16) => DType::F16,
            DType::F16 | DType::BF16 | DType::F64 => DType::F32,
            dt => return Err(Error::UnsupportedDTypeForOp(dt, "rand_uniform").bt()),
        };
        let params = RandParams {
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
            min_val: min as f32,
            max_val: max as f32,
            ne: count.try_into()?,
        };
        let shader =
            candle_wgpu_kernels::rand_uniform_shader(wgpu_kernel_dtype(kernel_dtype)?, WG_SIZE)
                .ok_or_else(|| {
                    Error::Msg("wgpu shader rand_uniform.wgsl not embedded".into()).bt()
                })?;
        let storage = self.run_rand_kernel(
            shape,
            kernel_dtype,
            shader,
            any_as_bytes(&params),
            "candle-wgpu-rand-uniform",
        )?;
        if kernel_dtype == dtype {
            Ok(storage)
        } else {
            storage.to_dtype(&Layout::contiguous(shape), dtype)
        }
    }

    fn rand_normal_gpu(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<WgpuStorage> {
        let count = shape.elem_count();
        if count == 0 {
            return self.zeros_impl(shape, dtype);
        }
        let seed = self.advance_rand_seed(count)?;
        let kernel_dtype = match dtype {
            DType::F32 => DType::F32,
            DType::F16 if self.inner.features.contains(wgpu::Features::SHADER_F16) => DType::F16,
            DType::F16 | DType::BF16 | DType::F64 => DType::F32,
            dt => return Err(Error::UnsupportedDTypeForOp(dt, "rand_normal").bt()),
        };
        let params = RandNormalParams {
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
            mean: mean as f32,
            std: std as f32,
            ne: count.try_into()?,
        };
        let shader =
            candle_wgpu_kernels::rand_normal_shader(wgpu_kernel_dtype(kernel_dtype)?, WG_SIZE)
                .ok_or_else(|| {
                    Error::Msg("wgpu shader rand_normal.wgsl not embedded".into()).bt()
                })?;
        let storage = self.run_rand_kernel(
            shape,
            kernel_dtype,
            shader,
            any_as_bytes(&params),
            "candle-wgpu-rand-normal",
        )?;
        if kernel_dtype == dtype {
            Ok(storage)
        } else {
            storage.to_dtype(&Layout::contiguous(shape), dtype)
        }
    }
}

fn wgpu_shader_f16_enabled(device: &WgpuDevice) -> bool {
    device.inner.features.contains(wgpu::Features::SHADER_F16)
}

fn wgpu_shader_f64_enabled(device: &WgpuDevice) -> bool {
    device.inner.features.contains(wgpu::Features::SHADER_F64)
}

fn wgpu_f16_emulates_f32(device: &WgpuDevice, dtype: DType) -> bool {
    dtype == DType::F16 && !wgpu_shader_f16_enabled(device)
}

fn unsupported(op: &'static str) -> Error {
    Error::Msg(format!("wgpu backend op {op} not implemented")).bt()
}

fn byte_len(dtype: DType, count: usize, op: &'static str) -> Result<usize> {
    let size = dtype.size_in_bytes();
    if size == 0 {
        Err(Error::UnsupportedDTypeForOp(dtype, op).bt())
    } else {
        Ok(size * count)
    }
}

fn wgpu_copy_size(size: usize) -> usize {
    size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
}

fn typed_as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

fn any_as_bytes<T>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

fn bytes_to_vec<T: Copy>(bytes: &[u8], count: usize) -> Result<Vec<T>> {
    let byte_len = count * std::mem::size_of::<T>();
    if bytes.len() != byte_len {
        crate::bail!("invalid byte length {}, expected {byte_len}", bytes.len())
    }
    let mut out = Vec::<T>::with_capacity(count);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), byte_len);
        out.set_len(count);
    }
    Ok(out)
}

fn cpu_storage_to_bytes(storage: &CpuStorage) -> Result<(DType, usize, Vec<u8>)> {
    macro_rules! typed {
        ($storage:expr, $dtype:expr) => {{
            let bytes = typed_as_bytes($storage).to_vec();
            Ok(($dtype, $storage.len(), bytes))
        }};
    }
    match storage {
        CpuStorage::U8(v) => typed!(v, DType::U8),
        CpuStorage::U32(v) => typed!(v, DType::U32),
        CpuStorage::I16(v) => typed!(v, DType::I16),
        CpuStorage::I32(v) => typed!(v, DType::I32),
        CpuStorage::I64(v) => typed!(v, DType::I64),
        CpuStorage::BF16(v) => typed!(v, DType::BF16),
        CpuStorage::F16(v) => typed!(v, DType::F16),
        CpuStorage::F32(v) => typed!(v, DType::F32),
        CpuStorage::F64(v) => typed!(v, DType::F64),
        CpuStorage::F8E4M3(v) => typed!(v, DType::F8E4M3),
        CpuStorage::F6E2M3(_) => {
            Err(Error::UnsupportedDTypeForOp(DType::F6E2M3, "wgpu upload").bt())
        }
        CpuStorage::F6E3M2(_) => {
            Err(Error::UnsupportedDTypeForOp(DType::F6E3M2, "wgpu upload").bt())
        }
        CpuStorage::F4(_) => Err(Error::UnsupportedDTypeForOp(DType::F4, "wgpu upload").bt()),
        CpuStorage::F8E8M0(v) => typed!(v, DType::F8E8M0),
    }
}

fn bytes_to_cpu_storage(dtype: DType, count: usize, bytes: &[u8]) -> Result<CpuStorage> {
    match dtype {
        DType::U8 => Ok(CpuStorage::U8(bytes_to_vec(bytes, count)?)),
        DType::U32 => Ok(CpuStorage::U32(bytes_to_vec(bytes, count)?)),
        DType::I16 => Ok(CpuStorage::I16(bytes_to_vec(bytes, count)?)),
        DType::I32 => Ok(CpuStorage::I32(bytes_to_vec(bytes, count)?)),
        DType::I64 => Ok(CpuStorage::I64(bytes_to_vec(bytes, count)?)),
        DType::BF16 => Ok(CpuStorage::BF16(bytes_to_vec(bytes, count)?)),
        DType::F16 => Ok(CpuStorage::F16(bytes_to_vec(bytes, count)?)),
        DType::F32 => Ok(CpuStorage::F32(bytes_to_vec(bytes, count)?)),
        DType::F64 => Ok(CpuStorage::F64(bytes_to_vec(bytes, count)?)),
        DType::F8E4M3 => Ok(CpuStorage::F8E4M3(bytes_to_vec(bytes, count)?)),
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 => {
            Err(Error::UnsupportedDTypeForOp(dtype, "wgpu download").bt())
        }
        DType::F8E8M0 => Ok(CpuStorage::F8E8M0(bytes_to_vec(bytes, count)?)),
    }
}

fn scalar_bytes(scalar: crate::scalar::Scalar, dtype: DType, op: &'static str) -> Result<Vec<u8>> {
    if scalar.dtype() != dtype {
        crate::bail!(
            "scalar dtype {:?} does not match storage dtype {:?}",
            scalar.dtype(),
            dtype
        )
    }
    let bytes = match scalar {
        crate::scalar::Scalar::U8(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::U32(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::I16(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::I32(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::I64(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::BF16(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::F16(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::F32(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::F64(v) => any_as_bytes(&v).to_vec(),
        crate::scalar::Scalar::F8E4M3(v) => any_as_bytes(&v).to_vec(),
    };
    if bytes.is_empty() {
        Err(Error::UnsupportedDTypeForOp(dtype, op).bt())
    } else {
        Ok(bytes)
    }
}

fn scalar_raw_words(
    scalar: crate::scalar::Scalar,
    dtype: DType,
    op: &'static str,
) -> Result<[u32; 2]> {
    let bytes = scalar_bytes(scalar, dtype, op)?;
    if bytes.len() > 8 {
        return Err(Error::UnsupportedDTypeForOp(dtype, op).bt());
    }
    let mut words = [0u32; 2];
    for (idx, byte) in bytes.into_iter().enumerate() {
        words[idx / 4] |= (byte as u32) << (8 * (idx % 4));
    }
    Ok(words)
}

fn wgpu_padded_write_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut padded = bytes.to_vec();
    padded.resize(wgpu_copy_size(bytes.len()), 0);
    padded
}

impl WgpuDevice {
    pub fn transfer_to_device(&self, storage: &WgpuStorage) -> Result<WgpuStorage> {
        let cpu = storage.to_cpu_storage()?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn prune_buffer_registry(&self) {
        if let Ok(mut registry) = self.inner.buffer_registry.lock() {
            registry.retain(|_, weak| weak.strong_count() > 0);
        }
    }

    /// Encode deferred computes then submit the active batch before any
    /// standalone `queue.submit`. Without this, copies race unsubmitted work.
    fn flush_before_standalone_submit(&self) -> Result<()> {
        // Encode any deferred dispatches onto the encoder first, then submit.
        {
            let mut slot = self
                .inner
                .active_batch
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            if let Some(batch) = slot.as_mut() {
                Self::encode_pending_dispatches(batch);
            }
        }
        let _ = self.flush_active_batch("standalone_submit")?;
        Ok(())
    }

    fn create_storage_buffer(&self, size: usize, label: &'static str) -> wgpu::Buffer {
        self.create_storage_buffer_arc(size, label).as_ref().clone()
    }

    /// Allocate (or recycle) a storage buffer, preserving the same `Arc` identity
    /// when recycled so bind-group cache keys stay stable across pool reuse.
    fn create_storage_buffer_arc(&self, size: usize, label: &'static str) -> Arc<wgpu::Buffer> {
        // Do not flush the active batch on every small alloc — that forced one
        // submit per output tensor. Large allocs still flush to bound memory.
        // Standalone copy/readback paths must call flush_before_standalone_submit.
        const LARGE_ALLOC: usize = 16 * 1024 * 1024;
        /// Hot ring covers common elementwise outputs (1024² f32 = 4MiB).
        const HOT_RING_MAX: usize = 32;
        let key = wgpu_copy_size(size) as u64;
        // Permanent hot ring only for the microbench elementwise size class
        // (1024² f32 = 4MiB). Broader ranges aliased matmul temporaries.
        const HOT_EXACT: u64 = 4 * 1024 * 1024;
        if key == HOT_EXACT || key == wgpu_copy_size(HOT_EXACT as usize) as u64 {
            if let Ok(mut rings) = self.inner.hot_rings.lock() {
                let ring = rings.entry(key).or_insert_with(|| WgpuHotRing {
                    buffers: Vec::new(),
                    free: Vec::new(),
                    pending_free: Vec::new(),
                });
                if let Some(arc) = ring.free.pop() {
                    return arc;
                }
                if ring.buffers.len() < HOT_RING_MAX {
                    let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("candle-wgpu-hot-ring"),
                        size: key,
                        usage: wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    let arc = Arc::new(buffer);
                    ring.buffers.push(Arc::clone(&arc));
                    return arc;
                }
            }
        }
        if key >= LARGE_ALLOC as u64 {
            let has_work = self
                .inner
                .active_batch
                .lock()
                .ok()
                .and_then(|slot| slot.as_ref().map(|b| b.dispatch_count > 0))
                .unwrap_or(false);
            if has_work {
                let _ = self.flush_active_batch("large_alloc");
            }
            let _ = self.cleanup_pending_submissions(false);
            self.prune_buffer_registry();
        }
        if let Ok(mut pool) = self.inner.storage_buffer_pool.lock() {
            if let Some(bucket) = pool.get_mut(&key) {
                if let Some(arc) = bucket.pop() {
                    return arc;
                }
            }
        }
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: key,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    /// After GPU drain, promote pending_free → free for hot-ring reuse.
    /// Reverse pending so LIFO matches first-alloc order (better BG-cache hits
    /// when sample N reuses the same dst sequence as sample 1).
    fn reset_hot_rings_if_idle(&self) {
        if let Ok(mut rings) = self.inner.hot_rings.lock() {
            for ring in rings.values_mut() {
                while let Some(b) = ring.pending_free.pop() {
                    ring.free.push(b);
                }
            }
        }
    }

    fn release_hot_ring_buffer(&self, buffer: &Arc<wgpu::Buffer>) {
        if let Ok(mut rings) = self.inner.hot_rings.lock() {
            for ring in rings.values_mut() {
                if ring.buffers.iter().any(|b| Arc::ptr_eq(b, buffer)) {
                    ring.pending_free.push(Arc::clone(buffer));
                    return;
                }
            }
        }
    }

    /// Return a free storage buffer to the size-class pool (last-owner path).
    fn recycle_storage_buffer(&self, buffer: &Arc<wgpu::Buffer>) {
        // Hot-ring buffers are permanent; never put them in the free pool.
        if let Ok(rings) = self.inner.hot_rings.lock() {
            for ring in rings.values() {
                if ring.buffers.iter().any(|b| Arc::ptr_eq(b, buffer)) {
                    return;
                }
            }
        }
        let key = buffer.size();
        if let Ok(mut pool) = self.inner.storage_buffer_pool.lock() {
            let bucket = pool.entry(key).or_default();
            if bucket.len() < 64 {
                bucket.push(Arc::clone(buffer));
            }
        }
    }

    /// Write params into a ring-buffered uniform and return the buffer handle.
    /// Ring size is large enough for several in-flight batches before reuse.
    /// Prefer [`Self::write_uniform_slot`] + dynamic offsets for elementwise
    /// batches (bind-group reuse).
    fn write_uniform_params(&self, bytes: &[u8]) -> Result<wgpu::Buffer> {
        // Large enough for elementwise batch20 × several in-flight submits
        // before GPU completion reclaims a slot's contents.
        const RING: usize = 128;
        const SLOT: u64 = 256;
        if bytes.len() as u64 > SLOT {
            // Oversized uniforms: allocate once (rare).
            let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-uniform-large"),
                size: wgpu_copy_size(bytes.len()) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.inner.queue.write_buffer(&buf, 0, bytes);
            return Ok(buf);
        }
        let mut ring = self
            .inner
            .uniform_ring
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        if ring.0.is_empty() {
            ring.0 = (0..RING)
                .map(|_| {
                    self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("candle-wgpu-uniform-ring"),
                        size: SLOT,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            ring.1 = 0;
        }
        let idx = ring.1 % ring.0.len();
        ring.1 = idx + 1;
        let buf = ring.0[idx].clone();
        self.inner.queue.write_buffer(&buf, 0, bytes);
        Ok(buf)
    }

    /// Write params into a single large uniform buffer at a rotating slot.
    /// Returns `(buffer, dynamic_offset)` for use with `uniform_entry_dyn` /
    /// `uniform_binding_dyn` and `set_bind_group(..., &[offset])`.
    fn write_uniform_slot(&self, bytes: &[u8]) -> Result<(wgpu::Buffer, u32)> {
        const RING_SLOTS: usize = 256;
        let slot = self.inner.uniform_dyn_slot;
        if bytes.len() as u64 > slot {
            // Oversized: fall back to a dedicated buffer (no dynamic offset).
            let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-uniform-large"),
                size: wgpu_copy_size(bytes.len()) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.inner.queue.write_buffer(&buf, 0, bytes);
            return Ok((buf, 0));
        }
        let mut guard = self
            .inner
            .uniform_dyn
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        if guard.is_none() {
            let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-uniform-dyn"),
                size: slot * RING_SLOTS as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *guard = Some((buf, 0));
        }
        let (buf, cursor) = guard.as_mut().unwrap();
        let idx = *cursor % RING_SLOTS;
        *cursor = idx + 1;
        let offset = idx as u64 * slot;
        // Bind group covers the full slot; pad so the entire range is defined.
        if (bytes.len() as u64) < slot {
            let mut padded = vec![0u8; slot as usize];
            padded[..bytes.len()].copy_from_slice(bytes);
            self.inner.queue.write_buffer(buf, offset, &padded);
        } else {
            self.inner.queue.write_buffer(buf, offset, bytes);
        }
        Ok((buf.clone(), offset as u32))
    }

    fn create_zeroed_storage_buffer(&self, size: usize, label: &'static str) -> wgpu::Buffer {
        // ponytail: 1MiB chunks + submit — caps in-flight write_buffer staging vs vec![0; nbytes].
        const ZERO_CHUNK: usize = 1024 * 1024;
        let copy_size = wgpu_copy_size(size);
        let buffer = self.create_storage_buffer(size, label);
        let zeros = vec![0u8; ZERO_CHUNK];
        let mut off = 0usize;
        while off < copy_size {
            let n = ZERO_CHUNK.min(copy_size - off);
            self.inner
                .queue
                .write_buffer(&buffer, off as u64, &zeros[..n]);
            off += n;
        }
        let _ = self.flush_before_standalone_submit();
        let _ = self.inner.queue.submit([]);
        buffer
    }

    fn trim_pipeline_cache(&self) -> Result<()> {
        let mut cache = self
            .inner
            .pipeline_cache
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        cache.clear();
        Ok(())
    }

    fn register_buffer(&self, buffer: wgpu::Buffer) -> Arc<wgpu::Buffer> {
        self.register_buffer_arc(Arc::new(buffer))
    }

    fn register_buffer_arc(&self, arc: Arc<wgpu::Buffer>) -> Arc<wgpu::Buffer> {
        let key = wgpu_buffer_key(arc.as_ref());
        if let Ok(mut registry) = self.inner.buffer_registry.lock() {
            registry.insert(key, Arc::downgrade(&arc));
        }
        arc
    }

    fn upgrade_registered_buffer(&self, buffer: &wgpu::Buffer) -> Option<Arc<wgpu::Buffer>> {
        let key = wgpu_buffer_key(buffer);
        let registry = self.inner.buffer_registry.lock().ok()?;
        registry.get(&key).and_then(|weak| weak.upgrade())
    }

    fn retain_from_bindings(
        &self,
        bindings: &[wgpu::BindGroupEntry<'_>],
    ) -> Vec<Arc<wgpu::Buffer>> {
        let mut retained = Vec::new();
        let mut seen = HashSet::new();
        for entry in bindings {
            let wgpu::BindingResource::Buffer(buf) = &entry.resource else {
                continue;
            };
            let key = wgpu_buffer_key(buf.buffer);
            if seen.insert(key) {
                if let Some(arc) = self.upgrade_registered_buffer(buf.buffer) {
                    retained.push(arc);
                }
            }
        }
        retained
    }

    fn cleanup_pending_submissions(&self, wait: bool) -> Result<()> {
        if wait {
            self.inner
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(Error::wrap)?;
        } else {
            let _ = self.inner.device.poll(wgpu::PollType::Poll);
        }
        let mut pending = self
            .inner
            .pending_submissions
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        let mut keep = Vec::with_capacity(pending.len());
        for submission in pending.drain(..) {
            if submission.completed.load(Ordering::Acquire) {
                // GPU finished — recycle sole-owner storage buffers.
                for buf in submission.retained_buffers {
                    if Arc::strong_count(&buf) == 1 {
                        self.recycle_storage_buffer(&buf);
                    }
                }
            } else {
                keep.push(submission);
            }
        }
        *pending = keep;
        if wait {
            self.prune_buffer_registry();
        }
        Ok(())
    }

    const MAX_BATCH_DISPATCHES: u32 = 32;

    fn begin_active_batch(&self) -> Result<WgpuActiveBatch> {
        let encoder = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("candle-wgpu-batch"),
            });
        Ok(WgpuActiveBatch {
            encoder,
            retained_buffers: Vec::new(),
            pending_dispatches: Vec::new(),
            dispatch_count: 0,
        })
    }

    /// Fold deferred computes into one compute pass on the batch encoder.
    fn encode_pending_dispatches(batch: &mut WgpuActiveBatch) {
        if batch.pending_dispatches.is_empty() {
            return;
        }
        {
            let mut pass = batch
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("candle-wgpu-batch-pass"),
                    timestamp_writes: None,
                });
            // Skip redundant set_pipeline when consecutive dispatches share one
            // (common for elementwise batch20 of the same op).
            let mut last_pipe: usize = 0;
            for d in &batch.pending_dispatches {
                let pipe_key = std::ptr::from_ref(d.pipeline.as_ref()) as usize;
                if pipe_key != last_pipe {
                    pass.set_pipeline(&d.pipeline.pipeline);
                    last_pipe = pipe_key;
                }
                pass.set_bind_group(0, &d.bind_group, &d.dynamic_offsets);
                pass.dispatch_workgroups(d.workgroups.0, d.workgroups.1, d.workgroups.2);
            }
        }
        batch.pending_dispatches.clear();
    }

    fn retain_buffers_into(batch: &mut WgpuActiveBatch, buffers: Vec<Arc<wgpu::Buffer>>) {
        let mut seen: HashSet<usize> = batch
            .retained_buffers
            .iter()
            .map(|buffer| wgpu_buffer_key(buffer.as_ref()))
            .collect();
        for buffer in buffers {
            let key = wgpu_buffer_key(buffer.as_ref());
            if seen.insert(key) {
                batch.retained_buffers.push(buffer);
            }
        }
    }

    fn flush_active_batch(&self, reason: &'static str) -> Result<bool> {
        let batch = {
            let mut slot = self
                .inner
                .active_batch
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            slot.take()
        };
        let Some(batch) = batch else {
            return Ok(false);
        };
        if batch.dispatch_count == 0 && batch.pending_dispatches.is_empty() {
            let mut slot = self
                .inner
                .active_batch
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            if slot.is_none() {
                *slot = Some(batch);
            }
            return Ok(false);
        }
        let mut batch = batch;
        Self::encode_pending_dispatches(&mut batch);
        let completed = Arc::new(AtomicBool::new(false));
        let done = completed.clone();
        self.inner
            .queue
            .on_submitted_work_done(move || done.store(true, Ordering::Release));
        self.inner.queue.submit([batch.encoder.finish()]);
        const MAX_IN_FLIGHT_SUBMISSIONS: usize = 32;
        let mut pending = self
            .inner
            .pending_submissions
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        if pending.len() >= MAX_IN_FLIGHT_SUBMISSIONS {
            drop(pending);
            self.cleanup_pending_submissions(true)?;
            pending = self
                .inner
                .pending_submissions
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
        }
        pending.push(WgpuPendingSubmission {
            retained_buffers: batch.retained_buffers,
            completed,
        });
        let _ = reason;
        Ok(true)
    }

    fn ensure_active_batch(&self) -> Result<()> {
        let mut slot = self
            .inner
            .active_batch
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        if slot.is_none() {
            *slot = Some(self.begin_active_batch()?);
        }
        Ok(())
    }

    fn read_buffer(&self, buffer: &wgpu::Buffer, size: usize) -> Result<Vec<u8>> {
        let copy_size = wgpu_copy_size(size);
        // ponytail: sync + drop cached pipelines before staging alloc; one retry on map failure
        self.synchronize()?;
        let _ = self.trim_pipeline_cache();
        let try_read = || -> Result<Vec<u8>> {
            let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-readback"),
                size: copy_size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder =
                self.inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("candle-wgpu-readback"),
                    });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, copy_size as u64);
            // synchronize() already flushed; keep for call paths that skip it.
            let _ = self.flush_before_standalone_submit();
            self.inner.queue.submit([encoder.finish()]);

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            self.synchronize()?;
            rx.recv()
                .map_err(Error::wrap)?
                .map_err(Error::wrap)
                .map_err(|e| {
                    Error::Msg(format!("wgpu readback map_async failed (size={size}): {e}")).bt()
                })?;
            let mut data = slice.get_mapped_range().to_vec();
            data.truncate(size);
            staging.unmap();
            Ok(data)
        };
        match try_read() {
            Ok(data) => Ok(data),
            Err(_) => {
                self.synchronize()?;
                let _ = self.trim_pipeline_cache();
                try_read()
            }
        }
    }

    fn run_compute(
        &self,
        shader: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        bindings: &[wgpu::BindGroupEntry<'_>],
        workgroups: u32,
        label: &'static str,
    ) -> Result<()> {
        let max_per_dim = wgpu_dispatch_wg_cap(self);
        let (wg_x, wg_y) = compute_2d_workgroups(workgroups, max_per_dim);
        self.run_compute_xyz(shader, entries, bindings, (wg_x, wg_y, 1), &[], label)
    }

    fn run_compute_linear(
        &self,
        shader: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        bindings: &[wgpu::BindGroupEntry<'_>],
        total_items: u32,
        label: &'static str,
    ) -> Result<()> {
        let (wg_x, wg_y) = linear_dispatch_workgroups(self, total_items);
        self.run_compute_xyz(shader, entries, bindings, (wg_x, wg_y, 1), &[], label)
    }

    fn run_compute_linear_dyn(
        &self,
        shader: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        bindings: &[wgpu::BindGroupEntry<'_>],
        total_items: u32,
        dynamic_offsets: &[u32],
        label: &'static str,
    ) -> Result<()> {
        let (wg_x, wg_y) = linear_dispatch_workgroups(self, total_items);
        self.run_compute_xyz(
            shader,
            entries,
            bindings,
            (wg_x, wg_y, 1),
            dynamic_offsets,
            label,
        )
    }

    fn run_compute_xyz(
        &self,
        shader: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        bindings: &[wgpu::BindGroupEntry<'_>],
        workgroups: (u32, u32, u32),
        dynamic_offsets: &[u32],
        label: &'static str,
    ) -> Result<()> {
        let (shader_hash, shader_len) = wgpu_shader_cache_key(shader);
        let entry_keys: Vec<(u32, WgpuBindingKindKey)> = entries
            .iter()
            .map(|entry| {
                let kind = match entry.ty {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset,
                        ..
                    } => {
                        if has_dynamic_offset {
                            // Storage dynamic offsets unused today.
                            WgpuBindingKindKey::Storage { read_only }
                        } else {
                            WgpuBindingKindKey::Storage { read_only }
                        }
                    }
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        ..
                    } => WgpuBindingKindKey::UniformDynamic,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        ..
                    } => WgpuBindingKindKey::Uniform,
                    _ => unreachable!("unsupported wgpu binding type for compute cache"),
                };
                (entry.binding, kind)
            })
            .collect();
        let cache_key = WgpuPipelineCacheKey {
            shader_hash,
            shader_len,
            entries: entry_keys,
        };
        let cached =
            {
                let mut cache = self
                    .inner
                    .pipeline_cache
                    .lock()
                    .map_err(|e| Error::wrap(e.to_string()))?;
                if let Some(cached) = cache.get(&cache_key) {
                    cached.clone()
                } else {
                    let module =
                        self.inner
                            .device
                            .create_shader_module(wgpu::ShaderModuleDescriptor {
                                label: Some(label),
                                source: wgpu::ShaderSource::Wgsl(shader.into()),
                            });
                    let bind_group_layout = self.inner.device.create_bind_group_layout(
                        &wgpu::BindGroupLayoutDescriptor {
                            label: Some(label),
                            entries,
                        },
                    );
                    let pipeline_layout =
                        self.inner
                            .device
                            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some(label),
                                bind_group_layouts: &[Some(&bind_group_layout)],
                                immediate_size: 0,
                            });
                    let pipeline = self.inner.device.create_compute_pipeline(
                        &wgpu::ComputePipelineDescriptor {
                            label: Some(label),
                            layout: Some(&pipeline_layout),
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        },
                    );
                    let cached = Arc::new(WgpuCachedPipeline {
                        bind_group_layout,
                        pipeline,
                    });
                    cache.insert(cache_key.clone(), cached.clone());
                    cached
                }
            };
        // Elementwise dyn-uniform path: cache bind groups by (shader, src, dst,
        // params). Hot-ring dst buffers keep stable Arc identity across
        // synchronize cycles so subsequent batches hit this cache.
        let bind_group = if !dynamic_offsets.is_empty() && bindings.len() >= 3 {
            let ptrs: Vec<usize> = bindings
                .iter()
                .filter_map(|e| match &e.resource {
                    wgpu::BindingResource::Buffer(bb) => {
                        Some(std::ptr::from_ref(bb.buffer) as usize)
                    }
                    other => {
                        let _ = other;
                        None
                    }
                })
                .collect();
            if ptrs.len() >= 3 {
                // Last ptr is the uniform params buffer; rest are storage.
                let storage_ptrs = ptrs[..ptrs.len() - 1].to_vec();
                let key = WgpuElemBgKey {
                    shader_hash: cache_key.shader_hash,
                    shader_len: cache_key.shader_len,
                    storage_ptrs,
                };
                let mut cache = self
                    .inner
                    .elem_bg_cache
                    .lock()
                    .map_err(|e| Error::wrap(e.to_string()))?;
                if let Some(bg) = cache.get(&key) {
                    self.inner
                        .elem_bg_hits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    bg.clone()
                } else {
                    self.inner
                        .elem_bg_misses
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let bg = self
                        .inner
                        .device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(label),
                            layout: &cached.bind_group_layout,
                            entries: bindings,
                        });
                    if cache.len() >= 256 {
                        let drop_n = cache.len() / 4;
                        let keys: Vec<_> = cache.keys().take(drop_n).cloned().collect();
                        for k in keys {
                            cache.remove(&k);
                        }
                    }
                    cache.insert(key, bg.clone());
                    bg
                }
            } else {
                self.inner
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(label),
                        layout: &cached.bind_group_layout,
                        entries: bindings,
                    })
            }
        } else {
            self.inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &cached.bind_group_layout,
                    entries: bindings,
                })
        };
        let batch_limit = {
            let slot = self
                .inner
                .active_batch
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            slot.as_ref()
                .map(|batch| batch.dispatch_count + 1 > Self::MAX_BATCH_DISPATCHES)
                .unwrap_or(false)
        };
        if batch_limit {
            self.flush_active_batch("batch_limit")?;
        }
        self.ensure_active_batch()?;
        let retained_buffers = self.retain_from_bindings(bindings);
        {
            let mut slot = self
                .inner
                .active_batch
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            let batch = slot
                .as_mut()
                .ok_or_else(|| Error::msg("wgpu active batch missing after ensure"))?;
            Self::retain_buffers_into(batch, retained_buffers);
            // Defer into one compute pass at encode/flush time — fewer
            // begin_compute_pass calls on elementwise batches.
            batch.pending_dispatches.push(WgpuPendingDispatch {
                pipeline: cached,
                bind_group,
                workgroups,
                dynamic_offsets: dynamic_offsets.to_vec(),
            });
            batch.dispatch_count += 1;
        }
        Ok(())
    }
}

fn dims4(layout: &Layout) -> Result<([u32; 4], [u32; 4])> {
    if layout.dims().len() > 4 {
        crate::bail!("wgpu backend supports up to rank-4 tensors for this op")
    }
    let mut dims = [1u32; 4];
    let mut strides = [0u32; 4];
    for (idx, dim) in layout.dims().iter().rev().enumerate() {
        dims[idx] = (*dim).try_into()?;
    }
    for (idx, stride) in layout.stride().iter().rev().enumerate() {
        strides[idx] = (*stride).try_into()?;
    }
    Ok((dims, strides))
}

fn contiguous_strides(dims: [u32; 4]) -> [u32; 4] {
    [1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]]
}

fn next_power_of_two_u32(value: usize, op: &'static str) -> Result<u32> {
    value
        .checked_next_power_of_two()
        .ok_or_else(|| Error::Msg(format!("wgpu backend op {op} dimension overflow")).bt())?
        .try_into()
        .map_err(Error::wrap)
}

fn compute_2d_workgroups(total_wg: u32, max_per_dim: u32) -> (u32, u32) {
    if total_wg == 0 {
        return (1, 1);
    }
    let max = max_per_dim.clamp(1, WGPU_DISPATCH_WG_CAP);
    let mut wg_x = total_wg.min(max);
    let mut wg_y = total_wg.div_ceil(wg_x);
    if wg_y > max {
        wg_y = max;
        wg_x = total_wg.div_ceil(wg_y);
    }
    (wg_x.max(1), wg_y.max(1))
}

fn linear_dispatch_workgroups(device: &WgpuDevice, total_items: u32) -> (u32, u32) {
    let total_wg = total_items.div_ceil(WG_SIZE);
    let max_per_dim = wgpu_dispatch_wg_cap(device);
    compute_2d_workgroups(total_wg, max_per_dim)
}

fn nearest_interp_weights(in_size: usize, out_size: usize) -> Vec<f32> {
    let mut weights = vec![0f32; in_size * out_size];
    let scale = in_size as f64 / out_size as f64;
    for out_idx in 0..out_size {
        let src_idx = usize::min(in_size - 1, (out_idx as f64 * scale) as usize);
        weights[src_idx * out_size + out_idx] = 1.0;
    }
    weights
}

fn bilinear_interp_weights(
    in_size: usize,
    out_size: usize,
    align_corners: bool,
    scale_factor: Option<f64>,
) -> Vec<f32> {
    let scale = if align_corners {
        if out_size > 1 {
            (in_size - 1) as f64 / (out_size - 1) as f64
        } else {
            0.0
        }
    } else if let Some(scale_factor) = scale_factor {
        1.0 / scale_factor
    } else {
        in_size as f64 / out_size as f64
    };
    let mut weights = vec![0f32; in_size * out_size];
    for out_idx in 0..out_size {
        let src = if align_corners {
            scale * out_idx as f64
        } else {
            scale * (out_idx as f64 + 0.5) - 0.5
        };
        let src = src.max(0.0);
        let idx0 = src.floor() as usize;
        let idx1 = (idx0 + 1).min(in_size - 1);
        let weight = (src - idx0 as f64).clamp(0.0, 1.0) as f32;
        weights[idx0 * out_size + out_idx] += 1.0 - weight;
        weights[idx1 * out_size + out_idx] += weight;
    }
    weights
}

fn wgpu_kernel_dtype(dtype: DType) -> Result<candle_wgpu_kernels::DType> {
    match dtype {
        DType::F32 => Ok(candle_wgpu_kernels::DType::F32),
        DType::F16 => Ok(candle_wgpu_kernels::DType::F16),
        _ => Err(Error::UnsupportedDTypeForOp(dtype, "wgpu shader").bt()),
    }
}

fn wgpu_quantized_dtype(dtype: GgmlDType) -> Result<candle_wgpu_kernels::QuantizedDType> {
    match dtype {
        GgmlDType::Q4_0 => Ok(candle_wgpu_kernels::QuantizedDType::Q4_0),
        GgmlDType::Q4_1 => Ok(candle_wgpu_kernels::QuantizedDType::Q4_1),
        GgmlDType::Q5_0 => Ok(candle_wgpu_kernels::QuantizedDType::Q5_0),
        GgmlDType::Q5_1 => Ok(candle_wgpu_kernels::QuantizedDType::Q5_1),
        GgmlDType::Q8_0 => Ok(candle_wgpu_kernels::QuantizedDType::Q8_0),
        GgmlDType::Q8_1 => Ok(candle_wgpu_kernels::QuantizedDType::Q8_1),
        GgmlDType::Q2K => Ok(candle_wgpu_kernels::QuantizedDType::Q2_K),
        GgmlDType::Q3K => Ok(candle_wgpu_kernels::QuantizedDType::Q3_K),
        GgmlDType::Q4K => Ok(candle_wgpu_kernels::QuantizedDType::Q4_K),
        GgmlDType::Q5K => Ok(candle_wgpu_kernels::QuantizedDType::Q5_K),
        GgmlDType::Q6K => Ok(candle_wgpu_kernels::QuantizedDType::Q6_K),
        GgmlDType::Q8K => Ok(candle_wgpu_kernels::QuantizedDType::Q8_K),
        other => crate::bail!("wgpu backend quantized dtype {other:?} is not supported"),
    }
}

fn unary_shader(op: &str, dtype: DType) -> Result<String> {
    if op == "recip" {
        return Ok(custom_unary_wgsl("1.0 / x"));
    }
    if op == "erf" {
        return Ok(erf_unary_wgsl());
    }
    let op = match op {
        "abs" => candle_wgpu_kernels::UnaryOp::Abs,
        "ceil" => candle_wgpu_kernels::UnaryOp::Ceil,
        "clamp" => candle_wgpu_kernels::UnaryOp::Clamp,
        "cos" => candle_wgpu_kernels::UnaryOp::Cos,
        "elu" => candle_wgpu_kernels::UnaryOp::Elu,
        "exp" => candle_wgpu_kernels::UnaryOp::Exp,
        "floor" => candle_wgpu_kernels::UnaryOp::Floor,
        "gelu" => candle_wgpu_kernels::UnaryOp::Gelu,
        "gelu_erf" => candle_wgpu_kernels::UnaryOp::GeluErf,
        "log" => candle_wgpu_kernels::UnaryOp::Log,
        "neg" => candle_wgpu_kernels::UnaryOp::Neg,
        "relu" => candle_wgpu_kernels::UnaryOp::Relu,
        "round" => candle_wgpu_kernels::UnaryOp::Round,
        "sign" => candle_wgpu_kernels::UnaryOp::Sgn,
        "sigmoid" => candle_wgpu_kernels::UnaryOp::Sigmoid,
        "silu" => candle_wgpu_kernels::UnaryOp::Silu,
        "sin" => candle_wgpu_kernels::UnaryOp::Sin,
        "sqr" => candle_wgpu_kernels::UnaryOp::Square,
        "sqrt" => candle_wgpu_kernels::UnaryOp::Sqrt,
        "tanh" => candle_wgpu_kernels::UnaryOp::Tanh,
        _ => return Err(Error::Msg(format!("wgpu backend op {op} not implemented")).bt()),
    };
    Ok(candle_wgpu_kernels::shader(
        candle_wgpu_kernels::ShaderOp::Unary(op),
        wgpu_kernel_dtype(dtype)?,
        WG_SIZE,
    ))
}

fn binary_shader(op: &str, dtype: DType) -> Result<String> {
    if dtype == DType::BF16 {
        return Ok(bf16_binary_wgsl(op));
    }
    if matches!(dtype, DType::U8 | DType::U32 | DType::I32 | DType::I64) {
        return custom_int_binary_wgsl(op, dtype);
    }
    if op == "maximum" {
        if !matches!(dtype, DType::F32 | DType::F16) {
            return Err(Error::UnsupportedDTypeForOp(dtype, "wgpu maximum").bt());
        }
        return custom_binary_wgsl("max(a, b)", dtype);
    }
    if op == "minimum" {
        if !matches!(dtype, DType::F32 | DType::F16) {
            return Err(Error::UnsupportedDTypeForOp(dtype, "wgpu minimum").bt());
        }
        return custom_binary_wgsl("min(a, b)", dtype);
    }
    let op = match op {
        "add" => candle_wgpu_kernels::BinaryOp::Add,
        "div" => candle_wgpu_kernels::BinaryOp::Div,
        "mul" => candle_wgpu_kernels::BinaryOp::Mul,
        "sub" => candle_wgpu_kernels::BinaryOp::Sub,
        _ => return Err(Error::Msg(format!("wgpu backend op {op} not implemented")).bt()),
    };
    Ok(candle_wgpu_kernels::shader(
        candle_wgpu_kernels::ShaderOp::Binary(op),
        wgpu_kernel_dtype(dtype)?,
        WG_SIZE,
    ))
}

// Integer binary ops for the dtypes the stock binary.wgsl cannot express.
// U32 is native; U8 packs four bytes per u32 word (one thread per output
// word); I64 is emulated with lo/hi u32 word pairs and carry arithmetic.
// Generated last-dim integer reduction kernel, one invocation per output row.
// U32 is native; U8 elements are unpacked from packed bytes and the result is
// repacked four-per-word; I64 elements are lo/hi u32 word pairs with carry
// arithmetic for sum and signed compares for extrema. Argmax/argmin return a
// contiguous u32 index buffer (first index on ties, matching CPU/CUDA).
fn int_reduce_wgsl(op: ReduceOp, dtype: DType) -> Result<String> {
    let is_arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);
    let want_max = matches!(op, ReduceOp::Max | ReduceOp::ArgMax);

    // All integer buffers are addressed as u32 words (U8 packed, I64 lo/hi).
    let src_decl = "@group(0) @binding(0) var<storage, read> src: array<u32>;";
    let dst_decl = "@group(0) @binding(1) var<storage, read_write> dst: array<u32>;";

    // Body computes one output element for row `r` over columns [0, kx).
    let body = match dtype {
        DType::U32 => {
            if is_arg {
                let cmp = if want_max { ">" } else { "<" };
                format!(
                    r#"
    var best = src[r * params.kx];
    var bidx: u32 = 0u;
    for (var c: u32 = 1u; c < params.kx; c = c + 1u) {{
        let v = src[r * params.kx + c];
        if (v {cmp} best) {{ best = v; bidx = c; }}
    }}
    dst[r] = bidx;
"#
                )
            } else {
                match op {
                    ReduceOp::Sum => r#"
    var acc: u32 = 0u;
    for (var c: u32 = 0u; c < params.kx; c = c + 1u) {
        acc = acc + src[r * params.kx + c];
    }
    dst[r] = acc;
"#
                    .to_string(),
                    _ => {
                        let cmp = if want_max { ">" } else { "<" };
                        format!(
                            r#"
    var acc = src[r * params.kx];
    for (var c: u32 = 1u; c < params.kx; c = c + 1u) {{
        let v = src[r * params.kx + c];
        if (v {cmp} acc) {{ acc = v; }}
    }}
    dst[r] = acc;
"#
                        )
                    }
                }
            }
        }
        DType::U8 => {
            if is_arg {
                // Source byte at logical index i: src[i/4] >> (8*(i%4)) & 0xff.
                let cmp = if want_max { ">" } else { "<" };
                format!(
                    r#"
    let bi0 = r * params.kx; var best = (src[bi0 / 4u] >> (8u * (bi0 % 4u))) & 0xffu;
    var bidx: u32 = 0u;
    for (var c: u32 = 1u; c < params.kx; c = c + 1u) {{
        let bi = r * params.kx + c; let bv = (src[bi / 4u] >> (8u * (bi % 4u))) & 0xffu;
        if (bv {cmp} best) {{ best = bv; bidx = c; }}
    }}
    dst[r] = bidx;
"#
                )
            } else {
                // Value result is one byte per row; pack four rows per output
                // word so concurrent invocations never share a destination word.
                return Ok(int_reduce_u8_packed_wgsl(want_max, op));
            }
        }
        DType::I64 => {
            if is_arg {
                let take = if want_max {
                    "(hi_v > bitcast<i32>(best_hi)) || (hi_v == bitcast<i32>(best_hi) && lo_v > best_lo)"
                } else {
                    "(hi_v < bitcast<i32>(best_hi)) || (hi_v == bitcast<i32>(best_hi) && lo_v < best_lo)"
                };
                format!(
                    r#"
    var best_lo = src[2u * (r * params.kx)];
    var best_hi = src[2u * (r * params.kx) + 1u];
    var bidx: u32 = 0u;
    for (var c: u32 = 1u; c < params.kx; c = c + 1u) {{
        let e = r * params.kx + c;
        let lo_v = src[2u * e]; let hi_v = bitcast<i32>(src[2u * e + 1u]);
        if ({take}) {{ best_lo = lo_v; best_hi = src[2u * e + 1u]; bidx = c; }}
    }}
    dst[r] = bidx;
"#
                )
            } else {
                match op {
                    ReduceOp::Sum => r#"
    var lo: u32 = 0u; var hi: u32 = 0u;
    for (var c: u32 = 0u; c < params.kx; c = c + 1u) {
        let e = r * params.kx + c;
        let a_lo = src[2u * e]; let a_hi = src[2u * e + 1u];
        let nlo = lo + a_lo;
        let carry = select(0u, 1u, nlo < lo);
        lo = nlo; hi = hi + a_hi + carry;
    }
    dst[2u * r] = lo; dst[2u * r + 1u] = hi;
"#
                    .to_string(),
                    _ => {
                        let take = if want_max {
                            "(hi_v > bitcast<i32>(acc_hi)) || (hi_v == bitcast<i32>(acc_hi) && lo_v > acc_lo)"
                        } else {
                            "(hi_v < bitcast<i32>(acc_hi)) || (hi_v == bitcast<i32>(acc_hi) && lo_v < acc_lo)"
                        };
                        format!(
                            r#"
    var acc_lo = src[2u * (r * params.kx)];
    var acc_hi = src[2u * (r * params.kx) + 1u];
    for (var c: u32 = 1u; c < params.kx; c = c + 1u) {{
        let e = r * params.kx + c;
        let lo_v = src[2u * e]; let hi_v = bitcast<i32>(src[2u * e + 1u]);
        if ({take}) {{ acc_lo = lo_v; acc_hi = src[2u * e + 1u]; }}
    }}
    dst[2u * r] = acc_lo; dst[2u * r + 1u] = acc_hi;
"#
                        )
                    }
                }
            }
        }
        other => return Err(Error::UnsupportedDTypeForOp(other, "wgpu int reduce").bt()),
    };

    Ok(format!(
        r#"
struct Params {{ rows: u32, kx: u32, }};
{src_decl}
{dst_decl}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let r = gid.x;
    if (r >= params.rows) {{ return; }}
{body}
}}
"#
    ))
}

// U8 sum/extrema: one invocation per output WORD (four rows) so packed byte
// writes never race on a shared destination word.
fn int_reduce_u8_packed_wgsl(want_max: bool, op: ReduceOp) -> String {
    let per_row = match op {
        ReduceOp::Sum => r#"
        var acc: u32 = 0u;
        for (var c: u32 = 0u; c < params.kx; c = c + 1u) {
            let bi = rr * params.kx + c; let bv = (src[bi / 4u] >> (8u * (bi % 4u))) & 0xffu;
            acc = (acc + bv) & 0xffu;
        }
"#
        .to_string(),
        _ => {
            let cmp = if want_max { ">" } else { "<" };
            format!(
                r#"
        let bi0 = rr * params.kx; var acc = (src[bi0 / 4u] >> (8u * (bi0 % 4u))) & 0xffu;
        for (var c: u32 = 1u; c < params.kx; c = c + 1u) {{
            let bi = rr * params.kx + c; let bv = (src[bi / 4u] >> (8u * (bi % 4u))) & 0xffu;
            if (bv {cmp} acc) {{ acc = bv; }}
        }}
"#
            )
        }
    };
    format!(
        r#"
struct Params {{ rows: u32, kx: u32, }};
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let word = gid.x;
    let base = word * 4u;
    if (base >= params.rows) {{ return; }}
    var out: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
        let rr = base + lane;
        if (rr >= params.rows) {{ break; }}
        {per_row}
        out = out | ((acc & 0xffu) << (8u * lane));
    }}
    dst[word] = out;
}}
"#
    )
}

fn u8_gather_last_dim_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

struct Params {{
    offset_src: u32,
    offset_idx: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,
    idx1: u32,
    idx2: u32,
}};

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let base = gid.x * 4u;
    let ne = params.n_rows * params.ne2;
    if (base >= ne) {{
        return;
    }}

    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
        let logical = base + lane;
        var b = 0u;
        if (logical < ne) {{
            let left = logical / params.n_rows;
            let pos = logical - left * params.n_rows;
            let idx_pos = params.offset_idx + pos * params.stride_idx0 + left * params.stride_idx1;
            let src_col = idx[idx_pos];
            let src_pos = params.offset_src + src_col * params.stride_src1 + left * params.stride_src2;
            b = (src[src_pos / 4u] >> (8u * (src_pos % 4u))) & 0xffu;
        }}
        w = w | (b << (8u * lane));
    }}
    dst[params.offset_dst / 4u + gid.x] = w;
}}
"#
    )
}

fn bf16_gather_last_dim_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

struct Params {{
    offset_src: u32,
    offset_idx: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,
    idx1: u32,
    idx2: u32,
}};

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let base = gid.x * 2u;
    let ne = params.n_rows * params.ne2;
    if (base >= ne) {{
        return;
    }}

    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 2u; lane = lane + 1u) {{
        let logical = base + lane;
        var h = 0u;
        if (logical < ne) {{
            let left = logical / params.n_rows;
            let pos = logical - left * params.n_rows;
            let idx_pos = params.offset_idx + pos * params.stride_idx0 + left * params.stride_idx1;
            let src_col = idx[idx_pos];
            let src_pos = params.offset_src + src_col * params.stride_src1 + left * params.stride_src2;
            h = (src[src_pos / 2u] >> (16u * (src_pos % 2u))) & 0xffffu;
        }}
        w = w | (h << (16u * lane));
    }}
    dst[params.offset_dst / 2u + gid.x] = w;
}}
"#
    )
}

fn i64_gather_last_dim_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

struct Params {{
    offset_src: u32,
    offset_idx: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,
    idx1: u32,
    idx2: u32,
}};

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let logical = gid.x;
    let ne = params.n_rows * params.ne2;
    if (logical >= ne) {{
        return;
    }}

    let left = logical / params.n_rows;
    let pos = logical - left * params.n_rows;
    let idx_pos = params.offset_idx + pos * params.stride_idx0 + left * params.stride_idx1;
    let src_col = idx[idx_pos];
    let src_pos = params.offset_src + src_col * params.stride_src1 + left * params.stride_src2;
    let dst_pos = params.offset_dst + pos * params.stride_dst1 + left * params.stride_dst2;
    dst[2u * dst_pos] = src[2u * src_pos];
    dst[2u * dst_pos + 1u] = src[2u * src_pos + 1u];
}}
"#
    )
}

fn f64_gather_last_dim_wgsl() -> String {
    i64_gather_last_dim_wgsl()
}

fn f64_cmp_helpers_wgsl() -> &'static str {
    r#"
fn f64_lt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    let a_neg = (a_hi & 0x80000000u) != 0u;
    let b_neg = (b_hi & 0x80000000u) != 0u;
    if (a_neg != b_neg) {
        return a_neg;
    }
    if (a_hi == b_hi) {
        if (a_lo == b_lo) {
            return false;
        }
        return select((a_lo < b_lo), (a_lo > b_lo), a_neg);
    }
    return select((a_hi < b_hi), (a_hi > b_hi), a_neg);
}

fn f64_gt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return f64_lt(b_lo, b_hi, a_lo, a_hi);
}

fn f64_le(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return !f64_lt(b_lo, b_hi, a_lo, a_hi);
}

fn f64_ge(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    return !f64_lt(a_lo, a_hi, b_lo, b_hi);
}
"#
}

fn i64_argsort_wgsl(workgroup_size: u32, asc: bool) -> String {
    let swap_compare_up = if asc { "i64_gt" } else { "i64_lt" };
    let swap_compare_down = if asc { "i64_lt" } else { "i64_gt" };
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {{
    offset_src: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    src_ne0: u32,
    ne1: u32,
    ne2: u32,
    ne0: u32,
    top_k: u32,
    npr: u32,
    nrows: u32
}};

@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shmem_idx: array<u32, {workgroup_size}>;

fn i64_lt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {{
    let ahi = bitcast<i32>(a_hi);
    let bhi = bitcast<i32>(b_hi);
    return (ahi < bhi) || ((ahi == bhi) && (a_lo < b_lo));
}}

fn i64_gt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {{
    let ahi = bitcast<i32>(a_hi);
    let bhi = bitcast<i32>(b_hi);
    return (ahi > bhi) || ((ahi == bhi) && (a_lo > b_lo));
}}

fn should_swap_up(a_idx: u32, b_idx: u32, row_base: u32) -> bool {{
    let a_oob = a_idx >= params.src_ne0;
    let b_oob = b_idx >= params.src_ne0;
    if (a_oob) {{
        return !b_oob;
    }}
    if (b_oob) {{
        return false;
    }}
    let a_pos = row_base + a_idx;
    let b_pos = row_base + b_idx;
    return {swap_compare_up}(src[2u * a_pos], src[2u * a_pos + 1u], src[2u * b_pos], src[2u * b_pos + 1u]);
}}

fn should_swap_down(a_idx: u32, b_idx: u32, row_base: u32) -> bool {{
    let a_oob = a_idx >= params.src_ne0;
    let b_oob = b_idx >= params.src_ne0;
    if (a_oob) {{
        return false;
    }}
    if (b_oob) {{
        return true;
    }}
    let a_pos = row_base + a_idx;
    let b_pos = row_base + b_idx;
    return {swap_compare_down}(src[2u * a_pos], src[2u * a_pos + 1u], src[2u * b_pos], src[2u * b_pos + 1u]);
}}

@compute @workgroup_size({workgroup_size})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let linear = wid.x + wid.y * num_wg.x;
    if (linear >= params.npr * params.nrows) {{
        return;
    }}
    let tile = linear % params.npr;
    var row = linear / params.npr;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_base = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let tile_base = tile * {workgroup_size}u;
    let idx = tile_base + lid.x;
    shmem_idx[lid.x] = select(params.src_ne0, idx, idx < params.src_ne0);
    workgroupBarrier();

    var k = 2u;
    while (k <= {workgroup_size}u) {{
        var j = k >> 1;
        while (j > 0) {{
            let ixj = lid.x ^ j;
            if (ixj > lid.x) {{
                let dir_up = (lid.x & k) == 0;
                let a_idx = shmem_idx[lid.x];
                let b_idx = shmem_idx[ixj];
                let should_swap = select(
                    should_swap_down(a_idx, b_idx, row_base),
                    should_swap_up(a_idx, b_idx, row_base),
                    dir_up);
                if (should_swap) {{
                    shmem_idx[lid.x] = b_idx;
                    shmem_idx[ixj] = a_idx;
                }}
            }}
            workgroupBarrier();
            j >>= 1;
        }}
        k <<= 1;
    }}

    let out_idx = tile * params.top_k + lid.x;
    if (out_idx < params.ne0 && lid.x < params.top_k) {{
        let row_dst = params.offset_dst +
            i1 * params.stride_dst1 +
            i2 * params.stride_dst2 +
            i3 * params.stride_dst3;
        dst[row_dst + out_idx] = shmem_idx[lid.x];
    }}
}}
"#
    )
}

fn f64_argsort_wgsl(workgroup_size: u32, asc: bool) -> String {
    let swap_compare_up = if asc { "f64_gt" } else { "f64_lt" };
    let swap_compare_down = if asc { "f64_lt" } else { "f64_gt" };
    let cmp_helpers = f64_cmp_helpers_wgsl();
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {{
    offset_src: u32,
    offset_dst: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    src_ne0: u32,
    ne1: u32,
    ne2: u32,
    ne0: u32,
    top_k: u32,
    npr: u32,
    nrows: u32
}};

@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shmem_idx: array<u32, {workgroup_size}>;

{cmp_helpers}

fn should_swap_up(a_idx: u32, b_idx: u32, row_base: u32) -> bool {{
    let a_oob = a_idx >= params.src_ne0;
    let b_oob = b_idx >= params.src_ne0;
    if (a_oob) {{
        return !b_oob;
    }}
    if (b_oob) {{
        return false;
    }}
    let a_pos = row_base + a_idx;
    let b_pos = row_base + b_idx;
    return {swap_compare_up}(src[2u * a_pos], src[2u * a_pos + 1u], src[2u * b_pos], src[2u * b_pos + 1u]);
}}

fn should_swap_down(a_idx: u32, b_idx: u32, row_base: u32) -> bool {{
    let a_oob = a_idx >= params.src_ne0;
    let b_oob = b_idx >= params.src_ne0;
    if (a_oob) {{
        return false;
    }}
    if (b_oob) {{
        return true;
    }}
    let a_pos = row_base + a_idx;
    let b_pos = row_base + b_idx;
    return {swap_compare_down}(src[2u * a_pos], src[2u * a_pos + 1u], src[2u * b_pos], src[2u * b_pos + 1u]);
}}

@compute @workgroup_size({workgroup_size})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let linear = wid.x + wid.y * num_wg.x;
    if (linear >= params.npr * params.nrows) {{
        return;
    }}
    let tile = linear % params.npr;
    var row = linear / params.npr;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_base = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let tile_base = tile * {workgroup_size}u;
    let idx = tile_base + lid.x;
    shmem_idx[lid.x] = select(params.src_ne0, idx, idx < params.src_ne0);
    workgroupBarrier();

    var k = 2u;
    while (k <= {workgroup_size}u) {{
        var j = k >> 1;
        while (j > 0) {{
            let ixj = lid.x ^ j;
            if (ixj > lid.x) {{
                let dir_up = (lid.x & k) == 0;
                let a_idx = shmem_idx[lid.x];
                let b_idx = shmem_idx[ixj];
                let should_swap = select(
                    should_swap_down(a_idx, b_idx, row_base),
                    should_swap_up(a_idx, b_idx, row_base),
                    dir_up);
                if (should_swap) {{
                    shmem_idx[lid.x] = b_idx;
                    shmem_idx[ixj] = a_idx;
                }}
            }}
            workgroupBarrier();
            j >>= 1;
        }}
        k <<= 1;
    }}

    let out_idx = tile * params.top_k + lid.x;
    if (out_idx < params.ne0 && lid.x < params.top_k) {{
        let row_dst = params.offset_dst +
            i1 * params.stride_dst1 +
            i2 * params.stride_dst2 +
            i3 * params.stride_dst3;
        dst[row_dst + out_idx] = shmem_idx[lid.x];
    }}
}}
"#
    )
}

fn i64_argsort_merge_wgsl(asc: bool) -> String {
    let cmp = if asc { "i64_le" } else { "i64_ge" };
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> idx_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> idx_out: array<u32>;

struct Params {{
    offset_src: u32,
    offset_in: u32,
    offset_out: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_idx1: u32,
    stride_idx2: u32,
    stride_idx3: u32,
    stride_out1: u32,
    stride_out2: u32,
    stride_out3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    top_k: u32,
    len: u32,
    nm: u32,
    nrows: u32
}};

@group(0) @binding(3) var<uniform> params: Params;

fn i64_lt(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {{
    let ahi = bitcast<i32>(a_hi);
    let bhi = bitcast<i32>(b_hi);
    return (ahi < bhi) || ((ahi == bhi) && (a_lo < b_lo));
}}

fn i64_le(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {{
    return i64_lt(a_lo, a_hi, b_lo, b_hi) || ((a_lo == b_lo) && (a_hi == b_hi));
}}

fn i64_ge(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {{
    return !i64_lt(a_lo, a_hi, b_lo, b_hi);
}}

fn take_left(a_idx: u32, b_idx: u32, row_base: u32) -> bool {{
    let a_pos = row_base + a_idx;
    let b_pos = row_base + b_idx;
    return {cmp}(src[2u * a_pos], src[2u * a_pos + 1u], src[2u * b_pos], src[2u * b_pos + 1u]);
}}

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let linear = wid.x + wid.y * num_wg.x;
    if (linear >= params.nm * params.nrows) {{
        return;
    }}

    let start = (linear % params.nm) * params.len * 2u;
    let len0 = min(params.len, params.ne0 - start);
    let rem1 = select(0u, params.ne0 - (start + params.len), params.ne0 > (start + params.len));
    let len1 = min(params.len, rem1);
    let total = len0 + len1;
    let chunk = (total + {WG_SIZE}u - 1u) / {WG_SIZE}u;
    let k0 = lid.x * chunk;
    let k1 = min(min(k0 + chunk, total), params.top_k);
    if (k0 >= params.top_k || k0 >= total) {{
        return;
    }}

    var row = linear / params.nm;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_src = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let row_in = params.offset_in +
        i1 * params.stride_idx1 +
        i2 * params.stride_idx2 +
        i3 * params.stride_idx3;

    let row_out = params.offset_out +
        i1 * params.stride_out1 +
        i2 * params.stride_out2 +
        i3 * params.stride_out3;

    var low: u32 = select(0u, k0 - len1, k0 > len1);
    var high: u32 = min(k0, len0);

    while (low < high) {{
        let mid = (low + high) >> 1u;
        let idx0 = idx_in[row_in + start + mid];
        let idx1 = idx_in[row_in + start + params.len + (k0 - mid - 1u)];
        if (take_left(idx0, idx1, row_src)) {{
            low = mid + 1u;
        }} else {{
            high = mid;
        }}
    }}

    var i = low;
    var j = k0 - i;
    var k = k0;
    while (k < k1) {{
        var take_l = false;
        if (i >= len0) {{
            take_l = false;
        }} else if (j >= len1) {{
            take_l = true;
        }} else {{
            let idx0 = idx_in[row_in + start + i];
            let idx1 = idx_in[row_in + start + params.len + j];
            take_l = take_left(idx0, idx1, row_src);
        }}

        let out_idx = select(
            idx_in[row_in + start + params.len + j],
            idx_in[row_in + start + i],
            take_l);
        idx_out[row_out + start + k] = out_idx;
        i = select(i, i + 1u, take_l);
        j = select(j + 1u, j, take_l);
        k += 1u;
    }}
}}
"#
    )
}

fn f64_argsort_merge_wgsl(asc: bool) -> String {
    let cmp = if asc { "f64_le" } else { "f64_ge" };
    let cmp_helpers = f64_cmp_helpers_wgsl();
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> idx_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> idx_out: array<u32>;

struct Params {{
    offset_src: u32,
    offset_in: u32,
    offset_out: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    stride_idx1: u32,
    stride_idx2: u32,
    stride_idx3: u32,
    stride_out1: u32,
    stride_out2: u32,
    stride_out3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    top_k: u32,
    len: u32,
    nm: u32,
    nrows: u32
}};

@group(0) @binding(3) var<uniform> params: Params;

{cmp_helpers}

fn take_left(a_idx: u32, b_idx: u32, row_base: u32) -> bool {{
    let a_pos = row_base + a_idx;
    let b_pos = row_base + b_idx;
    return {cmp}(src[2u * a_pos], src[2u * a_pos + 1u], src[2u * b_pos], src[2u * b_pos + 1u]);
}}

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {{
    let linear = wid.x + wid.y * num_wg.x;
    if (linear >= params.nm * params.nrows) {{
        return;
    }}

    let start = (linear % params.nm) * params.len * 2u;
    let len0 = min(params.len, params.ne0 - start);
    let rem1 = select(0u, params.ne0 - (start + params.len), params.ne0 > (start + params.len));
    let len1 = min(params.len, rem1);
    let total = len0 + len1;
    let chunk = (total + {WG_SIZE}u - 1u) / {WG_SIZE}u;
    let k0 = lid.x * chunk;
    let k1 = min(min(k0 + chunk, total), params.top_k);
    if (k0 >= params.top_k || k0 >= total) {{
        return;
    }}

    var row = linear / params.nm;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_src = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let row_in = params.offset_in +
        i1 * params.stride_idx1 +
        i2 * params.stride_idx2 +
        i3 * params.stride_idx3;

    let row_out = params.offset_out +
        i1 * params.stride_out1 +
        i2 * params.stride_out2 +
        i3 * params.stride_out3;

    var low: u32 = select(0u, k0 - len1, k0 > len1);
    var high: u32 = min(k0, len0);

    while (low < high) {{
        let mid = (low + high) >> 1u;
        let idx0 = idx_in[row_in + start + mid];
        let idx1 = idx_in[row_in + start + params.len + (k0 - mid - 1u)];
        if (take_left(idx0, idx1, row_src)) {{
            low = mid + 1u;
        }} else {{
            high = mid;
        }}
    }}

    var i = low;
    var j = k0 - i;
    var k = k0;
    while (k < k1) {{
        var take_l = false;
        if (i >= len0) {{
            take_l = false;
        }} else if (j >= len1) {{
            take_l = true;
        }} else {{
            let idx0 = idx_in[row_in + start + i];
            let idx1 = idx_in[row_in + start + params.len + j];
            take_l = take_left(idx0, idx1, row_src);
        }}

        let out_idx = select(
            idx_in[row_in + start + params.len + j],
            idx_in[row_in + start + i],
            take_l);
        idx_out[row_out + start + k] = out_idx;
        i = select(i, i + 1u, take_l);
        j = select(j + 1u, j, take_l);
        k += 1u;
    }}
}}
"#
    )
}

fn custom_int_binary_wgsl(op: &str, dtype: DType) -> Result<String> {
    const I64_WGSL_HELPERS: &str = r#"
fn u64_shl1_or(lo: u32, hi: u32, bit: u32) -> vec2<u32> {
    return vec2((lo << 1u) | bit, (hi << 1u) | (lo >> 31u));
}
fn u64_sub(lo_a: u32, hi_a: u32, lo_b: u32, hi_b: u32) -> vec2<u32> {
    let lo = lo_a - lo_b;
    let borrow = select(0u, 1u, lo_a < lo_b);
    let hi = hi_a - hi_b - borrow;
    return vec2(lo, hi);
}
fn u64_ge(lo_a: u32, hi_a: u32, lo_b: u32, hi_b: u32) -> bool {
    if (hi_a != hi_b) { return hi_a > hi_b; }
    return lo_a >= lo_b;
}
fn u64_neg(lo: u32, hi: u32) -> vec2<u32> {
    let lo_n = 0u - lo;
    let borrow = select(0u, 1u, lo != 0u);
    let hi_n = 0u - hi - borrow;
    return vec2(lo_n, hi_n);
}
fn u64_abs(lo: u32, hi: u32) -> vec2<u32> {
    if (bitcast<i32>(hi) < 0) { return u64_neg(lo, hi); }
    return vec2(lo, hi);
}
fn u64_div_unsigned(n_lo: u32, n_hi: u32, d_lo: u32, d_hi: u32) -> vec2<u32> {
    if (d_lo == 0u && d_hi == 0u) { return vec2(0u, 0u); }
    var r_lo = 0u;
    var r_hi = 0u;
    var q_lo = 0u;
    var q_hi = 0u;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let bit = select((n_lo >> u32(i)) & 1u, (n_hi >> u32(i - 32)) & 1u, i >= 32);
        let shifted = u64_shl1_or(r_lo, r_hi, bit);
        r_lo = shifted.x;
        r_hi = shifted.y;
        if (u64_ge(r_lo, r_hi, d_lo, d_hi)) {
            let sub = u64_sub(r_lo, r_hi, d_lo, d_hi);
            r_lo = sub.x;
            r_hi = sub.y;
            if (i < 32) {
                q_lo = q_lo | (1u << u32(i));
            } else {
                q_hi = q_hi | (1u << u32(i - 32));
            }
        }
    }
    return vec2(q_lo, q_hi);
}
fn i64_div_trunc(lo_a: u32, hi_a: u32, lo_b: u32, hi_b: u32) -> vec2<u32> {
    if (lo_b == 0u && hi_b == 0u) { return vec2(0u, 0u); }
    let neg_a = bitcast<i32>(hi_a) < 0;
    let neg_b = bitcast<i32>(hi_b) < 0;
    let aa = u64_abs(lo_a, hi_a);
    let bb = u64_abs(lo_b, hi_b);
    var q = u64_div_unsigned(aa.x, aa.y, bb.x, bb.y);
    if (neg_a != neg_b) { q = u64_neg(q.x, q.y); }
    return q;
}
"#;
    let indexing = r#"
fn src0_index(_i: u32) -> u32 {
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    return a_i0 * params.stride_src0_0 + a_i1 * params.stride_src0_1 +
           a_i2 * params.stride_src0_2 + a_i3 * params.stride_src0_3;
}

fn src1_index(_i: u32) -> u32 {
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    let b_i0 = a_i0 % params.b_ne0;
    let b_i1 = a_i1 % params.b_ne1;
    let b_i2 = a_i2 % params.b_ne2;
    let b_i3 = a_i3 % params.b_ne3;
    return b_i0 * params.stride_src1_0 + b_i1 * params.stride_src1_1 +
           b_i2 * params.stride_src1_2 + b_i3 * params.stride_src1_3;
}
"#;
    let params_struct = r#"
struct Params {
    ne: u32,
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,
    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,
    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read_write> src0: array<u32>;
@group(0) @binding(1) var<storage, read_write> src1: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
"#;
    let main = match dtype {
        DType::U32 => {
            let expr = match op {
                "add" => "a + b",
                "sub" => "a - b",
                "mul" => "a * b",
                "div" => "select(a / max(b, 1u), 0u, b == 0u)",
                "maximum" => "max(a, b)",
                "minimum" => "min(a, b)",
                _ => return Err(unsupported("binary int op")),
            };
            format!(
                r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let a = src0[params.offset_src0 + src0_index(gid.x)];
    let b = src1[params.offset_src1 + src1_index(gid.x)];
    dst[params.offset_dst + gid.x] = {expr};
}}"#
            )
        }
        DType::U8 => {
            let expr = match op {
                "add" => "(a + b) & 0xffu",
                "sub" => "(a - b) & 0xffu",
                "mul" => "(a * b) & 0xffu",
                "div" => "select(a / max(b, 1u), 0u, b == 0u)",
                "maximum" => "max(a, b)",
                "minimum" => "min(a, b)",
                _ => return Err(unsupported("binary int op")),
            };
            format!(
                r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let base = gid.x * 4u;
    if (base >= params.ne) {{ return; }}
    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
        let idx = base + lane;
        if (idx >= params.ne) {{ break; }}
        let ea = params.offset_src0 + src0_index(idx);
        let a = (src0[ea / 4u] >> (8u * (ea % 4u))) & 0xffu;
        let eb = params.offset_src1 + src1_index(idx);
        let b = (src1[eb / 4u] >> (8u * (eb % 4u))) & 0xffu;
        let r = ({expr}) & 0xffu;
        w = w | (r << (8u * lane));
    }}
    dst[params.offset_dst / 4u + gid.x] = w;
}}"#
            )
        }
        DType::I64 => {
            let compute = match op {
                "add" => {
                    r#"
    let lo = a_lo + b_lo;
    let carry = select(0u, 1u, lo < a_lo);
    let hi = a_hi + b_hi + carry;
"#
                }
                "sub" => {
                    r#"
    let lo = a_lo - b_lo;
    let borrow = select(0u, 1u, a_lo < b_lo);
    let hi = a_hi - b_hi - borrow;
"#
                }
                "mul" => {
                    // 64-bit low product from 32-bit limbs: hi limb cross
                    // products only contribute to the high word.
                    r#"
    let a0 = a_lo & 0xffffu; let a1 = a_lo >> 16u;
    let b0 = b_lo & 0xffffu; let b1 = b_lo >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = p01 + p10;
    let mid_carry = select(0u, 0x10000u, mid < p01);
    let lo = p00 + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < p00);
    let hi = p11 + (mid >> 16u) + mid_carry + lo_carry + a_lo * b_hi + a_hi * b_lo;
"#
                }
                "div" => {
                    r#"
    let q = i64_div_trunc(a_lo, a_hi, b_lo, b_hi);
    let lo = q.x;
    let hi = q.y;
"#
                }
                "maximum" | "minimum" => {
                    let pick_a = if op == "maximum" { "a_gt_b" } else { "!a_gt_b" };
                    return Ok(format!(
                        r#"{params_struct}
{indexing}
@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let ea = params.offset_src0 + src0_index(gid.x);
    let a_lo = src0[2u * ea]; let a_hi = src0[2u * ea + 1u];
    let eb = params.offset_src1 + src1_index(gid.x);
    let b_lo = src1[2u * eb]; let b_hi = src1[2u * eb + 1u];
    let a_gt_b = (bitcast<i32>(a_hi) > bitcast<i32>(b_hi)) || ((a_hi == b_hi) && (a_lo > b_lo));
    let lo = select(b_lo, a_lo, {pick_a});
    let hi = select(b_hi, a_hi, {pick_a});
    let d = params.offset_dst + gid.x;
    dst[2u * d] = lo;
    dst[2u * d + 1u] = hi;
}}
"#
                    ));
                }
                _ => return Err(unsupported("binary int op")),
            };
            format!(
                r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let ea = params.offset_src0 + src0_index(gid.x);
    let a_lo = src0[2u * ea]; let a_hi = src0[2u * ea + 1u];
    let eb = params.offset_src1 + src1_index(gid.x);
    let b_lo = src1[2u * eb]; let b_hi = src1[2u * eb + 1u];
    {compute}
    let d = params.offset_dst + gid.x;
    dst[2u * d] = lo;
    dst[2u * d + 1u] = hi;
}}"#
            )
        }
        DType::I32 => {
            let expr = match op {
                "add" => "a + b",
                "sub" => "a - b",
                "mul" => "a * b",
                "div" => "select(a / select(b, 1, b == 0), 0, b == 0)",
                "maximum" => "max(a, b)",
                "minimum" => "min(a, b)",
                _ => return Err(unsupported("binary int op")),
            };
            format!(
                r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let a = bitcast<i32>(src0[params.offset_src0 + src0_index(gid.x)]);
    let b = bitcast<i32>(src1[params.offset_src1 + src1_index(gid.x)]);
    dst[params.offset_dst + gid.x] = bitcast<u32>({expr});
}}"#
            )
        }
        other => return Err(Error::UnsupportedDTypeForOp(other, "wgpu binary int").bt()),
    };
    let helpers = if dtype == DType::I64 {
        I64_WGSL_HELPERS
    } else {
        ""
    };
    Ok(format!("{params_struct}\n{helpers}\n{indexing}\n{main}\n"))
}

fn copy_shader(src: DType, dst: DType) -> Result<String> {
    let defines = match (src, dst) {
        (DType::F32, DType::F32) => ["SRC_F32", "DST_F32"],
        (DType::F32, DType::F16) => ["SRC_F32", "DST_F16"],
        (DType::F32, DType::I32) => ["SRC_F32", "DST_I32"],
        (DType::F32, DType::U32) => ["SRC_F32", "DST_U32"],
        (DType::F16, DType::F32) => ["SRC_F16", "DST_F32"],
        (DType::F16, DType::F16) => ["SRC_F16", "DST_F16"],
        (DType::F16, DType::I32) => ["SRC_F16", "DST_I32"],
        (DType::F16, DType::U32) => ["SRC_F16", "DST_U32"],
        (DType::U32, DType::F32) => ["SRC_U32", "DST_F32"],
        (DType::U32, DType::F16) => ["SRC_U32", "DST_F16"],
        (DType::U32, DType::I32) => ["SRC_U32", "DST_I32"],
        (DType::U32, DType::U32) => ["SRC_U32", "DST_U32"],
        (DType::I32, DType::F32) => ["SRC_I32", "DST_F32"],
        (DType::I32, DType::F16) => ["SRC_I32", "DST_F16"],
        (DType::I32, DType::I32) => ["SRC_I32", "DST_I32"],
        (DType::I32, DType::U32) => ["SRC_I32", "DST_U32"],
        _ => return Err(unsupported("to_dtype")),
    };
    let source = candle_wgpu_kernels::get("cpy.wgsl")
        .ok_or_else(|| Error::Msg("wgpu shader cpy.wgsl not embedded".into()).bt())?
        .source()
        .replace("WG_SIZE", &WG_SIZE.to_string());
    let mut out = String::new();
    let mut active = true;
    let mut branch_taken = false;
    let mut replacements: Vec<(&str, &str)> = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed == "enable f16;" {
            if defines.iter().any(|d| d.ends_with("F16")) {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }
        if let Some(name) = trimmed.strip_prefix("#ifdef ") {
            active = defines.iter().any(|d| d == &name.trim());
            branch_taken = active;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#elif defined(") {
            let name = rest.trim_end_matches(')').trim();
            active = !branch_taken && defines.iter().any(|d| d == &name);
            branch_taken |= active;
            continue;
        }
        if trimmed == "#endif" {
            active = true;
            branch_taken = false;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#define ") {
            if active {
                let mut parts = rest.split_whitespace();
                if let (Some(name), Some(value)) = (parts.next(), parts.next()) {
                    replacements.push((name, value));
                }
            }
            continue;
        }
        if active {
            let mut expanded = line.to_string();
            for (name, value) in &replacements {
                expanded = expanded.replace(name, value);
            }
            out.push_str(&expanded);
            out.push('\n');
        }
    }
    Ok(out)
}

fn bf16_binary_wgsl(op: &str) -> String {
    let (ea, eb) = match op {
        "add" => ("a0 + b0", "a1 + b1"),
        "sub" => ("a0 - b0", "a1 - b1"),
        "mul" => ("a0 * b0", "a1 * b1"),
        "div" => ("a0 / b0", "a1 / b1"),
        "maximum" => ("max(a0, b0)", "max(a1, b1)"),
        "minimum" => ("min(a0, b0)", "min(a1, b1)"),
        other => panic!("bf16_binary_wgsl: unsupported op {other}"),
    };
    let expr0 = ea;
    let expr1 = eb;
    format!(
        r#"
fn bf16_bits(v: f32) -> u32 {{
    return ((bitcast<u32>(v) + (0x7fffu + ((bitcast<u32>(v) >> 16u) & 1u))) >> 16u) & 0xffffu;
}}
fn bf16_to_f32(word: u32, half: u32) -> f32 {{
    let shift = half * 16u;
    return bitcast<f32>(((word >> shift) & 0xffffu) << 16u);
}}
fn bf16_store(word: u32, half: u32, v: f32) -> u32 {{
    let p = bf16_bits(v);
    let shift = half * 16u;
    let mask = select(0x0000ffffu, 0xffff0000u, half == 0u);
    return (word & mask) | (p << shift);
}}

struct Params {{
    ne: u32,
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,
    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,
    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
    _pad0: u32,
}};

@group(0) @binding(0) var<storage, read_write> src0: array<u32>;
@group(0) @binding(1) var<storage, read_write> src1: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn src0_index(_i: u32) -> u32 {{
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    return a_i0 * params.stride_src0_0 + a_i1 * params.stride_src0_1 +
           a_i2 * params.stride_src0_2 + a_i3 * params.stride_src0_3;
}}

fn src1_index(_i: u32) -> u32 {{
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    let b_i0 = a_i0 % params.b_ne0;
    let b_i1 = a_i1 % params.b_ne1;
    let b_i2 = a_i2 % params.b_ne2;
    let b_i3 = a_i3 % params.b_ne3;
    return b_i0 * params.stride_src1_0 + b_i1 * params.stride_src1_1 +
           b_i2 * params.stride_src1_2 + b_i3 * params.stride_src1_3;
}}

@compute @workgroup_size({WG_SIZE})
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let pair = (wid.x + wid.y * num_wg.x) * {WG_SIZE}u + lid.x;
    let linear = pair * 2u;
    if (linear >= params.ne) {{ return; }}
    let i0s = params.offset_src0 + src0_index(linear);
    let i1s = params.offset_src1 + src1_index(linear);
    let a0 = bf16_to_f32(src0[i0s / 2u], i0s % 2u);
    let b0 = bf16_to_f32(src1[i1s / 2u], i1s % 2u);
    let out0 = {expr0};
    let out_i0 = params.offset_dst + linear;
    var word = bf16_store(0u, out_i0 % 2u, out0);
    if (linear + 1u < params.ne) {{
        let i0s1 = params.offset_src0 + src0_index(linear + 1u);
        let i1s1 = params.offset_src1 + src1_index(linear + 1u);
        let a1 = bf16_to_f32(src0[i0s1 / 2u], i0s1 % 2u);
        let b1 = bf16_to_f32(src1[i1s1 / 2u], i1s1 % 2u);
        let out1 = {expr1};
        word = bf16_store(word, 1u, out1);
    }}
    dst[out_i0 / 2u] = word;
}}
"#
    )
}

const BF16_WGSL_HELPERS: &str = r#"
fn bf16_bits(v: f32) -> u32 {
    return ((bitcast<u32>(v) + (0x7fffu + ((bitcast<u32>(v) >> 16u) & 1u))) >> 16u) & 0xffffu;
}
fn bf16_to_f32(word: u32, half: u32) -> f32 {
    let shift = half * 16u;
    return bitcast<f32>(((word >> shift) & 0xffffu) << 16u);
}
fn bf16_store_half(word: u32, half: u32, v: f32) -> u32 {
    let p = bf16_bits(v);
    let shift = half * 16u;
    let mask = select(0x0000ffffu, 0xffff0000u, half == 0u);
    return (word & mask) | (p << shift);
}
"#;

fn inject_wg_size(shader: String) -> String {
    shader.replace("WG_SIZE", &WG_SIZE.to_string())
}

fn bf16_softmax_shader() -> String {
    inject_wg_size(format!(
        r#"{BF16_WGSL_HELPERS}
struct Params {{
    offset_src0: u32,
    offset_src1: u32,
    offset_sinks: u32,
    offset_dst: u32,
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,
    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne12: u32,
    ne13: u32,
    scale: f32,
    max_bias: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
    row_base: u32,
}};
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
fn load_elem(i: u32) -> f32 {{
    return bf16_to_f32(src[i / 2u], i % 2u);
}}
fn store_pair(i0: u32, v0: f32, v1: f32, has_v1: bool) {{
    let wi = i0 / 2u;
    var word = bf16_store_half(0u, i0 % 2u, v0);
    if (has_v1) {{ word = bf16_store_half(word, 1u, v1); }}
    src[wi] = word;
}}
const CACHE_SIZE: u32 = 16;
var<workgroup> scratch: array<f32, WG_SIZE>;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    var i = wid.x + wid.y * num_wg.x + params.row_base;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_row = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01;
    let elems = (params.ne0 + WG_SIZE * 2u - 1u) / (WG_SIZE * 2u);
    var cache: array<f32, CACHE_SIZE>;
    var max_val = -1e30f;
    var col = lid.x * 2u;
    for (var j: u32 = 0u; j < elems; j++) {{
        if (col >= params.ne0) {{ break; }}
        let i0 = i_row + col;
        let v0 = load_elem(i0) * params.scale;
        max_val = max(max_val, v0);
        if (col + 1u < params.ne0) {{ max_val = max(max_val, load_elem(i0 + 1u) * params.scale); }}
        if (col < CACHE_SIZE) {{ cache[col] = v0; }}
        if (col + 1u < CACHE_SIZE) {{ cache[col + 1u] = load_elem(i0 + 1u) * params.scale; }}
        col += WG_SIZE * 2u;
    }}
    scratch[lid.x] = max_val;
    workgroupBarrier();
    var offset: u32 = WG_SIZE / 2u;
    while (offset > 0u) {{
        if (lid.x < offset) {{ scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + offset]); }}
        offset = offset / 2u;
        workgroupBarrier();
    }}
    let row_max = scratch[0];
    workgroupBarrier();
    var sum = 0.0f;
    col = lid.x * 2u;
    for (var j: u32 = 0u; j < elems; j++) {{
        if (col >= params.ne0) {{ break; }}
        let i0 = i_row + col;
        let has_v1 = col + 1u < params.ne0;
        let v0 = select(load_elem(i0) * params.scale, cache[col], col < CACHE_SIZE);
        let ex0 = exp(v0 - row_max);
        sum += ex0;
        var ex1 = 0.0f;
        if (has_v1) {{
            let v1 = select(load_elem(i0 + 1u) * params.scale, cache[col + 1u], col + 1u < CACHE_SIZE);
            ex1 = exp(v1 - row_max);
            sum += ex1;
        }}
        if (col < CACHE_SIZE) {{
            cache[col] = ex0;
            if (has_v1) {{ cache[col + 1u] = ex1; }}
        }} else {{
            store_pair(i0, ex0, ex1, has_v1);
        }}
        col += WG_SIZE * 2u;
    }}
    scratch[lid.x] = sum;
    workgroupBarrier();
    offset = WG_SIZE / 2u;
    while (offset > 0u) {{
        if (lid.x < offset) {{ scratch[lid.x] += scratch[lid.x + offset]; }}
        offset = offset / 2u;
        workgroupBarrier();
    }}
    let sum_recip = 1.0f / scratch[0];
    col = lid.x * 2u;
    for (var j: u32 = 0u; j < elems; j++) {{
        if (col >= params.ne0) {{ break; }}
        let i0 = i_row + col;
        let has_v1 = col + 1u < params.ne0;
        let out0 = select(load_elem(i0), cache[col], col < CACHE_SIZE) * sum_recip;
        var out1 = 0.0f;
        if (has_v1) {{
            out1 = select(load_elem(i0 + 1u), cache[col + 1u], col + 1u < CACHE_SIZE) * sum_recip;
        }}
        store_pair(i0, out0, out1, has_v1);
        col += WG_SIZE * 2u;
    }}
}}
"#
    ))
}

fn bf16_rms_norm_shader() -> String {
    inject_wg_size(format!(
        r#"{BF16_WGSL_HELPERS}
struct Params {{
    offset_rn_src: u32,
    offset_mul_src: u32,
    offset_dst: u32,
    stride_rn_src1: u32,
    stride_rn_src2: u32,
    stride_rn_src3: u32,
    stride_mul_src1: u32,
    stride_mul_src2: u32,
    stride_mul_src3: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    mul_src_ne0: u32,
    mul_src_ne1: u32,
    mul_src_ne2: u32,
    mul_src_ne3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
    eps: f32,
}};
@group(0) @binding(0) var<storage, read_write> rn_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> mul_src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> scratch: array<f32, WG_SIZE>;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    var i = wid.x + wid.y * num_wg.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_rn_src_row = params.offset_rn_src + i3 * params.stride_rn_src3 + i2 * params.stride_rn_src2 + i1 * params.stride_rn_src1;
    let i_mul_src_row = params.offset_mul_src + (i3 % params.mul_src_ne3) * params.stride_mul_src3 + (i2 % params.mul_src_ne2) * params.stride_mul_src2 + (i1 % params.mul_src_ne1) * params.stride_mul_src1;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;
    let elems = (params.ne0 + WG_SIZE * 2u - 1u) / (WG_SIZE * 2u);
    var sum = 0.0f;
    var col = lid.x * 2u;
    for (var j: u32 = 0u; j < elems; j++) {{
        if (col >= params.ne0) {{ break; }}
        let idx0 = i_rn_src_row + col;
        let v0 = bf16_to_f32(rn_src[idx0 / 2u], idx0 % 2u);
        sum += v0 * v0;
        if (col + 1u < params.ne0) {{
            let idx1 = i_rn_src_row + col + 1u;
            let v1 = bf16_to_f32(rn_src[idx1 / 2u], idx1 % 2u);
            sum += v1 * v1;
        }}
        col += WG_SIZE * 2u;
    }}
    scratch[lid.x] = sum;
    workgroupBarrier();
    var offset: u32 = WG_SIZE / 2u;
    while (offset > 0u) {{
        if (lid.x < offset) {{ scratch[lid.x] += scratch[lid.x + offset]; }}
        offset = offset / 2u;
        workgroupBarrier();
    }}
    let scale = 1.0f / sqrt(scratch[0] / f32(params.ne0) + params.eps);
    col = lid.x * 2u;
    for (var j: u32 = 0u; j < elems; j++) {{
        if (col >= params.ne0) {{ break; }}
        let idx0 = i_rn_src_row + col;
        let idx1 = i_mul_src_row + col % params.mul_src_ne0;
        let out0 = scale * bf16_to_f32(rn_src[idx0 / 2u], idx0 % 2u) * bf16_to_f32(mul_src[idx1 / 2u], idx1 % 2u);
        let out_i0 = i_dst_row + col;
        var word = bf16_store_half(0u, out_i0 % 2u, out0);
        if (col + 1u < params.ne0) {{
            let idx0b = i_rn_src_row + col + 1u;
            let idx1b = i_mul_src_row + (col + 1u) % params.mul_src_ne0;
            let out1 = scale * bf16_to_f32(rn_src[idx0b / 2u], idx0b % 2u) * bf16_to_f32(mul_src[idx1b / 2u], idx1b % 2u);
            word = bf16_store_half(word, 1u, out1);
        }}
        dst[out_i0 / 2u] = word;
        col += WG_SIZE * 2u;
    }}
}}
"#
    ))
}

fn bf16_scale_shader() -> String {
    inject_wg_size(format!(
        r#"{BF16_WGSL_HELPERS}
struct Params {{
    offset_src: u32,
    offset_dst: u32,
    ne: u32,
    scale: f32,
    bias: f32,
}};
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let pair = (wid.x + wid.y * num_wg.x) * WG_SIZE + lid.x;
    let i0 = pair * 2u;
    if (i0 >= params.ne) {{ return; }}
    let si0 = params.offset_src + i0;
    let si1 = si0 + 1u;
    let dst_wi = (params.offset_dst + i0) / 2u;
    var word = 0u;
    let v0 = bf16_to_f32(src[si0 / 2u], si0 % 2u) * params.scale + params.bias;
    word = bf16_store_half(word, si0 % 2u, v0);
    if (i0 + 1u < params.ne) {{
        let v1 = bf16_to_f32(src[si1 / 2u], si1 % 2u) * params.scale + params.bias;
        word = bf16_store_half(word, 1u, v1);
    }}
    dst[dst_wi] = word;
}}
"#
    ))
}

fn bf16_rope_shader() -> String {
    inject_wg_size(format!(
        r#"{BF16_WGSL_HELPERS}
struct Params {{
    offset_src0: u32,
    offset_src1: u32,
    offset_src2: u32,
    offset_dst: u32,
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    n_threads: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    n_dims: u32,
    mode: u32,
    theta_scale: f32,
    attn_factor: f32,
    freq_scale: f32,
    ext_factor: f32,
    corr_dim0: f32,
    corr_dim1: f32,
    sections0: u32,
    sections1: u32,
    sections2: u32,
    sections3: u32,
}};
@group(0) @binding(0) var<storage, read_write> src0: array<u32>;
@group(0) @binding(1) var<storage, read_write> src1: array<i32>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
fn freq_factor(i: u32) -> f32 {{ return 1.0f; }}
fn rotate(i_dst0: u32, i_dst1: u32, out0: f32, out1: f32) {{
    let wi0 = i_dst0 / 2u;
    let half0 = i_dst0 % 2u;
    let p0 = bf16_bits(out0);
    let mask0 = select(0x0000ffffu, 0xffff0000u, half0 == 0u);
    loop {{
        let old = atomicLoad(&dst[wi0]);
        let desired = (old & mask0) | (p0 << (half0 * 16u));
        let res = atomicCompareExchangeWeak(&dst[wi0], old, desired);
        if (res.exchanged) {{ break; }}
    }}
    let wi1 = i_dst1 / 2u;
    let half1 = i_dst1 % 2u;
    let p1 = bf16_bits(out1);
    let mask1 = select(0x0000ffffu, 0xffff0000u, half1 == 0u);
    loop {{
        let old = atomicLoad(&dst[wi1]);
        let desired = (old & mask1) | (p1 << (half1 * 16u));
        let res = atomicCompareExchangeWeak(&dst[wi1], old, desired);
        if (res.exchanged) {{ break; }}
    }}
}}
fn rope_yarn_ramp(low: f32, high: f32, i: u32) -> f32 {{
    let y = (f32(i / 2u) - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}}
fn rope_yarn(theta_extrap: f32, i: u32) -> vec2<f32> {{
    var mscale = params.attn_factor;
    var theta = params.freq_scale * theta_extrap;
    if (params.ext_factor != 0.0f) {{
        let ramp_mix = rope_yarn_ramp(params.corr_dim0, params.corr_dim1, i) * params.ext_factor;
        theta = theta * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / params.freq_scale);
    }}
    return vec2<f32>(cos(theta) * mscale, sin(theta) * mscale);
}}
fn pair_base(i0: u32, div_2: bool) -> u32 {{ if (div_2) {{ return i0 / 2u; }} else {{ return i0; }} }}
fn pair_offset(is_neox: bool, is_mrope: bool, is_vision: bool) -> u32 {{
    if (is_vision) {{ return params.n_dims; }}
    else if (is_neox || is_mrope) {{ return params.n_dims / 2u; }}
    else {{ return 1u; }}
}}
@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let linear = (wid.x + wid.y * num_wg.x) * WG_SIZE + lid.x;
    if (linear >= params.n_threads) {{ return; }}
    let is_neox = (params.mode & 2u) != 0u;
    let is_mrope = (params.mode & 8u) != 0u;
    let is_imrope = params.mode == 40u;
    let is_vision = params.mode == 24u;
    var i = linear * 2u;
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;
    let i_src_row = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;
    if (i0 >= params.n_dims && !is_vision) {{
        let i_src = i_src_row + i0;
        let i_dst = i_dst_row + i0;
        rotate(i_dst, i_dst + 1u, bf16_to_f32(src0[i_src / 2u], i_src % 2u), bf16_to_f32(src0[(i_src + 1u) / 2u], (i_src + 1u) % 2u));
        return;
    }}
    var theta_base_mult: u32 = 0u;
    var theta_scale_pwr: u32 = i0 / 2u;
    if (is_mrope) {{
        let sect_dims = params.sections0 + params.sections1 + params.sections2 + params.sections3;
        let sec_w = params.sections1 + params.sections0;
        let sec_e = params.sections2 + sec_w;
        let sector = (i0 / 2u) % sect_dims;
        if (sector >= params.sections0 && sector < sec_w) {{ theta_base_mult = 1u; }}
        else if (sector >= sec_w && sector < sec_e) {{ theta_base_mult = 2u; }}
        else if (sector >= sec_e) {{ theta_base_mult = 3u; }}
    }}
    let theta_base = f32(src1[params.offset_src1 + i2 + params.ne2 * theta_base_mult]) * pow(params.theta_scale, f32(theta_scale_pwr));
    let thetas = rope_yarn(theta_base / freq_factor(i0), i0);
    let po = pair_offset(is_neox, is_mrope, is_vision);
    let i_src = i_src_row + pair_base(i0, is_neox || is_mrope || is_vision);
    let i_dst = i_dst_row + pair_base(i0, is_neox || is_mrope || is_vision);
    let x0 = bf16_to_f32(src0[i_src / 2u], i_src % 2u);
    let x1 = bf16_to_f32(src0[(i_src + po) / 2u], (i_src + po) % 2u);
    rotate(i_dst, i_dst + po, x0 * thetas.x - x1 * thetas.y, x0 * thetas.y + x1 * thetas.x);
}}
"#
    ))
}

fn custom_binary_wgsl(expr: &str, dtype: DType) -> Result<String> {
    let (enable, data_type) = match dtype {
        DType::F32 => ("", "f32"),
        DType::F16 => ("enable f16;\n", "f16"),
        _ => return Err(Error::UnsupportedDTypeForOp(dtype, "wgpu binary").bt()),
    };
    Ok(format!(
        r#"
{enable}
struct Params {{
    ne: u32,
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,
    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,
    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
    _pad0: u32,
}};

@group(0) @binding(0) var<storage, read_write> src0: array<{data_type}>;
@group(0) @binding(1) var<storage, read_write> src1: array<{data_type}>;
@group(0) @binding(2) var<storage, read_write> dst: array<{data_type}>;
@group(0) @binding(3) var<uniform> params: Params;

fn src0_index(_i: u32) -> u32 {{
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    return a_i0 * params.stride_src0_0 + a_i1 * params.stride_src0_1 +
           a_i2 * params.stride_src0_2 + a_i3 * params.stride_src0_3;
}}

fn src1_index(_i: u32) -> u32 {{
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    let b_i0 = a_i0 % params.b_ne0;
    let b_i1 = a_i1 % params.b_ne1;
    let b_i2 = a_i2 % params.b_ne2;
    let b_i3 = a_i3 % params.b_ne3;
    return b_i0 * params.stride_src1_0 + b_i1 * params.stride_src1_1 +
           b_i2 * params.stride_src1_2 + b_i3 * params.stride_src1_3;
}}

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let a = src0[params.offset_src0 + src0_index(gid.x)];
    let b = src1[params.offset_src1 + src1_index(gid.x)];
    dst[params.offset_dst + gid.x] = {expr};
}}
"#
    ))
}

fn wgpu_scalar_type(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::U32 => Ok("u32"),
        DType::I32 => Ok("i32"),
        _ => Err(Error::UnsupportedDTypeForOp(dtype, "wgpu shader").bt()),
    }
}

fn custom_cmp_wgsl(op: CmpOp, dtype: DType) -> Result<String> {
    // U8 and I64 have no native WGSL scalar: U8 elements live as packed bytes
    // in u32 words, I64 elements as lo/hi u32 word pairs (two's complement).
    let (ty, load) = match dtype {
        DType::U8 => (
            "u32",
            r#"let ea = params.offset_src0 + src0_index(idx);
        let a = (src0[ea / 4u] >> (8u * (ea % 4u))) & 0xffu;
        let eb = params.offset_src1 + src1_index(idx);
        let b = (src1[eb / 4u] >> (8u * (eb % 4u))) & 0xffu;"#,
        ),
        DType::I64 => (
            "u32",
            r#"let ea = params.offset_src0 + src0_index(idx);
        let a_lo = src0[2u * ea]; let a_hi = src0[2u * ea + 1u];
        let eb = params.offset_src1 + src1_index(idx);
        let b_lo = src1[2u * eb]; let b_hi = src1[2u * eb + 1u];
        let i64_eq = (a_lo == b_lo) && (a_hi == b_hi);
        let i64_lt = (bitcast<i32>(a_hi) < bitcast<i32>(b_hi)) || ((a_hi == b_hi) && (a_lo < b_lo));"#,
        ),
        _ => (
            wgpu_scalar_type(dtype)?,
            r#"let a = src0[params.offset_src0 + src0_index(idx)];
        let b = src1[params.offset_src1 + src1_index(idx)];"#,
        ),
    };
    let prelude = if dtype == DType::F16 {
        "enable f16;\n"
    } else {
        ""
    };
    let pred = if dtype == DType::I64 {
        match op {
            CmpOp::Eq => "i64_eq",
            CmpOp::Ne => "!i64_eq",
            CmpOp::Lt => "i64_lt",
            CmpOp::Le => "i64_lt || i64_eq",
            CmpOp::Gt => "!(i64_lt || i64_eq)",
            CmpOp::Ge => "!i64_lt",
        }
    } else {
        match op {
            CmpOp::Eq => "a == b",
            CmpOp::Ne => "a != b",
            CmpOp::Lt => "a < b",
            CmpOp::Le => "a <= b",
            CmpOp::Gt => "a > b",
            CmpOp::Ge => "a >= b",
        }
    };
    Ok(format!(
        r#"{prelude}
struct Params {{
    ne: u32,
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,
    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,
    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
    _pad0: u32,
}};

@group(0) @binding(0) var<storage, read> src0: array<{ty}>;
@group(0) @binding(1) var<storage, read> src1: array<{ty}>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn src0_index(_i: u32) -> u32 {{
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    return a_i0 * params.stride_src0_0 + a_i1 * params.stride_src0_1 +
           a_i2 * params.stride_src0_2 + a_i3 * params.stride_src0_3;
}}

fn src1_index(_i: u32) -> u32 {{
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;
    let b_i0 = a_i0 % params.b_ne0;
    let b_i1 = a_i1 % params.b_ne1;
    let b_i2 = a_i2 % params.b_ne2;
    let b_i3 = a_i3 % params.b_ne3;
    return b_i0 * params.stride_src1_0 + b_i1 * params.stride_src1_1 +
           b_i2 * params.stride_src1_2 + b_i3 * params.stride_src1_3;
}}

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let packed_words = (params.ne + 3u) / 4u;
    if (gid.x >= packed_words) {{ return; }}
    let base = gid.x * 4u;
    var out_word: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
        let idx = base + lane;
        if (idx >= params.ne) {{ break; }}
        {load}
        let value = select(0u, 1u, {pred});
        out_word = out_word | (value << (lane * 8u));
    }}
    dst[params.offset_dst + gid.x] = out_word;
}}
"#
    ))
}

fn custom_where_u8_wgsl(dtype: DType) -> Result<String> {
    // U8 and I64 values have no native WGSL scalar; the value buffers are
    // u32 words (U8: four packed bytes per word, I64: lo/hi pair per element)
    // so element selection must address sub-word lanes explicitly.
    let standard_main = format!(
        r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let logical_idx = params.base + gid.x;
    let coords = decompose_idx(logical_idx);
    let pred = cond_is_true(params.offset_cond + cond_index(coords));
    let t = on_true[params.offset_true + true_index(coords)];
    let f = on_false[params.offset_false + false_index(coords) - params.base];
    dst[params.offset_dst + logical_idx - params.base] = select(f, t, pred);
}}"#
    );
    let u8_main = format!(
        r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let lane0 = params.base + gid.x * 4u;
    if (gid.x * 4u >= params.ne) {{ return; }}
    var out_word: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
        let logical_idx = lane0 + lane;
        if (logical_idx >= params.base + params.ne) {{ break; }}
        let coords = decompose_idx(logical_idx);
        let pred = cond_is_true(params.offset_cond + cond_index(coords));
        let t_idx = params.offset_true + true_index(coords);
        let f_idx = params.offset_false + false_index(coords) - params.base;
        let t_b = (on_true[t_idx / 4u] >> (8u * (t_idx % 4u))) & 0xffu;
        let f_b = (on_false[f_idx / 4u] >> (8u * (f_idx % 4u))) & 0xffu;
        let b = select(f_b, t_b, pred);
        out_word = out_word | (b << (8u * lane));
    }}
    dst[params.offset_dst / 4u + gid.x] = out_word;
}}"#
    );
    let i64_main = format!(
        r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let logical_idx = params.base + gid.x;
    let coords = decompose_idx(logical_idx);
    let pred = cond_is_true(params.offset_cond + cond_index(coords));
    let t_idx = params.offset_true + true_index(coords);
    let f_idx = params.offset_false + false_index(coords) - params.base;
    let lo = select(on_false[2u * f_idx], on_true[2u * t_idx], pred);
    let hi = select(on_false[2u * f_idx + 1u], on_true[2u * t_idx + 1u], pred);
    let dst_idx = params.offset_dst + logical_idx - params.base;
    dst[2u * dst_idx] = lo;
    dst[2u * dst_idx + 1u] = hi;
}}"#
    );
    let (ty, main) = match dtype {
        DType::U8 => ("u32", u8_main),
        DType::I64 => ("u32", i64_main),
        _ => (wgpu_scalar_type(dtype)?, standard_main),
    };
    let prelude = if dtype == DType::F16 {
        "enable f16;\n"
    } else {
        ""
    };
    Ok(format!(
        r#"{prelude}
struct Params {{
    ne: u32,
    offset_cond: u32,
    offset_true: u32,
    offset_false: u32,
    offset_dst: u32,
    stride_cond0: u32,
    stride_cond1: u32,
    stride_cond2: u32,
    stride_cond3: u32,
    stride_true0: u32,
    stride_true1: u32,
    stride_true2: u32,
    stride_true3: u32,
    stride_false0: u32,
    stride_false1: u32,
    stride_false2: u32,
    stride_false3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    base: u32,
}};

@group(0) @binding(0) var<storage, read> cond_words: array<u32>;
@group(0) @binding(1) var<storage, read> on_true: array<{ty}>;
@group(0) @binding(2) var<storage, read> on_false: array<{ty}>;
@group(0) @binding(3) var<storage, read_write> dst: array<{ty}>;
@group(0) @binding(4) var<uniform> params: Params;

fn decompose_idx(_i: u32) -> vec4<u32> {{
    var i = _i;
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;
    return vec4<u32>(i0, i1, i2, i3);
}}

fn cond_index(idx: vec4<u32>) -> u32 {{
    return idx.x * params.stride_cond0 + idx.y * params.stride_cond1 +
           idx.z * params.stride_cond2 + idx.w * params.stride_cond3;
}}

fn true_index(idx: vec4<u32>) -> u32 {{
    return idx.x * params.stride_true0 + idx.y * params.stride_true1 +
           idx.z * params.stride_true2 + idx.w * params.stride_true3;
}}

fn false_index(idx: vec4<u32>) -> u32 {{
    return idx.x * params.stride_false0 + idx.y * params.stride_false1 +
           idx.z * params.stride_false2 + idx.w * params.stride_false3;
}}

fn cond_is_true(idx: u32) -> bool {{
    let word = cond_words[idx / 4u];
    let shift = (idx % 4u) * 8u;
    return ((word >> shift) & 0xffu) != 0u;
}}

{main}
"#
    ))
}

fn custom_unary_wgsl(expr: &str) -> String {
    custom_unary_wgsl_body(&format!("dst[params.offset_dst + gid.x] = {expr};"))
}

fn custom_unary_wgsl_body(body: &str) -> String {
    format!(
        r#"
struct Params {{
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    _pad0: u32,
    _pad1: u32,
}};
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    var i = gid.x;
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;
    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;
    let x = src[params.offset_src + src_idx];
    {body}
}}
"#
    )
}

fn erf_unary_wgsl() -> String {
    custom_unary_wgsl_body(
        r#"let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
  let poly = (((((1.061405429 * t + -1.453152027) * t + 1.421413741) * t + -0.284496736) * t + 0.254829592) * t);
    dst[params.offset_dst + gid.x] = sign(x) * (1.0 - poly * exp(-ax * ax));"#,
    )
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Dynamic-offset uniform binding. Slot size must match `WgpuDevice::uniform_dyn_slot`.
fn uniform_entry_dyn(binding: u32, slot_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: NonZeroU64::new(slot_size),
        },
        count: None,
    }
}

fn buffer_binding<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Bind a dynamic-offset uniform slot (base offset 0, size = slot_size).
fn uniform_binding_dyn<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    slot_size: u64,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let size = NonZeroU64::new(slot_size).ok_or_else(|| {
        Error::Msg("wgpu dynamic uniform slot size must be non-zero".into()).bt()
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: Some(size),
        }),
    })
}

fn buffer_binding_range<'a>(
    binding: u32,
    buffer: &'a wgpu::Buffer,
    offset_bytes: u64,
    size_bytes: u64,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let size = NonZeroU64::new(size_bytes).ok_or_else(|| {
        Error::Msg("wgpu binding range size must be non-zero".into()).bt()
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: offset_bytes,
            size: Some(size),
        }),
    })
}

fn storage_buffer_binding<'a>(
    device: &WgpuDevice,
    binding: u32,
    buffer: &'a wgpu::Buffer,
    dtype: DType,
    start_elem: usize,
    num_elems: usize,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let elem_size = dtype.size_in_bytes();
    let offset_bytes = (start_elem * elem_size) as u64;
    let size_bytes = (num_elems * elem_size).next_multiple_of(4) as u64;
    if size_bytes == 0 {
        // Empty tensors use a 1-byte dummy buffer; wgpu requires non-zero bindings.
        return buffer_binding_range(binding, buffer, 0, buffer.size().max(1));
    }
    let align = device.inner.limits.min_storage_buffer_offset_alignment as u64;
    if !offset_bytes.is_multiple_of(align) {
        return Err(Error::Msg(format!(
            "wgpu buffer binding offset {offset_bytes} not aligned to {align}"
        ))
        .bt());
    }
    let max = device.inner.limits.max_storage_buffer_binding_size;
    if size_bytes > max {
        return Err(Error::Msg(format!(
            "wgpu storage buffer binding size {size_bytes} exceeds max {max}"
        ))
        .bt());
    }
    buffer_binding_range(binding, buffer, offset_bytes, size_bytes)
}

fn layout_storage_binding<'a>(
    device: &WgpuDevice,
    binding: u32,
    buffer: &'a wgpu::Buffer,
    layout: &Layout,
    dtype: DType,
    storage_count: usize,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let start = layout.start_offset();
    let num_elems = if layout.is_contiguous() {
        layout.shape().elem_count()
    } else {
        storage_count.saturating_sub(start)
    };
    storage_buffer_binding(device, binding, buffer, dtype, start, num_elems)
}

fn storage_layout_binding<'a>(
    storage: &'a WgpuStorage,
    layout: &Layout,
    binding: u32,
) -> Result<wgpu::BindGroupEntry<'a>> {
    layout_storage_binding(
        &storage.device,
        binding,
        &storage.buffer,
        layout,
        storage.dtype,
        storage.count,
    )
}

fn layout_binding_bytes(storage: &WgpuStorage, layout: &Layout) -> usize {
    let elem_size = storage.dtype.size_in_bytes().max(1);
    let start = layout.start_offset();
    let num_elems = if layout.is_contiguous() {
        layout.shape().elem_count()
    } else {
        storage.count.saturating_sub(start)
    };
    let raw = num_elems.saturating_mul(elem_size);
    raw.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
}

impl WgpuStorage {
    fn run_unary_like(&self, layout: &Layout, shader: &str, label: &'static str) -> Result<Self> {
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let params = UnaryParams {
            ne: count.try_into()?,
            offset_src: layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src0: strides[0],
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            _pad0: 0,
            _pad1: 0,
        };
        let slot = self.device.inner.uniform_dyn_slot;
        let (param_buffer, dyn_off) = self
            .device
            .write_uniform_slot(any_as_bytes(&params))?;
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry_dyn(2, slot),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            uniform_binding_dyn(2, &param_buffer, slot)?,
        ];
        self.device.run_compute_linear_dyn(
            shader,
            &entries,
            &bindings,
            count as u32,
            &[dyn_off],
            label,
        )?;
        Ok(dst)
    }

    fn run_scale(&self, layout: &Layout, scale: f32, bias: f32) -> Result<Self> {
        let count = layout.shape().elem_count();
        let params = ScaleParams {
            offset_src: 0,
            offset_dst: 0,
            stride_src1: 0,
            stride_src2: 0,
            stride_src3: 0,
            stride_dst1: 0,
            stride_dst2: 0,
            stride_dst3: 0,
            ne: count.try_into()?,
            ne0: 0,
            ne1: 0,
            ne2: 0,
            scale,
            bias,
        };
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        if self.dtype == DType::BF16 && layout.is_contiguous() && bias == 0.0 {
            // ponytail: in-place pair-write — saves one full attn-scores buffer at long prefill.
            let mut inplace_params = params;
            inplace_params.offset_src = 0;
            inplace_params.offset_dst = 0;
            let param_buffer = self
                .device
                .write_uniform_params(any_as_bytes(&inplace_params))?;
            let bindings = [
                storage_layout_binding(self, layout, 0)?,
                storage_layout_binding(self, layout, 1)?,
                buffer_binding(2, &param_buffer),
            ];
            let pairs = count.div_ceil(2) as u32;
            self.device.run_compute_linear(
                &bf16_scale_shader(),
                &entries,
                &bindings,
                pairs,
                "candle-wgpu-scale-bf16-inplace",
            )?;
            return Ok(WgpuStorage {
                buffer: self.buffer.clone(),
                device: self.device.clone(),
                count,
                dtype: self.dtype,
            });
        }
        let dst = if self.dtype == DType::BF16 {
            if !layout.is_contiguous() {
                let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
                <Self as BackendStorage>::copy_strided_src(self, &mut materialized, 0, layout)?;
                let mat_layout = Layout::contiguous(layout.shape());
                return materialized.run_scale(&mat_layout, scale, bias);
            }
            unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? }
        } else {
            let (dims, strides) = dims4(layout)?;
            let _ = (dims, strides);
            unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? }
        };
        let mut dispatch_params = params;
        dispatch_params.offset_src = 0;
        dispatch_params.offset_dst = 0;
        let shader = if self.dtype == DType::BF16 {
            bf16_scale_shader()
        } else {
            let (dims, strides) = dims4(layout)?;
            dispatch_params.stride_src1 = strides[1];
            dispatch_params.stride_src2 = strides[2];
            dispatch_params.stride_src3 = strides[3];
            dispatch_params.stride_dst1 = dims[0];
            dispatch_params.stride_dst2 = dims[0] * dims[1];
            dispatch_params.stride_dst3 = dims[0] * dims[1] * dims[2];
            dispatch_params.ne0 = dims[0];
            dispatch_params.ne1 = dims[1];
            dispatch_params.ne2 = dims[2];
            candle_wgpu_kernels::scale_shader(WG_SIZE)
                .ok_or_else(|| Error::Msg("wgpu shader scale.wgsl not embedded".into()).bt())?
        };
        let elem_size = self.dtype.size_in_bytes();
        let total_bytes = count * elem_size;
        let max_binding_bytes = self.device.inner.limits.max_storage_buffer_binding_size as usize;
        if layout.is_contiguous() && total_bytes > max_binding_bytes {
            let align = self.device.inner.limits.min_storage_buffer_offset_alignment as usize;
            let mut chunk_bytes_cap = max_binding_bytes;
            if chunk_bytes_cap > align {
                chunk_bytes_cap = (chunk_bytes_cap / align) * align;
            }
            if chunk_bytes_cap == 0 {
                return Err(Error::Msg("wgpu scale chunk size is zero".into()).bt());
            }
            let start_bytes = layout.start_offset() * elem_size;
            let mut base_bytes = 0usize;
            while base_bytes < total_bytes {
                let chunk_bytes = (total_bytes - base_bytes).min(chunk_bytes_cap);
                let chunk_elems = chunk_bytes / elem_size;
                let mut chunk_params = dispatch_params;
                chunk_params.offset_src = 0;
                chunk_params.offset_dst = 0;
                chunk_params.ne = chunk_elems.try_into()?;
                if self.dtype != DType::BF16 {
                    chunk_params.stride_src1 = 0;
                    chunk_params.stride_src2 = 0;
                    chunk_params.stride_src3 = 0;
                    chunk_params.stride_dst1 = 0;
                    chunk_params.stride_dst2 = 0;
                    chunk_params.stride_dst3 = 0;
                    chunk_params.ne0 = chunk_elems.try_into()?;
                    chunk_params.ne1 = 1;
                    chunk_params.ne2 = 1;
                }
                // Distinct ring slot per chunk so deferred multi-dispatch
                // does not share one uniform's last write.
                let param_buffer = self
                    .device
                    .write_uniform_params(any_as_bytes(&chunk_params))?;
                let chunk_bindings = [
                    buffer_binding_range(
                        0,
                        &self.buffer,
                        (start_bytes + base_bytes) as u64,
                        chunk_bytes as u64,
                    )?,
                    buffer_binding_range(1, &dst.buffer, base_bytes as u64, chunk_bytes as u64)?,
                    buffer_binding(2, &param_buffer),
                ];
                let work = if self.dtype == DType::BF16 {
                    chunk_elems.div_ceil(2) as u32
                } else {
                    chunk_elems as u32
                };
                self.device.run_compute_linear(
                    &shader,
                    &entries,
                    &chunk_bindings,
                    work,
                    "candle-wgpu-scale",
                )?;
                base_bytes += chunk_bytes;
            }
            return Ok(dst);
        }
        let param_buffer = self
            .device
            .write_uniform_params(any_as_bytes(&dispatch_params))?;
        let bindings = [
            storage_layout_binding(self, layout, 0)?,
            storage_layout_binding(&dst, &Layout::contiguous(layout.shape()), 1)?,
            buffer_binding(2, &param_buffer),
        ];
        let work = if self.dtype == DType::BF16 {
            count.div_ceil(2) as u32
        } else {
            count as u32
        };
        self.device
            .run_compute_linear(&shader, &entries, &bindings, work, "candle-wgpu-scale")?;
        Ok(dst)
    }

    fn run_clamp(&self, layout: &Layout, min: f32, max: f32) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu clamp").bt());
        }
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let params = ClampParams {
            ne: count.try_into()?,
            offset_src: layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src0: strides[0],
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            clamp_min: min,
            clamp_max: max,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-clamp-params"),
                size: std::mem::size_of::<ClampParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let shader = unary_shader("clamp", self.dtype)?;
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-clamp",
        )?;
        Ok(dst)
    }

    fn run_cmp_u8(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: CmpOp,
    ) -> Result<Self> {
        if !matches!(
            self.dtype,
            DType::F32 | DType::F16 | DType::U8 | DType::U32 | DType::I64
        ) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu cmp").bt());
        }
        if rhs.dtype != self.dtype {
            return Err(Error::UnsupportedDTypeForOp(rhs.dtype, "wgpu cmp").bt());
        }
        self.device
            .same_device(&rhs.device)
            .then_some(())
            .ok_or_else(|| {
                Error::DeviceMismatchBinaryOp {
                    lhs: self.device.location(),
                    rhs: rhs.device.location(),
                    op: "cmp",
                }
                .bt()
            })?;
        if lhs_layout.dims().len() > 4 || rhs_layout.dims().len() > 4 {
            let (lhs, lhs_l) = if lhs_layout.dims().len() > 4 {
                self.materialize_rank_gt4_compact(lhs_layout)?
            } else {
                (
                    self.try_clone(lhs_layout)?,
                    Layout::contiguous(lhs_layout.shape()),
                )
            };
            let (rhs, rhs_l) = if rhs_layout.dims().len() > 4 {
                rhs.materialize_rank_gt4_compact(rhs_layout)?
            } else {
                (
                    rhs.try_clone(rhs_layout)?,
                    Layout::contiguous(rhs_layout.shape()),
                )
            };
            return lhs.run_cmp_u8(&rhs, &lhs_l, &rhs_l, op);
        }
        let (lhs_dims, lhs_strides) = dims4(lhs_layout)?;
        let (rhs_dims, rhs_strides) = dims4(rhs_layout)?;
        let count = lhs_layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(lhs_layout.shape(), DType::U8)? };
        let params = BinaryParams {
            ne: count.try_into()?,
            offset_src0: lhs_layout.start_offset().try_into()?,
            offset_src1: rhs_layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src0_0: lhs_strides[0],
            stride_src0_1: lhs_strides[1],
            stride_src0_2: lhs_strides[2],
            stride_src0_3: lhs_strides[3],
            stride_src1_0: rhs_strides[0],
            stride_src1_1: rhs_strides[1],
            stride_src1_2: rhs_strides[2],
            stride_src1_3: rhs_strides[3],
            a_ne0: lhs_dims[0],
            a_ne1: lhs_dims[1],
            a_ne2: lhs_dims[2],
            b_ne0: rhs_dims[0],
            b_ne1: rhs_dims[1],
            b_ne2: rhs_dims[2],
            b_ne3: rhs_dims[3],
            _pad0: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-cmp-params"),
                size: std::mem::size_of::<BinaryParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &rhs.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let packed_words = count.div_ceil(4);
        let workgroups: u32 = packed_words.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            &custom_cmp_wgsl(op, self.dtype)?,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-cmp",
        )?;
        Ok(dst)
    }

    fn run_where_u8_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::U8 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu where_cond").bt());
        }
        if !matches!(
            t.dtype,
            DType::F32 | DType::F16 | DType::U8 | DType::U32 | DType::I64
        ) {
            return Err(Error::UnsupportedDTypeForOp(t.dtype, "wgpu where_cond").bt());
        }
        if f.dtype != t.dtype {
            return Err(Error::UnsupportedDTypeForOp(f.dtype, "wgpu where_cond").bt());
        }
        self.device
            .same_device(&t.device)
            .then_some(())
            .ok_or_else(|| {
                Error::DeviceMismatchBinaryOp {
                    lhs: self.device.location(),
                    rhs: t.device.location(),
                    op: "where_cond",
                }
                .bt()
            })?;
        self.device
            .same_device(&f.device)
            .then_some(())
            .ok_or_else(|| {
                Error::DeviceMismatchBinaryOp {
                    lhs: self.device.location(),
                    rhs: f.device.location(),
                    op: "where_cond",
                }
                .bt()
            })?;
        if layout.dims().len() > 4 || t_l.dims().len() > 4 || f_l.dims().len() > 4 {
            let (cond, cond_l) = if layout.dims().len() > 4 {
                self.materialize_rank_gt4_compact(layout)?
            } else {
                (self.try_clone(layout)?, Layout::contiguous(layout.shape()))
            };
            let (t, t_l) = if t_l.dims().len() > 4 {
                t.materialize_rank_gt4_compact(t_l)?
            } else {
                (t.try_clone(t_l)?, Layout::contiguous(t_l.shape()))
            };
            let (f, f_l) = if f_l.dims().len() > 4 {
                f.materialize_rank_gt4_compact(f_l)?
            } else {
                (f.try_clone(f_l)?, Layout::contiguous(f_l.shape()))
            };
            return cond.run_where_u8_cond(&cond_l, &t, &t_l, &f, &f_l);
        }
        let (dims, cond_strides) = dims4(layout)?;
        let (_, true_strides) = dims4(t_l)?;
        let (_, false_strides) = dims4(f_l)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { t.device.alloc_uninit(layout.shape(), t.dtype)? };
        let params = WhereParams {
            ne: count.try_into()?,
            offset_cond: 0,
            offset_true: 0,
            offset_false: 0,
            offset_dst: 0,
            stride_cond0: cond_strides[0],
            stride_cond1: cond_strides[1],
            stride_cond2: cond_strides[2],
            stride_cond3: cond_strides[3],
            stride_true0: true_strides[0],
            stride_true1: true_strides[1],
            stride_true2: true_strides[2],
            stride_true3: true_strides[3],
            stride_false0: false_strides[0],
            stride_false1: false_strides[1],
            stride_false2: false_strides[2],
            stride_false3: false_strides[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            base: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-where-params"),
                size: std::mem::size_of::<WhereParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ];
        let max_binding_bytes = self.device.inner.limits.max_storage_buffer_binding_size as usize;
        let dst_layout = Layout::contiguous(layout.shape());
        let needs_chunk = layout_binding_bytes(self, layout) > max_binding_bytes
            || layout_binding_bytes(t, t_l) > max_binding_bytes
            || layout_binding_bytes(f, f_l) > max_binding_bytes
            || layout_binding_bytes(&dst, &dst_layout) > max_binding_bytes;
        if needs_chunk {
            if !f_l.is_contiguous() {
                return Err(Error::Msg(
                    "wgpu where chunking requires a contiguous on_false operand".into(),
                )
                .bt());
            }
            let value_elem_size = t.dtype.size_in_bytes();
            let align = self.device.inner.limits.min_storage_buffer_offset_alignment as usize;
            let mut chunk_bytes_cap = max_binding_bytes;
            if chunk_bytes_cap > align {
                chunk_bytes_cap = (chunk_bytes_cap / align) * align;
            }
            if chunk_bytes_cap == 0 {
                return Err(Error::Msg("wgpu where chunk size is zero".into()).bt());
            }
            let chunk_elems_cap = chunk_bytes_cap / value_elem_size.max(1);
            if chunk_elems_cap == 0 {
                return Err(Error::Msg("wgpu where chunk elems is zero".into()).bt());
            }
            let mut base = 0usize;
            while base < count {
                let chunk_elems = (count - base).min(chunk_elems_cap);
                let chunk_shape = Shape::from(chunk_elems);
                let false_chunk_l =
                    Layout::contiguous_with_offset(chunk_shape.clone(), f_l.start_offset() + base);
                let dst_chunk_l = Layout::contiguous_with_offset(chunk_shape, base);
                let mut chunk_params = params;
                chunk_params.ne = chunk_elems.try_into()?;
                chunk_params.base = base.try_into()?;
                chunk_params.offset_false = 0;
                chunk_params.offset_dst = 0;
                self.device
                    .inner
                    .queue
                    .write_buffer(&param_buffer, 0, any_as_bytes(&chunk_params));
                let chunk_bindings = [
                    storage_layout_binding(self, layout, 0)?,
                    storage_layout_binding(t, t_l, 1)?,
                    storage_layout_binding(f, &false_chunk_l, 2)?,
                    storage_layout_binding(&dst, &dst_chunk_l, 3)?,
                    buffer_binding(4, &param_buffer),
                ];
                let workgroups = (chunk_elems as u32).div_ceil(WG_SIZE);
                t.device.run_compute(
                    &custom_where_u8_wgsl(t.dtype)?,
                    &entries,
                    &chunk_bindings,
                    workgroups,
                    "candle-wgpu-where",
                )?;
                base += chunk_elems;
            }
            return Ok(dst);
        }
        let bindings = [
            storage_layout_binding(self, layout, 0)?,
            storage_layout_binding(t, t_l, 1)?,
            storage_layout_binding(f, f_l, 2)?,
            storage_layout_binding(&dst, &dst_layout, 3)?,
            buffer_binding(4, &param_buffer),
        ];
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        t.device.run_compute(
            &custom_where_u8_wgsl(t.dtype)?,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-where",
        )?;
        Ok(dst)
    }

    fn run_fill_inplace(&self, layout: &Layout, value: f32) -> Result<()> {
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        if count == 0 {
            return Ok(());
        }
        let params = FillParams {
            ne: count.try_into()?,
            offset_src: layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src0: strides[0],
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            fill_val: value,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-fill-params"),
                size: std::mem::size_of::<FillParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [storage_entry(0, false), uniform_entry(1)];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &param_buffer),
        ];
        let shader =
            candle_wgpu_kernels::fill_inplace_shader(wgpu_kernel_dtype(self.dtype)?, WG_SIZE);
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        self.device
            .run_compute(&shader, &entries, &bindings, workgroups, "candle-wgpu-fill")
    }

    fn run_raw_fill_inplace(&self, layout: &Layout, value: [u32; 2]) -> Result<()> {
        let packed_dtype = matches!(
            self.dtype,
            DType::U8 | DType::F8E4M3 | DType::F8E8M0 | DType::I16 | DType::BF16 | DType::F16
        );
        if layout.dims().len() > 4 || (packed_dtype && !layout.is_contiguous()) {
            match layout.strided_blocks() {
                crate::StridedBlocks::SingleBlock { start_offset, len } => {
                    let block_layout =
                        Layout::contiguous_with_offset(Shape::from(len), start_offset);
                    return self.run_raw_fill_inplace(&block_layout, value);
                }
                crate::StridedBlocks::UniformBlocks {
                    start_offset,
                    block_len,
                    count,
                    src_stride,
                } => {
                    for i in 0..count {
                        let off = start_offset
                            .checked_add(i * src_stride)
                            .ok_or_else(|| Error::msg("wgpu fill uniform offset overflow"))?;
                        let block_layout =
                            Layout::contiguous_with_offset(Shape::from(block_len), off);
                        self.run_raw_fill_inplace(&block_layout, value)?;
                    }
                    return Ok(());
                }
                crate::StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len,
                } => {
                    if block_len == 0 {
                        return Ok(());
                    }
                    for start_offset in block_start_index {
                        let block_layout =
                            Layout::contiguous_with_offset(Shape::from(block_len), start_offset);
                        self.run_raw_fill_inplace(&block_layout, value)?;
                    }
                    return Ok(());
                }
            }
        }
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        if count == 0 {
            return Ok(());
        }
        let params = RawFillParams {
            ne: count.try_into()?,
            offset_dst: layout.start_offset().try_into()?,
            stride_dst0: strides[0],
            stride_dst1: strides[1],
            stride_dst2: strides[2],
            stride_dst3: strides[3],
            dst_ne0: dims[0],
            dst_ne1: dims[1],
            dst_ne2: dims[2],
            value0: value[0],
            value1: value[1],
        };
        let body = match self.dtype {
            DType::U8 | DType::F8E4M3 | DType::F8E8M0 => {
                r#"
    let word_idx = params.offset_dst / 4u + gid.x;
    let word_base = word_idx * 4u;
    if (word_base >= params.offset_dst + params.ne) { return; }
    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
        let elem_idx = word_base + lane;
        var bits: u32;
        if (elem_idx >= params.offset_dst && elem_idx < params.offset_dst + params.ne) {
            bits = params.value0 & 0xffu;
        } else {
            bits = (dst[word_idx] >> (8u * lane)) & 0xffu;
        }
        w = w | (bits << (8u * lane));
    }
    dst[word_idx] = w;
"#
            }
            DType::I16 | DType::BF16 | DType::F16 => {
                r#"
    let word_idx = params.offset_dst / 2u + gid.x;
    let word_base = word_idx * 2u;
    if (word_base >= params.offset_dst + params.ne) { return; }
    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 2u; lane = lane + 1u) {
        let elem_idx = word_base + lane;
        var bits: u32;
        if (elem_idx >= params.offset_dst && elem_idx < params.offset_dst + params.ne) {
            bits = params.value0 & 0xffffu;
        } else {
            bits = (dst[word_idx] >> (16u * lane)) & 0xffffu;
        }
        w = w | (bits << (16u * lane));
    }
    dst[word_idx] = w;
"#
            }
            DType::U32 | DType::I32 | DType::F32 => {
                r#"
    let e = params.offset_dst + dst_index(gid.x);
    dst[e] = params.value0;
"#
            }
            DType::I64 | DType::F64 => {
                r#"
    let e = params.offset_dst + dst_index(gid.x);
    dst[2u * e] = params.value0;
    dst[2u * e + 1u] = params.value1;
"#
            }
            dtype => return Err(Error::UnsupportedDTypeForOp(dtype, "wgpu const_set").bt()),
        };
        let dst_decl = "@group(0) @binding(0) var<storage, read_write> dst: array<u32>;";
        let shader = format!(
            r#"
struct Params {{
    ne: u32,
    offset_dst: u32,
    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32,
    value0: u32,
    value1: u32,
}};

{dst_decl}
@group(0) @binding(1) var<uniform> params: Params;

fn dst_index(_i: u32) -> u32 {{
    var i = _i;
    let i3 = i / (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    i = i % (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    let i2 = i / (params.dst_ne1 * params.dst_ne0);
    i = i % (params.dst_ne1 * params.dst_ne0);
    let i1 = i / params.dst_ne0;
    let i0 = i % params.dst_ne0;
    return i0 * params.stride_dst0 + i1 * params.stride_dst1 +
           i2 * params.stride_dst2 + i3 * params.stride_dst3;
}}

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
{body}
}}
"#
        );
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-raw-fill-params"),
                size: std::mem::size_of::<RawFillParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [storage_entry(0, false), uniform_entry(1)];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &param_buffer),
        ];
        let work_items = match self.dtype {
            DType::U8 | DType::F8E4M3 | DType::F8E8M0 => {
                (layout.start_offset() % 4 + count).div_ceil(4)
            }
            DType::I16 | DType::BF16 | DType::F16 => {
                (layout.start_offset() % 2 + count).div_ceil(2)
            }
            _ => count,
        };
        let workgroups: u32 = work_items.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-raw-fill",
        )
    }

    /// Emulated casts for dtypes WGSL cannot express directly: `U8` lives as
    /// four packed bytes per `u32` word and `I64` as a lo/hi `u32` pair.
    /// Float -> integer conversions follow Rust `as` semantics (saturating,
    /// NaN -> 0); integer -> integer conversions truncate like Rust `as`.
    fn run_emulated_cast(&self, layout: &Layout, dst_dtype: DType) -> Result<Self> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut materialized, 0, layout)?;
            let contiguous = Layout::contiguous(layout.shape());
            return materialized.run_emulated_cast(&contiguous, dst_dtype);
        }
        let ne = layout.shape().elem_count();
        // Loads the source element `i` as an f32 `v` plus, for I64 sources,
        // exact lo/hi words (`v` alone loses precision past 2^24).
        let load = match self.dtype {
            DType::F32 => "let v = bitcast<f32>(src[i]); let lo = 0u; let hi = 0u;",
            DType::F16 => {
                "let v = unpack2x16float(src[i / 2u])[i % 2u]; let lo = 0u; let hi = 0u;"
            }
            DType::BF16 => {
                "let v = bitcast<f32>(((src[i / 2u] >> (16u * (i % 2u))) & 0xffffu) << 16u); let lo = 0u; let hi = 0u;"
            }
            DType::U8 => {
                "let b = (src[i / 4u] >> (8u * (i % 4u))) & 0xffu; let v = f32(b); let lo = b; let hi = 0u;"
            }
            DType::U32 => "let sw = src[i]; let v = f32(sw); let lo = sw; let hi = 0u;",
            DType::I32 => {
                "let sw = bitcast<i32>(src[i]); let v = f32(sw); let lo = src[i]; let hi = select(0u, 0xffffffffu, sw < 0);"
            }
            DType::I64 => {
                "let lo = src[2u * i]; let hi = src[2u * i + 1u]; let v = f32(bitcast<i32>(hi)) * 4294967296.0 + f32(lo);"
            }
            DType::I16 => {
                "let u = (src[i / 2u] >> (16u * (i % 2u))) & 0xffffu; let sw = select(i32(u), i32(u) - 65536, (u & 0x8000u) != 0u); let v = f32(sw); let lo = src[i / 2u]; let hi = 0u;"
            }
            other => return Err(Error::UnsupportedDTypeForOp(other, "wgpu emulated cast").bt()),
        };
        let src_is_float = matches!(self.dtype, DType::F32 | DType::F16 | DType::BF16);
        // `conv_*` snippets produce the destination scalar from (v, lo, hi).
        let conv_u8 = if src_is_float {
            "select(u32(clamp(v, 0.0, 255.0)), 0u, v != v)"
        } else {
            "lo & 0xffu"
        };
        let conv_u32 = if src_is_float {
            "select(select(u32(clamp(v, 0.0, 4294967040.0)), 0xffffffffu, v >= 4294967296.0), 0u, v != v || v <= 0.0)"
        } else {
            "lo"
        };
        let conv_f32 = "v";
        let conv_bf16 =
            "((bitcast<u32>(v) + (0x7fffu + ((bitcast<u32>(v) >> 16u) & 1u))) >> 16u) & 0xffffu";
        let body = match dst_dtype {
            DType::U8 => format!(
                r#"
    if (gid.x >= params.chunk_wi) {{ return; }}
    let wi = params.wg_base + gid.x;
    let base = wi * 4u;
    if (base >= params.ne) {{ return; }}
    var w: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k++) {{
        let i = base + k;
        if (i < params.ne) {{
            {load}
            let b = {conv_u8};
            w |= (b & 0xffu) << (8u * k);
        }}
    }}
    dst[wi] = w;
"#
            ),
            DType::F16 => format!(
                r#"
    if (gid.x >= params.chunk_wi) {{ return; }}
    let wi = params.wg_base + gid.x;
    let base = wi * 2u;
    if (base >= params.ne) {{ return; }}
    var v0: f32 = 0.0;
    var v1: f32 = 0.0;
    {{
        let i = base;
        {load}
        v0 = {conv_f32};
    }}
    if (base + 1u < params.ne) {{
        let i = base + 1u;
        {load}
        v1 = {conv_f32};
    }}
    dst[wi] = pack2x16float(vec2<f32>(v0, v1));
"#
            ),
            DType::BF16 => format!(
                r#"
    if (gid.x >= params.chunk_wi) {{ return; }}
    let wi = params.wg_base + gid.x;
    let base = wi * 2u;
    if (base >= params.ne) {{ return; }}
    var w: u32 = 0u;
    {{
        let i = base;
        {load}
        let bf = {conv_bf16};
        w = w | bf;
    }}
    if (base + 1u < params.ne) {{
        let i = base + 1u;
        {load}
        let bf = {conv_bf16};
        w = w | (bf << 16u);
    }}
    dst[wi] = w;
"#
            ),
            DType::F32 => format!(
                r#"
    if (gid.x >= params.chunk_wi) {{ return; }}
    let i = params.wg_base + gid.x;
    if (i >= params.ne) {{ return; }}
    {load}
    dst[i] = bitcast<u32>({conv_f32});
"#
            ),
            DType::U32 => format!(
                r#"
    if (gid.x >= params.chunk_wi) {{ return; }}
    let i = params.wg_base + gid.x;
    if (i >= params.ne) {{ return; }}
    {load}
    dst[i] = {conv_u32};
"#
            ),
            DType::I64 => {
                let split = if src_is_float {
                    // Rust `as` saturates; f32 cannot exactly represent i64
                    // bounds, so clamp on the f32 side before splitting.
                    r#"
    var out_lo: u32 = 0u;
    var out_hi: u32 = 0u;
    if (v == v) {
        var x = clamp(v, -9223371487098961920.0, 9223371487098961920.0);
        let neg = x < 0.0;
        var a = abs(trunc(x));
        let hi_f = floor(a / 4294967296.0);
        let lo_f = a - hi_f * 4294967296.0;
        out_hi = u32(hi_f);
        out_lo = u32(lo_f);
        if (neg) {
            // two's complement negate
            out_lo = ~out_lo;
            out_hi = ~out_hi;
            if (out_lo == 0xffffffffu) { out_lo = 0u; out_hi += 1u; } else { out_lo += 1u; }
        }
    }
"#
                } else {
                    r#"
    let out_lo = lo;
    let out_hi = hi;
"#
                };
                format!(
                    r#"
    if (gid.x >= params.chunk_wi) {{ return; }}
    let i = params.wg_base + gid.x;
    if (i >= params.ne) {{ return; }}
    {load}
    {split}
    dst[2u * i] = out_lo;
    dst[2u * i + 1u] = out_hi;
"#
                )
            }
            DType::I16 => format!(
                r#"
    if (gid.x >= params.chunk_wi) {{ return; }}
    let wi = params.wg_base + gid.x;
    let base = wi * 2u;
    if (base >= params.ne) {{ return; }}
    var w: u32 = 0u;
    {{
        let i = base;
        {load}
        let si = i32(trunc(clamp(v, f32(-32768), f32(32767))));
        let u = bitcast<u32>(si) & 0xffffu;
        w = w | u;
    }}
    if (base + 1u < params.ne) {{
        let i = base + 1u;
        {load}
        let si = i32(trunc(clamp(v, f32(-32768), f32(32767))));
        let u = bitcast<u32>(si) & 0xffffu;
        w = w | (u << 16u);
    }}
    dst[wi] = w;
"#
            ),
            other => {
                return Err(Error::UnsupportedDTypeForOp(other, "wgpu emulated cast dst").bt())
            }
        };
        let shader = format!(
            r#"
struct Params {{
    ne: u32,
    wg_base: u32,
    chunk_wi: u32,
    _pad2: u32,
}};
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
{body}
}}
"#
        );
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), dst_dtype)? };
        let groups_of = match dst_dtype {
            DType::U8 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            _ => 1,
        };
        let work_items = ne.div_ceil(groups_of);
        let max_workgroups = wgpu_dispatch_wg_cap(&self.device) as usize;
        let max_work_items = max_workgroups * WG_SIZE as usize;
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let ne_u32: u32 = ne.try_into()?;
        let mut wg_base = 0usize;
        while wg_base < work_items {
            let chunk_wi = (work_items - wg_base).min(max_work_items);
            let params = F64CastParams {
                ne: ne_u32,
                _pad0: wg_base.try_into()?,
                _pad1: chunk_wi.try_into()?,
                _pad2: 0,
            };
            let param_buffer = self
                .device
                .inner
                .device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("candle-wgpu-emulated-cast-params"),
                    size: std::mem::size_of::<F64CastParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            self.device
                .inner
                .queue
                .write_buffer(&param_buffer, 0, any_as_bytes(&params));
            let bindings = [
                buffer_binding(0, &self.buffer),
                buffer_binding(1, &dst.buffer),
                buffer_binding(2, &param_buffer),
            ];
            let workgroups: u32 = chunk_wi.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
            self.device.run_compute(
                &shader,
                &entries,
                &bindings,
                workgroups,
                "candle-wgpu-emulated-cast",
            )?;
            wg_base += chunk_wi;
        }
        Ok(dst)
    }

    fn run_copy_to_dtype(&self, layout: &Layout, dtype: DType, shader: &str) -> Result<Self> {
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), dtype)? };
        self.run_copy_into(layout, &dst, 0, shader)?;
        Ok(dst)
    }

    fn materialize_to_f32(&self, layout: &Layout) -> Result<Self> {
        if self.dtype == DType::F32 {
            if layout.is_contiguous() && layout.start_offset() == 0 {
                self.try_clone(layout)
            } else if layout.dims().len() > 4 {
                let (materialized, _) = self.materialize_rank_gt4_compact(layout)?;
                Ok(materialized)
            } else {
                let mut materialized =
                    unsafe { self.device.alloc_uninit(layout.shape(), DType::F32)? };
                self.copy_strided_src(&mut materialized, 0, layout)?;
                Ok(materialized)
            }
        } else if layout.dims().len() > 4 {
            let (materialized, compact_l) = self.materialize_rank_gt4_compact(layout)?;
            materialized.to_dtype(&compact_l, DType::F32)
        } else {
            let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut materialized, 0, layout)?;
            let contiguous = Layout::contiguous(layout.shape());
            materialized.to_dtype(&contiguous, DType::F32)
        }
    }

    fn rank_gt4_batch_start_offset(layout: &Layout, mut batch_idx: usize) -> usize {
        let rank = layout.dims().len();
        let mut offset = layout.start_offset();
        for dim in (0..rank - 2).rev() {
            let idx = batch_idx % layout.dims()[dim];
            batch_idx /= layout.dims()[dim];
            offset += idx * layout.stride()[dim];
        }
        offset
    }

    fn compact_rank_gt4_shape(layout: &Layout) -> Shape {
        let rank = layout.dims().len();
        let prefix = layout.dims()[..rank - 3].iter().product::<usize>();
        Shape::from(vec![
            prefix,
            layout.dims()[rank - 3],
            layout.dims()[rank - 2],
            layout.dims()[rank - 1],
        ])
    }

    fn compact_rank_gt4_start_offset(layout: &Layout, mut prefix_idx: usize) -> usize {
        let rank = layout.dims().len();
        let mut offset = layout.start_offset();
        for dim in (0..rank - 3).rev() {
            let idx = prefix_idx % layout.dims()[dim];
            prefix_idx /= layout.dims()[dim];
            offset += idx * layout.stride()[dim];
        }
        offset
    }

    fn materialize_rank_gt4_compact(&self, layout: &Layout) -> Result<(Self, Layout)> {
        let compact_shape = Self::compact_rank_gt4_shape(layout);
        let compact_layout = Layout::contiguous(compact_shape.clone());
        let mut out = unsafe { self.device.alloc_uninit(&compact_shape, self.dtype)? };
        let rank = layout.dims().len();
        let slice_shape = Shape::from(vec![
            layout.dims()[rank - 3],
            layout.dims()[rank - 2],
            layout.dims()[rank - 1],
        ]);
        let slice_stride = vec![
            layout.stride()[rank - 3],
            layout.stride()[rank - 2],
            layout.stride()[rank - 1],
        ];
        let slice_len = slice_shape.elem_count();
        let prefix = compact_shape.dims()[0];
        for prefix_idx in 0..prefix {
            let slice_layout = Layout::new(
                slice_shape.clone(),
                slice_stride.clone(),
                Self::compact_rank_gt4_start_offset(layout, prefix_idx),
            );
            self.copy_strided_src(&mut out, prefix_idx * slice_len, &slice_layout)?;
        }
        Ok((out, compact_layout))
    }

    fn materialize_rank_gt4_matmul_operand(
        &self,
        layout: &Layout,
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let out_shape = Shape::from(vec![batch, rows, cols]);
        let mut out = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let matrix_shape = Shape::from(vec![rows, cols]);
        let rank = layout.dims().len();
        let matrix_stride = vec![layout.stride()[rank - 2], layout.stride()[rank - 1]];
        for batch_idx in 0..batch {
            let matrix_layout = Layout::new(
                matrix_shape.clone(),
                matrix_stride.clone(),
                Self::rank_gt4_batch_start_offset(layout, batch_idx),
            );
            self.copy_strided_src(&mut out, batch_idx * rows * cols, &matrix_layout)?;
        }
        Ok(out)
    }

    fn materialize_rank_gt4_matmul_operand_to_f32(
        &self,
        layout: &Layout,
        batch: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if self.dtype == DType::F32 {
            return self.materialize_rank_gt4_matmul_operand(layout, batch, rows, cols);
        }
        let out_shape = Shape::from(vec![batch, rows, cols]);
        let mut out = unsafe { self.device.alloc_uninit(&out_shape, DType::F32)? };
        let matrix_shape = Shape::from(vec![rows, cols]);
        let matrix_contiguous = Layout::contiguous(matrix_shape.clone());
        let rank = layout.dims().len();
        let matrix_stride = vec![layout.stride()[rank - 2], layout.stride()[rank - 1]];
        for batch_idx in 0..batch {
            let matrix_layout = Layout::new(
                matrix_shape.clone(),
                matrix_stride.clone(),
                Self::rank_gt4_batch_start_offset(layout, batch_idx),
            );
            let mut matrix = unsafe { self.device.alloc_uninit(&matrix_shape, self.dtype)? };
            self.copy_strided_src(&mut matrix, 0, &matrix_layout)?;
            let matrix_f32 = matrix.to_dtype(&matrix_contiguous, DType::F32)?;
            matrix_f32.copy_strided_src(&mut out, batch_idx * rows * cols, &matrix_contiguous)?;
        }
        Ok(out)
    }

    fn bf16_unary_via_f32(
        &self,
        layout: &Layout,
        f: impl FnOnce(&Self, &Layout) -> Result<Self>,
    ) -> Result<Self> {
        let f32_storage = self.materialize_to_f32(layout)?;
        let contiguous = if layout.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(layout))
        } else {
            Layout::contiguous(layout.shape())
        };
        let out_f32 = f(&f32_storage, &contiguous)?;
        out_f32.to_dtype(&contiguous, DType::BF16)
    }

    fn bf16_binary_via_f32(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        f: impl FnOnce(&Self, &Self, &Layout, &Layout) -> Result<Self>,
    ) -> Result<Self> {
        if rhs.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(rhs.dtype, "wgpu bf16 binary").bt());
        }
        self.device
            .same_device(&rhs.device)
            .then_some(())
            .ok_or_else(|| {
                Error::DeviceMismatchBinaryOp {
                    lhs: self.device.location(),
                    rhs: rhs.device.location(),
                    op: "bf16 binary",
                }
                .bt()
            })?;
        let lhs_f32 = self.materialize_to_f32(lhs_layout)?;
        let rhs_f32 = rhs.materialize_to_f32(rhs_layout)?;
        let lhs_contiguous = if lhs_layout.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(lhs_layout))
        } else {
            Layout::contiguous(lhs_layout.shape())
        };
        let rhs_contiguous = if rhs_layout.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(rhs_layout))
        } else {
            Layout::contiguous(rhs_layout.shape())
        };
        let out_f32 = f(&lhs_f32, &rhs_f32, &lhs_contiguous, &rhs_contiguous)?;
        out_f32.to_dtype(&lhs_contiguous, DType::BF16)
    }

    fn bf16_cmp_via_f32(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: CmpOp,
    ) -> Result<Self> {
        if rhs.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(rhs.dtype, "wgpu bf16 cmp").bt());
        }
        self.device
            .same_device(&rhs.device)
            .then_some(())
            .ok_or_else(|| {
                Error::DeviceMismatchBinaryOp {
                    lhs: self.device.location(),
                    rhs: rhs.device.location(),
                    op: "bf16 cmp",
                }
                .bt()
            })?;
        let lhs_f32 = self.materialize_to_f32(lhs_layout)?;
        let rhs_f32 = rhs.materialize_to_f32(rhs_layout)?;
        let lhs_contiguous = if lhs_layout.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(lhs_layout))
        } else {
            Layout::contiguous(lhs_layout.shape())
        };
        let rhs_contiguous = if rhs_layout.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(rhs_layout))
        } else {
            Layout::contiguous(rhs_layout.shape())
        };
        lhs_f32.run_cmp_u8(&rhs_f32, &lhs_contiguous, &rhs_contiguous, op)
    }

    fn bf16_where_via_f32(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if t.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(t.dtype, "wgpu bf16 where_cond").bt());
        }
        if f.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(f.dtype, "wgpu bf16 where_cond").bt());
        }
        let t_f32 = t.materialize_to_f32(t_l)?;
        let f_f32 = f.materialize_to_f32(f_l)?;
        let t_contiguous = if t_l.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(t_l))
        } else {
            Layout::contiguous(t_l.shape())
        };
        let f_contiguous = if f_l.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(f_l))
        } else {
            Layout::contiguous(f_l.shape())
        };
        let out_f32 =
            self.run_where_u8_cond(layout, &t_f32, &t_contiguous, &f_f32, &f_contiguous)?;
        let out_layout = if layout.dims().len() > 4 {
            Layout::contiguous(Self::compact_rank_gt4_shape(layout))
        } else {
            Layout::contiguous(layout.shape())
        };
        out_f32.to_dtype(&out_layout, DType::BF16)
    }

    fn gpu_resident_via_f32(
        &self,
        layout: &Layout,
        out_shape: &Shape,
        op_name: &'static str,
        f: impl FnOnce(&Self, &Layout) -> Result<Self>,
    ) -> Result<Self> {
        let src_dtype = self.dtype;
        match src_dtype {
            DType::F32 => f(self, layout),
            DType::F16 | DType::BF16 => {
                let src_f32 = self.materialize_to_f32(layout)?;
                let src_f32_l = Layout::contiguous(layout.shape());
                let out_f32 = f(&src_f32, &src_f32_l)?;
                let out_l = Layout::contiguous(out_shape);
                out_f32.to_dtype(&out_l, src_dtype)
            }
            _ => Err(Error::UnsupportedDTypeForOp(src_dtype, op_name).bt()),
        }
    }

    fn cuda_parity_conv_via_f32(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        out_shape: &Shape,
        op_name: &'static str,
        f: impl FnOnce(&Self, &Layout, &Self, &Layout) -> Result<Self>,
    ) -> Result<Self> {
        let src_dtype = self.dtype;
        if kernel.dtype != src_dtype {
            return Err(Error::UnsupportedDTypeForOp(kernel.dtype, op_name).bt());
        }
        match src_dtype {
            DType::F32 => f(self, layout, kernel, kernel_l),
            DType::F16 | DType::BF16 => {
                let input_f32 = self.materialize_to_f32(layout)?;
                let kernel_f32 = kernel.materialize_to_f32(kernel_l)?;
                let input_l = Layout::contiguous(layout.shape());
                let kernel_l = if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
                    kernel_l.clone()
                } else {
                    Layout::contiguous(kernel_l.shape())
                };
                let out_f32 = f(&input_f32, &input_l, &kernel_f32, &kernel_l)?;
                let out_l = Layout::contiguous(out_shape);
                out_f32.to_dtype(&out_l, src_dtype)
            }
            _ => Err(Error::UnsupportedDTypeForOp(src_dtype, op_name).bt()),
        }
    }

    // Strided same-dtype copy for dtypes WGSL cannot address as scalars:
    // U8 elements are packed four-per-u32-word, BF16/I16 two-per-u32-word,
    // and I64/F64 elements are lo/hi word pairs. The destination is written
    // contiguously starting at dst_offset.
    fn run_emulated_strided_copy_into(
        &self,
        layout: &Layout,
        dst: &Self,
        dst_offset: usize,
    ) -> Result<()> {
        let count = layout.shape().elem_count();
        if count == 0 {
            return Ok(());
        }
        let (src_dims, src_strides) = dims4(layout)?;
        let body = match self.dtype {
            DType::BF16 | DType::I16 => {
                // One thread per packed 16-bit output word: gathers two strided
                // source halfwords, so concurrent threads never share a word.
                r#"
    if (gid.x >= params.ne) { return; }
    let wi = params.elem_base + gid.x;
    let word_idx = params.offset_dst / 2u + wi;
    let word_base = word_idx * 2u;
    if (word_base >= params.offset_dst + params.ne_total) { return; }
    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 2u; lane = lane + 1u) {
        let elem_idx = word_base + lane;
        var bits: u32;
        if (elem_idx >= params.offset_dst && elem_idx < params.offset_dst + params.ne_total) {
            let logical_idx = elem_idx - params.offset_dst;
            let e = params.offset_src + src_index(logical_idx);
            bits = (src[e / 2u] >> (16u * (e % 2u))) & 0xffffu;
        } else {
            bits = (dst[word_idx] >> (16u * lane)) & 0xffffu;
        }
        w = w | (bits << (16u * lane));
    }
    dst[word_idx] = w;
"#
            }
            DType::U8 => {
                // One thread per packed output word: gathers four strided
                // source bytes, so concurrent threads never share a word.
                // Bytes past ne_total in the final word are merged from the
                // existing destination so neighbors in dst are not clobbered.
                r#"
    if (gid.x >= params.ne) { return; }
    let wi = params.elem_base + gid.x;
    let base = wi * 4u;
    if (base >= params.ne_total) { return; }
    let word_idx = params.offset_dst / 4u + wi;
    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
        let logical_idx = base + lane;
        var b: u32;
        if (logical_idx < params.ne_total) {
            let e = params.offset_src + src_index(logical_idx);
            b = (src[e / 4u] >> (8u * (e % 4u))) & 0xffu;
        } else {
            b = (dst[word_idx] >> (8u * lane)) & 0xffu;
        }
        w = w | (b << (8u * lane));
    }
    dst[word_idx] = w;
"#
            }
            DType::I64 | DType::F64 => {
                r#"
    if (gid.x >= params.ne) { return; }
    let i = params.elem_base + gid.x;
    if (i >= params.ne_total) { return; }
    let e = params.offset_src + src_index(i);
    let d = params.offset_dst + i;
    dst[2u * d] = src[2u * e];
    dst[2u * d + 1u] = src[2u * e + 1u];
"#
            }
            other => {
                return Err(Error::UnsupportedDTypeForOp(other, "wgpu emulated strided copy").bt())
            }
        };
        let shader = format!(
            r#"
struct Params {{
    ne: u32,
    ne_total: u32,
    offset_src: u32,
    offset_dst: u32,
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    elem_base: u32,
    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,
}};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn src_index(_i: u32) -> u32 {{
    var i = _i;
    let i3 = i / (params.src_ne2 * params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne2 * params.src_ne1 * params.src_ne0);
    let i2 = i / (params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne1 * params.src_ne0);
    let i1 = i / params.src_ne0;
    let i0 = i % params.src_ne0;
    return i0 * params.stride_src0 + i1 * params.stride_src1 +
           i2 * params.stride_src2 + i3 * params.stride_src3;
}}

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{{body}}}
"#
        );
        if self.dtype == DType::U8
            && (!layout.start_offset().is_multiple_of(4) || !dst_offset.is_multiple_of(4))
        {
            // Byte-level sub-word offsets would force read-modify-write on
            // shared destination words; keep that case on the safe path.
            return Err(unsupported("emulated u8 copy with sub-word offset"));
        }
        let work_items = match self.dtype {
            DType::BF16 | DType::I16 => (dst_offset % 2 + count).div_ceil(2),
            DType::U8 => count.div_ceil(4),
            _ => count,
        };
        let max_workgroups = wgpu_dispatch_wg_cap(&self.device) as usize;
        let max_work_items = max_workgroups * WG_SIZE as usize;
        let entries = [
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let count_u32: u32 = count.try_into()?;
        let mut wi_base = 0usize;
        while wi_base < work_items {
            let chunk_wi = (work_items - wi_base).min(max_work_items);
            let params = EmulatedStridedCopyParams {
                ne: chunk_wi.try_into()?,
                ne_total: count_u32,
                offset_src: layout.start_offset().try_into()?,
                offset_dst: dst_offset.try_into()?,
                stride_src0: src_strides[0],
                stride_src1: src_strides[1],
                stride_src2: src_strides[2],
                stride_src3: src_strides[3],
                elem_base: wi_base.try_into()?,
                src_ne0: src_dims[0],
                src_ne1: src_dims[1],
                src_ne2: src_dims[2],
            };
            let param_buffer = self
                .device
                .inner
                .device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("candle-wgpu-emulated-copy-params"),
                    size: std::mem::size_of::<EmulatedStridedCopyParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            self.device
                .inner
                .queue
                .write_buffer(&param_buffer, 0, any_as_bytes(&params));
            let bindings = [
                buffer_binding(0, &self.buffer),
                buffer_binding(1, &dst.buffer),
                buffer_binding(2, &param_buffer),
            ];
            let workgroups: u32 = chunk_wi.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
            self.device.run_compute(
                &shader,
                &entries,
                &bindings,
                workgroups,
                "candle-wgpu-emulated-copy",
            )?;
            wi_base += chunk_wi;
        }
        Ok(())
    }

    fn run_f64_f32_cast(&self, layout: &Layout, dst_dtype: DType) -> Result<Self> {
        if !matches!(
            (self.dtype, dst_dtype),
            (DType::F32, DType::F64) | (DType::F64, DType::F32) | (DType::F64, DType::F64)
        ) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu f64 cast").bt());
        }
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut materialized, 0, layout)?;
            let contiguous = Layout::contiguous(layout.shape());
            return materialized.run_f64_f32_cast(&contiguous, dst_dtype);
        }
        let ne = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), dst_dtype)? };
        let body = match (self.dtype, dst_dtype) {
            (DType::F32, DType::F64) => {
                r#"
    if (gid.x >= params.ne) { return; }
    let words = f32_to_f64_words(src[gid.x]);
    dst[2u * gid.x] = words.x;
    dst[2u * gid.x + 1u] = words.y;
"#
            }
            (DType::F64, DType::F32) => {
                r#"
    if (gid.x >= params.ne) { return; }
    dst[gid.x] = f64_to_f32_bits(src[2u * gid.x], src[2u * gid.x + 1u]);
"#
            }
            (DType::F64, DType::F64) => {
                r#"
    if (gid.x >= params.ne) { return; }
    dst[2u * gid.x] = src[2u * gid.x];
    dst[2u * gid.x + 1u] = src[2u * gid.x + 1u];
"#
            }
            _ => unreachable!(),
        };
        let shader = format!(
            r#"
struct Params {{
    ne: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn highest_bit(v: u32) -> u32 {{
    var out = 0u;
    for (var bit = 0u; bit < 23u; bit = bit + 1u) {{
        if ((v & (1u << bit)) != 0u) {{
            out = bit;
        }}
    }}
    return out;
}}

fn shift_left_words(v: u32, shift: u32) -> vec2<u32> {{
    if (shift >= 32u) {{
        return vec2<u32>(0u, v << (shift - 32u));
    }}
    return vec2<u32>(v << shift, v >> (32u - shift));
}}

fn f32_to_f64_words(bits: u32) -> vec2<u32> {{
    let sign = bits >> 31u;
    let exp = (bits >> 23u) & 0xffu;
    let frac = bits & 0x7fffffu;
    if (exp == 0xffu) {{
        let nan_hi = select(0u, 0x80000u | (frac >> 3u), frac != 0u);
        return vec2<u32>(frac << 29u, (sign << 31u) | (0x7ffu << 20u) | nan_hi);
    }}
    if (exp == 0u) {{
        if (frac == 0u) {{
            return vec2<u32>(0u, sign << 31u);
        }}
        let h = highest_bit(frac);
        let exp64 = h + 874u;
        let rem = frac ^ (1u << h);
        let words = shift_left_words(rem, 52u - h);
        return vec2<u32>(words.x, (sign << 31u) | (exp64 << 20u) | (words.y & 0xfffffu));
    }}
    let exp64 = exp + 896u;
    return vec2<u32>(frac << 29u, (sign << 31u) | (exp64 << 20u) | (frac >> 3u));
}}

fn round_shift_right_words(hi: u32, lo: u32, shift: u32) -> u32 {{
    var base = 0u;
    var guard = false;
    var sticky = false;
    if (shift < 32u) {{
        base = (hi << (32u - shift)) | (lo >> shift);
        guard = ((lo >> (shift - 1u)) & 1u) != 0u;
        sticky = (lo & ((1u << (shift - 1u)) - 1u)) != 0u;
    }} else if (shift == 32u) {{
        base = hi;
        guard = (lo & 0x80000000u) != 0u;
        sticky = (lo & 0x7fffffffu) != 0u;
    }} else {{
        let s = shift - 32u;
        base = hi >> s;
        guard = ((hi >> (s - 1u)) & 1u) != 0u;
        sticky = ((hi & ((1u << (s - 1u)) - 1u)) != 0u) || (lo != 0u);
    }}
    if (guard && (sticky || ((base & 1u) != 0u))) {{
        base = base + 1u;
    }}
    return base;
}}

fn f64_to_f32_bits(lo: u32, hi: u32) -> u32 {{
    let sign = hi >> 31u;
    let exp = (hi >> 20u) & 0x7ffu;
    let frac_hi = hi & 0xfffffu;
    if (exp == 0x7ffu) {{
        let is_nan = (frac_hi != 0u) || (lo != 0u);
        let payload = select(0u, 0x400000u | ((frac_hi << 3u) | (lo >> 29u)), is_nan);
        return (sign << 31u) | 0x7f800000u | payload;
    }}
    if (exp == 0u) {{
        return sign << 31u;
    }}
    let exp32_signed = i32(exp) - 1023 + 127;
    let sig_hi = (1u << 20u) | frac_hi;
    if (exp32_signed >= 255) {{
        return (sign << 31u) | 0x7f800000u;
    }}
    if (exp32_signed <= 0) {{
        if (exp32_signed < -23) {{
            return sign << 31u;
        }}
        let shift = u32(30 - exp32_signed);
        let mant = round_shift_right_words(sig_hi, lo, shift);
        if (mant >= (1u << 23u)) {{
            return (sign << 31u) | (1u << 23u);
        }}
        return (sign << 31u) | mant;
    }}
    var mant24 = (1u << 23u) | (frac_hi << 3u) | (lo >> 29u);
    let rem = lo & 0x1fffffffu;
    if ((rem > 0x10000000u) || (rem == 0x10000000u && ((mant24 & 1u) != 0u))) {{
        mant24 = mant24 + 1u;
    }}
    var exp32 = u32(exp32_signed);
    if (mant24 == (1u << 24u)) {{
        mant24 = 1u << 23u;
        exp32 = exp32 + 1u;
        if (exp32 >= 255u) {{
            return (sign << 31u) | 0x7f800000u;
        }}
    }}
    return (sign << 31u) | (exp32 << 23u) | (mant24 & 0x7fffffu);
}}

@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
{body}
}}
"#
        );
        let params = F64CastParams {
            ne: ne.try_into()?,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-f64-cast-params"),
                size: std::mem::size_of::<F64CastParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let workgroups: u32 = ne.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-f64-cast",
        )?;
        Ok(dst)
    }

    fn run_copy_into(
        &self,
        layout: &Layout,
        dst: &Self,
        dst_offset: usize,
        shader: &str,
    ) -> Result<()> {
        let count = layout.shape().elem_count();
        if count == 0 {
            return Ok(());
        }

        let max_workgroups = wgpu_dispatch_wg_cap(&self.device) as usize;
        let max_elems_per_dispatch = max_workgroups * WG_SIZE as usize;
        let chunk_linear = count > max_elems_per_dispatch && layout.is_contiguous();

        let (src_dims, src_strides) = dims4(layout)?;
        let dst_strides = contiguous_strides(src_dims);
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];

        let mut processed = 0usize;
        while processed < count {
            let chunk = (count - processed).min(max_elems_per_dispatch);
            let params = if chunk_linear {
                CopyParams {
                    ne: chunk.try_into()?,
                    offset_src: (layout.start_offset() + processed).try_into()?,
                    offset_dst: (dst_offset + processed).try_into()?,
                    stride_src0: 1,
                    stride_src1: 0,
                    stride_src2: 0,
                    stride_src3: 0,
                    stride_dst0: 1,
                    stride_dst1: 0,
                    stride_dst2: 0,
                    stride_dst3: 0,
                    src_ne0: chunk.try_into()?,
                    src_ne1: 1,
                    src_ne2: 1,
                    dst_ne0: chunk.try_into()?,
                    dst_ne1: 1,
                    dst_ne2: 1,
                    elem_base: 0,
                }
            } else {
                CopyParams {
                    ne: chunk.try_into()?,
                    offset_src: layout.start_offset().try_into()?,
                    offset_dst: dst_offset.try_into()?,
                    stride_src0: src_strides[0],
                    stride_src1: src_strides[1],
                    stride_src2: src_strides[2],
                    stride_src3: src_strides[3],
                    stride_dst0: dst_strides[0],
                    stride_dst1: dst_strides[1],
                    stride_dst2: dst_strides[2],
                    stride_dst3: dst_strides[3],
                    src_ne0: src_dims[0],
                    src_ne1: src_dims[1],
                    src_ne2: src_dims[2],
                    dst_ne0: src_dims[0],
                    dst_ne1: src_dims[1],
                    dst_ne2: src_dims[2],
                    elem_base: processed.try_into()?,
                }
            };
            let param_buffer = self
                .device
                .inner
                .device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("candle-wgpu-copy-params"),
                    size: std::mem::size_of::<CopyParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            self.device
                .inner
                .queue
                .write_buffer(&param_buffer, 0, any_as_bytes(&params));
            let bindings = [
                buffer_binding(0, &self.buffer),
                buffer_binding(1, &dst.buffer),
                buffer_binding(2, &param_buffer),
            ];
            let workgroups: u32 = chunk.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
            self.device
                .run_compute(shader, &entries, &bindings, workgroups, "candle-wgpu-copy")?;
            processed += chunk;
        }
        Ok(())
    }

    fn run_argmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        let rank = layout.dims().len();
        if rank == 0 {
            return Ok(unsafe { self.device.alloc_uninit(&Shape::from(&[] as &[usize]), DType::U32)? });
        }
        let ne0 = *layout.dims().last().unwrap_or(&1);
        let mut dst_dims = layout.dims().to_vec();
        dst_dims[rank - 1] = 1;
        let dst_shape = Shape::from(dst_dims);
        let rows = dst_shape.elem_count();
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::U32)? };
        let params = ArgMaxParams {
            offset_src: layout.start_offset().try_into()?,
            offset_dst: 0,
            ne0: ne0.try_into()?,
            _pad0: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-argmax-params"),
                size: std::mem::size_of::<ArgMaxParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::argmax_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader argmax.wgsl not embedded".into()).bt())?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.try_into()?,
            "candle-wgpu-argmax",
        )?;
        Ok(dst)
    }

    fn run_reduce_extrema_last_dim(&self, layout: &Layout, op: ReduceOp) -> Result<Self> {
        let mut materialized_storage;
        let materialized_layout;
        let src;
        let src_layout;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            src = self;
            src_layout = layout;
        } else {
            materialized_storage = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut materialized_storage, 0, layout)?;
            materialized_layout = Layout::contiguous(layout.shape());
            src = &materialized_storage;
            src_layout = &materialized_layout;
        }

        let rank = src_layout.dims().len();
        let mut ids_dims = src_layout.dims().to_vec();
        ids_dims[rank - 1] = 1;
        let ids_layout = Layout::contiguous(Shape::from(ids_dims));

        match op {
            ReduceOp::ArgMax => src.run_argmax_last_dim(src_layout),
            ReduceOp::Max => {
                let ids = src.run_argmax_last_dim(src_layout)?;
                <Self as BackendStorage>::gather(src, src_layout, &ids, &ids_layout, rank - 1)
            }
            ReduceOp::ArgMin | ReduceOp::Min => {
                let shader = unary_shader("neg", DType::F32)?;
                let neg = src.run_unary_like(src_layout, &shader, "candle-wgpu-reduce-neg")?;
                let neg_layout = Layout::contiguous(src_layout.shape());
                let ids = neg.run_argmax_last_dim(&neg_layout)?;
                if op == ReduceOp::ArgMin {
                    Ok(ids)
                } else {
                    <Self as BackendStorage>::gather(src, src_layout, &ids, &ids_layout, rank - 1)
                }
            }
            ReduceOp::Sum => {
                let rank = layout.dims().len();
                let perm = (0..rank).filter(|&i| i != rank - 1).chain(std::iter::once(rank - 1));
                let perm = perm.collect::<Vec<_>>();
                let perm_shape = perm.iter().map(|&i| layout.dims()[i]).collect::<Vec<_>>();
                let perm_stride = perm.iter().map(|&i| layout.stride()[i]).collect::<Vec<_>>();
                let perm_layout = Layout::new(
                    Shape::from(perm_shape.clone()),
                    perm_stride,
                    layout.start_offset(),
                );
                let mut permuted = unsafe {
                    self.device
                        .alloc_uninit(&Shape::from(perm_shape.clone()), self.dtype)?
                };
                <Self as BackendStorage>::copy_strided_src(self, &mut permuted, 0, &perm_layout)?;
                let permuted_layout = Layout::contiguous(Shape::from(perm_shape));
                <Self as BackendStorage>::reduce_op(&permuted, op, &permuted_layout, &[rank - 1])
            },
        }
    }

    fn run_reduce_non_last_dim(&self, op: ReduceOp, layout: &Layout, dim: usize) -> Result<Self> {
        let rank = layout.dims().len();
        let perm = (0..rank).filter(|&i| i != dim).chain(std::iter::once(dim));
        let perm = perm.collect::<Vec<_>>();
        let perm_shape = perm.iter().map(|&i| layout.dims()[i]).collect::<Vec<_>>();
        let perm_stride = perm.iter().map(|&i| layout.stride()[i]).collect::<Vec<_>>();
        let perm_layout = Layout::new(
            Shape::from(perm_shape.clone()),
            perm_stride,
            layout.start_offset(),
        );
        let mut permuted = unsafe {
            self.device
                .alloc_uninit(&Shape::from(perm_shape.clone()), self.dtype)?
        };
        <Self as BackendStorage>::copy_strided_src(self, &mut permuted, 0, &perm_layout)?;
        let permuted_layout = Layout::contiguous(Shape::from(perm_shape));
        <Self as BackendStorage>::reduce_op(&permuted, op, &permuted_layout, &[rank - 1])
    }

    fn run_reduce_multi_dim(
        &self,
        op: ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> Result<Self> {
        let mut current_shape = layout.dims().to_vec();
        let mut current_layout = layout.clone();
        let mut current_storage = None;
        for &dim in reduce_dims {
            let src = current_storage.as_ref().unwrap_or(self);
            let reduced = <Self as BackendStorage>::reduce_op(src, op, &current_layout, &[dim])?;
            current_shape[dim] = 1;
            current_layout = Layout::contiguous(Shape::from(current_shape.clone()));
            current_storage = Some(reduced);
        }
        match current_storage {
            Some(v) => Ok(v),
            None => self.try_clone(layout),
        }
    }

    pub(crate) fn argsort_last_dim(
        &self,
        layout: &Layout,
        asc: bool,
        last_dim: usize,
    ) -> Result<Self> {
        if matches!(
            self.dtype,
            DType::F16 | DType::BF16 | DType::U8 | DType::I16 | DType::I32 | DType::F8E4M3
        ) {
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape());
            return src_f32.argsort_last_dim_f32(&src_f32_layout, asc, last_dim);
        }
        match self.dtype {
            DType::F32 => self.argsort_last_dim_typed(layout, asc, last_dim, WgpuArgsortDType::F32),
            DType::U32 => self.argsort_last_dim_typed(layout, asc, last_dim, WgpuArgsortDType::U32),
            DType::I64 => self.argsort_last_dim_typed(layout, asc, last_dim, WgpuArgsortDType::I64),
            DType::F64 => self.argsort_last_dim_typed(layout, asc, last_dim, WgpuArgsortDType::F64),
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu argsort").bt()),
        }
    }

    fn argsort_last_dim_f32(&self, layout: &Layout, asc: bool, last_dim: usize) -> Result<Self> {
        self.argsort_last_dim_typed(layout, asc, last_dim, WgpuArgsortDType::F32)
    }

    fn argsort_last_dim_typed(
        &self,
        layout: &Layout,
        asc: bool,
        _last_dim: usize,
        sort_dtype: WgpuArgsortDType,
    ) -> Result<Self> {
        let expected = match sort_dtype {
            WgpuArgsortDType::F32 => DType::F32,
            WgpuArgsortDType::U32 => DType::U32,
            WgpuArgsortDType::I64 => DType::I64,
            WgpuArgsortDType::F64 => DType::F64,
        };
        if self.dtype != expected {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu argsort").bt());
        }
        let mut materialized_argsort;
        let argsort_layout_buf;
        let argsort_layout;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            argsort_layout = layout;
        } else {
            materialized_argsort = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut materialized_argsort, 0, layout)?;
            argsort_layout_buf = Layout::contiguous(layout.shape());
            argsort_layout = &argsort_layout_buf;
        }
        let last_dim = argsort_layout.dims().last().copied().unwrap_or(0);
        if last_dim == 0 {
            return Ok(unsafe { self.device.alloc_uninit(layout.shape(), DType::U32)? });
        }
        let workgroup_size = next_power_of_two_u32(last_dim.min(WG_SIZE as usize), "argsort")?;
        let (dims, strides) = dims4(argsort_layout)?;
        let count = argsort_layout.shape().elem_count();
        let nrows = count / last_dim;
        let dst = unsafe { self.device.alloc_uninit(argsort_layout.shape(), DType::U32)? };
        let dst_strides = contiguous_strides(dims);
        let npr = last_dim.div_ceil(workgroup_size as usize);
        let top_k = if npr == 1 {
            last_dim.try_into()?
        } else {
            workgroup_size
        };
        let params = ArgsortParams {
            offset_src: argsort_layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            stride_dst1: dst_strides[1],
            stride_dst2: dst_strides[2],
            stride_dst3: dst_strides[3],
            src_ne0: last_dim.try_into()?,
            ne1: dims[1],
            ne2: dims[2],
            ne0: last_dim.try_into()?,
            top_k,
            npr: npr.try_into()?,
            nrows: nrows.try_into()?,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-argsort-params"),
                size: std::mem::size_of::<ArgsortParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let shader = match sort_dtype {
            WgpuArgsortDType::F32 => candle_wgpu_kernels::argsort_shader(workgroup_size, asc)
                .ok_or_else(|| Error::Msg("wgpu shader argsort.wgsl not embedded".into()).bt())?,
            WgpuArgsortDType::U32 => candle_wgpu_kernels::argsort_u32_shader(workgroup_size, asc)
                .ok_or_else(|| {
                Error::Msg("wgpu shader argsort.wgsl not embedded".into()).bt()
            })?,
            WgpuArgsortDType::I64 => i64_argsort_wgsl(workgroup_size, asc),
            WgpuArgsortDType::F64 => f64_argsort_wgsl(workgroup_size, asc),
        };
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            (npr * nrows).try_into()?,
            "candle-wgpu-argsort",
        )?;
        if npr == 1 {
            return Ok(dst);
        }

        let mut current = dst;
        let mut scratch = unsafe { self.device.alloc_uninit(layout.shape(), DType::U32)? };
        let idx_layout = Layout::contiguous(layout.shape());
        let mut len = workgroup_size as usize;
        while len < last_dim {
            let nm = last_dim.div_ceil(2 * len);
            self.run_argsort_merge_pass(
                &current,
                &scratch,
                layout,
                &idx_layout,
                asc,
                len,
                nm,
                nrows,
            )?;
            std::mem::swap(&mut current, &mut scratch);
            len *= 2;
        }
        Ok(current)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_argsort_merge_pass(
        &self,
        idx_in: &Self,
        idx_out: &Self,
        layout: &Layout,
        idx_layout: &Layout,
        asc: bool,
        len: usize,
        nm: usize,
        nrows: usize,
    ) -> Result<()> {
        let (dims, strides) = dims4(layout)?;
        let (_, idx_strides) = dims4(idx_layout)?;
        let params = ArgsortMergeParams {
            offset_src: layout.start_offset().try_into()?,
            offset_in: 0,
            offset_out: 0,
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            stride_idx1: idx_strides[1],
            stride_idx2: idx_strides[2],
            stride_idx3: idx_strides[3],
            stride_out1: idx_strides[1],
            stride_out2: idx_strides[2],
            stride_out3: idx_strides[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            top_k: dims[0],
            len: len.try_into()?,
            nm: nm.try_into()?,
            nrows: nrows.try_into()?,
            _pad0: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-argsort-merge-params"),
                size: std::mem::size_of::<ArgsortMergeParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &idx_in.buffer),
            buffer_binding(2, &idx_out.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = match self.dtype {
            DType::U32 => {
                candle_wgpu_kernels::argsort_u32_merge_shader(WG_SIZE, asc).ok_or_else(|| {
                    Error::Msg("wgpu shader argsort_merge.wgsl not embedded".into()).bt()
                })?
            }
            DType::I64 => i64_argsort_merge_wgsl(asc),
            DType::F64 => f64_argsort_merge_wgsl(asc),
            _ => candle_wgpu_kernels::argsort_merge_shader(WG_SIZE, asc).ok_or_else(|| {
                Error::Msg("wgpu shader argsort_merge.wgsl not embedded".into()).bt()
            })?,
        };
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            (nm * nrows).try_into()?,
            "candle-wgpu-argsort-merge",
        )?;
        Ok(())
    }

    fn run_cumsum_last_dim(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu cumsum").bt());
        }
        let mut materialized_cumsum;
        let cumsum_layout_buf;
        let cumsum_layout;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            cumsum_layout = layout;
        } else {
            materialized_cumsum = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut materialized_cumsum, 0, layout)?;
            cumsum_layout_buf = Layout::contiguous(layout.shape());
            cumsum_layout = &cumsum_layout_buf;
        }
        let ne0 = *cumsum_layout
            .dims()
            .last()
            .ok_or_else(|| Error::Msg("cumsum scalar".into()))?;
        let rows = cumsum_layout.shape().elem_count() / ne0;
        let dst = unsafe { self.device.alloc_uninit(cumsum_layout.shape(), self.dtype)? };
        let params = CumsumParams {
            offset_src: cumsum_layout.start_offset().try_into()?,
            offset_dst: 0,
            ne0: ne0.try_into()?,
            _pad0: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-cumsum-params"),
                size: std::mem::size_of::<CumsumParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::cumsum_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader cumsum.wgsl not embedded".into()).bt())?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.try_into()?,
            "candle-wgpu-cumsum",
        )?;
        Ok(dst)
    }

    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        if !layout.is_contiguous() {
            let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut materialized, 0, layout)?;
            let mat_layout = Layout::contiguous(layout.shape());
            return materialized.softmax_last_dim(&mat_layout);
        }
        if self.dtype == DType::F16 {
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape().clone());
            let out_f32 = src_f32.softmax_last_dim(&src_f32_layout)?;
            return out_f32.to_dtype(&src_f32_layout, self.dtype);
        }
        if self.dtype == DType::BF16 {
            let (dims, strides) = dims4(layout)?;
            let count = layout.shape().elem_count();
            let dst_strides = contiguous_strides(dims);
            let offset = layout.start_offset().try_into()?;
            let params = SoftmaxParams {
                offset_src0: offset,
                offset_src1: 0,
                offset_sinks: 0,
                offset_dst: offset,
                stride_src01: strides[1],
                stride_src02: strides[2],
                stride_src03: strides[3],
                stride_src11: 0,
                stride_src12: 0,
                stride_src13: 0,
                stride_dst1: dst_strides[1],
                stride_dst2: dst_strides[2],
                stride_dst3: dst_strides[3],
                ne: count.try_into()?,
                ne0: dims[0],
                ne1: dims[1],
                ne2: dims[2],
                ne12: 1,
                ne13: 1,
                scale: 1.0,
                max_bias: 0.0,
                n_head_log2: 0.0,
                m0: 0.0,
                m1: 0.0,
                row_base: 0,
            };
            let param_buffer = self
                .device
                .inner
                .device
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("candle-wgpu-softmax-params"),
                    size: std::mem::size_of::<SoftmaxParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            self.device
                .inner
                .queue
                .write_buffer(&param_buffer, 0, any_as_bytes(&params));
            let entries = [storage_entry(0, false), uniform_entry(1)];
            let bindings = [
                buffer_binding(0, &self.buffer),
                buffer_binding(1, &param_buffer),
            ];
            let rows = count / layout.dims()[layout.dims().len() - 1];
            self.device.run_compute(
                &bf16_softmax_shader(),
                &entries,
                &bindings,
                rows.try_into()?,
                "candle-wgpu-softmax-bf16",
            )?;
            return Ok(WgpuStorage {
                buffer: self.buffer.clone(),
                device: self.device.clone(),
                count,
                dtype: self.dtype,
            });
        }
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu softmax").bt());
        }
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let dst_strides = contiguous_strides(dims);
        let params = SoftmaxParams {
            offset_src0: layout.start_offset().try_into()?,
            offset_src1: 0,
            offset_sinks: 0,
            offset_dst: 0,
            stride_src01: strides[1],
            stride_src02: strides[2],
            stride_src03: strides[3],
            stride_src11: 0,
            stride_src12: 0,
            stride_src13: 0,
            stride_dst1: dst_strides[1],
            stride_dst2: dst_strides[2],
            stride_dst3: dst_strides[3],
            ne: count.try_into()?,
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            ne12: 1,
            ne13: 1,
            scale: 1.0,
            max_bias: 0.0,
            n_head_log2: 0.0,
            m0: 0.0,
            m1: 0.0,
            row_base: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-softmax-params"),
                size: std::mem::size_of::<SoftmaxParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let rows = count / layout.dims()[layout.dims().len() - 1];
        let shader = candle_wgpu_kernels::softmax_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader soft_max.wgsl not embedded".into()).bt())?;
        let elem_size = self.dtype.size_in_bytes();
        let row_elems = layout.dims()[layout.dims().len() - 1];
        let row_bytes = row_elems * elem_size;
        let total_bytes = count * elem_size;
        let max_binding_bytes = self.device.inner.limits.max_storage_buffer_binding_size as usize;
        if total_bytes > max_binding_bytes {
            let align = self.device.inner.limits.min_storage_buffer_offset_alignment as usize;
            let mut row_bytes_cap = max_binding_bytes;
            if row_bytes_cap > align {
                row_bytes_cap = (row_bytes_cap / align) * align;
            }
            let rows_per_chunk = (row_bytes_cap / row_bytes.max(1)).max(1);
            let mut row_base = 0usize;
            while row_base < rows {
                let chunk_rows = (rows - row_base).min(rows_per_chunk);
                let chunk_bytes = chunk_rows * row_bytes;
                let src_byte_off = (layout.start_offset() + row_base * row_elems) * elem_size;
                let dst_byte_off = row_base * row_elems * elem_size;
                let mut chunk_params = params;
                chunk_params.offset_src0 = 0;
                chunk_params.offset_dst = 0;
                chunk_params.row_base = row_base.try_into()?;
                self.device
                    .inner
                    .queue
                    .write_buffer(&param_buffer, 0, any_as_bytes(&chunk_params));
                let chunk_bindings = [
                    buffer_binding_range(0, &self.buffer, src_byte_off as u64, chunk_bytes as u64)?,
                    buffer_binding_range(1, &dst.buffer, dst_byte_off as u64, chunk_bytes as u64)?,
                    buffer_binding(2, &param_buffer),
                ];
                self.device.run_compute(
                    &shader,
                    &entries,
                    &chunk_bindings,
                    chunk_rows.try_into()?,
                    "candle-wgpu-softmax",
                )?;
                row_base += chunk_rows;
            }
            return Ok(dst);
        }
        let bindings = [
            storage_layout_binding(self, layout, 0)?,
            storage_layout_binding(&dst, &Layout::contiguous(layout.shape()), 1)?,
            buffer_binding(2, &param_buffer),
        ];
        let mut ranged_params = params;
        ranged_params.offset_src0 = 0;
        ranged_params.offset_dst = 0;
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&ranged_params));
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.try_into()?,
            "candle-wgpu-softmax",
        )?;
        Ok(dst)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn(
        q: &WgpuStorage,
        q_layout: &Layout,
        k: &WgpuStorage,
        k_layout: &Layout,
        v: &WgpuStorage,
        v_layout: &Layout,
        scale: f32,
        causal: bool,
    ) -> Result<WgpuStorage> {
        use crate::DType;

        let ensure_contiguous = |src: &WgpuStorage, l: &Layout| -> Result<(WgpuStorage, Layout)> {
            if !l.is_contiguous() || l.start_offset() != 0 {
                let shape_ref = l.shape();
                let mut tmp = unsafe { src.device.alloc_uninit(shape_ref, src.dtype)? };
                src.copy_strided_src(&mut tmp, 0, l)?;
                Ok((tmp, Layout::contiguous(shape_ref.clone())))
            } else {
                Ok((src.try_clone(l)?, l.clone()))
            }
        };

        let (q_buf, q_l) = ensure_contiguous(q, q_layout)?;
        let (k_buf, k_l) = ensure_contiguous(k, k_layout)?;
        let (v_buf, v_l) = ensure_contiguous(v, v_layout)?;

        let dims_q = q_l.dims();
        let (b, h, seq_q, head_dim) = if dims_q.len() == 4 {
            (dims_q[0], dims_q[1], dims_q[2], dims_q[3])
        } else {
            return Err(Error::Msg("flash_attn expects 4D Q tensor [B,H,S,D]".into()).bt());
        };
        let seq_kv = k_l.dims()[2];
        let head_dim_v = v_l.dims()[3];

        let out_shape = Shape::from_dims(&[b, h, seq_q, head_dim_v]);
        let dst = unsafe { q.device.alloc_uninit(&out_shape, DType::F32)? };

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct FlashAttnParams {
            seq_q: u32,
            seq_kv: u32,
            head_dim: u32,
            head_dim_v: u32,
            num_heads: u32,
            num_kv_heads: u32,
            batch_size: u32,
            scale: f32,
            causal: u32,
        }

        let params = FlashAttnParams {
            seq_q: seq_q as u32,
            seq_kv: seq_kv as u32,
            head_dim: head_dim as u32,
            head_dim_v: head_dim_v as u32,
            num_heads: h as u32,
            num_kv_heads: k_l.dims()[1] as u32,
            batch_size: b as u32,
            scale,
            causal: if causal { 1 } else { 0 },
        };

        let param_buffer = q.device.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle-wgpu-flash-attn-params"),
            size: std::mem::size_of::<FlashAttnParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        q.device.inner.queue.write_buffer(&param_buffer, 0, any_as_bytes(&params));

        let entries = [
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ];
        let bindings = [
            buffer_binding(0, &q_buf.buffer),
            buffer_binding(1, &k_buf.buffer),
            buffer_binding(2, &v_buf.buffer),
            buffer_binding(3, &dst.buffer),
            buffer_binding(4, &param_buffer),
        ];

        let shader_source = candle_wgpu_kernels::get("flash_attn_simple.wgsl")
            .ok_or_else(|| Error::Msg("wgpu flash_attn_simple shader not found".into()).bt())?
            .source();
        let total_q_rows = (b * h * seq_q) as u32;
        q.device.run_compute_linear(
            shader_source,
            &entries,
            &bindings,
            total_q_rows,
            "candle-wgpu-flash-attn",
        )?;

        Ok(dst)
    }

    pub fn ggml_rope(
        &self,
        layout: &Layout,
        pos: &Self,
        pos_layout: &Layout,
        n_dims: usize,
        freq_base: f32,
        mode: u32,
    ) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 && self.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu rope").bt());
        }
        if wgpu_f16_emulates_f32(&self.device, self.dtype) {
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let src_l = Layout::contiguous(layout.shape().clone());
            let out_f32 = src_f32.ggml_rope(&src_l, pos, pos_layout, n_dims, freq_base, mode)?;
            return out_f32.to_dtype(&src_l, DType::F16);
        }
        if pos.dtype != DType::I32 {
            return Err(Error::UnsupportedDTypeForOp(pos.dtype, "wgpu rope positions").bt());
        }
        if !pos_layout.is_contiguous() || pos_layout.start_offset() != 0 {
            let mut pos_buf = unsafe { pos.device.alloc_uninit(pos_layout.shape(), pos.dtype)? };
            <Self as BackendStorage>::copy_strided_src(pos, &mut pos_buf, 0, pos_layout)?;
            let pos_contig_l = Layout::contiguous(pos_layout.shape());
            return self.ggml_rope(layout, &pos_buf, &pos_contig_l, n_dims, freq_base, mode);
        }
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let in_place = layout.is_contiguous();
        let offset_src0: u32 = layout.start_offset().try_into()?;
        let dst_strides = contiguous_strides(dims);
        let dst = if in_place {
            WgpuStorage {
                buffer: self.buffer.clone(),
                device: self.device.clone(),
                count,
                dtype: self.dtype,
            }
        } else {
            unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? }
        };
        let offset_dst = if in_place { offset_src0 } else { 0 };
        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        let params = WgpuRopeParams {
            offset_src0,
            offset_src1: pos_layout.start_offset().try_into()?,
            offset_src2: 0,
            offset_dst,
            stride_src01: strides[1],
            stride_src02: strides[2],
            stride_src03: strides[3],
            stride_dst1: dst_strides[1],
            stride_dst2: dst_strides[2],
            stride_dst3: dst_strides[3],
            n_threads: (count / 2).try_into()?,
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            n_dims: n_dims.try_into()?,
            mode,
            theta_scale,
            attn_factor: 1.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            corr_dim0: 0.0,
            corr_dim1: 0.0,
            sections0: 0,
            sections1: 0,
            sections2: 0,
            sections3: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-rope-params"),
                size: std::mem::size_of::<WgpuRopeParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &pos.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = if self.dtype == DType::BF16 {
            bf16_rope_shader()
        } else {
            candle_wgpu_kernels::rope_shader(wgpu_kernel_dtype(self.dtype)?, WG_SIZE)
                .ok_or_else(|| Error::Msg("wgpu shader rope.wgsl not embedded".into()).bt())?
        };
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            ((count / 2) as u32).div_ceil(WG_SIZE),
            "candle-wgpu-rope",
        )?;
        Ok(dst)
    }

    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_layout: &Layout,
        eps: f32,
    ) -> Result<Self> {
        // CUDA parity: mixed dtypes and F16 use a GPU-resident F32 hub.
        // rms_norm WGSL only supports f32/bf16 — always upconvert F16.
        if self.dtype != alpha.dtype
            || self.dtype == DType::F16
            || alpha.dtype == DType::F16
        {
            let out_dtype = self.dtype;
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let alpha_f32 = alpha.to_dtype(alpha_layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape().clone());
            let alpha_f32_layout = Layout::contiguous(alpha_layout.shape().clone());
            let out_f32 = src_f32.rms_norm(&src_f32_layout, &alpha_f32, &alpha_f32_layout, eps)?;
            return out_f32.to_dtype(&src_f32_layout, out_dtype);
        }
        if self.dtype != DType::F32 && self.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu rms_norm").bt());
        }
        if self.dtype == DType::BF16 && alpha.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(alpha.dtype, "wgpu rms_norm").bt());
        }
        if !layout.is_contiguous() || layout.start_offset() != 0
            || !alpha_layout.is_contiguous() || alpha_layout.start_offset() != 0
        {
            let src_shape = layout.shape().clone();
            let alpha_shape = alpha_layout.shape().clone();
            let mut src = unsafe { self.device.alloc_uninit(&src_shape, self.dtype)? };
            let mut alpha_tmp = unsafe { alpha.device.alloc_uninit(&alpha_shape, alpha.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut src, 0, layout)?;
            <Self as BackendStorage>::copy_strided_src(alpha, &mut alpha_tmp, 0, alpha_layout)?;
            let src_layout = Layout::contiguous(src_shape);
            let alpha_layout = Layout::contiguous(alpha_shape);
            return src.rms_norm(&src_layout, &alpha_tmp, &alpha_layout, eps);
        }
        let (dims, strides) = dims4(layout)?;
        let (alpha_dims, alpha_strides) = dims4(alpha_layout)?;
        let count = layout.shape().elem_count();
        let dst_strides = contiguous_strides(dims);
        let offset_rn = layout.start_offset().try_into()?;
        let params = RmsNormMulParams {
            offset_rn_src: offset_rn,
            offset_mul_src: alpha_layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_rn_src1: strides[1],
            stride_rn_src2: strides[2],
            stride_rn_src3: strides[3],
            stride_mul_src1: alpha_strides[1],
            stride_mul_src2: alpha_strides[2],
            stride_mul_src3: alpha_strides[3],
            stride_dst1: dst_strides[1],
            stride_dst2: dst_strides[2],
            stride_dst3: dst_strides[3],
            mul_src_ne0: alpha_dims[0],
            mul_src_ne1: alpha_dims[1],
            mul_src_ne2: alpha_dims[2],
            mul_src_ne3: alpha_dims[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            ne3: dims[3],
            eps,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-rms-norm-params"),
                size: std::mem::size_of::<RmsNormMulParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        if self.dtype == DType::BF16 {
            // ponytail: in-place — shader reads then overwrites rn_src row-wise.
            let mut inplace_params = params;
            inplace_params.offset_dst = offset_rn;
            self.device
                .inner
                .queue
                .write_buffer(&param_buffer, 0, any_as_bytes(&inplace_params));
            let bindings = [
                buffer_binding(0, &self.buffer),
                buffer_binding(1, &alpha.buffer),
                buffer_binding(2, &self.buffer),
                buffer_binding(3, &param_buffer),
            ];
            let rows = count / layout.dims()[layout.dims().len() - 1];
            let rows_u32: u32 = rows.try_into()?;
            let max_per_dim = wgpu_dispatch_wg_cap(&self.device);
            let (wg_x, wg_y) = compute_2d_workgroups(rows_u32, max_per_dim);
            self.device.run_compute_xyz(
                &bf16_rms_norm_shader(),
                &entries,
                &bindings,
                (wg_x, wg_y, 1),
                &[],
                "candle-wgpu-rms-norm-bf16-inplace",
            )?;
            return Ok(WgpuStorage {
                buffer: self.buffer.clone(),
                device: self.device.clone(),
                count,
                dtype: self.dtype,
            });
        }
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &alpha.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::rms_norm_mul_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader rms_norm_mul.wgsl not embedded".into()).bt())?;
        let rows = count / layout.dims()[layout.dims().len() - 1];
        let rows_u32: u32 = rows.try_into()?;
        let max_per_dim = wgpu_dispatch_wg_cap(&self.device);
        let (wg_x, wg_y) = compute_2d_workgroups(rows_u32, max_per_dim);
        self.device.run_compute_xyz(
            &shader,
            &entries,
            &bindings,
            (wg_x, wg_y, 1),
            &[],
            "candle-wgpu-rms-norm",
        )?;
        Ok(dst)
    }

    pub fn layer_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_layout: &Layout,
        beta: &Self,
        beta_layout: &Layout,
        eps: f32,
    ) -> Result<Self> {
        if self.dtype != DType::F32
            || alpha.dtype != DType::F32
            || beta.dtype != DType::F32
        {
            let out_dtype = self.dtype;
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let alpha_f32 = alpha.to_dtype(alpha_layout, DType::F32)?;
            let beta_f32 = beta.to_dtype(beta_layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape().clone());
            let alpha_f32_layout = Layout::contiguous(alpha_layout.shape().clone());
            let beta_f32_layout = Layout::contiguous(beta_layout.shape().clone());
            let out_f32 =
                src_f32.layer_norm(&src_f32_layout, &alpha_f32, &alpha_f32_layout, &beta_f32, &beta_f32_layout, eps)?;
            if out_dtype == DType::F32 {
                return Ok(out_f32);
            }
            return out_f32.to_dtype(&src_f32_layout, out_dtype);
        }
        if !layout.is_contiguous() || layout.start_offset() != 0
            || !alpha_layout.is_contiguous() || alpha_layout.start_offset() != 0
            || !beta_layout.is_contiguous() || beta_layout.start_offset() != 0
        {
            let src_shape = layout.shape().clone();
            let alpha_shape = alpha_layout.shape().clone();
            let beta_shape = beta_layout.shape().clone();
            let mut src = unsafe { self.device.alloc_uninit(&src_shape, self.dtype)? };
            let mut alpha_tmp = unsafe { alpha.device.alloc_uninit(&alpha_shape, alpha.dtype)? };
            let mut beta_tmp = unsafe { beta.device.alloc_uninit(&beta_shape, beta.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut src, 0, layout)?;
            <Self as BackendStorage>::copy_strided_src(alpha, &mut alpha_tmp, 0, alpha_layout)?;
            <Self as BackendStorage>::copy_strided_src(beta, &mut beta_tmp, 0, beta_layout)?;
            let src_layout = Layout::contiguous(src_shape);
            let alpha_layout = Layout::contiguous(alpha_shape);
            let beta_layout = Layout::contiguous(beta_shape);
            return src.layer_norm(&src_layout, &alpha_tmp, &alpha_layout, &beta_tmp, &beta_layout, eps);
        }
        let (dims, strides) = dims4(layout)?;
        let (alpha_dims, alpha_strides) = dims4(alpha_layout)?;
        let (_beta_dims, beta_strides) = dims4(beta_layout)?;
        let count = layout.shape().elem_count();
        let dst_strides = contiguous_strides(dims);

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct LayerNormParams {
            offset_src: u32,
            offset_alpha: u32,
            offset_beta: u32,
            offset_dst: u32,
            stride_src1: u32,
            stride_src2: u32,
            stride_src3: u32,
            stride_alpha1: u32,
            stride_alpha2: u32,
            stride_alpha3: u32,
            stride_beta1: u32,
            stride_beta2: u32,
            stride_beta3: u32,
            stride_dst1: u32,
            stride_dst2: u32,
            stride_dst3: u32,
            alpha_ne0: u32,
            alpha_ne1: u32,
            alpha_ne2: u32,
            alpha_ne3: u32,
            ne0: u32,
            ne1: u32,
            ne2: u32,
            ne3: u32,
            eps: f32,
        }

        let params = LayerNormParams {
            offset_src: layout.start_offset().try_into()?,
            offset_alpha: alpha_layout.start_offset().try_into()?,
            offset_beta: beta_layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            stride_alpha1: alpha_strides[1],
            stride_alpha2: alpha_strides[2],
            stride_alpha3: alpha_strides[3],
            stride_beta1: beta_strides[1],
            stride_beta2: beta_strides[2],
            stride_beta3: beta_strides[3],
            stride_dst1: dst_strides[1],
            stride_dst2: dst_strides[2],
            stride_dst3: dst_strides[3],
            alpha_ne0: alpha_dims[0],
            alpha_ne1: alpha_dims[1],
            alpha_ne2: alpha_dims[2],
            alpha_ne3: alpha_dims[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            ne3: dims[3],
            eps,
        };

        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-layernorm-params"),
                size: std::mem::size_of::<LayerNormParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));

        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
            uniform_entry(4),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &alpha.buffer),
            buffer_binding(2, &beta.buffer),
            buffer_binding(3, &dst.buffer),
            buffer_binding(4, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::LAYERNORM_WGSL
            .source()
            .replace("WG_SIZE", &WG_SIZE.to_string());
        let rows = count / dims[0] as usize;
        let rows_u32: u32 = rows.try_into()?;
        let max_per_dim = wgpu_dispatch_wg_cap(&self.device);
        let (wg_x, wg_y) = compute_2d_workgroups(rows_u32, max_per_dim);
        self.device.run_compute_xyz(
            &shader,
            &entries,
            &bindings,
            (wg_x, wg_y, 1),
            &[],
            "candle-wgpu-layernorm",
        )?;
        Ok(dst)
    }

    pub fn sigmoid(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu sigmoid").bt());
        }
        if wgpu_f16_emulates_f32(&self.device, self.dtype) {
            let src_f32 = self.materialize_to_f32(layout)?;
            let src_l = Layout::contiguous(layout.shape());
            let out_f32 = src_f32.sigmoid(&src_l)?;
            return out_f32.to_dtype(&src_l, DType::F16);
        }
        let shader = unary_shader("sigmoid", self.dtype)?;
        self.run_unary_like(layout, &shader, "candle-wgpu-sigmoid")
    }

    fn materialize_contiguous(&self, layout: &Layout) -> Result<(Self, Layout)> {
        if layout.is_contiguous() && layout.start_offset() == 0 {
            return Ok((self.try_clone(layout)?, layout.clone()));
        }
        let mut compact = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        self.copy_strided_src(&mut compact, 0, layout)?;
        Ok((compact, Layout::contiguous(layout.shape())))
    }

    fn normalize_index_ids(&self, ids_l: &Layout) -> Result<(Self, Layout)> {
        let (ids, ids_l) = self.materialize_contiguous(ids_l)?;
        let normalized = match ids.dtype {
            DType::U32 => ids,
            DType::U8 | DType::I64 => ids.run_emulated_cast(&ids_l, DType::U32)?,
            dt => return Err(Error::UnsupportedDTypeForOp(dt, "wgpu index ids").bt()),
        };
        Ok((normalized, ids_l))
    }

    fn run_index_select_f32(
        &self,
        ids: &Self,
        src_l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let (src, src_l) = self.materialize_contiguous(src_l)?;
        let (ids, ids_l) = ids.normalize_index_ids(ids_l)?;
        if !matches!(
            src.dtype,
            DType::F32
                | DType::F16
                | DType::U8
                | DType::U32
                | DType::I64
                | DType::F64
                | DType::BF16
        ) {
            return Err(Error::UnsupportedDTypeForOp(src.dtype, "wgpu index_select").bt());
        }
        let use_f32_hub = match src.dtype {
            DType::F32 => false,
            DType::F16 => wgpu_f16_emulates_f32(&src.device, src.dtype),
            _ => true,
        };
        if use_f32_hub {
            let src_f32 = src.to_dtype(&src_l, DType::F32)?;
            let src_f32_l = Layout::contiguous(src_l.shape());
            let out_f32 = src_f32.run_index_select_f32(&ids, &src_f32_l, &ids_l, dim)?;
            if src.dtype == src_f32.dtype {
                return Ok(out_f32);
            }
            let ids_total: usize = ids_l.dims().iter().product();
            let mut dst_dims = src_l.dims().to_vec();
            dst_dims[dim] = ids_total;
            return out_f32.to_dtype(&Layout::contiguous(Shape::from(dst_dims)), src.dtype);
        }
        // Flatten multi-dim ids to 1D
        let ids_len: usize = ids_l.dims().iter().product();
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim = src_l.dims()[dim];
        let dst_el = left_size * ids_len * right_size;
        let mut dst_dims = src_l.dims().to_vec();
        dst_dims[dim] = ids_len;
        let dst_shape = Shape::from(dst_dims);
        let dst_layout = Layout::contiguous(dst_shape.clone());
        let tmp_dtype = if src.dtype == DType::F16 {
            DType::F32
        } else {
            src.dtype
        };
        let dst = unsafe { src.device.alloc_uninit(&dst_shape, tmp_dtype)? };
        let params = GetRowsParams {
            offset_src: src_l.start_offset().try_into()?,
            offset_idx: ids_l.start_offset().try_into()?,
            offset_dst: 0,
            stride_src1: right_size.try_into()?,
            stride_src2: (src_dim * right_size).try_into()?,
            stride_src3: (src_dim * right_size * left_size).try_into()?,
            stride_idx0: ids_l.stride()[0].try_into()?,
            stride_idx1: 0,
            stride_idx2: 0,
            stride_dst1: right_size.try_into()?,
            stride_dst2: (ids_len * right_size).try_into()?,
            stride_dst3: (left_size * ids_len * right_size).try_into()?,
            ne0: right_size.try_into()?,
            n_rows: ids_len.try_into()?,
            ne2: left_size.try_into()?,
            ne3: 1,
            idx1: 1,
            idx2: 1,
        };
        let param_buffer = src
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-get-rows-params"),
                size: std::mem::size_of::<GetRowsParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        src.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &src.buffer),
            buffer_binding(1, &ids.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = match src.dtype {
            DType::F32 => candle_wgpu_kernels::get_rows_f32_shader(WG_SIZE),
            DType::F16 => candle_wgpu_kernels::get_rows_f16_shader(WG_SIZE),
            DType::BF16 => Some(bf16_gather_last_dim_wgsl()),
            DType::U8 => Some(u8_gather_last_dim_wgsl()),
            DType::U32 => candle_wgpu_kernels::get_rows_u32_shader(WG_SIZE),
            DType::I64 => Some(i64_gather_last_dim_wgsl()),
            DType::F64 => Some(f64_gather_last_dim_wgsl()),
            _ => None,
        }
        .ok_or_else(|| Error::Msg("wgpu shader get_rows.wgsl not embedded".into()).bt())?;
        let rows: u32 = (left_size * ids_len).try_into()?;
        let work_items = match src.dtype {
            DType::U8 => rows.div_ceil(4),
            DType::BF16 => rows.div_ceil(2),
            _ => rows,
        };
        src.device.run_compute(
            &shader,
            &entries,
            &bindings,
            work_items.div_ceil(WG_SIZE),
            "candle-wgpu-get-rows",
        )?;
        debug_assert_eq!(dst.count, dst_el);
        if src.dtype == DType::F16 {
            let copy = copy_shader(DType::F32, DType::F16)?;
            dst.run_copy_to_dtype(&dst_layout, DType::F16, &copy)
        } else {
            Ok(dst)
        }
    }

    fn run_gather_last_dim_f32(&self, ids: &Self, src_l: &Layout, ids_l: &Layout) -> Result<Self> {
        let (src, src_l) = self.materialize_contiguous(src_l)?;
        let (ids, ids_l) = ids.normalize_index_ids(ids_l)?;
        if !matches!(
            src.dtype,
            DType::F32
                | DType::F16
                | DType::U8
                | DType::U32
                | DType::I64
                | DType::F64
                | DType::BF16
        ) {
            return Err(Error::UnsupportedDTypeForOp(src.dtype, "wgpu gather").bt());
        }
        if wgpu_f16_emulates_f32(&src.device, src.dtype) {
            let src_f32 = src.to_dtype(&src_l, DType::F32)?;
            let src_f32_l = Layout::contiguous(src_l.shape());
            return src_f32.run_gather_last_dim_f32(&ids, &src_f32_l, &ids_l);
        }
        let rank = src_l.dims().len();
        if rank == 0 {
            return Err(Error::UnsupportedDTypeForOp(src.dtype, "wgpu gather rank-0").bt());
        }
        // Flatten mismatched-rank ids to 1D
        let ids_dim = if ids_l.dims().len() == rank {
            ids_l.dims()[rank - 1]
        } else {
            ids_l.dims().iter().product()
        };
        let left_size: usize = if ids_l.dims().len() == rank {
            ids_l.dims()[..rank - 1].iter().product()
        } else {
            1
        };
        let src_dim = src_l.dims()[rank - 1];
        let dst_shape = ids_l.shape().clone();
        let tmp_dtype = if src.dtype == DType::F16 {
            DType::F32
        } else {
            src.dtype
        };
        let dst = unsafe { src.device.alloc_uninit(&dst_shape, tmp_dtype)? };
        let params = GetRowsParams {
            offset_src: src_l.start_offset().try_into()?,
            offset_idx: ids_l.start_offset().try_into()?,
            offset_dst: 0,
            stride_src1: 1,
            stride_src2: src_dim.try_into()?,
            stride_src3: (src_dim * left_size).try_into()?,
            stride_idx0: 1,
            stride_idx1: ids_dim.try_into()?,
            stride_idx2: 0,
            stride_dst1: 1,
            stride_dst2: ids_dim.try_into()?,
            stride_dst3: (ids_dim * left_size).try_into()?,
            ne0: 1,
            n_rows: ids_dim.try_into()?,
            ne2: left_size.try_into()?,
            ne3: 1,
            idx1: left_size.try_into()?,
            idx2: 1,
        };
        let param_buffer = src
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-gather-params"),
                size: std::mem::size_of::<GetRowsParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        src.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &src.buffer),
            buffer_binding(1, &ids.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = match src.dtype {
            DType::F32 => candle_wgpu_kernels::get_rows_f32_shader(WG_SIZE),
            DType::F16 => candle_wgpu_kernels::get_rows_f16_shader(WG_SIZE),
            DType::BF16 => Some(bf16_gather_last_dim_wgsl()),
            DType::U8 => Some(u8_gather_last_dim_wgsl()),
            DType::U32 => candle_wgpu_kernels::get_rows_u32_shader(WG_SIZE),
            DType::I64 => Some(i64_gather_last_dim_wgsl()),
            DType::F64 => Some(f64_gather_last_dim_wgsl()),
            _ => None,
        }
        .ok_or_else(|| Error::Msg("wgpu shader get_rows.wgsl not embedded".into()).bt())?;
        let rows: u32 = (left_size * ids_dim).try_into()?;
        let work_items = match src.dtype {
            DType::U8 => rows.div_ceil(4),
            DType::BF16 => rows.div_ceil(2),
            _ => rows,
        };
        src.device.run_compute(
            &shader,
            &entries,
            &bindings,
            work_items.div_ceil(WG_SIZE),
            "candle-wgpu-gather",
        )?;
        if src.dtype == DType::F16 {
            let dst_layout = Layout::contiguous(dst_shape.clone());
            let copy = copy_shader(DType::F32, DType::F16)?;
            dst.run_copy_to_dtype(&dst_layout, DType::F16, &copy)
        } else {
            Ok(dst)
        }
    }

    fn run_scatter_set_last_dim_f32(
        &mut self,
        dst_l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
    ) -> Result<()> {
        let (ids, ids_l) = ids.normalize_index_ids(ids_l)?;
        let (src, src_l) = src.materialize_contiguous(src_l)?;
        if !dst_l.is_contiguous() || dst_l.start_offset() != 0 {
            let mut compact = unsafe { self.device.alloc_uninit(dst_l.shape(), self.dtype)? };
            self.copy_strided_src(&mut compact, 0, dst_l)?;
            let dst_cl = Layout::contiguous(dst_l.shape());
            compact.run_scatter_set_last_dim_f32(&dst_cl, &ids, &ids_l, &src, &src_l)?;
            compact.copy_strided_src(self, dst_l.start_offset(), dst_l)?;
            return Ok(());
        }
        if (self.dtype != DType::F32 && self.dtype != DType::F16 && self.dtype != DType::U32)
            || self.dtype != src.dtype
        {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu scatter_set").bt());
        }
        if wgpu_f16_emulates_f32(&self.device, self.dtype) {
            let mut dst_f32 = self.to_dtype(dst_l, DType::F32)?;
            let src_f32 = src.to_dtype(&src_l, DType::F32)?;
            let dst_f32_layout = Layout::contiguous(dst_l.shape().clone());
            let src_f32_layout = Layout::contiguous(src_l.shape().clone());
            dst_f32.run_scatter_set_last_dim_f32(
                &dst_f32_layout,
                &ids,
                &ids_l,
                &src_f32,
                &src_f32_layout,
            )?;
            *self = dst_f32.to_dtype(&dst_f32_layout, DType::F16)?;
            return Ok(());
        }
        let rank = dst_l.dims().len();
        if rank == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu scatter_set rank-0").bt());
        }
        let ids_dim = if ids_l.dims().len() == rank {
            ids_l.dims()[rank - 1]
        } else {
            ids_l.dims().iter().product()
        };
        let left_size: usize = if ids_l.dims().len() == rank {
            ids_l.dims()[..rank - 1].iter().product()
        } else {
            1
        };
        let dst_dim = dst_l.dims()[rank - 1];
        let params = GetRowsParams {
            offset_src: src_l.start_offset().try_into()?,
            offset_idx: ids_l.start_offset().try_into()?,
            offset_dst: dst_l.start_offset().try_into()?,
            stride_src1: 1,
            stride_src2: ids_dim.try_into()?,
            stride_src3: (ids_dim * left_size).try_into()?,
            stride_idx0: 1,
            stride_idx1: ids_dim.try_into()?,
            stride_idx2: 0,
            stride_dst1: 1,
            stride_dst2: dst_dim.try_into()?,
            stride_dst3: (dst_dim * left_size).try_into()?,
            ne0: 1,
            n_rows: ids_dim.try_into()?,
            ne2: left_size.try_into()?,
            ne3: 1,
            idx1: left_size.try_into()?,
            idx2: 1,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-scatter-set-params"),
                size: std::mem::size_of::<GetRowsParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &src.buffer),
            buffer_binding(1, &ids.buffer),
            buffer_binding(2, &self.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = match self.dtype {
            DType::F32 => candle_wgpu_kernels::set_rows_f32_shader(WG_SIZE),
            DType::F16 => candle_wgpu_kernels::set_rows_f16_shader(WG_SIZE),
            DType::U32 => candle_wgpu_kernels::set_rows_u32_shader(WG_SIZE),
            _ => None,
        }
        .ok_or_else(|| Error::Msg("wgpu shader set_rows.wgsl not embedded".into()).bt())?;
        let rows: u32 = (left_size * ids_dim).try_into()?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.div_ceil(WG_SIZE),
            "candle-wgpu-scatter-set",
        )?;
        Ok(())
    }

    fn run_scatter_add_last_dim_f32(
        &mut self,
        dst_l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
    ) -> Result<()> {
        let (ids, ids_l) = ids.normalize_index_ids(ids_l)?;
        let (src, src_l) = src.materialize_contiguous(src_l)?;
        if !dst_l.is_contiguous() || dst_l.start_offset() != 0 {
            let mut compact = unsafe { self.device.alloc_uninit(dst_l.shape(), self.dtype)? };
            self.copy_strided_src(&mut compact, 0, dst_l)?;
            let dst_cl = Layout::contiguous(dst_l.shape());
            compact.run_scatter_add_last_dim_f32(&dst_cl, &ids, &ids_l, &src, &src_l)?;
            compact.copy_strided_src(self, dst_l.start_offset(), dst_l)?;
            return Ok(());
        }
        if self.dtype != DType::F32 || self.dtype != src.dtype {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu scatter_add").bt());
        }
        let rank = dst_l.dims().len();
        if rank == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu scatter_add rank-0").bt());
        }
        let ids_dim = if src_l.dims().len() == rank {
            src_l.dims()[rank - 1]
        } else {
            src_l.dims().iter().product()
        };
        let left_size: usize = if src_l.dims().len() == rank {
            src_l.dims()[..rank - 1].iter().product()
        } else {
            1
        };
        // `scatter_add` passes ids shaped like `src`; `index_add` passes one
        // rank-1 id row shared by every leading row, which maps to a zero ids
        // row stride in the shader.
        let ids_row_stride: usize = if ids_l.dims().len() == rank {
            ids_dim
        } else if ids_l.dims() == [ids_dim] || ids_l.dims().len() == 1 {
            0
        } else {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu scatter_add ids shape mismatch").bt());
        };
        let dst_dim = dst_l.dims()[rank - 1];
        let params = GetRowsParams {
            offset_src: src_l.start_offset().try_into()?,
            offset_idx: ids_l.start_offset().try_into()?,
            offset_dst: dst_l.start_offset().try_into()?,
            stride_src1: 1,
            stride_src2: ids_dim.try_into()?,
            stride_src3: (ids_dim * left_size).try_into()?,
            stride_idx0: 1,
            stride_idx1: ids_row_stride.try_into()?,
            stride_idx2: 0,
            stride_dst1: 1,
            stride_dst2: dst_dim.try_into()?,
            stride_dst3: (dst_dim * left_size).try_into()?,
            ne0: 1,
            n_rows: ids_dim.try_into()?,
            ne2: left_size.try_into()?,
            ne3: 1,
            idx1: left_size.try_into()?,
            idx2: 1,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-scatter-add-params"),
                size: std::mem::size_of::<GetRowsParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &src.buffer),
            buffer_binding(1, &ids.buffer),
            buffer_binding(2, &self.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::set_rows_add_f32_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader set_rows.wgsl not embedded".into()).bt())?;
        let rows: u32 = (left_size * ids_dim).try_into()?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.div_ceil(WG_SIZE),
            "candle-wgpu-scatter-add",
        )?;
        Ok(())
    }

    fn run_matmul_f32(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != rhs.dtype {
            let promote_to = if self.dtype == DType::F64 || rhs.dtype == DType::F64 {
                DType::F64
            } else if self.dtype == DType::F32 || rhs.dtype == DType::F32 {
                DType::F32
            } else if self.dtype == DType::BF16 || rhs.dtype == DType::BF16 {
                DType::BF16
            } else {
                DType::F16
            };
            let lhs = if self.dtype != promote_to {
                self.to_dtype(lhs_l, promote_to)?
            } else {
                self.try_clone(lhs_l)?
            };
            let rhs = if rhs.dtype != promote_to {
                rhs.to_dtype(rhs_l, promote_to)?
            } else {
                rhs.try_clone(rhs_l)?
            };
            let flat_lhs = Layout::contiguous(lhs_l.shape().clone());
            let flat_rhs = Layout::contiguous(rhs_l.shape().clone());
            return lhs.run_matmul_f32(&rhs, (b, m, n, k), &flat_lhs, &flat_rhs);
        }
        if self.dtype == DType::BF16 {
            let lhs_f32 = self.to_dtype(lhs_l, DType::F32)?;
            let rhs_f32 = rhs.to_dtype(rhs_l, DType::F32)?;
            let lhs_f32_l = Layout::contiguous(lhs_l.shape().clone());
            let rhs_f32_l = Layout::contiguous(rhs_l.shape().clone());
            let out_f32 = lhs_f32.run_matmul_f32(&rhs_f32, (b, m, n, k), &lhs_f32_l, &rhs_f32_l)?;
            let out_l = Layout::contiguous(Shape::from(vec![b, m, n]));
            return out_f32.to_dtype(&out_l, DType::BF16);
        }
        // ponytail: F16 matmul shader writes f32 dst — must upconvert to avoid buffer overflow.
        if self.dtype == DType::F16 {
            let lhs_f32 = self.to_dtype(lhs_l, DType::F32)?;
            let rhs_f32 = rhs.to_dtype(rhs_l, DType::F32)?;
            let lhs_f32_l = Layout::contiguous(lhs_l.shape().clone());
            let rhs_f32_l = Layout::contiguous(rhs_l.shape().clone());
            let out_f32 = lhs_f32.run_matmul_f32(&rhs_f32, (b, m, n, k), &lhs_f32_l, &rhs_f32_l)?;
            let out_l = Layout::contiguous(Shape::from(vec![b, m, n]));
            return out_f32.to_dtype(&out_l, DType::F16);
        }
        if self.dtype != DType::F32
            && self.dtype != DType::F16
            && self.dtype != DType::BF16
            && self.dtype != DType::F64
        {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu matmul").bt());
        }
        let rank = lhs_l.dims().len();
        if rank == rhs_l.dims().len() && rank > 4 {
            if b != lhs_l.dims()[..rank - 2].iter().product::<usize>()
                || b != rhs_l.dims()[..rank - 2].iter().product::<usize>()
            {
                return Err(unsupported("matmul batch"));
            }
            if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
                let lhs_flat_l = Layout::contiguous_with_offset((b, m, k), lhs_l.start_offset());
                let rhs_flat_l = Layout::contiguous_with_offset((b, k, n), rhs_l.start_offset());
                return self.run_matmul_f32(rhs, (b, m, n, k), &lhs_flat_l, &rhs_flat_l);
            }
            let lhs_l = if wgpu_f16_emulates_f32(&self.device, self.dtype) {
                let lhs = self.materialize_rank_gt4_matmul_operand_to_f32(lhs_l, b, m, k)?;
                let rhs = rhs.materialize_rank_gt4_matmul_operand_to_f32(rhs_l, b, k, n)?;
                let lhs_l = Layout::contiguous(Shape::from(vec![b, m, k]));
                let rhs_l = Layout::contiguous(Shape::from(vec![b, k, n]));
                let out = lhs.run_matmul_f32(&rhs, (b, m, n, k), &lhs_l, &rhs_l)?;
                let out_l = Layout::contiguous(Shape::from(vec![b, m, n]));
                return out.to_dtype(&out_l, self.dtype);
            } else {
                self.materialize_rank_gt4_matmul_operand(lhs_l, b, m, k)?
            };
            let rhs = rhs.materialize_rank_gt4_matmul_operand(rhs_l, b, k, n)?;
            let lhs_flat_l = Layout::contiguous(Shape::from(vec![b, m, k]));
            let rhs_flat_l = Layout::contiguous(Shape::from(vec![b, k, n]));
            return lhs_l.run_matmul_f32(&rhs, (b, m, n, k), &lhs_flat_l, &rhs_flat_l);
        }
        if rank != rhs_l.dims().len() || rank < 2 {
            return Err(unsupported("matmul rank"));
        }
        if b != lhs_l.dims()[..rank - 2].iter().product::<usize>() {
            return Err(unsupported("matmul batch"));
        }
        if wgpu_f16_emulates_f32(&self.device, self.dtype) {
            let lhs_f32 = self.materialize_to_f32(lhs_l)?;
            let rhs_f32 = rhs.materialize_to_f32(rhs_l)?;
            let lhs_f32_l = Layout::contiguous(lhs_l.shape().clone());
            let rhs_f32_l = Layout::contiguous(rhs_l.shape().clone());
            let out_f32 = lhs_f32.run_matmul_f32(&rhs_f32, (b, m, n, k), &lhs_f32_l, &rhs_f32_l)?;
            let out_l = Layout::contiguous(Shape::from(vec![b, m, n]));
            return out_f32.to_dtype(&out_l, self.dtype);
        }

        let mut lhs_contiguous = None;
        let (lhs, lhs_layout) = if lhs_l.is_contiguous() && lhs_l.start_offset() == 0 {
            (self, lhs_l.clone())
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(lhs_l.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, lhs_l)?;
            lhs_contiguous = Some(tmp);
            (
                lhs_contiguous.as_ref().unwrap(),
                Layout::contiguous(lhs_l.shape().clone()),
            )
        };

        // Large F32 GEMM binds contiguous (K,N) RHS as a virtual B^T via
        // stride_0k — measured faster than materializing on RTX 3060 (including
        // coopmat path; a full B^T of 4096² is far costlier than strided loads).
        let will_use_warptile = self.dtype == DType::F32 && m >= 64 && n >= 64 && k >= 64;
        let rhs_skip_transpose = will_use_warptile
            && rhs_l.is_contiguous()
            && rhs_l.start_offset() == 0
            && rhs_l.dims().len() == rank
            && rhs_l.dims()[rank - 2] == k
            && rhs_l.dims()[rank - 1] == n;

        let rhs_t_src_layout = rhs_l.transpose(rank - 2, rank - 1)?;
        let rhs_t_materialized = if rhs_skip_transpose
            || (rhs_t_src_layout.is_contiguous() && rhs_t_src_layout.start_offset() == 0)
        {
            None
        } else {
            let rhs_t_shape = rhs_t_src_layout.shape().clone();
            let mut tmp = unsafe { rhs.device.alloc_uninit(&rhs_t_shape, rhs.dtype)? };
            rhs.copy_strided_src(&mut tmp, 0, &rhs_t_src_layout)?;
            Some(tmp)
        };
        // Physical (...,K,N): B^T[n_i,k_i] @ k_i*N + n_i → stride_01=1, stride_0k=N.
        // Contiguous B^T (...,N,K): stride_01=K, stride_0k=1.
        let (rhs_t, rhs_t_layout, stride_0k, stride_01_rhs) =
            if let Some(ref tmp) = rhs_t_materialized {
                (
                    tmp,
                    Layout::contiguous(rhs_t_src_layout.shape().clone()),
                    1,
                    k,
                )
            } else if rhs_skip_transpose {
                (rhs, rhs_l.clone(), n, 1)
            } else {
                (rhs, rhs_t_src_layout.clone(), 1, k)
            };

        let lhs_stride = lhs_layout.stride();
        let rhs_view_stride = rhs_t_layout.stride();
        let bs02 = if rank >= 3 {
            lhs_layout.dims()[rank - 3]
        } else {
            1
        };
        let bs03 = if rank >= 4 {
            lhs_layout.dims()[rank - 4]
        } else {
            1
        };
        let lhs_stride_batch_inner = if rank >= 3 {
            lhs_stride[rank - 3]
        } else {
            m * k
        };
        let lhs_stride_batch_outer = if rank >= 4 {
            lhs_stride[rank - 4]
        } else {
            b * m * k
        };
        // Batch strides from the bound RHS view (physical K,N or B^T).
        // Rank-2 batch stride is always k*n elements (same for either layout).
        let rhs_stride_batch_inner = if rank >= 3 {
            rhs_view_stride[rank - 3]
        } else {
            k * n
        };
        let rhs_stride_batch_outer = if rank >= 4 {
            rhs_view_stride[rank - 4]
        } else {
            b * n * k
        };

        let dst_shape = Shape::from(vec![b, m, n]);
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };
        let dst_layout = Layout::contiguous(dst_shape);
        // Contiguous LHS (..., M, K) has unit K stride.
        let lhs_k_stride = 1usize;
        let params = MulMatParams {
            offset_src0: 0,
            offset_src1: 0,
            offset_dst: 0,
            m: n.try_into()?,
            n: m.try_into()?,
            k: k.try_into()?,
            stride_01: stride_01_rhs.try_into()?,
            stride_11: lhs_stride[rank - 2].try_into()?,
            stride_02: rhs_stride_batch_inner.try_into()?,
            stride_12: lhs_stride_batch_inner.try_into()?,
            stride_03: rhs_stride_batch_outer.try_into()?,
            stride_13: lhs_stride_batch_outer.try_into()?,
            bs02: bs02.try_into()?,
            bs03: bs03.try_into()?,
            broadcast2: 1,
            broadcast3: 1,
            stride_0k: stride_0k.try_into()?,
            stride_1k: lhs_k_stride.try_into()?,
            _pad0: 0,
            _pad1: 0,
        };
        let param_buffer = self
            .device
            .write_uniform_params(any_as_bytes(&params))?;
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        // Register-tiled GEMM (mul_mat_reg_tile) is dramatically faster than the
        // naive mul_mat.wgsl path for large dense F32/F16 problems. It reuses the
        // f16 workgroup cache path and therefore requires SHADER_F16.
        let shader_storage;
        let matmul_label: &'static str;
        let mut use_reg_tile = false;
        let mut use_warptile = false;
        let shader: &str = match self.dtype {
            DType::F32 => {
                // Cooperative matrix: Ampere+ Vulkan exposes 16×16 f16 A/B → f32 C.
                // Mixed-precision for large GEMMs; small squares stay full-f32 warptile.
                let coop_ok = self.device.inner.coop_matmul_enabled
                    && self
                        .device
                        .inner
                        .features
                        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
                    && self
                        .device
                        .inner
                        .features
                        .contains(wgpu::Features::SHADER_F16)
                    && m.is_multiple_of(16)
                    && n.is_multiple_of(16)
                    && k.is_multiple_of(16)
                    && m >= 64
                    && n >= 64
                    && k >= 64
                    // Exclude small square smokes (64³) where f16 error exceeds
                    // tight abs tols; allow tall/wide (e.g. 64×4096) and ≥128².
                    && (m >= 128 || n >= 128)
                    && params.stride_1k == 1;
                if coop_ok {
                    use_warptile = true;
                    // 128×64 dual-MMA + dbuf for large squares and tall/wide
                    // (64×4096). 128×32 tall split and coop64-for-skinny both
                    // regressed vs dual on RTX 3060. Keep 64×64 for mid squares.
                    if m.max(n) >= 512 && m.min(n) >= 64 {
                        matmul_label = "candle-wgpu-matmul-coop";
                        candle_wgpu_kernels::matmul_coop_shader().ok_or_else(|| {
                            Error::Msg("wgpu shader mul_mat_coop.wgsl not embedded".into()).bt()
                        })?
                    } else {
                        matmul_label = "candle-wgpu-matmul-coop64";
                        candle_wgpu_kernels::matmul_coop_64_shader().ok_or_else(|| {
                            Error::Msg("wgpu shader mul_mat_coop_64.wgsl not embedded".into()).bt()
                        })?
                    }
                } else if m >= 64 && n >= 64 && k >= 64 && params.stride_1k == 1 {
                    matmul_label = "candle-wgpu-matmul-warptile";
                    use_warptile = true;
                    candle_wgpu_kernels::matmul_warptile_shader().ok_or_else(|| {
                        Error::Msg("wgpu shader mul_mat_warptile.wgsl not embedded".into()).bt()
                    })?
                } else if m.max(n) >= 32 && k >= 32 {
                    matmul_label = "candle-wgpu-matmul-fast";
                    use_reg_tile = true;
                    // VEC loads assume contiguous K (unit stride_0k / stride_1k).
                    let vectorized = m.is_multiple_of(4)
                        && n.is_multiple_of(4)
                        && params.stride_0k == 1
                        && params.stride_1k == 1;
                    shader_storage = candle_wgpu_kernels::matmul_fast_shader(
                        wgpu_kernel_dtype(DType::F32)?,
                        vectorized,
                    )
                    .ok_or_else(|| {
                        Error::Msg("wgpu shader mul_mat_reg_tile.wgsl not embedded".into()).bt()
                    })?;
                    &shader_storage
                } else {
                    matmul_label = "candle-wgpu-matmul";
                    shader_storage = candle_wgpu_kernels::matmul_f32_shader().ok_or_else(|| {
                        Error::Msg("wgpu shader mul_mat.wgsl not embedded".into()).bt()
                    })?;
                    &shader_storage
                }
            }
            DType::F16 => {
                // f16 reg-tile is valid (shmem is f16). Restrict to tile-aligned
                // dims for residual-edge correctness under tight smoke tols.
                let tile_ok = m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(32);
                if self
                    .device
                    .inner
                    .features
                    .contains(wgpu::Features::SHADER_F16)
                    && tile_ok
                    && m.max(n) >= 64
                {
                    matmul_label = "candle-wgpu-matmul-fast";
                    use_reg_tile = true;
                    shader_storage = candle_wgpu_kernels::matmul_fast_shader(
                        wgpu_kernel_dtype(DType::F16)?,
                        false,
                    )
                    .ok_or_else(|| {
                        Error::Msg("wgpu shader mul_mat_reg_tile.wgsl not embedded".into()).bt()
                    })?;
                    &shader_storage
                } else {
                    matmul_label = "candle-wgpu-matmul";
                    shader_storage = candle_wgpu_kernels::matmul_f16_shader().ok_or_else(|| {
                        Error::Msg("wgpu shader mul_mat.wgsl not embedded".into()).bt()
                    })?;
                    &shader_storage
                }
            }
            DType::BF16 => {
                matmul_label = "candle-wgpu-matmul-bf16";
                shader_storage = candle_wgpu_kernels::matmul_bf16_shader().ok_or_else(|| {
                    Error::Msg("wgpu shader mul_mat_bf16.wgsl not embedded".into()).bt()
                })?;
                &shader_storage
            }
            DType::F64 => {
                matmul_label = "candle-wgpu-matmul-f64";
                candle_wgpu_kernels::matmul_f64_shader().ok_or_else(|| {
                    Error::Msg("wgpu shader mul_mat_f64.wgsl not embedded".into()).bt()
                })?
            }
            _ => {
                return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu matmul").bt());
            }
        };
        let max_binding_bytes = self.device.inner.limits.max_storage_buffer_binding_size as usize;
        let dst_bytes = byte_len(self.dtype, b * m * n, "wgpu matmul dst")?;
        let lhs_bytes = layout_binding_bytes(lhs, &lhs_layout);
        let rhs_bytes = layout_binding_bytes(rhs_t, &rhs_t_layout);
        let elem_size = self.dtype.size_in_bytes();
        let matrix_elems = m * n;
        let matrix_bytes = matrix_elems * elem_size;
        let lhs_matrix_bytes = m * k * elem_size;
        let rhs_matrix_bytes = n * k * elem_size;
        let needs_batch_chunk = dst_bytes > max_binding_bytes
            || lhs_bytes > max_binding_bytes
            || rhs_bytes > max_binding_bytes;
        if needs_batch_chunk {
            if b <= 1 {
                return Err(Error::Msg(format!(
                    "wgpu matmul binding exceeds max_storage_buffer_binding_size (dst={dst_bytes}, lhs={lhs_bytes}, rhs={rhs_bytes}, max={max_binding_bytes})"
                ))
                .bt());
            }
            if matrix_bytes > max_binding_bytes
                || lhs_matrix_bytes > max_binding_bytes
                || rhs_matrix_bytes > max_binding_bytes
            {
                return Err(Error::Msg(format!(
                    "wgpu matmul per-batch binding exceeds max_storage_buffer_binding_size (dst={matrix_bytes}, lhs={lhs_matrix_bytes}, rhs={rhs_matrix_bytes}, max={max_binding_bytes})"
                ))
                .bt());
            }
            let max_per_dim = wgpu_dispatch_wg_cap(&self.device);
            let total_wg = (m * n).try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
            let (wg_x, wg_y) = compute_2d_workgroups(total_wg, max_per_dim);
            let mut chunk_params = params;
            chunk_params.bs02 = 1;
            chunk_params.bs03 = 1;
            chunk_params.broadcast2 = 1;
            chunk_params.broadcast3 = 1;
            let matrix_elems_u64 = matrix_elems as u64;
            let lhs_matrix_elems_u64 = (m * k) as u64;
            let rhs_matrix_elems_u64 = (n * k) as u64;
            let elem_size_u64 = elem_size as u64;
            for batch_idx in 0..b {
                let dst2_idx = batch_idx % bs02;
                let dst3_idx = batch_idx / bs02;
                let src02_idx = dst2_idx;
                let src03_idx = dst3_idx;
                let src12_idx = dst2_idx;
                let src13_idx = dst3_idx;
                let rhs_elem_off = params.offset_src0 as u64
                    + (src03_idx as u64 * rhs_stride_batch_outer as u64
                        + src02_idx as u64 * rhs_stride_batch_inner as u64);
                let lhs_elem_off = params.offset_src1 as u64
                    + (src13_idx as u64 * lhs_stride_batch_outer as u64
                        + src12_idx as u64 * lhs_stride_batch_inner as u64);
                let mut dispatch_params = chunk_params;
                dispatch_params.offset_src0 = 0;
                dispatch_params.offset_src1 = 0;
                dispatch_params.offset_dst = 0;
                self.device
                    .inner
                    .queue
                    .write_buffer(&param_buffer, 0, any_as_bytes(&dispatch_params));
                let dst_offset_bytes = (batch_idx as u64)
                    .saturating_mul(matrix_elems_u64)
                    .saturating_mul(elem_size_u64);
                let batch_bindings = [
                    buffer_binding_range(
                        0,
                        &rhs_t.buffer,
                        rhs_elem_off.saturating_mul(elem_size_u64),
                        rhs_matrix_elems_u64.saturating_mul(elem_size_u64),
                    )?,
                    buffer_binding_range(
                        1,
                        &lhs.buffer,
                        lhs_elem_off.saturating_mul(elem_size_u64),
                        lhs_matrix_elems_u64.saturating_mul(elem_size_u64),
                    )?,
                    buffer_binding_range(
                        2,
                        &dst.buffer,
                        dst_offset_bytes,
                        matrix_elems_u64.saturating_mul(elem_size_u64),
                    )?,
                    buffer_binding(3, &param_buffer),
                ];
                self.device.run_compute_xyz(
                    shader,
                    &entries,
                    &batch_bindings,
                    (wg_x, wg_y, 1),
                    &[],
                    matmul_label,
                )?;
            }
        } else {
            let rhs_bind = storage_layout_binding(rhs_t, &rhs_t_layout, 0)?;
            let lhs_bind = storage_layout_binding(lhs, &lhs_layout, 1)?;
            let dst_bind = storage_layout_binding(&dst, &dst_layout, 2)?;
            let bindings = [rhs_bind, lhs_bind, dst_bind, buffer_binding(3, &param_buffer)];
            let total_wg = if use_warptile {
                // Coop 128×64 dual / coop 64×64 / warptile 64×64 — pick by label.
                let (bm, bn, _) = if matmul_label.ends_with("-coop") {
                    candle_wgpu_kernels::matmul_coop_tile_shape()
                } else if matmul_label.contains("coop64") {
                    candle_wgpu_kernels::matmul_coop_64_tile_shape()
                } else {
                    candle_wgpu_kernels::matmul_warptile_tile_shape()
                };
                // BM along candle N (params.m), BN along candle M (params.n).
                n.div_ceil(bm as usize)
                    .checked_mul(m.div_ceil(bn as usize))
                    .and_then(|v| v.checked_mul(b))
                    .ok_or_else(|| {
                        Error::Msg("wgpu backend op matmul warptile workgroup overflow".into()).bt()
                    })?
                    .try_into()?
            } else if use_reg_tile {
                // params.m/n are swapped relative to candle (m,n): see MulMatParams above.
                let (tile_m, tile_n, wg_size_m, wg_size_n, _) =
                    candle_wgpu_kernels::matmul_fast_tile_shape();
                let tile_m_s = (tile_m * wg_size_m) as usize;
                let tile_n_s = (tile_n * wg_size_n) as usize;
                // params.m == n (cols), params.n == m (rows)
                m.div_ceil(tile_n_s)
                    .checked_mul(n.div_ceil(tile_m_s))
                    .and_then(|v| v.checked_mul(b))
                    .ok_or_else(|| {
                        Error::Msg("wgpu backend op matmul workgroup overflow".into()).bt()
                    })?
                    .try_into()?
            } else {
                (b * m * n).try_into().map(|v: u32| v.div_ceil(WG_SIZE))?
            };
            let max_per_dim = wgpu_dispatch_wg_cap(&self.device);
            let (wg_x, wg_y) = compute_2d_workgroups(total_wg, max_per_dim);
            self.device
                .run_compute_xyz(shader, &entries, &bindings, (wg_x, wg_y, 1), &[], matmul_label)?;
        }
        drop(lhs_contiguous);
        Ok(dst)
    }

    fn run_conv1d_f32(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu conv1d").bt());
        }
        let src_stride = layout.stride();
        let kernel_stride = kernel_l.stride();
        let input_l = Layout::new(
            Shape::from(vec![params.b_size, params.c_in, 1, params.l_in]),
            vec![
                src_stride[0],
                src_stride[1],
                src_stride[2] * params.l_in,
                src_stride[2],
            ],
            layout.start_offset(),
        );
        let kernel_2d_l = Layout::new(
            Shape::from(vec![params.c_out, params.c_in, 1, params.k_size]),
            vec![
                kernel_stride[0],
                kernel_stride[1],
                kernel_stride[2] * params.k_size,
                kernel_stride[2],
            ],
            kernel_l.start_offset(),
        );
        let (input_dims, input_strides) = dims4(&input_l)?;
        let (kernel_dims, kernel_strides) = dims4(&kernel_2d_l)?;
        let out_shape = Shape::from(params.out_dims());
        let out_shader_shape = Shape::from(vec![params.b_size, params.c_out, 1, params.l_out()]);
        let out_shader_layout = Layout::contiguous(out_shader_shape);
        let (out_dims, out_strides) = dims4(&out_shader_layout)?;
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let shader = candle_wgpu_kernels::conv2d_f32_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader conv2d.wgsl not embedded".into()).bt())?;
        let params = Conv2dParams {
            offset_w: kernel_l.start_offset().try_into()?,
            offset_i: layout.start_offset().try_into()?,
            offset_o: 0,
            sw0: kernel_strides[0],
            sw1: kernel_strides[1],
            sw2: kernel_strides[2],
            sw3: kernel_strides[3],
            si0: input_strides[0],
            si1: input_strides[1],
            si2: input_strides[2],
            si3: input_strides[3],
            so0: out_strides[0],
            so1: out_strides[1],
            so2: out_strides[2],
            so3: out_strides[3],
            kw: kernel_dims[0],
            kh: kernel_dims[1],
            ic: kernel_dims[2],
            iw: input_dims[0],
            ih: input_dims[1],
            ow: out_dims[0],
            oh: out_dims[1],
            oc_out: out_dims[2],
            n_out: out_dims[3],
            s0: params.stride.try_into()?,
            s1: 1,
            p0: params.padding.try_into()?,
            p1: 0,
            d0: params.dilation.try_into()?,
            d1: 1,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-conv1d-params"),
                size: std::mem::size_of::<Conv2dParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &kernel.buffer),
            buffer_binding(1, &self.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let workgroups = (out_shape.elem_count() as u32).div_ceil(WG_SIZE);
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-conv1d",
        )?;
        Ok(dst)
    }

    fn run_conv2d_f32(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu conv2d").bt());
        }
        let (input_dims, input_strides) = dims4(layout)?;
        let (kernel_dims, kernel_strides) = dims4(kernel_l)?;
        let out_shape = Shape::from(params.out_dims());
        let out_layout = Layout::contiguous(out_shape.clone());
        let (out_dims, out_strides) = dims4(&out_layout)?;
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let shader = candle_wgpu_kernels::conv2d_f32_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader conv2d.wgsl not embedded".into()).bt())?;
        let params = Conv2dParams {
            offset_w: kernel_l.start_offset().try_into()?,
            offset_i: layout.start_offset().try_into()?,
            offset_o: 0,
            sw0: kernel_strides[0],
            sw1: kernel_strides[1],
            sw2: kernel_strides[2],
            sw3: kernel_strides[3],
            si0: input_strides[0],
            si1: input_strides[1],
            si2: input_strides[2],
            si3: input_strides[3],
            so0: out_strides[0],
            so1: out_strides[1],
            so2: out_strides[2],
            so3: out_strides[3],
            kw: kernel_dims[0],
            kh: kernel_dims[1],
            ic: kernel_dims[2],
            iw: input_dims[0],
            ih: input_dims[1],
            ow: out_dims[0],
            oh: out_dims[1],
            oc_out: out_dims[2],
            n_out: out_dims[3],
            s0: params.stride.try_into()?,
            s1: params.stride.try_into()?,
            p0: params.padding.try_into()?,
            p1: params.padding.try_into()?,
            d0: params.dilation.try_into()?,
            d1: params.dilation.try_into()?,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-conv2d-params"),
                size: std::mem::size_of::<Conv2dParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &kernel.buffer),
            buffer_binding(1, &self.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let workgroups = (out_shape.elem_count() as u32).div_ceil(WG_SIZE);
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-conv2d",
        )?;
        Ok(dst)
    }

    fn run_conv_transpose1d_f32(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let src_dtype = self.dtype;
        if src_dtype != kernel.dtype {
            return Err(Error::UnsupportedDTypeForOp(src_dtype, "wgpu conv_transpose1d").bt());
        }
        if !matches!(
            src_dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::F64 | DType::U8
        ) {
            return Err(Error::UnsupportedDTypeForOp(src_dtype, "wgpu conv_transpose1d").bt());
        }
        let src_len = params
            .l_in
            .checked_mul(params.k_size)
            .ok_or_else(|| Error::Msg("wgpu conv_transpose1d src_len overflow".into()).bt())?;
        let l_out = params.l_out();
        if l_out > u32::MAX as usize {
            return Err(Error::Msg("wgpu conv_transpose1d output too large".into()).bt());
        }

        let input_f32 = self.materialize_to_f32(layout)?;
        let input_f32_l = Layout::contiguous(layout.shape());
        let kernel_f32 = kernel.materialize_to_f32(kernel_l)?;
        let kernel_mm_l = Layout::contiguous((params.c_in, params.c_out * params.k_size));

        let input_t_view_l = input_f32_l.transpose(1, 2)?;
        let mut input_t_owned = None;
        let (input_t, input_mm_l) =
            if input_t_view_l.is_contiguous() && input_t_view_l.start_offset() == 0 {
                (
                    &input_f32,
                    Layout::contiguous((params.b_size * params.l_in, params.c_in)),
                )
            } else {
                let mut tmp = unsafe {
                    self.device
                        .alloc_uninit(input_t_view_l.shape(), DType::F32)?
                };
                input_f32.copy_strided_src(&mut tmp, 0, &input_t_view_l)?;
                input_t_owned = Some(tmp);
                (
                    input_t_owned.as_ref().unwrap(),
                    Layout::contiguous((params.b_size * params.l_in, params.c_in)),
                )
            };

        let cols = input_t.matmul(
            &kernel_f32,
            (
                1,
                params.b_size * params.l_in,
                params.c_out * params.k_size,
                params.c_in,
            ),
            &input_mm_l,
            &kernel_mm_l,
        )?;
        let cols_l = Layout::contiguous((params.b_size, params.l_in, params.c_out, params.k_size));
        let src_perm_l = cols_l.permute(&[0, 2, 1, 3])?;
        let mut src_owned = None;
        let (src, src_l) = if src_perm_l.is_contiguous() && src_perm_l.start_offset() == 0 {
            (
                &cols,
                Layout::contiguous((params.b_size * params.c_out, src_len)),
            )
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(src_perm_l.shape(), DType::F32)? };
            cols.copy_strided_src(&mut tmp, 0, &src_perm_l)?;
            src_owned = Some(tmp);
            (
                src_owned.as_ref().unwrap(),
                Layout::contiguous((params.b_size * params.c_out, src_len)),
            )
        };

        let mut ids = Vec::with_capacity(src_len);
        let mut mask = Vec::with_capacity(src_len);
        for t_in in 0..params.l_in {
            let base = t_in * params.stride;
            for k in 0..params.k_size {
                let pos = base + k * params.dilation;
                if pos >= params.padding {
                    let out_idx = pos - params.padding;
                    if out_idx < l_out {
                        ids.push(out_idx as u32);
                        mask.push(1f32);
                        continue;
                    }
                }
                ids.push(0);
                mask.push(0f32);
            }
        }

        let ids_storage = self.device.storage_from_slice(&ids)?;
        let ids_l = Layout::contiguous(src_len);
        let mask_storage = self.device.storage_from_slice(&mask)?;
        let mask_l = Layout::contiguous((1, src_len))
            .broadcast_as((params.b_size * params.c_out, src_len))?;
        let src_masked = src.binary_impl::<Mul>(&mask_storage, &src_l, &mask_l)?;

        let out_shape = Shape::from(vec![params.b_size * params.c_out, l_out]);
        let out_l = Layout::contiguous(out_shape.clone());
        let zeros = self.device.zeros_impl(&out_shape, DType::F32)?;
        let out = zeros.index_add(&out_l, &ids_storage, &ids_l, &src_masked, &src_l, 1)?;

        let final_shape = Shape::from(params.out_dims());
        let final_l = Layout::contiguous(final_shape.clone());
        let mut result = unsafe { self.device.alloc_uninit(&final_shape, DType::F32)? };
        out.copy_strided_src(&mut result, 0, &final_l)?;
        drop(input_t_owned);
        drop(src_owned);
        if src_dtype == DType::F32 {
            Ok(result)
        } else {
            result.to_dtype(&final_l, src_dtype)
        }
    }

    fn run_conv_transpose2d_f32(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let src_dtype = self.dtype;
        if src_dtype != kernel.dtype {
            return Err(Error::UnsupportedDTypeForOp(src_dtype, "wgpu conv_transpose2d").bt());
        }
        if !matches!(
            src_dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::F64 | DType::U8
        ) {
            return Err(Error::UnsupportedDTypeForOp(src_dtype, "wgpu conv_transpose2d").bt());
        }
        let input_spatial = params.i_h.checked_mul(params.i_w).ok_or_else(|| {
            Error::Msg("wgpu conv_transpose2d input_spatial overflow".into()).bt()
        })?;
        let kernel_spatial = params.k_h.checked_mul(params.k_w).ok_or_else(|| {
            Error::Msg("wgpu conv_transpose2d kernel_spatial overflow".into()).bt()
        })?;
        let src_len = input_spatial
            .checked_mul(kernel_spatial)
            .ok_or_else(|| Error::Msg("wgpu conv_transpose2d src_len overflow".into()).bt())?;
        let out_h = params.out_h();
        let out_w = params.out_w();
        let out_spatial = out_h
            .checked_mul(out_w)
            .ok_or_else(|| Error::Msg("wgpu conv_transpose2d out_spatial overflow".into()).bt())?;
        if out_spatial > u32::MAX as usize {
            return Err(Error::Msg("wgpu conv_transpose2d output too large".into()).bt());
        }

        let input_f32 = self.materialize_to_f32(layout)?;
        let input_f32_l = Layout::contiguous(layout.shape());
        let kernel_f32 = kernel.materialize_to_f32(kernel_l)?;
        let kernel_mm_l = Layout::contiguous((params.c_in, params.c_out * kernel_spatial));

        let input_hw_view_l = input_f32_l.permute(&[0, 2, 3, 1])?;
        let mut input_hw_owned = None;
        let (input_hw, input_mm_l) =
            if input_hw_view_l.is_contiguous() && input_hw_view_l.start_offset() == 0 {
                (
                    &input_f32,
                    Layout::contiguous((params.b_size * input_spatial, params.c_in)),
                )
            } else {
                let mut tmp = unsafe {
                    self.device
                        .alloc_uninit(input_hw_view_l.shape(), DType::F32)?
                };
                input_f32.copy_strided_src(&mut tmp, 0, &input_hw_view_l)?;
                input_hw_owned = Some(tmp);
                (
                    input_hw_owned.as_ref().unwrap(),
                    Layout::contiguous((params.b_size * input_spatial, params.c_in)),
                )
            };

        let cols = input_hw.matmul(
            &kernel_f32,
            (
                1,
                params.b_size * input_spatial,
                params.c_out * kernel_spatial,
                params.c_in,
            ),
            &input_mm_l,
            &kernel_mm_l,
        )?;
        let cols_l =
            Layout::contiguous((params.b_size, input_spatial, params.c_out, kernel_spatial));
        let src_perm_l = cols_l.permute(&[0, 2, 1, 3])?;
        let mut src_owned = None;
        let (src, src_l) = if src_perm_l.is_contiguous() && src_perm_l.start_offset() == 0 {
            (
                &cols,
                Layout::contiguous((params.b_size * params.c_out, src_len)),
            )
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(src_perm_l.shape(), DType::F32)? };
            cols.copy_strided_src(&mut tmp, 0, &src_perm_l)?;
            src_owned = Some(tmp);
            (
                src_owned.as_ref().unwrap(),
                Layout::contiguous((params.b_size * params.c_out, src_len)),
            )
        };

        let mut ids = Vec::with_capacity(src_len);
        let mut mask = Vec::with_capacity(src_len);
        for i_h in 0..params.i_h {
            let base_h = i_h * params.stride;
            for i_w in 0..params.i_w {
                let base_w = i_w * params.stride;
                for k_h in 0..params.k_h {
                    let out_h_idx = base_h + k_h * params.dilation;
                    for k_w in 0..params.k_w {
                        let out_w_idx = base_w + k_w * params.dilation;
                        if out_h_idx >= params.padding && out_w_idx >= params.padding {
                            let out_h_idx = out_h_idx - params.padding;
                            let out_w_idx = out_w_idx - params.padding;
                            if out_h_idx < out_h && out_w_idx < out_w {
                                ids.push((out_h_idx * out_w + out_w_idx) as u32);
                                mask.push(1f32);
                                continue;
                            }
                        }
                        ids.push(0);
                        mask.push(0f32);
                    }
                }
            }
        }

        let ids_storage = self.device.storage_from_slice(&ids)?;
        let ids_l = Layout::contiguous(src_len);
        let mask_storage = self.device.storage_from_slice(&mask)?;
        let mask_l = Layout::contiguous((1, src_len))
            .broadcast_as((params.b_size * params.c_out, src_len))?;
        let src_masked = src.binary_impl::<Mul>(&mask_storage, &src_l, &mask_l)?;

        let out_shape = Shape::from(vec![params.b_size * params.c_out, out_spatial]);
        let out_l = Layout::contiguous(out_shape.clone());
        let zeros = self.device.zeros_impl(&out_shape, DType::F32)?;
        let out = zeros.index_add(&out_l, &ids_storage, &ids_l, &src_masked, &src_l, 1)?;

        let final_shape = Shape::from(params.out_dims());
        let final_l = Layout::contiguous(final_shape.clone());
        let mut result = unsafe { self.device.alloc_uninit(&final_shape, DType::F32)? };
        out.copy_strided_src(&mut result, 0, &final_l)?;
        drop(input_hw_owned);
        drop(src_owned);
        if src_dtype == DType::F32 {
            Ok(result)
        } else {
            result.to_dtype(&final_l, src_dtype)
        }
    }

    fn run_upsample_nearest1d_f32(&self, layout: &Layout, out_l: usize) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu upsample_nearest1d").bt());
        }
        let (b, c, l) = layout.shape().dims3()?;
        let rows = b * c;
        let mut src_contiguous = None;
        let (src, src_l) = if layout.is_contiguous() && layout.start_offset() == 0 {
            (self, Layout::contiguous((rows, l)))
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            src_contiguous = Some(tmp);
            (
                src_contiguous.as_ref().unwrap(),
                Layout::contiguous((rows, l)),
            )
        };
        let weights = nearest_interp_weights(l, out_l);
        let weight_storage = self.device.storage_from_slice(&weights)?;
        let weight_l = Layout::contiguous((l, out_l));
        let out = src.matmul(&weight_storage, (1, rows, out_l, l), &src_l, &weight_l)?;
        drop(src_contiguous);
        Ok(out)
    }

    fn run_upsample2d_f32(
        &self,
        layout: &Layout,
        out_h: usize,
        out_w: usize,
        h_weights: Vec<f32>,
        w_weights: Vec<f32>,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu upsample2d").bt());
        }
        let (b, c, h, w) = layout.shape().dims4()?;
        let bc = b * c;
        let mut src_contiguous = None;
        let (src, src_l) = if layout.is_contiguous() && layout.start_offset() == 0 {
            (self, Layout::contiguous((bc * h, w)))
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            src_contiguous = Some(tmp);
            (
                src_contiguous.as_ref().unwrap(),
                Layout::contiguous((bc * h, w)),
            )
        };

        let w_storage = self.device.storage_from_slice(&w_weights)?;
        let w_l = Layout::contiguous((w, out_w));
        let width = src.matmul(&w_storage, (1, bc * h, out_w, w), &src_l, &w_l)?;

        let width_l = Layout::new(
            Shape::from(vec![bc, out_w, h]),
            vec![h * out_w, 1, out_w],
            0,
        );
        let h_storage = self.device.storage_from_slice(&h_weights)?;
        let h_l = Layout::new(Shape::from(vec![bc, h, out_h]), vec![0, out_h, 1], 0);
        let height = width.matmul(&h_storage, (bc, out_w, out_h, h), &width_l, &h_l)?;

        let height_l = Layout::contiguous((bc, out_w, out_h)).transpose(1, 2)?;
        let out_shape = Shape::from(vec![b, c, out_h, out_w]);
        let mut out = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        height.copy_strided_src(&mut out, 0, &height_l)?;
        drop(src_contiguous);
        Ok(out)
    }

    fn run_pool2d_im2col_f32(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        max_pool: bool,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu pool2d").bt());
        }
        let (b, c, h, w) = layout.shape().dims4()?;
        let out_h = (h - kernel_size.0) / stride.0 + 1;
        let out_w = (w - kernel_size.1) / stride.1 + 1;
        let bc = b * c;
        let k = kernel_size.0 * kernel_size.1;

        let mut src_contiguous = None;
        let (input, input_l) = if layout.is_contiguous() && layout.start_offset() == 0 {
            (self, Layout::contiguous((bc, 1, h, w)))
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            src_contiguous = Some(tmp);
            (
                src_contiguous.as_ref().unwrap(),
                Layout::contiguous((bc, 1, h, w)),
            )
        };

        let col_shape = Shape::from(vec![bc, out_h, out_w, k]);
        let col_l = Layout::contiguous(col_shape.clone());
        let col = unsafe { self.device.alloc_uninit(&col_shape, self.dtype)? };
        let (input_dims, input_strides) = dims4(&input_l)?;
        let (col_dims, col_strides) = dims4(&col_l)?;
        let shader = candle_wgpu_kernels::im2col_f32_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader im2col.wgsl not embedded".into()).bt())?;
        let params = Im2ColParams {
            offset_i: 0,
            offset_o: 0,
            si0: input_strides[0],
            si1: input_strides[1],
            si2: input_strides[2],
            si3: input_strides[3],
            so0: col_strides[0],
            so1: col_strides[1],
            so2: col_strides[2],
            so3: col_strides[3],
            kw: kernel_size.1.try_into()?,
            kh: kernel_size.0.try_into()?,
            ic: 1,
            iw: input_dims[0],
            ih: input_dims[1],
            n: input_dims[3],
            ow: col_dims[2],
            oh: col_dims[1],
            s0: stride.1.try_into()?,
            s1: stride.0.try_into()?,
            p0: 0,
            p1: 0,
            d0: 1,
            d1: 1,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-pool2d-im2col-params"),
                size: std::mem::size_of::<Im2ColParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &input.buffer),
            buffer_binding(1, &col.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let workgroups = (col_shape.elem_count() as u32).div_ceil(WG_SIZE);
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-pool2d-im2col",
        )?;
        drop(src_contiguous);

        let out_col_shape = Shape::from(vec![bc, out_h, out_w, 1]);
        let out_col_l = Layout::contiguous(out_col_shape.clone());
        if !max_pool {
            let sum = <Self as BackendStorage>::reduce_op(&col, ReduceOp::Sum, &col_l, &[3])?;
            return <Self as BackendStorage>::affine(&sum, &out_col_l, 1.0 / k as f64, 0.0);
        }

        let col_strides = col_l.stride();
        let mut acc = None;
        for k_idx in 0..k {
            let col_k_l = Layout::new(
                out_col_shape.clone(),
                vec![
                    col_strides[0],
                    col_strides[1],
                    col_strides[2],
                    col_strides[3],
                ],
                k_idx,
            );
            let mut col_k = unsafe { self.device.alloc_uninit(&out_col_shape, self.dtype)? };
            col.copy_strided_src(&mut col_k, 0, &col_k_l)?;
            acc = Some(if let Some(prev) = acc.take() {
                <Self as BackendStorage>::binary_impl::<crate::op::Maximum>(
                    &prev, &col_k, &out_col_l, &out_col_l,
                )?
            } else {
                col_k
            });
        }
        acc.ok_or_else(|| Error::Msg("pool2d empty kernel".into()).bt())
    }

    pub(crate) fn quantized_index_select_f32(
        &self,
        qdtype: GgmlDType,
        src_shape: &Shape,
        ids: &Self,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if !self
            .device
            .inner
            .features
            .contains(wgpu::Features::SHADER_F16)
        {
            return Err(unsupported("quantized index_select requires shader-f16"));
        }
        if ids.dtype != DType::U32 {
            return Err(
                Error::UnsupportedDTypeForOp(ids.dtype, "wgpu quantized index_select ids").bt(),
            );
        }
        let (ids_data, ids_l) = if !ids_l.is_contiguous() || ids_l.start_offset() != 0 {
            let mut tmp = unsafe { ids.device.alloc_uninit(ids_l.shape(), ids.dtype)? };
            <Self as BackendStorage>::copy_strided_src(ids, &mut tmp, 0, ids_l)?;
            (tmp, Layout::contiguous(ids_l.shape().clone()))
        } else {
            (ids.clone(), ids_l.clone())
        };
        let ids_len = match ids_l.dims() {
            [ids_len] => *ids_len,
            _ => {
                // Flatten multi-dim ids to 1D
                ids_l.shape().elem_count()
            }
        };
        let dims = src_shape.dims();
        if dim >= dims.len() {
            crate::bail!("index_select dim {dim} out of range for {src_shape:?}")
        }
        let block_size = qdtype.block_size();
        let right_size: usize = dims[dim + 1..].iter().product();
        if !right_size.is_multiple_of(block_size) {
            crate::bail!(
                "wgpu quantized index_select requires block-aligned rows, got right_size={right_size}, block_size={block_size}"
            )
        }
        let left_size: usize = dims[..dim].iter().product();
        let src_dim = dims[dim];
        let mut dst_dims = dims.to_vec();
        dst_dims[dim] = ids_len;
        let dst_shape = Shape::from(dst_dims);
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::F32)? };
        let params = GetRowsParams {
            offset_src: 0,
            offset_idx: ids_l.start_offset().try_into()?,
            offset_dst: 0,
            stride_src1: (right_size / block_size).try_into()?,
            stride_src2: (src_dim * right_size / block_size).try_into()?,
            stride_src3: (src_dim * right_size * left_size / block_size).try_into()?,
            stride_idx0: ids_l.stride()[0].try_into()?,
            stride_idx1: 0,
            stride_idx2: 0,
            stride_dst1: right_size.try_into()?,
            stride_dst2: (ids_len * right_size).try_into()?,
            stride_dst3: (left_size * ids_len * right_size).try_into()?,
            ne0: right_size.try_into()?,
            n_rows: ids_len.try_into()?,
            ne2: left_size.try_into()?,
            ne3: 1,
            idx1: 1,
            idx2: 1,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-quant-get-rows-params"),
                size: std::mem::size_of::<GetRowsParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &ids_data.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::quantized_get_rows_f32_shader(
            wgpu_quantized_dtype(qdtype)?,
            WG_SIZE,
        )
        .ok_or_else(|| Error::Msg("wgpu quantized get_rows shader not embedded".into()).bt())?;
        let rows: u32 = (left_size * ids_len).try_into()?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.div_ceil(WG_SIZE),
            "candle-wgpu-quant-get-rows",
        )?;
        Ok(dst)
    }

    /// Dequantize GGUF raw-float payloads (`F32`/`F16`/`BF16` stored under the
    /// GGML dtype enum) without leaving the GPU. `F32` is a plain device copy;
    /// `F16`/`BF16` decode the packed 16-bit halves in WGSL via
    /// `unpack2x16float` / exponent-shift, so no `SHADER_F16` feature or CPU
    /// round-trip is required.
    pub(crate) fn quantized_raw_float_dequantize_f32(
        &self,
        qdtype: GgmlDType,
        elem_count: usize,
    ) -> Result<Self> {
        if self.dtype != DType::U8 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu raw-float dequantize").bt());
        }
        let dst_shape = Shape::from(elem_count);
        if qdtype == GgmlDType::F32 {
            let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::F32)? };
            let size = byte_len(DType::F32, elem_count, "wgpu raw-float dequantize")?;
            let mut encoder =
                self.device
                    .inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("candle-wgpu-quant-f32-copy"),
                    });
            encoder.copy_buffer_to_buffer(
                &self.buffer,
                0,
                &dst.buffer,
                0,
                wgpu_copy_size(size) as u64,
            );
            self.device.flush_before_standalone_submit()?;
            self.device.inner.queue.submit([encoder.finish()]);
            return Ok(dst);
        }
        let decode = match qdtype {
            GgmlDType::F16 => "unpack2x16float(word)[half_idx]",
            // bf16 -> f32: the 16-bit payload is the top half of the f32 bits.
            GgmlDType::BF16 => "bitcast<f32>(half_bits << 16u)",
            _ => {
                return Err(Error::Msg(format!(
                    "wgpu raw-float dequantize expects F32/F16/BF16, got {qdtype:?}"
                ))
                .bt())
            }
        };
        let shader = format!(
            r#"
struct Params {{
    ne: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}};
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.ne) {{ return; }}
    let word = src[gid.x / 2u];
    let half_idx = gid.x % 2u;
    let half_bits = (word >> (16u * half_idx)) & 0xffffu;
    dst[gid.x] = {decode};
}}
"#
        );
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::F32)? };
        // Reuse the 4-u32 uniform block layout; the shader only reads the
        // first field (`ne`), which maps onto `offset_src` here.
        let params = ArgMaxParams {
            offset_src: elem_count.try_into()?,
            offset_dst: 0,
            ne0: 0,
            _pad0: 0,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-quant-raw-float-params"),
                size: std::mem::size_of::<ArgMaxParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let workgroups: u32 = elem_count.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-quant-raw-float",
        )?;
        Ok(dst)
    }

    pub(crate) fn quantized_matmul(
        &self,
        qdtype: GgmlDType,
        qshape: &Shape,
        storage: &Self,
        layout: &Layout,
    ) -> Result<(Self, Shape)> {
        if !self
            .device
            .inner
            .features
            .contains(wgpu::Features::SHADER_F16)
        {
            return Err(unsupported("quantized matmul requires shader-f16"));
        }
        if storage.dtype != DType::F32 && storage.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(storage.dtype, "wgpu quantized matmul").bt());
        }
        if storage.dtype == DType::F16
            && !storage
                .device
                .inner
                .features
                .contains(wgpu::Features::SHADER_F16)
        {
            return Err(unsupported("quantized matmul f16"));
        }
        let rank = layout.dims().len();
        if rank < 2 {
            return Err(unsupported("quantized matmul rank"));
        }
        let (n, k) = qshape.dims2()?;
        let input_m = layout.dims()[rank - 2];
        let last_k = layout.dims()[rank - 1];
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with quantized weights {qshape:?}")
        }
        let mut dst_dims = layout.dims().to_vec();
        dst_dims.pop();
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        if rank > 2 {
            let mut src_contiguous = None;
            let (src, _src_layout) = if layout.is_contiguous() && layout.start_offset() == 0 {
                (storage, layout.clone())
            } else {
                let mut tmp =
                    unsafe { storage.device.alloc_uninit(layout.shape(), storage.dtype)? };
                storage.copy_strided_src(&mut tmp, 0, layout)?;
                src_contiguous = Some(tmp);
                (
                    src_contiguous.as_ref().unwrap(),
                    Layout::contiguous(layout.shape().clone()),
                )
            };
            let flat_m = layout.shape().elem_count() / k;
            let flat_layout = Layout::contiguous(Shape::from((flat_m, k)));
            let (dst, _) = self.quantized_matmul(qdtype, qshape, src, &flat_layout)?;
            drop(src_contiguous);
            return Ok((dst, dst_shape));
        }
        if input_m == 1 {
            return self.quantized_matvec(qdtype, qshape, storage, layout);
        }

        let mut src_contiguous = None;
        let (src, src_layout) = if layout.is_contiguous() && layout.start_offset() == 0 {
            (storage, layout.clone())
        } else {
            let mut tmp = unsafe { storage.device.alloc_uninit(layout.shape(), storage.dtype)? };
            storage.copy_strided_src(&mut tmp, 0, layout)?;
            src_contiguous = Some(tmp);
            (
                src_contiguous.as_ref().unwrap(),
                Layout::contiguous(layout.shape().clone()),
            )
        };

        let src_stride = src_layout.stride();
        let batch_inner = if rank >= 3 {
            src_layout.dims()[rank - 3]
        } else {
            1
        };
        let batch_outer = if rank >= 4 {
            src_layout.dims()[rank - 4]
        } else {
            1
        };
        let batch_count = batch_inner * batch_outer;
        let src_stride_batch_inner = if rank >= 3 {
            src_stride[rank - 3]
        } else {
            input_m * k
        };
        let src_stride_batch_outer = if rank >= 4 {
            src_stride[rank - 4]
        } else {
            batch_inner * input_m * k
        };

        let dst = unsafe { storage.device.alloc_uninit(&dst_shape, DType::F32)? };
        let params = MulMatParams {
            offset_src0: 0,
            offset_src1: src_layout.start_offset().try_into()?,
            offset_dst: 0,
            m: n.try_into()?,
            n: input_m.try_into()?,
            k: k.try_into()?,
            stride_01: (k / qdtype.block_size()).try_into()?,
            stride_11: src_stride[rank - 2].try_into()?,
            stride_02: 0,
            stride_12: src_stride_batch_inner.try_into()?,
            stride_03: 0,
            stride_13: src_stride_batch_outer.try_into()?,
            bs02: 1,
            bs03: 1,
            broadcast2: batch_inner.try_into()?,
            broadcast3: batch_outer.try_into()?,
            stride_0k: 1,
            stride_1k: 1,
            _pad0: 0,
            _pad1: 0,
        };
        let param_buffer = storage
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-quant-matmul-params"),
                size: std::mem::size_of::<MulMatParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        storage
            .device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &src.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::quantized_matmul_fast_shader(
            wgpu_quantized_dtype(qdtype)?,
            wgpu_kernel_dtype(src.dtype)?,
        )
        .ok_or_else(|| {
            Error::Msg("wgpu quantized mul_mat_reg_tile shader not embedded".into()).bt()
        })?;
        let (_, _, wg_size_m, wg_size_n, _) =
            candle_wgpu_kernels::quantized_matmul_fast_tile_shape();
        let tile_m_s = 4usize * wg_size_m as usize;
        let tile_n_s = 4usize * wg_size_n as usize;
        let total_wg = input_m
            .div_ceil(tile_n_s)
            .checked_mul(n.div_ceil(tile_m_s))
            .and_then(|v| v.checked_mul(batch_count))
            .ok_or_else(|| {
                Error::Msg("wgpu backend op quantized matmul workgroup overflow".into()).bt()
            })?;
        let total_wg: u32 = total_wg.try_into()?;
        let (wg_x, wg_y) = compute_2d_workgroups(total_wg, wgpu_dispatch_wg_cap(&storage.device));
        storage.device.run_compute_xyz(
            &shader,
            &entries,
            &bindings,
            (wg_x, wg_y, 1),
            &[],
            "candle-wgpu-quant-matmul",
        )?;
        drop(src_contiguous);
        Ok((dst, dst_shape))
    }

    fn quantized_matvec(
        &self,
        qdtype: GgmlDType,
        qshape: &Shape,
        storage: &Self,
        layout: &Layout,
    ) -> Result<(Self, Shape)> {
        if !self
            .device
            .inner
            .features
            .contains(wgpu::Features::SHADER_F16)
        {
            return Err(unsupported("quantized matvec requires shader-f16"));
        }
        if storage.dtype != DType::F32 && storage.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(storage.dtype, "wgpu quantized matvec").bt());
        }
        let rank = layout.dims().len();
        let (n, k) = qshape.dims2()?;
        let input_m = layout.dims()[rank - 2];
        if input_m != 1 {
            crate::bail!("wgpu quantized matvec expects input_m == 1, got {input_m}");
        }
        let last_k = layout.dims()[rank - 1];
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with quantized weights {qshape:?}")
        }

        let mut src_contiguous = None;
        let (src, src_layout) = if layout.is_contiguous() && layout.start_offset() == 0 {
            (storage, layout.clone())
        } else {
            let mut tmp = unsafe { storage.device.alloc_uninit(layout.shape(), storage.dtype)? };
            storage.copy_strided_src(&mut tmp, 0, layout)?;
            src_contiguous = Some(tmp);
            (
                src_contiguous.as_ref().unwrap(),
                Layout::contiguous(layout.shape().clone()),
            )
        };

        let src_stride = src_layout.stride();
        let batch_inner = if rank >= 3 {
            src_layout.dims()[rank - 3]
        } else {
            1
        };
        let batch_outer = if rank >= 4 {
            src_layout.dims()[rank - 4]
        } else {
            1
        };
        let batch_count = batch_inner * batch_outer;
        let src_stride_batch_inner = if rank >= 3 {
            src_stride[rank - 3]
        } else {
            input_m * k
        };
        let src_stride_batch_outer = if rank >= 4 {
            src_stride[rank - 4]
        } else {
            batch_inner * input_m * k
        };

        let mut dst_dims = src_layout.dims().to_vec();
        dst_dims.pop();
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        let dst = unsafe { storage.device.alloc_uninit(&dst_shape, DType::F32)? };
        let params = MulMatParams {
            offset_src0: 0,
            offset_src1: src_layout.start_offset().try_into()?,
            offset_dst: 0,
            m: n.try_into()?,
            n: input_m.try_into()?,
            k: k.try_into()?,
            stride_01: (k / qdtype.block_size()).try_into()?,
            stride_11: src_stride[rank - 2].try_into()?,
            stride_02: 0,
            stride_12: src_stride_batch_inner.try_into()?,
            stride_03: 0,
            stride_13: src_stride_batch_outer.try_into()?,
            bs02: 1,
            bs03: 1,
            broadcast2: batch_inner.try_into()?,
            broadcast3: batch_outer.try_into()?,
            stride_0k: 1,
            stride_1k: 1,
            _pad0: 0,
            _pad1: 0,
        };
        let param_buffer = storage
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-quant-matvec-params"),
                size: std::mem::size_of::<MulMatParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        storage
            .device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry(3),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &src.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let qdtype = wgpu_quantized_dtype(qdtype)?;
        let shader =
            candle_wgpu_kernels::quantized_matvec_shader(qdtype, wgpu_kernel_dtype(src.dtype)?)
                .ok_or_else(|| {
                    Error::Msg("wgpu quantized mul_mat_vec shader not embedded".into()).bt()
                })?;
        let outputs_per_wg = candle_wgpu_kernels::quantized_matvec_outputs_per_wg(qdtype) as usize;
        let total_wg = n
            .div_ceil(outputs_per_wg)
            .checked_mul(batch_count)
            .ok_or_else(|| {
                Error::Msg("wgpu backend op quantized matvec workgroup overflow".into()).bt()
            })?;
        let total_wg: u32 = total_wg.try_into()?;
        let (wg_x, wg_y) = compute_2d_workgroups(total_wg, wgpu_dispatch_wg_cap(&storage.device));
        storage.device.run_compute_xyz(
            &shader,
            &entries,
            &bindings,
            (wg_x, wg_y, 1),
            &[],
            "candle-wgpu-quant-matvec",
        )?;
        drop(src_contiguous);
        Ok((dst, dst_shape))
    }

    // Integer reductions (u8/u32/i64), fully GPU-resident. Mirrors the float
    // structure: multi-dim folds one dim at a time, non-last-dim permutes via
    // the GPU strided copy, last-dim runs a generated per-dtype WGSL kernel.
    fn run_int_reduce(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        let rank = layout.dims().len();
        if rank == 0 {
            return self.try_clone(layout);
        }
        if reduce_dims.is_empty() {
            return self.try_clone(layout);
        }
        for &dim in reduce_dims {
            if dim >= rank {
                crate::bail!("wgpu backend int reduce got out-of-range dim {dim} for rank {rank}")
            }
        }
        if reduce_dims.len() > 1 {
            let mut current_layout = layout.clone();
            let mut current_shape = layout.dims().to_vec();
            let mut current: Option<Self> = None;
            for &dim in reduce_dims {
                let src = current.as_ref().map_or(self, |s| s);
                let reduced = src.run_int_reduce(op, &current_layout, &[dim])?;
                current_shape[dim] = 1;
                current_layout = Layout::contiguous(Shape::from(current_shape.clone()));
                current = Some(reduced);
            }
            return match current {
                Some(v) => Ok(v),
                None => self.try_clone(layout),
            };
        }
        let dim = reduce_dims[0];
        if dim != rank - 1 {
            let perm = (0..rank)
                .filter(|&i| i != dim)
                .chain(std::iter::once(dim))
                .collect::<Vec<_>>();
            let perm_shape = perm.iter().map(|&i| layout.dims()[i]).collect::<Vec<_>>();
            let perm_stride = perm.iter().map(|&i| layout.stride()[i]).collect::<Vec<_>>();
            let perm_layout = Layout::new(
                Shape::from(perm_shape.clone()),
                perm_stride,
                layout.start_offset(),
            );
            let mut permuted = unsafe {
                self.device
                    .alloc_uninit(&Shape::from(perm_shape.clone()), self.dtype)?
            };
            self.copy_strided_src(&mut permuted, 0, &perm_layout)?;
            let permuted_layout = Layout::contiguous(Shape::from(perm_shape));
            return permuted.run_int_reduce(op, &permuted_layout, &[rank - 1]);
        }
        // Last-dim reduction: materialize contiguous, then dispatch.
        let mut materialized;
        let src;
        let src_layout;
        let contiguous_layout;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            src = self;
            src_layout = layout;
        } else {
            materialized = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut materialized, 0, layout)?;
            contiguous_layout = Layout::contiguous(layout.shape());
            src = &materialized;
            src_layout = &contiguous_layout;
        }
        src.run_int_reduce_last_dim(op, src_layout)
    }

    fn run_int_reduce_last_dim(&self, op: ReduceOp, layout: &Layout) -> Result<Self> {
        let rank = layout.dims().len();
        if rank == 0 {
            if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                return Ok(unsafe { self.device.alloc_uninit(&Shape::from(&[] as &[usize]), DType::U32)? });
            }
            return self.try_clone(layout);
        }
        let kx = *layout.dims().last().unwrap_or(&1);
        let mut dst_dims = layout.dims().to_vec();
        dst_dims[rank - 1] = 1;
        let dst_shape = Shape::from(dst_dims);
        let rows = dst_shape.elem_count();
        let is_arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);
        let dst_dtype = if is_arg { DType::U32 } else { self.dtype };
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, dst_dtype)? };

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct IntReduceParams {
            rows: u32,
            kx: u32,
        }
        let params = IntReduceParams {
            rows: rows.try_into()?,
            kx: kx.try_into()?,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-int-reduce-params"),
                size: std::mem::size_of::<IntReduceParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let shader = int_reduce_wgsl(op, self.dtype)?;
        // The U8 value-reduce kernel runs one invocation per packed output
        // word (four rows); everything else runs one invocation per row.
        let work_items = if self.dtype == DType::U8 && !is_arg {
            (rows as u32).div_ceil(4)
        } else {
            rows as u32
        };
        let workgroups = work_items.div_ceil(WG_SIZE);
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-int-reduce",
        )?;
        Ok(dst)
    }
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        if layout.is_contiguous()
            && layout.start_offset() == 0
            && layout.shape().elem_count() == self.count
        {
            let size = byte_len(self.dtype, self.count, "wgpu clone")?;
            let buffer = self
                .device
                .register_buffer(self.device.create_storage_buffer(size, "candle-wgpu-clone"));
            let mut encoder =
                self.device
                    .inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("candle-wgpu-clone"),
                    });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, wgpu_copy_size(size) as u64);
            let completed = Arc::new(AtomicBool::new(false));
            let done = completed.clone();
            self.device
                .inner
                .queue
                .on_submitted_work_done(move || done.store(true, Ordering::Release));
            self.device.flush_before_standalone_submit()?;
            self.device.inner.queue.submit([encoder.finish()]);
            if let Ok(mut pending) = self.device.inner.pending_submissions.lock() {
                pending.push(WgpuPendingSubmission {
                    retained_buffers: vec![self.buffer.clone(), buffer.clone()],
                    completed,
                });
            }
            return Ok(Self {
                buffer,
                device: self.device.clone(),
                count: self.count,
                dtype: self.dtype,
            });
        }
        let mut out = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        Self::copy_strided_src(self, &mut out, 0, layout)?;
        Ok(out)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let size = byte_len(self.dtype, self.count, "wgpu download")?;
        let bytes = self.device.read_buffer(&self.buffer, size)?;
        bytes_to_cpu_storage(self.dtype, self.count, &bytes)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let gpu = match self.dtype {
            DType::F32 => self.run_scale(layout, mul as f32, add as f32),
            DType::F16 => {
                let src_f32 = self.materialize_to_f32(layout)?;
                let src_f32_layout = if layout.dims().len() > 4 {
                    Layout::contiguous(Self::compact_rank_gt4_shape(layout))
                } else {
                    Layout::contiguous(layout.shape())
                };
                let scaled = src_f32.run_scale(&src_f32_layout, mul as f32, add as f32)?;
                scaled.to_dtype(&src_f32_layout, DType::F16)
            }
            DType::BF16 => {
                let src_f32 = self.materialize_to_f32(layout)?;
                let src_f32_layout = if layout.dims().len() > 4 {
                    Layout::contiguous(Self::compact_rank_gt4_shape(layout))
                } else {
                    Layout::contiguous(layout.shape())
                };
                let scaled = src_f32.run_scale(&src_f32_layout, mul as f32, add as f32)?;
                scaled.to_dtype(&src_f32_layout, DType::BF16)
            }
            _ => Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu affine").bt()),
        };
        match gpu {
            Ok(out) => Ok(out),
            Err(err) => Err(err),
        }
    }
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if self.dtype == DType::BF16 {
            return self.bf16_unary_via_f32(layout, |src, src_l| src.powf(src_l, e));
        }
        if self.dtype == DType::F16
            && !self
                .device
                .inner
                .features
                .contains(wgpu::Features::SHADER_F16)
        {
            let src_f32 = self.materialize_to_f32(layout)?;
            let contiguous = if layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(layout))
            } else {
                Layout::contiguous(layout.shape())
            };
            let out_f32 = src_f32.powf(&contiguous, e)?;
            return out_f32.to_dtype(&contiguous, DType::F16);
        }
        let shader = custom_unary_wgsl(&format!("pow(x, {:?})", e as f32));
        self.run_unary_like(layout, &shader, "candle-wgpu-powf")
    }
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype == DType::F16
            && !self
                .device
                .inner
                .features
                .contains(wgpu::Features::SHADER_F16)
        {
            let src_f32 = self.materialize_to_f32(layout)?;
            let src_l = Layout::contiguous(layout.shape());
            let out_f32 = src_f32.elu(&src_l, alpha)?;
            return out_f32.to_dtype(&src_l, DType::F16);
        }
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            if self.dtype == DType::BF16 {
                return self.bf16_unary_via_f32(layout, |src, src_l| src.elu(src_l, alpha));
            }
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu elu").bt());
        }
        if alpha != 1.0 {
            let a = alpha as f32;
            let shader = custom_unary_wgsl(&format!("select({a} * (exp(x) - 1.0), x, x > 0.0)"));
            return self.run_unary_like(layout, &shader, "candle-wgpu-elu-alpha");
        }
        let shader = unary_shader("elu", self.dtype)?;
        self.run_unary_like(layout, &shader, "candle-wgpu-elu")
    }
    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if matches!(self.dtype, DType::U8 | DType::U32 | DType::I64) {
            return self.run_int_reduce(op, layout, reduce_dims);
        }
        if self.dtype == DType::BF16 {
            return self.materialize_to_f32(layout).and_then(|src_f32| {
                let contiguous = Layout::contiguous(layout.shape());
                let reduced = src_f32.reduce_op(op, &contiguous, reduce_dims)?;
                if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                    Ok(reduced)
                } else {
                    let mut out_dims = layout.dims().to_vec();
                    for &dim in reduce_dims {
                        out_dims[dim] = 1;
                    }
                    let out_layout = Layout::contiguous(Shape::from(out_dims));
                    reduced.to_dtype(&out_layout, DType::BF16)
                }
            });
        }
        if matches!(self.dtype, DType::F16) {
            let materialized_f32 = self.materialize_to_f32(layout)?;
            let contiguous_layout = Layout::contiguous(layout.shape());
            let reduced = materialized_f32.reduce_op(op, &contiguous_layout, reduce_dims)?;
            if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                return Ok(reduced);
            }
            let mut out_dims = layout.dims().to_vec();
            for &dim in reduce_dims {
                out_dims[dim] = 1;
            }
            let out_layout = Layout::contiguous(Shape::from(out_dims));
            return reduced.to_dtype(&out_layout, DType::F16);
        }
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu reduce").bt());
        }
        let rank = layout.dims().len();
        if rank == 0 {
            return self.try_clone(layout);
        }
        if reduce_dims.is_empty() {
            return self.try_clone(layout);
        }
        for &dim in reduce_dims {
            if dim >= rank {
                crate::bail!("wgpu backend op reduce got out-of-range dim {dim} for rank {rank}")
            }
        }
        if reduce_dims.len() > 1 {
            return self.run_reduce_multi_dim(op, layout, reduce_dims);
        }
        let dim = reduce_dims[0];
        if dim != rank - 1 {
            return self.run_reduce_non_last_dim(op, layout, dim);
        }
        match op {
            ReduceOp::ArgMax | ReduceOp::ArgMin | ReduceOp::Max | ReduceOp::Min => {
                return self.run_reduce_extrema_last_dim(layout, op);
            }
            ReduceOp::Sum => {}
        }
        let (dims, strides) = dims4(layout)?;
        let mut dst_dims = layout.dims().to_vec();
        dst_dims[rank - 1] = 1;
        let dst_shape = Shape::from(dst_dims);
        let rows = dst_shape.elem_count();
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };
        let params = SumRowsParams {
            offset_src: layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-sum-rows-params"),
                size: std::mem::size_of::<SumRowsParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        self.device
            .inner
            .queue
            .write_buffer(&param_buffer, 0, any_as_bytes(&params));
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            uniform_entry(2),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::get("sum_rows.wgsl")
            .ok_or_else(|| Error::Msg("wgpu shader sum_rows.wgsl not embedded".into()).bt())?
            .source()
            .replace("WG_SIZE", &WG_SIZE.to_string());
        match self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows as u32,
            "candle-wgpu-sum-rows",
        ) {
            Ok(()) => Ok(dst),
            Err(err) => Err(err),
        }
    }
    fn cumsum_last_dim(&self, layout: &Layout) -> Result<Self> {
        if self.dtype == DType::F32 && !layout.is_contiguous() {
            let src_f32 = self.materialize_to_f32(layout)?;
            let src_f32_layout = if layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(layout))
            } else {
                Layout::contiguous(layout.shape())
            };
            return src_f32.run_cumsum_last_dim(&src_f32_layout);
        }
        if self.dtype == DType::BF16 {
            return self.bf16_unary_via_f32(layout, |src, src_l| src.cumsum_last_dim(src_l));
        }
        if self.dtype == DType::F16 {
            let gpu = || -> Result<Self> {
                let src_f32 = self.materialize_to_f32(layout)?;
                let src_f32_layout = if layout.dims().len() > 4 {
                    Layout::contiguous(Self::compact_rank_gt4_shape(layout))
                } else {
                    Layout::contiguous(layout.shape())
                };
                let out_f32 = src_f32.cumsum_last_dim(&src_f32_layout)?;
                out_f32.to_dtype(&src_f32_layout, DType::F16)
            };
            return gpu();
        }
        self.run_cumsum_last_dim(layout)
    }
    fn clamp(&self, layout: &Layout, min: f32, max: f32) -> Result<Self> {
        if self.dtype == DType::BF16 {
            return match self
                .bf16_unary_via_f32(layout, |src, src_l| src.run_clamp(src_l, min, max))
            {
                Ok(out) => Ok(out),
                Err(err) => Err(err),
            };
        }
        if self.dtype == DType::F16 {
            let src_f32 = self.materialize_to_f32(layout)?;
            let contiguous = if layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(layout))
            } else {
                Layout::contiguous(layout.shape())
            };
            let out_f32 = src_f32.clamp(&contiguous, min, max)?;
            return out_f32.to_dtype(&contiguous, DType::F16);
        }
        self.run_clamp(layout, min, max)
    }
    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if self.dtype == DType::BF16 {
            return self.bf16_cmp_via_f32(rhs, lhs_l, rhs_l, op);
        }
        if wgpu_f16_emulates_f32(&self.device, self.dtype) {
            let lhs_f32 = self.materialize_to_f32(lhs_l)?;
            let rhs_f32 = rhs.materialize_to_f32(rhs_l)?;
            let lhs_l = Layout::contiguous(lhs_l.shape());
            let rhs_l = Layout::contiguous(rhs_l.shape());
            return lhs_f32.run_cmp_u8(&rhs_f32, &lhs_l, &rhs_l, op);
        }
        self.run_cmp_u8(rhs, lhs_l, rhs_l, op)
    }
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if layout.dims().len() > 4 {
            let (materialized, compact_layout) = self.materialize_rank_gt4_compact(layout)?;
            return materialized.to_dtype(&compact_layout, dtype);
        }
        let f16_involved = self.dtype == DType::F16 || dtype == DType::F16;
        let gpu = if f16_involved && !wgpu_shader_f16_enabled(&self.device) {
            if matches!(
                (self.dtype, dtype),
                (DType::F32, DType::F64) | (DType::F64, DType::F32) | (DType::F64, DType::F64)
            ) {
                self.run_f64_f32_cast(layout, dtype)
            } else if dtype == DType::F64 || self.dtype == DType::F64 {
                let f32 = self.materialize_to_f32(layout)?;
                f32.to_dtype(&Layout::contiguous(layout.shape()), dtype)
            } else {
                self.run_emulated_cast(layout, dtype)
            }
        } else if matches!(
            (self.dtype, dtype),
            (DType::F32, DType::F64) | (DType::F64, DType::F32) | (DType::F64, DType::F64)
        ) {
            self.run_f64_f32_cast(layout, dtype)
        } else {
            match copy_shader(self.dtype, dtype) {
                Ok(shader) => self.run_copy_to_dtype(layout, dtype, &shader),
                Err(_) => {
                    if self.dtype == DType::F64 || dtype == DType::F64 {
                        let layout_c = Layout::contiguous(layout.shape());
                        let f32 = if self.dtype == DType::F32 {
                            if layout.is_contiguous() && layout.start_offset() == 0 {
                                self.try_clone(layout)?
                            } else {
                                self.materialize_to_f32(layout)?
                            }
                        } else {
                            self.materialize_to_f32(layout)?
                        };
                        f32.to_dtype(&layout_c, dtype)
                    } else {
                        self.run_emulated_cast(layout, dtype)
                    }
                }
            }
        };
        match gpu {
            Ok(out) => Ok(out),
            Err(err) => Err(err),
        }
    }
    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype == DType::F16
            && (matches!(B::NAME, "erf" | "recip") || !wgpu_shader_f16_enabled(&self.device))
        {
            let src_f32 = self.materialize_to_f32(layout)?;
            let src_l = if layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(layout))
            } else {
                Layout::contiguous(layout.shape())
            };
            let out_f32 = src_f32.unary_impl::<B>(&src_l)?;
            return out_f32.to_dtype(&src_l, DType::F16);
        }
        if self.dtype == DType::F64 {
            let src_l = if layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(layout))
            } else {
                Layout::contiguous(layout.shape())
            };
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let out_f32 = src_f32.unary_impl::<B>(&src_l)?;
            return out_f32.to_dtype(&src_l, DType::F64);
        }
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            if self.dtype == DType::BF16 {
                return match self
                    .bf16_unary_via_f32(layout, |src, src_l| src.unary_impl::<B>(src_l))
                {
                    Ok(out) => Ok(out),
                    Err(err) => Err(err),
                };
            }
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "op").bt());
        }
        if layout.dims().len() > 4 {
            let (src, src_l) = self.materialize_rank_gt4_compact(layout)?;
            return src.unary_impl::<B>(&src_l);
        }
        let shader = unary_shader(B::NAME, self.dtype)?;
        self.run_unary_like(layout, &shader, "candle-wgpu-unary")
    }
    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype == DType::F16
            && !self
                .device
                .inner
                .features
                .contains(wgpu::Features::SHADER_F16)
        {
            let lhs_f32 = self.materialize_to_f32(lhs_layout)?;
            let rhs_f32 = rhs.materialize_to_f32(rhs_layout)?;
            let lhs_l = if lhs_layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(lhs_layout))
            } else {
                Layout::contiguous(lhs_layout.shape())
            };
            let rhs_l = if rhs_layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(rhs_layout))
            } else {
                Layout::contiguous(rhs_layout.shape())
            };
            let out_f32 = lhs_f32.binary_impl::<B>(&rhs_f32, &lhs_l, &rhs_l)?;
            return out_f32.to_dtype(&lhs_l, DType::F16);
        }
        if self.dtype == DType::F64 {
            let lhs_l = if lhs_layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(lhs_layout))
            } else {
                Layout::contiguous(lhs_layout.shape())
            };
            let rhs_l = if rhs_layout.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(rhs_layout))
            } else {
                Layout::contiguous(rhs_layout.shape())
            };
            let lhs_f32 = self.to_dtype(lhs_layout, DType::F32)?;
            let rhs_f32 = rhs.to_dtype(rhs_layout, DType::F32)?;
            let out_f32 = lhs_f32.binary_impl::<B>(&rhs_f32, &lhs_l, &rhs_l)?;
            return out_f32.to_dtype(&lhs_l, DType::F64);
        }
        if !matches!(
            self.dtype,
            DType::F32 | DType::F16 | DType::BF16 | DType::U8 | DType::U32 | DType::I32 | DType::I64
        ) {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "op").bt());
        }
        if self.dtype == DType::BF16 && rhs.dtype != DType::BF16 {
            return match self.bf16_binary_via_f32(
                rhs,
                lhs_layout,
                rhs_layout,
                |lhs, rhs, lhs_l, rhs_l| lhs.binary_impl::<B>(rhs, lhs_l, rhs_l),
            ) {
                Ok(out) => Ok(out),
                Err(err) => Err(err),
            };
        }
        if rhs.dtype != self.dtype {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "op").bt());
        }
        self.device
            .same_device(&rhs.device)
            .then_some(())
            .ok_or_else(|| {
                Error::DeviceMismatchBinaryOp {
                    lhs: self.device.location(),
                    rhs: rhs.device.location(),
                    op: B::NAME,
                }
                .bt()
            })?;
        if lhs_layout.dims().len() > 4 || rhs_layout.dims().len() > 4 {
            let (lhs, lhs_l) = if lhs_layout.dims().len() > 4 {
                self.materialize_rank_gt4_compact(lhs_layout)?
            } else {
                (
                    self.try_clone(lhs_layout)?,
                    Layout::contiguous(lhs_layout.shape()),
                )
            };
            let (rhs, rhs_l) = if rhs_layout.dims().len() > 4 {
                rhs.materialize_rank_gt4_compact(rhs_layout)?
            } else {
                (
                    rhs.try_clone(rhs_layout)?,
                    Layout::contiguous(rhs_layout.shape()),
                )
            };
            return lhs.binary_impl::<B>(&rhs, &lhs_l, &rhs_l);
        }
        let (lhs_dims, lhs_strides) = dims4(lhs_layout)?;
        let (rhs_dims, rhs_strides) = dims4(rhs_layout)?;
        let count = lhs_layout.shape().elem_count();

        let dst = unsafe { self.device.alloc_uninit(lhs_layout.shape(), self.dtype)? };
        let params = BinaryParams {
            ne: count.try_into()?,
            offset_src0: lhs_layout.start_offset().try_into()?,
            offset_src1: rhs_layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src0_0: lhs_strides[0],
            stride_src0_1: lhs_strides[1],
            stride_src0_2: lhs_strides[2],
            stride_src0_3: lhs_strides[3],
            stride_src1_0: rhs_strides[0],
            stride_src1_1: rhs_strides[1],
            stride_src1_2: rhs_strides[2],
            stride_src1_3: rhs_strides[3],
            a_ne0: lhs_dims[0],
            a_ne1: lhs_dims[1],
            a_ne2: lhs_dims[2],
            b_ne0: rhs_dims[0],
            b_ne1: rhs_dims[1],
            b_ne2: rhs_dims[2],
            b_ne3: rhs_dims[3],
            _pad0: 0,
        };
        let slot = self.device.inner.uniform_dyn_slot;
        let (param_buffer, dyn_off) = self
            .device
            .write_uniform_slot(any_as_bytes(&params))?;
        let entries = [
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, false),
            uniform_entry_dyn(3, slot),
        ];
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &rhs.buffer),
            buffer_binding(2, &dst.buffer),
            uniform_binding_dyn(3, &param_buffer, slot)?,
        ];
        let work_items = match self.dtype {
            DType::U8 => count.div_ceil(4),
            DType::BF16 => count.div_ceil(2),
            _ => count,
        };
        match self.device.run_compute_linear_dyn(
            &binary_shader(B::NAME, self.dtype)?,
            &entries,
            &bindings,
            work_items as u32,
            &[dyn_off],
            "candle-wgpu-binary",
        ) {
            Ok(()) => Ok(dst),
            Err(err) => Err(err),
        }
    }
    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if t.dtype == DType::BF16 {
            return self.bf16_where_via_f32(layout, t, t_l, f, f_l);
        }
        if wgpu_f16_emulates_f32(&t.device, t.dtype) {
            let t_f32 = t.materialize_to_f32(t_l)?;
            let f_f32 = f.materialize_to_f32(f_l)?;
            let layout_l = if layout.dims().len() > 4 {
                let (cond, cond_l) = self.materialize_rank_gt4_compact(layout)?;
                let out_f32 = cond.run_where_u8_cond(&cond_l, &t_f32, t_l, &f_f32, f_l)?;
                return out_f32.to_dtype(&cond_l, DType::F16);
            } else {
                Layout::contiguous(layout.shape())
            };
            let t_l = if t_l.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(t_l))
            } else {
                Layout::contiguous(t_l.shape())
            };
            let f_l = if f_l.dims().len() > 4 {
                Layout::contiguous(Self::compact_rank_gt4_shape(f_l))
            } else {
                Layout::contiguous(f_l.shape())
            };
            let out_f32 = self.run_where_u8_cond(layout, &t_f32, &t_l, &f_f32, &f_l)?;
            return out_f32.to_dtype(&layout_l, DType::F16);
        }
        self.run_where_u8_cond(layout, t, t_l, f, f_l)
    }
    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let out_shape = Shape::from(params.out_dims());
        self.cuda_parity_conv_via_f32(
            layout,
            kernel,
            kernel_l,
            &out_shape,
            "wgpu conv1d",
            |input, input_l, kernel, kernel_l| {
                input.run_conv1d_f32(input_l, kernel, kernel_l, params)
            },
        )
    }
    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.run_conv_transpose1d_f32(layout, kernel, kernel_l, params)
    }
    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let out_shape = Shape::from(params.out_dims());
        self.cuda_parity_conv_via_f32(
            layout,
            kernel,
            kernel_l,
            &out_shape,
            "wgpu conv2d",
            |input, input_l, kernel, kernel_l| {
                input.run_conv2d_f32(input_l, kernel, kernel_l, params)
            },
        )
    }
    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.run_conv_transpose2d_f32(layout, kernel, kernel_l, params)
    }
    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let (b, c, h, w) = layout.shape().dims4()?;
        let out_shape = Shape::from(vec![
            b,
            c,
            (h - kernel_size.0) / stride.0 + 1,
            (w - kernel_size.1) / stride.1 + 1,
        ]);
        self.gpu_resident_via_f32(layout, &out_shape, "wgpu avg_pool2d", |src, src_l| {
            src.run_pool2d_im2col_f32(src_l, kernel_size, stride, false)
        })
    }
    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let (b, c, h, w) = layout.shape().dims4()?;
        let out_shape = Shape::from(vec![
            b,
            c,
            (h - kernel_size.0) / stride.0 + 1,
            (w - kernel_size.1) / stride.1 + 1,
        ]);
        self.gpu_resident_via_f32(layout, &out_shape, "wgpu max_pool2d", |src, src_l| {
            src.run_pool2d_im2col_f32(src_l, kernel_size, stride, true)
        })
    }
    fn upsample_nearest1d(&self, layout: &Layout, out_l: usize) -> Result<Self> {
        self.run_upsample_nearest1d_f32(layout, out_l)
    }
    fn upsample_nearest2d(&self, layout: &Layout, out_h: usize, out_w: usize) -> Result<Self> {
        let (b, c, h, w) = layout.shape().dims4()?;
        let out_shape = Shape::from(vec![b, c, out_h, out_w]);
        self.gpu_resident_via_f32(
            layout,
            &out_shape,
            "wgpu upsample_nearest2d",
            |src, src_l| {
                src.run_upsample2d_f32(
                    src_l,
                    out_h,
                    out_w,
                    nearest_interp_weights(h, out_h),
                    nearest_interp_weights(w, out_w),
                )
            },
        )
    }
    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let (b, c, h, w) = layout.shape().dims4()?;
        let out_shape = Shape::from(vec![b, c, out_h, out_w]);
        self.gpu_resident_via_f32(
            layout,
            &out_shape,
            "wgpu upsample_bilinear2d",
            |src, src_l| {
                src.run_upsample2d_f32(
                    src_l,
                    out_h,
                    out_w,
                    bilinear_interp_weights(h, out_h, align_corners, scale_h),
                    bilinear_interp_weights(w, out_w, align_corners, scale_w),
                )
            },
        )
    }
    fn gather(&self, src_l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let rank = src_l.dims().len();
        if dim + 1 == rank {
            return self.run_gather_last_dim_f32(ids, src_l, ids_l);
        }
        if rank == 0 {
            crate::bail!("wgpu gather requires rank >= 1");
        }
        let mut perm: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
        perm.push(dim);
        let mut inv_perm = vec![0usize; rank];
        for (idx, &src_dim) in perm.iter().enumerate() {
            inv_perm[src_dim] = idx;
        }
        let permute_and_gather = |storage: &Self, layout: &Layout| -> Result<(Self, Layout)> {
            let permuted_l = layout.permute(&perm)?;
            let mut compact = unsafe {
                storage
                    .device
                    .alloc_uninit(permuted_l.shape(), storage.dtype)?
            };
            storage.copy_strided_src(&mut compact, 0, &permuted_l)?;
            Ok((compact, Layout::contiguous(permuted_l.shape())))
        };
        let (self_p, self_pl) = permute_and_gather(self, src_l)?;
        let (ids_p, ids_pl) = permute_and_gather(ids, ids_l)?;
        let out = self_p.run_gather_last_dim_f32(&ids_p, &self_pl, &ids_pl)?;
        let out_l = Layout::contiguous(ids_pl.shape()).permute(&inv_perm)?;
        let mut result = unsafe { self.device.alloc_uninit(out_l.shape(), out.dtype)? };
        out.copy_strided_src(&mut result, 0, &out_l)?;
        Ok(result)
    }
    fn scatter_set(
        &mut self,
        dst_l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let rank = dst_l.dims().len();
        if dim + 1 == rank {
            return self.run_scatter_set_last_dim_f32(dst_l, ids, ids_l, src, src_l);
        }
        let mut perm: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
        perm.push(dim);
        let mut inv_perm = vec![0usize; rank];
        for (idx, &src_dim) in perm.iter().enumerate() {
            inv_perm[src_dim] = idx;
        }
        let permute_contiguous = |storage: &Self, layout: &Layout| -> Result<(Self, Layout)> {
            let permuted_l = layout.permute(&perm)?;
            let mut compact = unsafe {
                storage
                    .device
                    .alloc_uninit(permuted_l.shape(), storage.dtype)?
            };
            storage.copy_strided_src(&mut compact, 0, &permuted_l)?;
            Ok((compact, Layout::contiguous(permuted_l.shape())))
        };
        let dst_perm_l = dst_l.permute(&perm)?;
        let mut dst_p = unsafe { self.device.alloc_uninit(dst_perm_l.shape(), self.dtype)? };
        self.copy_strided_src(&mut dst_p, 0, &dst_perm_l)?;
        let (ids_p, ids_pl) = permute_contiguous(ids, ids_l)?;
        let (src_p, src_pl) = permute_contiguous(src, src_l)?;
        let dst_pl = Layout::contiguous(dst_perm_l.shape());
        dst_p.run_scatter_set_last_dim_f32(&dst_pl, &ids_p, &ids_pl, &src_p, &src_pl)?;
        let inv_l = dst_pl.permute(&inv_perm)?;
        dst_p.copy_strided_src(self, dst_l.start_offset(), &inv_l)?;
        Ok(())
    }
    fn scatter_add_set(
        &mut self,
        dst_l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let rank = dst_l.dims().len();
        if dim + 1 == rank {
            return self.run_scatter_add_last_dim_f32(dst_l, ids, ids_l, src, src_l);
        }
        let mut perm: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
        perm.push(dim);
        let mut inv_perm = vec![0usize; rank];
        for (idx, &src_dim) in perm.iter().enumerate() {
            inv_perm[src_dim] = idx;
        }
        let permute_contiguous = |storage: &Self, layout: &Layout| -> Result<(Self, Layout)> {
            let permuted_l = layout.permute(&perm)?;
            let mut compact = unsafe {
                storage
                    .device
                    .alloc_uninit(permuted_l.shape(), storage.dtype)?
            };
            storage.copy_strided_src(&mut compact, 0, &permuted_l)?;
            Ok((compact, Layout::contiguous(permuted_l.shape())))
        };
        let dst_perm_l = dst_l.permute(&perm)?;
        let mut dst_p = unsafe { self.device.alloc_uninit(dst_perm_l.shape(), self.dtype)? };
        self.copy_strided_src(&mut dst_p, 0, &dst_perm_l)?;
        let (ids_p, ids_pl) = permute_contiguous(ids, ids_l)?;
        let (src_p, src_pl) = permute_contiguous(src, src_l)?;
        let dst_pl = Layout::contiguous(dst_perm_l.shape());
        dst_p.run_scatter_add_last_dim_f32(&dst_pl, &ids_p, &ids_pl, &src_p, &src_pl)?;
        let inv_l = dst_pl.permute(&inv_perm)?;
        dst_p.copy_strided_src(self, dst_l.start_offset(), &inv_l)?;
        Ok(())
    }
    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        self.run_index_select_f32(ids, src_l, ids_l, dim)
    }
    fn index_add(
        &self,
        dst_l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if dim + 1 == dst_l.dims().len() {
            let mut out = self.try_clone(dst_l)?;
            out.run_scatter_add_last_dim_f32(dst_l, ids, ids_l, src, src_l)?;
            return Ok(out);
        }
        let rank = dst_l.dims().len();
        let mut perm: Vec<usize> = (0..rank).filter(|&d| d != dim).collect();
        perm.push(dim);
        let mut inv_perm = vec![0usize; rank];
        for (idx, &src_dim) in perm.iter().enumerate() {
            inv_perm[src_dim] = idx;
        }
        let permute_contiguous = |storage: &Self, layout: &Layout| -> Result<(Self, Layout)> {
            let permuted_l = layout.permute(&perm)?;
            let mut compact = unsafe {
                storage
                    .device
                    .alloc_uninit(permuted_l.shape(), storage.dtype)?
            };
            storage.copy_strided_src(&mut compact, 0, &permuted_l)?;
            Ok((compact, Layout::contiguous(permuted_l.shape())))
        };
        let dst_perm_l = dst_l.permute(&perm)?;
        let mut dst_p = unsafe { self.device.alloc_uninit(dst_perm_l.shape(), self.dtype)? };
        self.copy_strided_src(&mut dst_p, 0, &dst_perm_l)?;
        let (ids_p, ids_pl) = permute_contiguous(ids, ids_l)?;
        let (src_p, src_pl) = permute_contiguous(src, src_l)?;
        let dst_pl = Layout::contiguous(dst_perm_l.shape());
        dst_p.run_scatter_add_last_dim_f32(&dst_pl, &ids_p, &ids_pl, &src_p, &src_pl)?;
        let inv_l = dst_pl.permute(&inv_perm)?;
        let mut out = self.try_clone(dst_l)?;
        dst_p.copy_strided_src(&mut out, dst_l.start_offset(), &inv_l)?;
        Ok(out)
    }
    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.run_matmul_f32(rhs, bmnk, lhs_l, rhs_l)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if self.dtype != dst.dtype {
            crate::bail!(
                "copy with inconsistent dtypes {:?} {:?}",
                self.dtype,
                dst.dtype
            )
        }
        if !src_l.is_contiguous() {
            if src_l.dims().len() > 4
                && dst_offset == 0
                && dst.count == src_l.shape().elem_count()
                && matches!(
                    self.dtype,
                    DType::F32
                        | DType::F16
                        | DType::BF16
                        | DType::U8
                        | DType::U32
                        | DType::I16
                        | DType::I64
                )
            {
                let (materialized, _) = self.materialize_rank_gt4_compact(src_l)?;
                *dst = materialized;
                return Ok(());
            }
            if self.dtype == DType::F16
                && !self
                    .device
                    .inner
                    .features
                    .contains(wgpu::Features::SHADER_F16)
            {
                let src_f32 = self.materialize_to_f32(src_l)?;
                let mut dst_f32 = unsafe { dst.device.alloc_uninit(src_l.shape(), DType::F32)? };
                src_f32.copy_strided_src(
                    &mut dst_f32,
                    dst_offset,
                    &Layout::contiguous(src_l.shape()),
                )?;
                *dst = dst_f32.to_dtype(&Layout::contiguous(src_l.shape()), DType::F16)?;
                return Ok(());
            }
            match self.dtype {
                DType::F32 | DType::F16 | DType::U32 | DType::I32 => {
                    let shader = copy_shader(self.dtype, self.dtype)?;
                    match self.run_copy_into(src_l, dst, dst_offset, &shader) {
                        Ok(()) => return Ok(()),
                        Err(err) => return Err(err),
                    }
                }
                DType::BF16 | DType::U8 | DType::I16 | DType::I64 | DType::F64 => {
                    match self.run_emulated_strided_copy_into(src_l, dst, dst_offset) {
                        Ok(()) => return Ok(()),
                        Err(err) => return Err(err),
                    }
                }
                _ => {
                    return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu copy_strided").bt());
                }
            }
        }
        let elem_size = self.dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu copy").bt());
        }
        let src_offset = src_l.start_offset() * elem_size;
        let size = src_l.shape().elem_count() * elem_size;
        let dst_byte_offset = dst_offset * elem_size;
        if !src_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
            || !dst_byte_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
            || !size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
        {
            return match self.dtype {
                DType::F32 | DType::F16 | DType::U32 | DType::I32 => {
                    let shader = copy_shader(self.dtype, self.dtype)?;
                    self.run_copy_into(src_l, dst, dst_offset, &shader)
                }
                DType::BF16 | DType::U8 | DType::I16 | DType::I64 | DType::F64 => {
                    self.run_emulated_strided_copy_into(src_l, dst, dst_offset)
                }
                _ => Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu copy_strided").bt()),
            };
        }
        let mut encoder =
            self.device
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("candle-wgpu-copy"),
                });
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            src_offset as u64,
            &dst.buffer,
            dst_byte_offset as u64,
            size as u64,
        );
        self.device.flush_before_standalone_submit()?;
        self.device.inner.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if self.dtype != dst.dtype {
            crate::bail!(
                "copy2d with inconsistent dtypes {:?} {:?}",
                self.dtype,
                dst.dtype
            )
        }
        let elem_size = self.dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu copy2d").bt());
        }
        let mut encoder =
            self.device
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("candle-wgpu-copy2d"),
                });
        for i1 in 0..d1 {
            let src = (i1 * src_stride1 + src_offset) * elem_size;
            let dst_offset = (i1 * dst_stride1 + dst_offset) * elem_size;
            encoder.copy_buffer_to_buffer(
                &self.buffer,
                src as u64,
                &dst.buffer,
                dst_offset as u64,
                (d2 * elem_size) as u64,
            );
        }
        self.device.flush_before_standalone_submit()?;
        self.device.inner.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        match (self.dtype, scalar) {
            (DType::F32, crate::scalar::Scalar::F32(value)) if layout.dims().len() <= 4 => {
                return self.run_fill_inplace(layout, value);
            }
            (DType::F16, crate::scalar::Scalar::F16(value)) if layout.dims().len() <= 4 => {
                if !self
                    .device
                    .inner
                    .features
                    .contains(wgpu::Features::SHADER_F16)
                {
                    let value = scalar_raw_words(scalar, self.dtype, "wgpu const_set")?;
                    return self.run_raw_fill_inplace(layout, value);
                }
                return self.run_fill_inplace(layout, value.to_f32());
            }
            _ => {}
        }
        let scalar = scalar_raw_words(scalar, self.dtype, "wgpu const_set")?;
        self.run_raw_fill_inplace(layout, scalar)
    }
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        // Prefer native Vulkan under wgpu when available — DX12 compute has
        // historically shown higher dispatch/overhead for our GEMM kernels on
        // NVIDIA Windows. Fall back to all backends if Vulkan is unavailable.
        let backend_pref = std::env::var("CANDLE_WGPU_BACKENDS")
            .ok()
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_else(|| "vulkan,primary".into());
        instance_desc.backends = if backend_pref.contains("all") {
            wgpu::Backends::all()
        } else if backend_pref.contains("dx12") && !backend_pref.contains("vulkan") {
            wgpu::Backends::DX12
        } else if backend_pref.contains("vulkan") && !backend_pref.contains("dx12") {
            wgpu::Backends::VULKAN
        } else {
            // Default: try Vulkan first by restricting enumeration order later;
            // keep all backends so we can fall back.
            wgpu::Backends::all()
        };
        let backends = instance_desc.backends;
        let instance = wgpu::Instance::new(instance_desc);
        let mut adapters = pollster::block_on(instance.enumerate_adapters(backends));
        if adapters.is_empty() && backends != wgpu::Backends::all() {
            // Fallback if preferred backend had no adapters.
            let mut fallback = wgpu::InstanceDescriptor::new_without_display_handle();
            fallback.backends = wgpu::Backends::all();
            let fb_backends = fallback.backends;
            let fb_instance = wgpu::Instance::new(fallback);
            adapters = pollster::block_on(fb_instance.enumerate_adapters(fb_backends));
        }
        if adapters.is_empty() {
            crate::bail!("no wgpu adapters found")
        }
        // Prefer Vulkan adapters over DX12/GL when multiple backends enumerate.
        adapters.sort_by_key(|a| {
            let b = a.get_info().backend;
            match b {
                wgpu::Backend::Vulkan => 0u8,
                wgpu::Backend::Metal => 1,
                wgpu::Backend::Dx12 => 2,
                wgpu::Backend::Gl => 3,
                _ => 4,
            }
        });
        let requested_name = std::env::var("CANDLE_WGPU_ADAPTER_NAME")
            .ok()
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty());
        let selected_index = if let Some(requested_name) = requested_name {
            let requested_name = requested_name.to_ascii_lowercase();
            adapters
                .iter()
                .position(|adapter| {
                    adapter
                        .get_info()
                        .name
                        .to_ascii_lowercase()
                        .contains(&requested_name)
                })
                .ok_or_else(|| {
                    Error::msg(format!(
                        "no wgpu adapter matching CANDLE_WGPU_ADAPTER_NAME={requested_name:?}"
                    ))
                })?
        } else {
            let mut preferred = adapters
                .iter()
                .enumerate()
                .filter(|(_, adapter)| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
                .map(|(index, _)| index);
            preferred
                .nth(ordinal)
                .or_else(|| {
                    adapters
                        .iter()
                        .position(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
                })
                .unwrap_or(ordinal.min(adapters.len() - 1))
        };
        let adapter = adapters.swap_remove(selected_index);
        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();
        let adapter_limits = adapter.limits();
        if !adapter_features.contains(wgpu::Features::SHADER_F64) {
            return Err(Error::msg(format!(
                "wgpu adapter {:?} does not support SHADER_F64 (required for native f64)",
                adapter_info.name
            )));
        }
        // Request optional accelerate features when the adapter exposes them.
        // EXPERIMENTAL_COOPERATIVE_MATRIX enables tensor-core 8×8 f32 MMA on
        // Vulkan (NVIDIA/AMD) for dense GEMM warptile paths.
        let optional = wgpu::Features::SHADER_F16
            | wgpu::Features::SUBGROUP
            | wgpu::Features::SUBGROUP_BARRIER
            | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        let required_features =
            wgpu::Features::SHADER_F64 | (adapter_features & optional);
        // SAFETY: cooperative matrix is experimental; we only enable it when
        // the adapter advertises EXPERIMENTAL_COOPERATIVE_MATRIX and fall back
        // to software warptile otherwise. Report bugs to gfx-rs/wgpu if hit.
        let experimental_features = if required_features
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        {
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("candle-wgpu"),
            required_features,
            required_limits: adapter_limits.clone(),
            experimental_features,
            ..Default::default()
        }))
        .map_err(Error::wrap)?;
        // Cache env once — env::var on every matmul was measurable host cost.
        let coop_matmul_enabled = std::env::var("CANDLE_WGPU_COOP_MATMUL")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        // WebGPU requires dynamic uniform offsets multiple of this alignment.
        let uniform_dyn_slot =
            u64::from(adapter_limits.min_uniform_buffer_offset_alignment).max(256);
        Ok(Self {
            inner: Arc::new(WgpuInner {
                ordinal,
                adapter_name: adapter_info.name,
                adapter_backend: format!("{:?}", adapter_info.backend),
                adapter_driver: adapter_info.driver,
                adapter_driver_info: adapter_info.driver_info,
                adapter_pci_bus_id: adapter_info.device_pci_bus_id,
                device,
                queue,
                features: required_features,
                limits: adapter_limits,
                coop_matmul_enabled,
                seed_value: RwLock::new(299_792_458),
                pipeline_cache: Mutex::new(HashMap::new()),
                buffer_registry: Mutex::new(HashMap::new()),
                storage_buffer_pool: Mutex::new(HashMap::new()),
                pending_submissions: Mutex::new(Vec::new()),
                active_batch: Mutex::new(None),
                uniform_ring: Mutex::new((Vec::new(), 0)),
                uniform_dyn: Mutex::new(None),
                uniform_dyn_slot,
                hot_rings: Mutex::new(HashMap::new()),
                elem_bg_cache: Mutex::new(HashMap::new()),
                elem_bg_hits: std::sync::atomic::AtomicU64::new(0),
                elem_bg_misses: std::sync::atomic::AtomicU64::new(0),
            }),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Wgpu {
            gpu_id: self.inner.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let size = byte_len(dtype, count, "wgpu zeros")?;
        // wgpu can't allocate 0-byte buffers; use a 1-byte dummy for empty shapes.
        let alloc_size = size.max(1);
        let buffer =
            self.register_buffer(self.create_zeroed_storage_buffer(alloc_size, "candle-wgpu-zeros"));
        Ok(WgpuStorage {
            buffer,
            device: self.clone(),
            count,
            dtype,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let size = byte_len(dtype, count, "wgpu alloc_uninit")?;
        // wgpu can't allocate 0-byte buffers; use a 1-byte dummy for empty shapes.
        let alloc_size = size.max(1);
        // 1024² f32 = 4 MiB — common matmul dst size. Waiting here forced a
        // full GPU stall on every GEMM. Only block for truly large buffers, and
        // only if there is actually in-flight work.
        if alloc_size >= 16 * 1024 * 1024 {
            let flushed = self.flush_active_batch("large_alloc")?;
            if flushed {
                self.cleanup_pending_submissions(true)?;
            } else {
                self.cleanup_pending_submissions(false)?;
            }
            self.prune_buffer_registry();
        }
        let buffer = self.register_buffer_arc(
            self.create_storage_buffer_arc(alloc_size, "candle-wgpu-alloc-uninit"),
        );
        Ok(WgpuStorage {
            buffer,
            device: self.clone(),
            count,
            dtype,
        })
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&T::to_cpu_storage(data))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let (dtype, count, bytes) = cpu_storage_to_bytes(storage)?;
        let buffer = self.register_buffer_arc(
            self.create_storage_buffer_arc(bytes.len(), "candle-wgpu-upload"),
        );
        self.inner
            .queue
            .write_buffer(&buffer, 0, &wgpu_padded_write_bytes(&bytes));
        Ok(WgpuStorage {
            buffer,
            device: self.clone(),
            count,
            dtype,
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self::Storage> {
        self.rand_uniform_gpu(shape, dtype, min, max)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        self.rand_normal_gpu(shape, dtype, mean, std)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let mut guard = self
            .inner
            .seed_value
            .write()
            .map_err(|_| Error::msg("wgpu seed lock poisoned"))?;
        *guard = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let guard = self
            .inner
            .seed_value
            .read()
            .map_err(|_| Error::msg("wgpu seed lock poisoned"))?;
        Ok(*guard)
    }

    fn synchronize(&self) -> Result<()> {
        self.flush_active_batch("synchronize")?;
        self.cleanup_pending_submissions(true)?;
        // Rewind hot rings; keep elem_bg_cache so (src, dst_i) pairs hit.
        self.reset_hot_rings_if_idle();
        if std::env::var_os("CANDLE_DEBUG_ELEM_BG").is_some() {
            let h = self
                .inner
                .elem_bg_hits
                .load(std::sync::atomic::Ordering::Relaxed);
            let m = self
                .inner
                .elem_bg_misses
                .load(std::sync::atomic::Ordering::Relaxed);
            eprintln!("candle-wgpu elem_bg hits={h} misses={m}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod compute_2d_self_check {
    use super::{compute_2d_workgroups, WGPU_DISPATCH_WG_CAP};

    #[test]
    fn both_axes_within_limit() {
        for (total, reported_max) in [
            (1u32, 65535),
            (65535, 65535),
            (65536, 65535),
            (65536, 65536),
            (262_144, 65535),
            (1_048_576, 65536),
        ] {
            let (x, y) = compute_2d_workgroups(total, reported_max);
            assert!(
                x <= WGPU_DISPATCH_WG_CAP && y <= WGPU_DISPATCH_WG_CAP,
                "total={total} reported_max={reported_max} -> ({x},{y})"
            );
            assert!(
                x.saturating_mul(y) >= total.max(1),
                "total={total} reported_max={reported_max} -> ({x},{y})"
            );
        }
    }
}
