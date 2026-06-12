use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, WithDType};
use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex, RwLock};
use tracing::trace_span;

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlHeadParams {
    kx: u32,
    ky: u32,
    param1: f32,
    param2: f32,
    param3: f32,
    param4: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanArgsortParams {
    ncols: u32,
    ncols_padded: u32,
    ncols_padded_log2: u32,
    nrows: u32,
    order: u32,
    outer_start: u32,
    outer_end: u32,
    inner_start: u32,
    inner_end: u32,
}

const VULKAN_ARGSORT_NUM_PIPELINES: u32 = 11;
const VULKAN_ARGSORT_WG_UNROLL_FACTOR: u32 = 2;
const VULKAN_MUL_MAT_VEC_MAX_COLS: usize = 8;
const VULKAN_DENSE_MUL_MAT_VEC_MAX_ROWS: usize = 8;

/// Mirror of llama.cpp's `ggml-vulkan.cpp` dense-GEMM dispatch rule: the
/// row-looped `mul_mat_vec` family is only profitable while the number of
/// output rows stays at or below `mul_mat_vec_max_cols` (8). Larger dense
/// GEMMs route to the tiled `matmul_f32_f32` shader so one dispatch covers the
/// whole tile grid instead of `m / 8` host-side dispatches.
fn vulkan_dense_gemm_prefers_tiled(m: usize, n: usize, k: usize) -> bool {
    let _ = (n, k);
    m > VULKAN_DENSE_MUL_MAT_VEC_MAX_ROWS
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlBinaryParams {
    ne: u32,
    ne00: u32,
    ne01: u32,
    ne02: u32,
    ne03: u32,
    nb00: u32,
    nb01: u32,
    nb02: u32,
    nb03: u32,
    ne10: u32,
    ne11: u32,
    ne12: u32,
    ne13: u32,
    nb10: u32,
    nb11: u32,
    nb12: u32,
    nb13: u32,
    ne20: u32,
    ne21: u32,
    ne22: u32,
    ne23: u32,
    nb20: u32,
    nb21: u32,
    nb22: u32,
    nb23: u32,
    misalign_offsets: u32,
    param1: f32,
    param2: f32,
    param3: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlUnaryParams {
    ne: u32,
    ne00: u32,
    ne01: u32,
    ne02: u32,
    ne03: u32,
    nb00: u32,
    nb01: u32,
    nb02: u32,
    nb03: u32,
    ne10: u32,
    ne11: u32,
    ne12: u32,
    ne13: u32,
    nb10: u32,
    nb11: u32,
    nb12: u32,
    nb13: u32,
    misalign_offsets: u32,
    param1: f32,
    param2: f32,
    ne0_012mp: u32,
    ne0_012l: u32,
    ne0_01mp: u32,
    ne0_01l: u32,
    ne0_0mp: u32,
    ne0_0l: u32,
    ne1_012mp: u32,
    ne1_012l: u32,
    ne1_01mp: u32,
    ne1_01l: u32,
    ne1_0mp: u32,
    ne1_0l: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanWhereU8Params {
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
    offset_cond: u32,
    offset_true: u32,
    offset_false: u32,
    cond_nb0: u32,
    cond_nb1: u32,
    cond_nb2: u32,
    cond_nb3: u32,
    true_nb0: u32,
    true_nb1: u32,
    true_nb2: u32,
    true_nb3: u32,
    false_nb0: u32,
    false_nb1: u32,
    false_nb2: u32,
    false_nb3: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlSumRowsParams {
    n_cols: u32,
    ne01: u32,
    ne02: u32,
    nb01: u32,
    nb02: u32,
    nb03: u32,
    nb11: u32,
    nb12: u32,
    nb13: u32,
    weight: f32,
    misalign_offsets: u32,
    ne0_12mp: u32,
    ne0_12l: u32,
    ne0_1mp: u32,
    ne0_1l: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlSoftmaxParams {
    kx: u32,
    ky: u32,
    ne00: u32,
    ne01: u32,
    ne02: u32,
    ne12: u32,
    ne13: u32,
    nb11: u32,
    nb12: u32,
    nb13: u32,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: u32,
    nrows_x: u32,
    has_sinks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanRopeParams {
    rope_mode: u32,
    nrows: u32,
    n_dims: u32,
    freq_scale: f32,
    freq_base: f32,
    ext_factor: f32,
    attn_factor: f32,
    corr_dims: [f32; 2],
    theta_scale: f32,
    has_ff: u32,
    sections: [i32; 4],
    is_imrope: u32,
    is_back: u32,
    set_rows_stride: u32,
    ne00: u32,
    ne01: u32,
    ne02: u32,
    nb01: u32,
    nb02: u32,
    nb03: u32,
    nb11: u32,
    nb12: u32,
    nb13: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanMatmulParams {
    m: u32,
    n: u32,
    k: u32,
    stride_a: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    base_work_group_z: u32,
    num_batches: u32,
    k_split: u32,
    ne02: u32,
    ne12: u32,
    broadcast2: u32,
    broadcast3: u32,
    padded_n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanMatVecParams {
    ncols: u32,
    stride_a: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    fusion_flags: u32,
    base_work_group_y: u32,
    ne02: u32,
    ne12: u32,
    broadcast2: u32,
    broadcast3: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanMatVecIdParams {
    ncols: u32,
    stride_a: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    fusion_flags: u32,
    nei0: u32,
    ne11: u32,
    expert_i1: u32,
    nbi1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanQuantizeQ8_1Params {
    ne: u32,
    num_blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanDequantizeParams {
    m: u32,
    k: u32,
    stride_a: u32,
    stride_b: u32,
    nel: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanRepackQ8_1ToQ8_0Params {
    num_blocks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanIm2ColParams {
    dst_addr: [u32; 2],
    batch_offset: u32,
    offset_delta: u32,
    ic: u32,
    iw: u32,
    ih: u32,
    ow: u32,
    oh: u32,
    kw: u32,
    kh: u32,
    oh_batch: u32,
    chw: u32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    d0: i32,
    d1: i32,
    batch_ic: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanPool2dParams {
    iw: u32,
    ih: u32,
    ow: u32,
    oh: u32,
    oc: u32,
    pelements: u32,
    op: u32,
    k0: i32,
    k1: i32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VulkanConv2dParams {
    cout: u32,
    cin: u32,
    n: u32,
    w: u32,
    h: u32,
    ow: u32,
    oh: u32,
    nb01: u32,
    nb02: u32,
    nb03: u32,
    nb11: u32,
    nb12: u32,
    nb13: u32,
    nb1: u32,
    nb2: u32,
    nb3: u32,
    owmp: u32,
    owl: u32,
    owohmp: u32,
    owohl: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        Self::Message(e)
    }
}

#[derive(Debug, Clone)]
pub struct VulkanDevice {
    inner: Arc<VulkanInner>,
}

struct VulkanInner {
    ordinal: usize,
    physical_device_name: String,
    physical_device_type: vk::PhysicalDeviceType,
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    vendor_id: u32,
    integer_dot_product: bool,
    subgroup_arithmetic: bool,
    subgroup_size: u32,
    subgroup_size_control: bool,
    compute_full_subgroups: bool,
    subgroup_min_size: u32,
    subgroup_max_size: u32,
    max_workgroup_size_log2: u32,
    max_workgroup_count_x: u32,
    max_workgroup_count_y: u32,
    max_push_constants_size: u32,
    robust_buffer_access: bool,
    vulkan_memory_model: bool,
    device: ash::Device,
    driver_pipeline_cache: vk::PipelineCache,
    queue_family_index: u32,
    queue: vk::Queue,
    transfer_queue_family_index: Option<u32>,
    transfer_queue: Option<vk::Queue>,
    allocator: Mutex<Option<Allocator>>,
    seed_value: RwLock<u64>,
    pipeline_cache: Mutex<HashMap<VulkanPipelineCacheKey, Arc<VulkanCachedPipeline>>>,
    pending_submissions: Mutex<Vec<VulkanPendingSubmission>>,
    active_compute_batch: Mutex<Option<VulkanActiveBatch>>,
    active_transfer_batch: Mutex<Option<VulkanActiveBatch>>,
    reusable_compute_submissions: Mutex<Vec<VulkanSubmissionResources>>,
    reusable_transfer_submissions: Mutex<Vec<VulkanSubmissionResources>>,
    deferred_buffer_frees: Mutex<Vec<VulkanDeferredBuffer>>,
}

impl VulkanDevice {
    pub fn physical_device_name(&self) -> &str {
        &self.inner.physical_device_name
    }

    pub fn physical_device_type(&self) -> vk::PhysicalDeviceType {
        self.inner.physical_device_type
    }

    pub fn integer_dot_product_supported(&self) -> bool {
        self.inner.integer_dot_product
    }

    pub fn subgroup_arithmetic_supported(&self) -> bool {
        self.inner.subgroup_arithmetic
    }

    pub fn subgroup_size(&self) -> u32 {
        self.inner.subgroup_size
    }

    pub fn subgroup_size_control_supported(&self) -> bool {
        self.inner.subgroup_size_control
    }

    pub fn subgroup_min_size(&self) -> u32 {
        self.inner.subgroup_min_size
    }

    pub fn subgroup_max_size(&self) -> u32 {
        self.inner.subgroup_max_size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmissionQueueKind {
    Compute,
    Transfer,
}

struct VulkanPendingSubmission {
    resources: VulkanSubmissionResources,
    queue_kind: SubmissionQueueKind,
    dispatch_count: u32,
    copy_count: u32,
    transfer_bytes: usize,
    compute_bytes: usize,
    retained_buffers: Vec<Arc<VulkanBuffer>>,
}

struct VulkanActiveBatch {
    resources: VulkanSubmissionResources,
    queue_kind: SubmissionQueueKind,
    dispatch_count: u32,
    copy_count: u32,
    descriptor_set_count: u32,
    allocated_descriptor_set_count: u32,
    storage_descriptor_count: u32,
    transfer_bytes: usize,
    compute_bytes: usize,
    retained_buffers: Vec<Arc<VulkanBuffer>>,
    cached_descriptor_sets: HashMap<vk::DescriptorSetLayout, SmallVec<[vk::DescriptorSet; 8]>>,
}

struct VulkanSubmissionResources {
    fence: vk::Fence,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
}

impl VulkanActiveBatch {
    fn has_commands(&self) -> bool {
        self.dispatch_count > 0 || self.copy_count > 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VulkanPipelineCacheKey {
    shader_hash: u64,
    shader_len_words: usize,
    binding_signature: SmallVec<[u32; 8]>,
    push_constant_len: u32,
    specialization_u32: SmallVec<[(u32, u32); 8]>,
    require_full_subgroups: bool,
    required_subgroup_size: Option<u32>,
}

struct VulkanCachedPipeline {
    shader: vk::ShaderModule,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

struct VulkanDeferredBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
}

struct VulkanInitGuard {
    instance: Option<ash::Instance>,
    device: Option<ash::Device>,
    driver_pipeline_cache: vk::PipelineCache,
}

impl VulkanInitGuard {
    fn new(instance: ash::Instance) -> Self {
        Self {
            instance: Some(instance),
            device: None,
            driver_pipeline_cache: vk::PipelineCache::null(),
        }
    }

    fn attach_device(&mut self, device: ash::Device) {
        self.device = Some(device);
    }

    fn attach_pipeline_cache(&mut self, pipeline_cache: vk::PipelineCache) {
        self.driver_pipeline_cache = pipeline_cache;
    }

    fn disarm(mut self) {
        self.instance = None;
        self.device = None;
        self.driver_pipeline_cache = vk::PipelineCache::null();
    }
}

impl Drop for VulkanInitGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(device) = self.device.as_ref() {
                let _ = device.device_wait_idle();
                if self.driver_pipeline_cache != vk::PipelineCache::null() {
                    device.destroy_pipeline_cache(self.driver_pipeline_cache, None);
                }
                device.destroy_device(None);
            }
            if let Some(instance) = self.instance.as_ref() {
                instance.destroy_instance(None);
            }
        }
    }
}

impl std::fmt::Debug for VulkanInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanInner")
            .field("ordinal", &self.ordinal)
            .field("physical_device", &self.physical_device)
            .field("vendor_id", &self.vendor_id)
            .field("integer_dot_product", &self.integer_dot_product)
            .field("subgroup_arithmetic", &self.subgroup_arithmetic)
            .field("subgroup_size", &self.subgroup_size)
            .field("subgroup_size_control", &self.subgroup_size_control)
            .field("compute_full_subgroups", &self.compute_full_subgroups)
            .field("subgroup_min_size", &self.subgroup_min_size)
            .field("subgroup_max_size", &self.subgroup_max_size)
            .field("max_workgroup_size_log2", &self.max_workgroup_size_log2)
            .field("max_workgroup_count_x", &self.max_workgroup_count_x)
            .field("max_workgroup_count_y", &self.max_workgroup_count_y)
            .field("max_push_constants_size", &self.max_push_constants_size)
            .field("robust_buffer_access", &self.robust_buffer_access)
            .field("vulkan_memory_model", &self.vulkan_memory_model)
            .field("queue_family_index", &self.queue_family_index)
            .field(
                "transfer_queue_family_index",
                &self.transfer_queue_family_index,
            )
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct VulkanBuffer {
    device: VulkanDevice,
    buffer: vk::Buffer,
    allocation: Mutex<Option<Allocation>>,
    size: usize,
}

#[derive(Debug, Clone)]
pub struct VulkanStorage {
    buffer: Arc<VulkanBuffer>,
    device: VulkanDevice,
    count: usize,
    dtype: DType,
}

fn unsupported(op: &'static str) -> Error {
    Error::Msg(format!("vulkan backend op {op} not implemented")).bt()
}

fn should_cpu_fallback(err: &Error) -> bool {
    fn matches_fallback(err: &Error) -> bool {
        match err {
            Error::UnsupportedDTypeForOp(..) => true,
            Error::Msg(msg) => msg.contains("vulkan backend op") && msg.contains("not implemented"),
            Error::Context { inner, .. }
            | Error::WithPath { inner, .. }
            | Error::WithBacktrace { inner, .. } => matches_fallback(inner),
            _ => false,
        }
    }
    let fallback = matches_fallback(err);
    if fallback {
        // Every backend-internal CPU recovery must be visible to the global
        // fallback counter, otherwise native-required tests pass while work
        // silently runs on the CPU.
        crate::storage::record_vulkan_cpu_fallback(err);
    }
    fallback
}

pub fn shader_source(name: &str) -> Option<&'static str> {
    candle_vulkan_kernels::get(name).map(|module| module.source())
}

fn vulkan_spirv_exists(name: &str) -> bool {
    candle_vulkan_kernels::spirv(name).is_some()
}

fn vulkan_quantized_stem(dtype: GgmlDType) -> Result<&'static str> {
    match dtype {
        GgmlDType::Q4_0 => Ok("q4_0"),
        GgmlDType::Q4_1 => Ok("q4_1"),
        GgmlDType::Q5_0 => Ok("q5_0"),
        GgmlDType::Q5_1 => Ok("q5_1"),
        GgmlDType::Q8_0 => Ok("q8_0"),
        GgmlDType::Q8_1 => Ok("q8_1"),
        GgmlDType::Q2K => Ok("q2_k"),
        GgmlDType::Q3K => Ok("q3_k"),
        GgmlDType::Q4K => Ok("q4_k"),
        GgmlDType::Q5K => Ok("q5_k"),
        GgmlDType::Q6K => Ok("q6_k"),
        other => crate::bail!("vulkan backend quantized dtype {other:?} is not supported"),
    }
}

fn vulkan_quantized_vec_rows(dtype: GgmlDType) -> Result<u32> {
    match dtype {
        GgmlDType::Q8_0 => Ok(1),
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_1
        | GgmlDType::Q2K
        | GgmlDType::Q3K
        | GgmlDType::Q4K
        | GgmlDType::Q5K
        | GgmlDType::Q6K => Ok(2),
        other => crate::bail!("vulkan backend quantized dtype {other:?} is not supported"),
    }
}

fn vulkan_supports_q8_1_rhs(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

fn vulkan_is_k_quant(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
    )
}

const VULKAN_VENDOR_ID_NVIDIA: u32 = 0x10DE;
const VULKAN_VENDOR_ID_AMD: u32 = 0x1002;
const VULKAN_VENDOR_ID_INTEL: u32 = 0x8086;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VulkanDmmvWorkgroup {
    Subgroup,
    Large,
}

fn vulkan_should_use_mmvq(device: &VulkanDevice, qdtype: GgmlDType, n: usize, k: usize) -> bool {
    if matches!(qdtype, GgmlDType::Q3K | GgmlDType::Q6K) {
        return false;
    }
    if n > 1 {
        return true;
    }
    match device.inner.vendor_id {
        VULKAN_VENDOR_ID_NVIDIA => {
            if matches!(qdtype, GgmlDType::Q2K) {
                return true;
            }
            if k <= 4096 {
                return false;
            }
            !matches!(qdtype, GgmlDType::Q8_0)
        }
        VULKAN_VENDOR_ID_AMD => {
            if k < 2048 {
                return false;
            }
            !matches!(qdtype, GgmlDType::Q8_0)
        }
        VULKAN_VENDOR_ID_INTEL => {
            if k < 2048 {
                return false;
            }
            !matches!(qdtype, GgmlDType::Q4_0 | GgmlDType::Q5_1)
        }
        _ => true,
    }
}

fn vulkan_use_subgroups(device: &VulkanDevice) -> bool {
    device.inner.subgroup_arithmetic
}

fn vulkan_has_subgroup_min_16(device: &VulkanDevice) -> bool {
    device.inner.subgroup_size_control
        && device.inner.subgroup_min_size <= 16
        && device.inner.subgroup_max_size >= 16
}

fn vulkan_dmmv_subgroup_size(device: &VulkanDevice) -> u32 {
    if device.inner.vendor_id == VULKAN_VENDOR_ID_INTEL && vulkan_has_subgroup_min_16(device) {
        16
    } else {
        device.inner.subgroup_size.max(1)
    }
}

fn vulkan_dmmv_subgroup_size16(device: &VulkanDevice) -> u32 {
    vulkan_dmmv_subgroup_size(device).max(16)
}

fn vulkan_dmmv_use_subgroups(device: &VulkanDevice, qdtype: GgmlDType) -> bool {
    let use_subgroups = vulkan_use_subgroups(device);
    if vulkan_is_k_quant(qdtype) {
        use_subgroups && vulkan_has_subgroup_min_16(device)
    } else {
        use_subgroups
    }
}

fn vulkan_supports_quantized_matvec_weight(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

fn vulkan_dmmv_workgroup(
    device: &VulkanDevice,
    qdtype: GgmlDType,
    m: usize,
    k: usize,
    q8_1_rhs: bool,
) -> VulkanDmmvWorkgroup {
    if qdtype == GgmlDType::F32 {
        return VulkanDmmvWorkgroup::Subgroup;
    }
    let mut workgroup = VulkanDmmvWorkgroup::Subgroup;
    if matches!(
        device.inner.vendor_id,
        VULKAN_VENDOR_ID_NVIDIA | VULKAN_VENDOR_ID_INTEL
    ) {
        if qdtype == GgmlDType::Q6K {
            if m < 4096 && k >= 1024 {
                workgroup = VulkanDmmvWorkgroup::Large;
            }
        } else if m <= 8192 && k >= 1024 {
            workgroup = VulkanDmmvWorkgroup::Large;
        }
    }
    if q8_1_rhs && device.inner.vendor_id == VULKAN_VENDOR_ID_INTEL {
        workgroup = VulkanDmmvWorkgroup::Subgroup;
    }
    workgroup
}

fn vulkan_dmmv_shader_name(
    device: &VulkanDevice,
    qdtype: GgmlDType,
    base_name: String,
    workgroup: VulkanDmmvWorkgroup,
) -> String {
    if !vulkan_dmmv_use_subgroups(device, qdtype) {
        return base_name;
    }
    match workgroup {
        VulkanDmmvWorkgroup::Subgroup => format!("{base_name}_subgroup"),
        VulkanDmmvWorkgroup::Large => format!("{base_name}_subgroup_no_shmem"),
    }
}

fn vulkan_dmmv_block_size(
    device: &VulkanDevice,
    qdtype: GgmlDType,
    workgroup: VulkanDmmvWorkgroup,
) -> u32 {
    let subgroup = if vulkan_is_k_quant(qdtype) {
        vulkan_dmmv_subgroup_size16(device)
    } else {
        vulkan_dmmv_subgroup_size(device)
    };
    match workgroup {
        VulkanDmmvWorkgroup::Subgroup => subgroup,
        VulkanDmmvWorkgroup::Large => subgroup.saturating_mul(4),
    }
}

fn vulkan_dmmv_rows_per_group(
    device: &VulkanDevice,
    qdtype: GgmlDType,
    q8_1_rhs: bool,
) -> Result<u32> {
    let rm_stdq_int = if device.inner.vendor_id == VULKAN_VENDOR_ID_INTEL {
        2
    } else {
        1
    };
    let rm_kq_int = 1;
    let rows = if q8_1_rhs {
        match qdtype {
            GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0 => rm_stdq_int,
            GgmlDType::Q2K => 2 * rm_kq_int,
            GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => rm_kq_int,
            other => {
                crate::bail!("vulkan backend q8_1 rhs rows/group unsupported for {other:?}")
            }
        }
    } else {
        vulkan_quantized_vec_rows(qdtype)?
    };
    Ok(rows)
}

fn vulkan_q8_1_rhs_matmul_shader_name(qdtype: GgmlDType) -> Result<Option<String>> {
    if !vulkan_supports_q8_1_rhs(qdtype) {
        return Ok(None);
    }
    let name = format!("matmul_{}_q8_1", vulkan_quantized_stem(qdtype)?);
    Ok(vulkan_spirv_exists(&name).then_some(name))
}

fn vulkan_q8_1_rhs_matvec_shader_name(
    device: &VulkanDevice,
    qdtype: GgmlDType,
    n: usize,
    k: usize,
    indexed: bool,
) -> Result<Option<(String, VulkanDmmvWorkgroup)>> {
    if !vulkan_supports_q8_1_rhs(qdtype) || !vulkan_should_use_mmvq(device, qdtype, n, k) {
        return Ok(None);
    }
    let workgroup = vulkan_dmmv_workgroup(device, qdtype, n, k, true);
    let base_name = if indexed {
        format!("mul_mat_vec_id_{}_q8_1_f32", vulkan_quantized_stem(qdtype)?)
    } else {
        format!("mul_mat_vec_{}_q8_1_f32", vulkan_quantized_stem(qdtype)?)
    };
    let name = vulkan_dmmv_shader_name(device, qdtype, base_name, workgroup);
    Ok(vulkan_spirv_exists(&name).then_some((name, workgroup)))
}

fn vulkan_q8_1_x4_bytes(elem_count: usize) -> Result<usize> {
    elem_count
        .div_ceil(128)
        .checked_mul(GgmlDType::Q8_1.type_size() * 4)
        .ok_or_else(|| Error::msg("vulkan q8_1 temp size overflow"))
}

fn quantize_f32_storage_to_q8_1_x4(
    device: &VulkanDevice,
    src: &VulkanStorage,
    elem_count: usize,
) -> Result<Arc<VulkanBuffer>> {
    if src.dtype != DType::F32 {
        return Err(Error::UnsupportedDTypeForOp(src.dtype, "vulkan quantize_q8_1").bt());
    }
    if !elem_count.is_multiple_of(4) {
        crate::bail!("vulkan quantize_q8_1 expects element count divisible by 4, got {elem_count}")
    }
    let num_blocks: u32 = elem_count.div_ceil(128).try_into()?;
    let out_size = vulkan_q8_1_x4_bytes(elem_count)?;
    let out = device.create_buffer(out_size, "candle-vulkan-quantize-q8_1-x4")?;
    let spirv = candle_vulkan_kernels::spirv("quantize_q8_1_x4")
        .ok_or_else(|| Error::Msg("vulkan shader quantize_q8_1_x4 not generated".into()).bt())?;
    let params = VulkanQuantizeQ8_1Params {
        ne: elem_count.try_into()?,
        num_blocks,
    };
    let workgroups_x = num_blocks.min(device.inner.max_workgroup_count_x.max(1));
    device.run_compute_specialized(
        spirv,
        &[
            VulkanBinding::Storage(&src.buffer),
            VulkanBinding::Storage(&out),
        ],
        Some(any_as_bytes(&params)),
        (workgroups_x, 1, 1),
        Some(&[(0, 32)]),
    )?;
    Ok(out)
}

fn repack_q8_1_storage_to_q8_0(
    device: &VulkanDevice,
    src: &VulkanStorage,
    elem_count: usize,
) -> Result<VulkanStorage> {
    if src.dtype != DType::U8 {
        return Err(Error::UnsupportedDTypeForOp(src.dtype, "vulkan repack_q8_1_to_q8_0").bt());
    }
    if !elem_count.is_multiple_of(GgmlDType::Q8_1.block_size()) {
        crate::bail!(
            "vulkan repack_q8_1_to_q8_0 expects element count divisible by {}, got {elem_count}",
            GgmlDType::Q8_1.block_size()
        )
    }
    let num_blocks = elem_count / GgmlDType::Q8_1.block_size();
    let out_count = num_blocks * GgmlDType::Q8_0.type_size();
    let out = unsafe { device.alloc_uninit(&Shape::from(out_count), DType::U8)? };
    let spirv = candle_vulkan_kernels::spirv("repack_q8_1_to_q8_0")
        .ok_or_else(|| Error::Msg("vulkan shader repack_q8_1_to_q8_0 not generated".into()).bt())?;
    let params = VulkanRepackQ8_1ToQ8_0Params {
        num_blocks: num_blocks.try_into()?,
    };
    let workgroups_x = (num_blocks as u32)
        .div_ceil(128)
        .min(device.inner.max_workgroup_count_x.max(1));
    device.run_compute(
        spirv,
        &[
            VulkanBinding::Storage(&src.buffer),
            VulkanBinding::Storage(&out.buffer),
        ],
        Some(any_as_bytes(&params)),
        workgroups_x,
    )?;
    Ok(out)
}

fn byte_len(dtype: DType, count: usize, op: &'static str) -> Result<usize> {
    let size = dtype.size_in_bytes();
    if size == 0 {
        Err(Error::UnsupportedDTypeForOp(dtype, op).bt())
    } else {
        Ok(size * count)
    }
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
            Err(Error::UnsupportedDTypeForOp(DType::F6E2M3, "vulkan upload").bt())
        }
        CpuStorage::F6E3M2(_) => {
            Err(Error::UnsupportedDTypeForOp(DType::F6E3M2, "vulkan upload").bt())
        }
        CpuStorage::F4(_) => Err(Error::UnsupportedDTypeForOp(DType::F4, "vulkan upload").bt()),
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
            Err(Error::UnsupportedDTypeForOp(dtype, "vulkan download").bt())
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

fn hash_spirv_words(words: &[u32]) -> u64 {
    // Stable process-local digest used only for in-memory pipeline cache keys.
    let mut hash = 0xcbf29ce484222325u64;
    for &word in words {
        for byte in word.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

impl VulkanDevice {
    const SUBMISSION_DESCRIPTOR_CAPACITY: u32 = 64;
    const MAX_REUSABLE_SUBMISSIONS_PER_QUEUE: usize = 64;
    const MAX_BATCH_DISPATCHES: u32 = 32;
    const MAX_BATCH_COPIES: u32 = 64;
    const DESCRIPTOR_SET_ALLOC_CHUNK: u32 = 8;
    const MAX_BATCH_DESCRIPTOR_SETS: u32 = Self::MAX_BATCH_DISPATCHES;
    const MAX_ALLOCATED_DESCRIPTOR_SETS_PER_BATCH: u32 =
        Self::MAX_BATCH_DISPATCHES * Self::DESCRIPTOR_SET_ALLOC_CHUNK;
    const MAX_BATCH_STORAGE_DESCRIPTORS: u32 =
        Self::MAX_BATCH_DESCRIPTOR_SETS * Self::SUBMISSION_DESCRIPTOR_CAPACITY;
    const MAX_ALLOCATED_STORAGE_DESCRIPTORS_PER_BATCH: u32 =
        Self::MAX_ALLOCATED_DESCRIPTOR_SETS_PER_BATCH * Self::SUBMISSION_DESCRIPTOR_CAPACITY;
    const MAX_BATCH_TRANSFER_BYTES: usize = 64 * 1024 * 1024;
    const MAX_BATCH_COMPUTE_BYTES: usize = 512 * 1024 * 1024;

    fn copy_queue_and_family(
        &self,
        prefer_transfer: bool,
    ) -> (vk::Queue, u32, SubmissionQueueKind) {
        if prefer_transfer {
            if let (Some(queue), Some(family)) = (
                self.inner.transfer_queue,
                self.inner.transfer_queue_family_index,
            ) {
                return (queue, family, SubmissionQueueKind::Transfer);
            }
        }
        (
            self.inner.queue,
            self.inner.queue_family_index,
            SubmissionQueueKind::Compute,
        )
    }

    fn create_submission_resources(
        &self,
        queue_kind: SubmissionQueueKind,
        queue_family_index: u32,
    ) -> Result<VulkanSubmissionResources> {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );
        let command_pool = unsafe {
            self.inner
                .device
                .create_command_pool(&command_pool_info, None)
        }
        .map_err(Error::wrap)?;
        let command_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { self.inner.device.allocate_command_buffers(&command_alloc) }
            .map_err(|err| {
                unsafe {
                    self.inner.device.destroy_command_pool(command_pool, None);
                }
                Error::wrap(err)
            })?[0];

        let (max_sets, descriptor_count) = match queue_kind {
            SubmissionQueueKind::Compute => (
                Self::MAX_ALLOCATED_DESCRIPTOR_SETS_PER_BATCH.max(1),
                Self::MAX_ALLOCATED_STORAGE_DESCRIPTORS_PER_BATCH.max(1),
            ),
            SubmissionQueueKind::Transfer => (1, 1),
        };
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(descriptor_count)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            self.inner
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
        }
        .map_err(|err| {
            unsafe {
                self.inner.device.destroy_command_pool(command_pool, None);
            }
            Error::wrap(err)
        })?;

        let fence = unsafe {
            self.inner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|err| {
            unsafe {
                self.inner
                    .device
                    .destroy_descriptor_pool(descriptor_pool, None);
                self.inner.device.destroy_command_pool(command_pool, None);
            }
            Error::wrap(err)
        })?;

        Ok(VulkanSubmissionResources {
            fence,
            command_pool,
            command_buffer,
            descriptor_pool,
        })
    }

    fn acquire_submission_resources(
        &self,
        queue_kind: SubmissionQueueKind,
        queue_family_index: u32,
    ) -> Result<VulkanSubmissionResources> {
        let pool = match queue_kind {
            SubmissionQueueKind::Compute => &self.inner.reusable_compute_submissions,
            SubmissionQueueKind::Transfer => &self.inner.reusable_transfer_submissions,
        };
        let resources =
            if let Some(resources) = pool.lock().map_err(|e| Error::wrap(e.to_string()))?.pop() {
                resources
            } else {
                self.create_submission_resources(queue_kind, queue_family_index)?
            };
        unsafe {
            self.inner
                .device
                .reset_fences(std::slice::from_ref(&resources.fence))
                .map_err(Error::wrap)?;
            self.inner
                .device
                .reset_command_pool(resources.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(Error::wrap)?;
            self.inner
                .device
                .reset_descriptor_pool(
                    resources.descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .map_err(Error::wrap)?;
        }
        Ok(resources)
    }

    fn recycle_submission_resources(
        &self,
        queue_kind: SubmissionQueueKind,
        resources: VulkanSubmissionResources,
    ) -> Result<()> {
        let pool = match queue_kind {
            SubmissionQueueKind::Compute => &self.inner.reusable_compute_submissions,
            SubmissionQueueKind::Transfer => &self.inner.reusable_transfer_submissions,
        };
        let mut pool = pool.lock().map_err(|e| Error::wrap(e.to_string()))?;
        if pool.len() >= Self::MAX_REUSABLE_SUBMISSIONS_PER_QUEUE {
            drop(pool);
            unsafe {
                self.inner.device.destroy_fence(resources.fence, None);
                self.inner
                    .device
                    .destroy_command_pool(resources.command_pool, None);
                self.inner
                    .device
                    .destroy_descriptor_pool(resources.descriptor_pool, None);
            }
            return Ok(());
        }
        pool.push(resources);
        Ok(())
    }

    fn active_batch_slot(
        &self,
        queue_kind: SubmissionQueueKind,
    ) -> &Mutex<Option<VulkanActiveBatch>> {
        match queue_kind {
            SubmissionQueueKind::Compute => &self.inner.active_compute_batch,
            SubmissionQueueKind::Transfer => &self.inner.active_transfer_batch,
        }
    }

    fn begin_active_batch(
        &self,
        queue_kind: SubmissionQueueKind,
        queue_family_index: u32,
    ) -> Result<VulkanActiveBatch> {
        let resources = self.acquire_submission_resources(queue_kind, queue_family_index)?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.inner
                .device
                .begin_command_buffer(resources.command_buffer, &begin_info)
                .map_err(Error::wrap)?;
        }
        Ok(VulkanActiveBatch {
            resources,
            queue_kind,
            dispatch_count: 0,
            copy_count: 0,
            descriptor_set_count: 0,
            allocated_descriptor_set_count: 0,
            storage_descriptor_count: 0,
            transfer_bytes: 0,
            compute_bytes: 0,
            retained_buffers: Vec::new(),
            cached_descriptor_sets: HashMap::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_active_batch_capacity(
        &self,
        queue_kind: SubmissionQueueKind,
        queue_family_index: u32,
        dispatches_to_add: u32,
        copies_to_add: u32,
        descriptor_sets_to_add: u32,
        storage_descriptors_to_add: u32,
        transfer_bytes_to_add: usize,
        compute_bytes_to_add: usize,
    ) -> Result<()> {
        loop {
            let mut should_create = false;
            let mut should_flush = false;
            {
                let slot = self
                    .active_batch_slot(queue_kind)
                    .lock()
                    .map_err(|e| Error::wrap(e.to_string()))?;
                if let Some(batch) = slot.as_ref() {
                    should_flush = batch.dispatch_count + dispatches_to_add
                        > Self::MAX_BATCH_DISPATCHES
                        || batch.copy_count + copies_to_add > Self::MAX_BATCH_COPIES
                        || batch.descriptor_set_count + descriptor_sets_to_add
                            > Self::MAX_BATCH_DESCRIPTOR_SETS
                        || batch.storage_descriptor_count + storage_descriptors_to_add
                            > Self::MAX_BATCH_STORAGE_DESCRIPTORS
                        || batch.transfer_bytes + transfer_bytes_to_add
                            > Self::MAX_BATCH_TRANSFER_BYTES
                        || batch.compute_bytes + compute_bytes_to_add
                            > Self::MAX_BATCH_COMPUTE_BYTES;
                } else {
                    should_create = true;
                }
            }
            if should_flush {
                self.flush_active_batch(queue_kind, "batch_limit")?;
                continue;
            }
            if should_create {
                let batch = self.begin_active_batch(queue_kind, queue_family_index)?;
                let mut slot = self
                    .active_batch_slot(queue_kind)
                    .lock()
                    .map_err(|e| Error::wrap(e.to_string()))?;
                if slot.is_none() {
                    *slot = Some(batch);
                }
            }
            return Ok(());
        }
    }

    fn flush_active_batch(
        &self,
        queue_kind: SubmissionQueueKind,
        reason: &'static str,
    ) -> Result<bool> {
        let batch = {
            let mut slot = self
                .active_batch_slot(queue_kind)
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            slot.take()
        };
        let Some(batch) = batch else {
            return Ok(false);
        };
        if !batch.has_commands() {
            self.recycle_submission_resources(queue_kind, batch.resources)?;
            return Ok(false);
        }
        unsafe {
            self.inner
                .device
                .end_command_buffer(batch.resources.command_buffer)
                .map_err(Error::wrap)?;
        }
        let queue = match queue_kind {
            SubmissionQueueKind::Compute => self.inner.queue,
            SubmissionQueueKind::Transfer => self
                .inner
                .transfer_queue
                .ok_or_else(|| Error::msg("missing transfer queue for active batch flush"))?,
        };
        let queue_name = match queue_kind {
            SubmissionQueueKind::Compute => "compute",
            SubmissionQueueKind::Transfer => "transfer",
        };
        let _flush_span = trace_span!(
            "vulkan.batch.flush",
            queue = queue_name,
            reason = reason,
            dispatches_in_batch = batch.dispatch_count,
            copies_in_batch = batch.copy_count,
            transfer_bytes_in_batch = batch.transfer_bytes,
            compute_bytes_in_batch = batch.compute_bytes
        )
        .entered();
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&batch.resources.command_buffer));
        let _submit_span = if batch.dispatch_count > 0 {
            Some(
                trace_span!(
                    "vulkan.dispatch.submit",
                    dispatches_in_batch = batch.dispatch_count,
                    copies_in_batch = batch.copy_count
                )
                .entered(),
            )
        } else {
            Some(
                trace_span!(
                    "vulkan.copy.submit",
                    copies_in_batch = batch.copy_count,
                    queue = queue_name
                )
                .entered(),
            )
        };
        unsafe {
            self.inner
                .device
                .queue_submit(
                    queue,
                    std::slice::from_ref(&submit_info),
                    batch.resources.fence,
                )
                .map_err(Error::wrap)?;
        }
        self.inner
            .pending_submissions
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?
            .push(VulkanPendingSubmission {
                resources: batch.resources,
                queue_kind: batch.queue_kind,
                dispatch_count: batch.dispatch_count,
                copy_count: batch.copy_count,
                transfer_bytes: batch.transfer_bytes,
                compute_bytes: batch.compute_bytes,
                retained_buffers: batch.retained_buffers,
            });
        Ok(true)
    }

    fn flush_all_active_batches(&self, reason: &'static str) -> Result<()> {
        self.flush_active_batch(SubmissionQueueKind::Compute, reason)?;
        if self.inner.transfer_queue.is_some() {
            self.flush_active_batch(SubmissionQueueKind::Transfer, reason)?;
        }
        Ok(())
    }

    fn cmd_batch_memory_barrier(&self, command_buffer: vk::CommandBuffer) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            self.inner.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[],
            );
        }
    }

    fn wait_for_transfer_dependencies(&self) -> Result<()> {
        if self.inner.transfer_queue.is_none() {
            return Ok(());
        }
        self.flush_active_batch(SubmissionQueueKind::Transfer, "transfer_dependency")?;
        let has_pending_transfer = self
            .inner
            .pending_submissions
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?
            .iter()
            .any(|submission| submission.queue_kind == SubmissionQueueKind::Transfer);
        if has_pending_transfer {
            let _wait_span = trace_span!("vulkan.transfer.wait_before_compute").entered();
            self.cleanup_pending_submissions_for_queue(SubmissionQueueKind::Transfer, true)?;
        }
        Ok(())
    }

    pub fn transfer_to_device(&self, storage: &VulkanStorage) -> Result<VulkanStorage> {
        let cpu = storage.to_cpu_storage()?;
        self.storage_from_cpu_storage(&cpu)
    }

    pub fn vendor_id(&self) -> u32 {
        self.inner.vendor_id
    }

    fn create_buffer_with_location(
        &self,
        size: usize,
        name: &'static str,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Arc<VulkanBuffer>> {
        self.cleanup_pending_submissions(false)?;
        let mut info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage);
        let queue_family_indices: Option<[u32; 2]> = self
            .inner
            .transfer_queue_family_index
            .filter(|family| *family != self.inner.queue_family_index)
            .map(|transfer_family| [self.inner.queue_family_index, transfer_family]);
        if let Some(queue_family_indices) = queue_family_indices.as_ref() {
            info = info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(queue_family_indices);
        } else {
            info = info.sharing_mode(vk::SharingMode::EXCLUSIVE);
        }
        let buffer =
            unsafe { self.inner.device.create_buffer(&info, None) }.map_err(Error::wrap)?;
        let requirements = unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };
        let allocate_once = || -> Result<Allocation> {
            let mut allocator = self
                .inner
                .allocator
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            let allocator = allocator
                .as_mut()
                .ok_or_else(|| Error::msg("vulkan allocator already dropped"))?;
            allocator
                .allocate(&AllocationCreateDesc {
                    name,
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(Error::wrap)
        };

        let allocation = match allocate_once() {
            Ok(allocation) => allocation,
            Err(first_err) => {
                // Long-running async submit bursts can temporarily retain GPU allocations
                // until fences are observed. Synchronize once and retry allocation.
                self.synchronize_pending()?;
                match allocate_once() {
                    Ok(allocation) => allocation,
                    Err(second_err) => {
                        unsafe {
                            self.inner.device.destroy_buffer(buffer, None);
                        }
                        return Err(Error::Msg(format!(
                            "vulkan allocation retry failed after synchronize: first={first_err}, second={second_err}"
                        ))
                        .bt());
                    }
                }
            }
        };
        unsafe {
            self.inner
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(Error::wrap)?;
        }
        Ok(Arc::new(VulkanBuffer {
            device: self.clone(),
            buffer,
            allocation: Mutex::new(Some(allocation)),
            size,
        }))
    }

    fn create_buffer(&self, size: usize, name: &'static str) -> Result<Arc<VulkanBuffer>> {
        self.create_buffer_with_location(
            size,
            name,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )
    }

    fn create_upload_staging_buffer(&self, size: usize) -> Result<Arc<VulkanBuffer>> {
        self.create_buffer_with_location(
            size,
            "candle-vulkan-upload-staging",
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )
    }

    fn create_readback_staging_buffer(&self, size: usize) -> Result<Arc<VulkanBuffer>> {
        self.create_buffer_with_location(
            size,
            "candle-vulkan-readback-staging",
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )
    }

    fn destroy_deferred_buffers(&self) -> Result<()> {
        let deferred = {
            let mut deferred = self
                .inner
                .deferred_buffer_frees
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            std::mem::take(&mut *deferred)
        };
        if deferred.is_empty() {
            return Ok(());
        }
        let mut allocator = self
            .inner
            .allocator
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        let allocator = allocator
            .as_mut()
            .ok_or_else(|| Error::msg("vulkan allocator already dropped"))?;
        for deferred in deferred {
            unsafe {
                self.inner.device.destroy_buffer(deferred.buffer, None);
            }
            allocator.free(deferred.allocation).map_err(Error::wrap)?;
        }
        Ok(())
    }

    fn cleanup_pending_submissions_for_queue(
        &self,
        queue_kind: SubmissionQueueKind,
        wait: bool,
    ) -> Result<()> {
        self.cleanup_pending_submissions_impl(wait, Some(queue_kind))
    }

    fn cleanup_pending_submissions(&self, wait: bool) -> Result<()> {
        self.cleanup_pending_submissions_impl(wait, None)
    }

    fn cleanup_pending_submissions_impl(
        &self,
        wait: bool,
        only_queue_kind: Option<SubmissionQueueKind>,
    ) -> Result<()> {
        let ready = {
            let mut pending = self
                .inner
                .pending_submissions
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            let mut ready = Vec::new();
            let mut idx = 0;
            while idx < pending.len() {
                if let Some(queue_kind) = only_queue_kind {
                    if pending[idx].queue_kind != queue_kind {
                        idx += 1;
                        continue;
                    }
                }
                let is_ready = if wait {
                    true
                } else {
                    unsafe {
                        self.inner
                            .device
                            .get_fence_status(pending[idx].resources.fence)
                    }
                    .map_err(Error::wrap)?
                };
                if is_ready {
                    ready.push(pending.swap_remove(idx));
                } else {
                    idx += 1;
                }
            }
            ready
        };
        for submission in ready {
            if wait {
                let _wait_span = trace_span!(
                    "vulkan.submit.wait_fence",
                    dispatches_in_batch = submission.dispatch_count,
                    copies_in_batch = submission.copy_count,
                    transfer_bytes_in_batch = submission.transfer_bytes,
                    compute_bytes_in_batch = submission.compute_bytes
                )
                .entered();
                unsafe {
                    self.inner
                        .device
                        .wait_for_fences(
                            std::slice::from_ref(&submission.resources.fence),
                            true,
                            u64::MAX,
                        )
                        .map_err(Error::wrap)?;
                }
            }
            drop(submission.retained_buffers);
            self.recycle_submission_resources(submission.queue_kind, submission.resources)?;
        }
        let pending_empty = self
            .inner
            .pending_submissions
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?
            .is_empty();
        let active_batches_empty = self
            .inner
            .active_compute_batch
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?
            .is_none()
            && self
                .inner
                .active_transfer_batch
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?
                .is_none();
        if pending_empty && active_batches_empty {
            self.destroy_deferred_buffers()?;
        }
        Ok(())
    }

    fn synchronize_pending(&self) -> Result<()> {
        let _wait_span = trace_span!(
            "vulkan.submit.wait_fences",
            wait_transfer = self.inner.transfer_queue.is_some()
        )
        .entered();
        self.flush_all_active_batches("synchronize")?;
        self.cleanup_pending_submissions(true)
    }

    fn map_write_host_buffer(&self, buffer: &VulkanBuffer, bytes: &[u8]) -> Result<()> {
        if bytes.len() > buffer.size {
            crate::bail!("vulkan write larger than buffer")
        }
        let allocation = buffer
            .allocation
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        let allocation = allocation
            .as_ref()
            .ok_or_else(|| Error::msg("freed vulkan allocation"))?;
        unsafe {
            let mapped_ptr = if let Some(ptr) = allocation.mapped_ptr() {
                ptr.cast::<u8>().as_ptr()
            } else {
                self.inner
                    .device
                    .map_memory(
                        allocation.memory(),
                        allocation.offset(),
                        buffer.size as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(Error::wrap)?
                    .cast::<u8>()
            };
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped_ptr, bytes.len());
            if bytes.len() < buffer.size {
                std::ptr::write_bytes(mapped_ptr.add(bytes.len()), 0, buffer.size - bytes.len());
            }
            let range = vk::MappedMemoryRange::default()
                .memory(allocation.memory())
                .offset(allocation.offset())
                .size(buffer.size as u64);
            self.inner
                .device
                .flush_mapped_memory_ranges(&[range])
                .map_err(Error::wrap)?;
            if allocation.mapped_ptr().is_none() {
                self.inner.device.unmap_memory(allocation.memory());
            }
        }
        Ok(())
    }

    fn map_read_host_buffer(&self, buffer: &VulkanBuffer) -> Result<Vec<u8>> {
        let allocation = buffer
            .allocation
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        let allocation = allocation
            .as_ref()
            .ok_or_else(|| Error::msg("freed vulkan allocation"))?;
        unsafe {
            let range = vk::MappedMemoryRange::default()
                .memory(allocation.memory())
                .offset(allocation.offset())
                .size(buffer.size as u64);
            self.inner
                .device
                .invalidate_mapped_memory_ranges(&[range])
                .map_err(Error::wrap)?;
            let mapped_ptr = if let Some(ptr) = allocation.mapped_ptr() {
                ptr.cast::<u8>().as_ptr()
            } else {
                self.inner
                    .device
                    .map_memory(
                        allocation.memory(),
                        allocation.offset(),
                        buffer.size as u64,
                        vk::MemoryMapFlags::empty(),
                    )
                    .map_err(Error::wrap)?
                    .cast::<u8>()
            };
            let bytes = std::slice::from_raw_parts(mapped_ptr, buffer.size).to_vec();
            if allocation.mapped_ptr().is_none() {
                self.inner.device.unmap_memory(allocation.memory());
            }
            Ok(bytes)
        }
    }

    fn submit_copy_regions_and_track(
        &self,
        src: &Arc<VulkanBuffer>,
        dst: &Arc<VulkanBuffer>,
        regions: &[vk::BufferCopy],
        prefer_transfer: bool,
    ) -> Result<()> {
        if regions.is_empty() {
            return Ok(());
        }
        self.cleanup_pending_submissions(false)?;
        let (queue, queue_family_index, queue_kind) = self.copy_queue_and_family(prefer_transfer);
        if queue_kind == SubmissionQueueKind::Compute {
            self.wait_for_transfer_dependencies()?;
        } else {
            self.flush_active_batch(SubmissionQueueKind::Compute, "transfer_copy_dependency")?;
            self.cleanup_pending_submissions_for_queue(SubmissionQueueKind::Compute, true)?;
        }
        let total_bytes = regions.iter().try_fold(0usize, |acc, region| {
            acc.checked_add(region.size as usize)
                .ok_or_else(|| Error::msg("vulkan copy batch byte count overflow"))
        })?;
        let (transfer_bytes_to_add, compute_bytes_to_add) = match queue_kind {
            SubmissionQueueKind::Transfer => (total_bytes, 0),
            SubmissionQueueKind::Compute => (0, total_bytes),
        };
        self.ensure_active_batch_capacity(
            queue_kind,
            queue_family_index,
            0,
            1,
            0,
            0,
            transfer_bytes_to_add,
            compute_bytes_to_add,
        )?;
        {
            let mut slot = self
                .active_batch_slot(queue_kind)
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            let batch = slot
                .as_mut()
                .ok_or_else(|| Error::msg("vulkan active batch missing after ensure"))?;
            unsafe {
                self.inner.device.cmd_copy_buffer(
                    batch.resources.command_buffer,
                    src.buffer,
                    dst.buffer,
                    regions,
                );
            }
            if queue_kind == SubmissionQueueKind::Compute {
                self.cmd_batch_memory_barrier(batch.resources.command_buffer);
            }
            batch.copy_count += 1;
            batch.transfer_bytes += transfer_bytes_to_add;
            batch.compute_bytes += compute_bytes_to_add;
            batch.retained_buffers.push(src.clone());
            batch.retained_buffers.push(dst.clone());
        }
        let _ = queue;
        Ok(())
    }

    fn submit_copy_region_and_track(
        &self,
        src: &Arc<VulkanBuffer>,
        dst: &Arc<VulkanBuffer>,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        prefer_transfer: bool,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        let region = vk::BufferCopy::default()
            .src_offset(src_offset as u64)
            .dst_offset(dst_offset as u64)
            .size(size as u64);
        self.submit_copy_regions_and_track(src, dst, std::slice::from_ref(&region), prefer_transfer)
    }

    fn submit_copy_and_track(
        &self,
        src: &Arc<VulkanBuffer>,
        dst: &Arc<VulkanBuffer>,
        size: usize,
        prefer_transfer: bool,
    ) -> Result<()> {
        self.submit_copy_region_and_track(src, dst, 0, 0, size, prefer_transfer)
    }

    fn write_buffer(&self, buffer: &Arc<VulkanBuffer>, bytes: &[u8]) -> Result<()> {
        let _upload_span = trace_span!(
            "vulkan.upload",
            requested_bytes = bytes.len(),
            buffer_bytes = buffer.size
        )
        .entered();
        let staging = self.create_upload_staging_buffer(buffer.size)?;
        self.map_write_host_buffer(staging.as_ref(), bytes)?;
        self.submit_copy_and_track(&staging, buffer, buffer.size, true)?;
        Ok(())
    }

    fn read_buffer(&self, buffer: &Arc<VulkanBuffer>) -> Result<Vec<u8>> {
        let _readback_span = trace_span!("vulkan.readback", buffer_bytes = buffer.size).entered();
        self.flush_all_active_batches("read_buffer_before_wait")?;
        self.cleanup_pending_submissions(true)?;
        let staging = self.create_readback_staging_buffer(buffer.size)?;
        self.submit_copy_and_track(buffer, &staging, buffer.size, true)?;
        self.flush_active_batch(
            self.copy_queue_and_family(true).2,
            "read_buffer_after_copy_record",
        )?;
        self.cleanup_pending_submissions(true)?;
        self.map_read_host_buffer(staging.as_ref())
    }

    fn run_compute(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: u32,
    ) -> Result<()> {
        self.run_compute_3d(spirv, bindings, push_constants, (workgroups, 1, 1))
    }

    fn run_compute_3d(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        self.run_compute_specialized_with_options(
            spirv,
            bindings,
            push_constants,
            workgroups,
            None,
            false,
            None,
        )
    }

    fn run_compute_specialized(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: (u32, u32, u32),
        specialization_u32: Option<&[(u32, u32)]>,
    ) -> Result<()> {
        self.run_compute_specialized_with_options(
            spirv,
            bindings,
            push_constants,
            workgroups,
            specialization_u32,
            false,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_compute_specialized_with_options(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: (u32, u32, u32),
        specialization_u32: Option<&[(u32, u32)]>,
        require_full_subgroups: bool,
        required_subgroup_size: Option<u32>,
    ) -> Result<()> {
        unsafe {
            self.run_compute_with_shader(
                spirv,
                bindings,
                push_constants,
                workgroups,
                specialization_u32,
                require_full_subgroups,
                required_subgroup_size,
            )?
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn run_compute_with_shader(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: (u32, u32, u32),
        specialization_u32: Option<&[(u32, u32)]>,
        require_full_subgroups: bool,
        required_subgroup_size: Option<u32>,
    ) -> Result<()> {
        let push_constants_len = push_constants.map(|bytes| bytes.len() as u32).unwrap_or(0);
        if push_constants_len > self.inner.max_push_constants_size {
            crate::bail!(
                "vulkan push constants too large: {} > {}",
                push_constants_len,
                self.inner.max_push_constants_size
            );
        }
        let _dispatch_span = trace_span!(
            "vulkan.dispatch",
            bindings = bindings.len(),
            push_constants = push_constants_len,
            wg_x = workgroups.0,
            wg_y = workgroups.1,
            wg_z = workgroups.2
        )
        .entered();
        self.wait_for_transfer_dependencies()?;
        self.cleanup_pending_submissions(false)?;
        let binding_signature = bindings
            .iter()
            .map(|binding| binding.descriptor_type().as_raw() as u32)
            .collect::<SmallVec<[u32; 8]>>();
        let cache_key = VulkanPipelineCacheKey {
            shader_hash: hash_spirv_words(spirv),
            shader_len_words: spirv.len(),
            binding_signature,
            push_constant_len: push_constants_len,
            specialization_u32: specialization_u32
                .map(|specs| specs.iter().copied().collect::<SmallVec<[(u32, u32); 8]>>())
                .unwrap_or_default(),
            require_full_subgroups,
            required_subgroup_size,
        };
        let (cached, pipeline_cache_hit) = {
            let _lookup_span = trace_span!(
                "vulkan.pipeline.lookup",
                shader_words = spirv.len(),
                bindings = bindings.len(),
                push_constants = push_constants_len
            )
            .entered();
            let mut cache = self
                .inner
                .pipeline_cache
                .lock()
                .map_err(|e| Error::wrap(e.to_string()))?;
            if let Some(cached) = cache.get(&cache_key) {
                (cached.clone(), true)
            } else {
                let _create_span = trace_span!(
                    "vulkan.pipeline.create",
                    shader_words = spirv.len(),
                    bindings = bindings.len(),
                    push_constants = push_constants_len
                )
                .entered();
                let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv);
                let shader = self
                    .inner
                    .device
                    .create_shader_module(&shader_info, None)
                    .map_err(Error::wrap)?;
                let layout_bindings = bindings
                    .iter()
                    .enumerate()
                    .map(|(binding, entry)| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(binding as u32)
                            .descriptor_type(entry.descriptor_type())
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    })
                    .collect::<SmallVec<[vk::DescriptorSetLayoutBinding; 8]>>();
                let set_layout_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
                let descriptor_set_layout = self
                    .inner
                    .device
                    .create_descriptor_set_layout(&set_layout_info, None)
                    .map_err(Error::wrap)?;
                let push_constant_ranges = push_constants
                    .map(|bytes| {
                        SmallVec::<[vk::PushConstantRange; 1]>::from_buf([
                            vk::PushConstantRange::default()
                                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                                .offset(0)
                                .size(bytes.len() as u32),
                        ])
                    })
                    .unwrap_or_default();
                let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                    .push_constant_ranges(&push_constant_ranges);
                let pipeline_layout = self
                    .inner
                    .device
                    .create_pipeline_layout(&pipeline_layout_info, None)
                    .map_err(Error::wrap)?;
                let entry = CString::new("main").map_err(Error::wrap)?;
                let mut spec_entries = SmallVec::<[vk::SpecializationMapEntry; 8]>::new();
                let mut spec_data = SmallVec::<[u8; 32]>::new();
                let spec_info;
                let mut stage_flags = vk::PipelineShaderStageCreateFlags::empty();
                if require_full_subgroups && self.inner.compute_full_subgroups {
                    stage_flags |= vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS_EXT;
                }
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .flags(stage_flags)
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader)
                    .name(&entry);
                if let Some(specialization_u32) = specialization_u32 {
                    for (idx, &(constant_id, value)) in specialization_u32.iter().enumerate() {
                        spec_entries.push(
                            vk::SpecializationMapEntry::default()
                                .constant_id(constant_id)
                                .offset((idx * std::mem::size_of::<u32>()) as u32)
                                .size(std::mem::size_of::<u32>()),
                        );
                        spec_data.extend_from_slice(&value.to_ne_bytes());
                    }
                    spec_info = vk::SpecializationInfo::default()
                        .map_entries(&spec_entries)
                        .data(&spec_data);
                    stage = stage.specialization_info(&spec_info);
                }
                let mut required_subgroup_info;
                if let Some(required_subgroup_size) = required_subgroup_size {
                    if self.inner.subgroup_size_control
                        && self.inner.subgroup_min_size <= required_subgroup_size
                        && required_subgroup_size <= self.inner.subgroup_max_size
                    {
                        required_subgroup_info =
                            vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT::default()
                                .required_subgroup_size(required_subgroup_size);
                        stage = stage.push_next(&mut required_subgroup_info);
                    }
                }
                let pipeline_info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout);
                let pipelines = self
                    .inner
                    .device
                    .create_compute_pipelines(
                        self.inner.driver_pipeline_cache,
                        &[pipeline_info],
                        None,
                    )
                    .map_err(|(_, e)| Error::wrap(e))?;
                let cached = Arc::new(VulkanCachedPipeline {
                    shader,
                    pipeline: pipelines[0],
                    pipeline_layout,
                    descriptor_set_layout,
                });
                cache.insert(cache_key, cached.clone());
                (cached, false)
            }
        };

        let mut storage_count = 0;
        for binding in bindings {
            match binding {
                VulkanBinding::Storage(_) => storage_count += 1,
            }
        }
        if storage_count > Self::SUBMISSION_DESCRIPTOR_CAPACITY {
            crate::bail!(
                "vulkan descriptor requirement {} exceeds submission capacity {}",
                storage_count,
                Self::SUBMISSION_DESCRIPTOR_CAPACITY
            );
        }
        let compute_bytes = bindings.iter().try_fold(0usize, |acc, binding| {
            acc.checked_add(binding.buffer().size)
                .ok_or_else(|| Error::msg("vulkan compute batch byte count overflow"))
        })?;
        self.ensure_active_batch_capacity(
            SubmissionQueueKind::Compute,
            self.inner.queue_family_index,
            1,
            0,
            1,
            storage_count,
            0,
            compute_bytes,
        )?;
        let mut slot = self
            .active_batch_slot(SubmissionQueueKind::Compute)
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        let batch = slot
            .as_mut()
            .ok_or_else(|| Error::msg("vulkan compute batch missing after ensure"))?;
        let descriptor_set = if let Some(cached_sets) = batch
            .cached_descriptor_sets
            .get_mut(&cached.descriptor_set_layout)
        {
            if let Some(descriptor_set) = cached_sets.pop() {
                descriptor_set
            } else {
                let remaining_capacity = Self::MAX_ALLOCATED_DESCRIPTOR_SETS_PER_BATCH
                    .saturating_sub(batch.allocated_descriptor_set_count);
                if remaining_capacity == 0 {
                    crate::bail!("vulkan descriptor set cache exhausted inside active batch")
                }
                let alloc_count = remaining_capacity.min(Self::DESCRIPTOR_SET_ALLOC_CHUNK) as usize;
                let set_layouts = SmallVec::<[vk::DescriptorSetLayout; 8]>::from_elem(
                    cached.descriptor_set_layout,
                    alloc_count,
                );
                let set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(batch.resources.descriptor_pool)
                    .set_layouts(&set_layouts);
                let mut descriptor_sets = self
                    .inner
                    .device
                    .allocate_descriptor_sets(&set_alloc_info)
                    .map_err(Error::wrap)?;
                batch.allocated_descriptor_set_count += alloc_count as u32;
                let descriptor_set = descriptor_sets
                    .pop()
                    .ok_or_else(|| Error::msg("vulkan descriptor allocation returned no sets"))?;
                if !descriptor_sets.is_empty() {
                    batch
                        .cached_descriptor_sets
                        .entry(cached.descriptor_set_layout)
                        .or_default()
                        .extend(descriptor_sets);
                }
                descriptor_set
            }
        } else {
            let remaining_capacity = Self::MAX_ALLOCATED_DESCRIPTOR_SETS_PER_BATCH
                .saturating_sub(batch.allocated_descriptor_set_count);
            if remaining_capacity == 0 {
                crate::bail!("vulkan descriptor set cache exhausted inside active batch")
            }
            let alloc_count = remaining_capacity.min(Self::DESCRIPTOR_SET_ALLOC_CHUNK) as usize;
            let set_layouts = SmallVec::<[vk::DescriptorSetLayout; 8]>::from_elem(
                cached.descriptor_set_layout,
                alloc_count,
            );
            let set_alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(batch.resources.descriptor_pool)
                .set_layouts(&set_layouts);
            let mut descriptor_sets = self
                .inner
                .device
                .allocate_descriptor_sets(&set_alloc_info)
                .map_err(Error::wrap)?;
            batch.allocated_descriptor_set_count += alloc_count as u32;
            let descriptor_set = descriptor_sets
                .pop()
                .ok_or_else(|| Error::msg("vulkan descriptor allocation returned no sets"))?;
            if !descriptor_sets.is_empty() {
                batch
                    .cached_descriptor_sets
                    .entry(cached.descriptor_set_layout)
                    .or_default()
                    .extend(descriptor_sets);
            }
            descriptor_set
        };
        let buffer_infos = bindings
            .iter()
            .map(|binding| {
                let buffer = binding.buffer();
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.size as u64)
            })
            .collect::<SmallVec<[vk::DescriptorBufferInfo; 8]>>();
        let writes = bindings
            .iter()
            .enumerate()
            .map(|(binding, entry)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(entry.descriptor_type())
                    .buffer_info(std::slice::from_ref(&buffer_infos[binding]))
            })
            .collect::<SmallVec<[vk::WriteDescriptorSet<'_>; 8]>>();
        self.inner.device.update_descriptor_sets(&writes, &[]);
        let command_buffer = batch.resources.command_buffer;
        self.inner.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            cached.pipeline,
        );
        self.inner.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            cached.pipeline_layout,
            0,
            std::slice::from_ref(&descriptor_set),
            &[],
        );
        if let Some(bytes) = push_constants {
            self.inner.device.cmd_push_constants(
                command_buffer,
                cached.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
        self.inner
            .device
            .cmd_dispatch(command_buffer, workgroups.0, workgroups.1, workgroups.2);
        self.cmd_batch_memory_barrier(command_buffer);
        batch.dispatch_count += 1;
        batch.descriptor_set_count += 1;
        batch.storage_descriptor_count += storage_count;
        batch.compute_bytes += compute_bytes;
        batch
            .retained_buffers
            .extend(bindings.iter().map(VulkanBinding::retained_buffer));
        let _ = pipeline_cache_hit;
        Ok(())
    }
}

enum VulkanBinding<'a> {
    Storage(&'a Arc<VulkanBuffer>),
}

impl VulkanBinding<'_> {
    fn buffer(&self) -> &VulkanBuffer {
        match self {
            Self::Storage(buffer) => buffer.as_ref(),
        }
    }

    fn descriptor_type(&self) -> vk::DescriptorType {
        match self {
            Self::Storage(_) => vk::DescriptorType::STORAGE_BUFFER,
        }
    }

    fn retained_buffer(&self) -> Arc<VulkanBuffer> {
        match self {
            Self::Storage(buffer) => (*buffer).clone(),
        }
    }
}

fn dims4_ggml(layout: &Layout) -> Result<([u32; 4], [u32; 4])> {
    if layout.dims().len() > 4 {
        crate::bail!("vulkan backend supports up to rank-4 tensors for this op")
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

fn contiguous_strides_ggml(dims: [u32; 4]) -> [u32; 4] {
    [1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]]
}

fn ggml_linear_workgroups(count: usize) -> Result<(u32, u32, u32)> {
    let count: u32 = count.try_into()?;
    Ok(if count > 262_144 {
        (1, 512, count.div_ceil(262_144))
    } else if count > 512 {
        (1, count.div_ceil(512), 1)
    } else {
        (1, 1, 1)
    })
}

fn next_power_of_two_u32(value: usize, op: &'static str) -> Result<u32> {
    value
        .checked_next_power_of_two()
        .ok_or_else(|| Error::Msg(format!("vulkan backend op {op} dimension overflow")).bt())?
        .try_into()
        .map_err(Error::wrap)
}

fn floor_log2_u32(value: u32) -> u32 {
    u32::BITS - 1 - value.leading_zeros()
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

fn fastdiv_values(d: u32) -> (u32, u32) {
    let d = d.max(1);
    let mut l = 0u32;
    while l < 32 && (1u32 << l) < d {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - u64::from(d))) / u64::from(d) + 1) as u32;
    (mp, l)
}

#[derive(Clone, Copy)]
enum VulkanUnaryKind {
    Head,
    Generic,
}

fn unary_spirv(op: &str, dtype: DType) -> Result<(&'static [u32], VulkanUnaryKind)> {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        _ => return Err(Error::UnsupportedDTypeForOp(dtype, "vulkan unary").bt()),
    };
    let stem = match op {
        "abs" => "abs",
        "ceil" => "ceil",
        "cos" if dtype == DType::F32 => "cos",
        "exp" => "exp",
        "floor" => "floor",
        "gelu" => "gelu",
        "gelu_erf" => "gelu_erf",
        "gelu_quick" => "gelu_quick",
        "hardsigmoid" => "hardsigmoid",
        "hardswish" => "hardswish",
        "log" => "log",
        "neg" => "neg",
        "relu" => "relu",
        "round" => "round",
        "sign" => "sgn",
        "sigmoid" => "sigmoid",
        "silu" => "silu",
        "sin" if dtype == DType::F32 => "sin",
        "sqr" if dtype == DType::F32 => "sqr",
        "sqrt" if dtype == DType::F32 => "sqrt",
        "tanh" => "tanh",
        _ => return Err(unsupported("unary")),
    };
    let name = format!("{stem}_{suffix}");
    let kind = match op {
        "cos" | "log" | "sin" | "sqr" | "sqrt" => VulkanUnaryKind::Generic,
        _ => VulkanUnaryKind::Head,
    };
    let spirv = candle_vulkan_kernels::spirv(&name)
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated")).bt())?;
    Ok((spirv, kind))
}

fn binary_spirv(op: &str, dtype: DType) -> Result<&'static [u32]> {
    let suffix = match dtype {
        DType::F32 => "f32_f32_f32",
        DType::F16 => "f16_f16_f16",
        DType::U8 | DType::U32 | DType::I64 => {
            // One opcode-switched Candle-owned shader per integer dtype.
            let name = match dtype {
                DType::U8 => "binary_int_u8",
                DType::U32 => "binary_int_u32",
                _ => "binary_int_i64",
            };
            binary_int_opcode(op)?;
            return candle_vulkan_kernels::spirv(name)
                .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated")).bt());
        }
        _ => return Err(Error::UnsupportedDTypeForOp(dtype, "vulkan binary").bt()),
    };
    let stem = match op {
        "add" => "add",
        "div" => "div",
        "mul" => "mul",
        "sub" => "sub",
        _ => return Err(unsupported("binary")),
    };
    let name = format!("{stem}_{suffix}");
    candle_vulkan_kernels::spirv(&name)
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated")).bt())
}

fn binary_int_opcode(op: &str) -> Result<i32> {
    match op {
        "add" => Ok(0),
        "sub" => Ok(1),
        "mul" => Ok(2),
        "div" => Ok(3),
        "maximum" => Ok(4),
        "minimum" => Ok(5),
        _ => Err(unsupported("binary int")),
    }
}

fn copy_spirv(src: DType, dst: DType) -> Result<&'static [u32]> {
    let name = match (src, dst) {
        (DType::F32, DType::F32) => "cpy_f32_f32",
        (DType::F32, DType::I32) => "cpy_f32_i32",
        (DType::I32, DType::F32) => "cpy_i32_f32",
        (DType::U32, DType::F32) => "cpy_u32_f32",
        (DType::F32, DType::F16) => "cpy_f32_f16",
        (DType::F16, DType::F32) => "cpy_f16_f32",
        (DType::F16, DType::F16) => "cpy_f16_f16",
        // Candle-owned integer cast family; the copied ggml generator only
        // covers the float pairs above.
        (DType::U8, DType::U8) => "convert_u8_u8",
        (DType::U32, DType::U32) => "convert_u32_u32",
        (DType::I64, DType::I64) => "convert_i64_i64",
        (DType::F32, DType::U8) => "convert_f32_u8",
        (DType::F32, DType::U32) => "convert_f32_u32",
        (DType::F32, DType::I64) => "convert_f32_i64",
        (DType::F16, DType::U8) => "convert_f16_u8",
        (DType::F16, DType::U32) => "convert_f16_u32",
        (DType::F16, DType::I64) => "convert_f16_i64",
        (DType::U8, DType::F32) => "convert_u8_f32",
        (DType::U8, DType::F16) => "convert_u8_f16",
        (DType::U8, DType::U32) => "convert_u8_u32",
        (DType::U8, DType::I64) => "convert_u8_i64",
        (DType::U32, DType::F16) => "convert_u32_f16",
        (DType::U32, DType::U8) => "convert_u32_u8",
        (DType::U32, DType::I64) => "convert_u32_i64",
        (DType::I64, DType::F32) => "convert_i64_f32",
        (DType::I64, DType::F16) => "convert_i64_f16",
        (DType::I64, DType::U8) => "convert_i64_u8",
        (DType::I64, DType::U32) => "convert_i64_u32",
        _ => {
            return Err(Error::Msg(format!(
                "vulkan backend op to_dtype {src:?}->{dst:?} not implemented"
            ))
            .bt())
        }
    };
    candle_vulkan_kernels::spirv(name)
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated")).bt())
}

fn quantized_dequant_spirv(qdtype: GgmlDType) -> Result<&'static [u32]> {
    let name = match qdtype {
        GgmlDType::Q4_0 => "dequant_q4_0",
        GgmlDType::Q4_1 => "dequant_q4_1",
        GgmlDType::Q5_0 => "dequant_q5_0",
        GgmlDType::Q5_1 => "dequant_q5_1",
        GgmlDType::Q8_0 => "dequant_q8_0",
        GgmlDType::Q8K => "dequant_q8_k_f32",
        GgmlDType::Q2K => "dequant_q2_k",
        GgmlDType::Q3K => "dequant_q3_k",
        GgmlDType::Q4K => "dequant_q4_k",
        GgmlDType::Q5K => "dequant_q5_k",
        GgmlDType::Q6K => "dequant_q6_k",
        _ => return Err(unsupported("quantized dequantize")),
    };
    candle_vulkan_kernels::spirv(name)
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated")).bt())
}

fn quantized_dequant_blocks_per_workgroup(qdtype: GgmlDType) -> Result<u32> {
    match qdtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            Ok(4)
        }
        GgmlDType::Q8K => Ok(1),
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => {
            Ok(256)
        }
        _ => Err(unsupported("quantized dequantize")),
    }
}

fn cmp_opcode(op: CmpOp) -> i32 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

impl VulkanStorage {
    pub(crate) fn quantized_dequantize_f32(
        &self,
        qdtype: GgmlDType,
        elem_count: usize,
    ) -> Result<Self> {
        if !elem_count.is_multiple_of(qdtype.block_size()) {
            crate::bail!(
                "vulkan quantized dequantize expects element count divisible by block size {}, got {elem_count}",
                qdtype.block_size()
            )
        }
        let flat_shape = Shape::from(elem_count);
        let flat_layout = Layout::contiguous(flat_shape.clone());
        match qdtype {
            // GGUF files store norm/bias tensors as raw float blobs under the
            // GGML dtype enum. These need no dequant math, only a device-side
            // reinterpret (and cast for f16/bf16), so keep them on the GPU
            // instead of routing through the CPU fallback.
            GgmlDType::F32 => {
                let dst = unsafe { self.device.alloc_uninit(&flat_shape, DType::F32)? };
                self.device.submit_copy_region_and_track(
                    &self.buffer,
                    &dst.buffer,
                    0,
                    0,
                    elem_count * DType::F32.size_in_bytes(),
                    false,
                )?;
                Ok(dst)
            }
            GgmlDType::F16 => {
                let dst_f16 = unsafe { self.device.alloc_uninit(&flat_shape, DType::F16)? };
                self.device.submit_copy_region_and_track(
                    &self.buffer,
                    &dst_f16.buffer,
                    0,
                    0,
                    elem_count * DType::F16.size_in_bytes(),
                    false,
                )?;
                dst_f16.to_dtype(&flat_layout, DType::F32)
            }
            GgmlDType::BF16 => {
                let dst_bf16 = unsafe { self.device.alloc_uninit(&flat_shape, DType::BF16)? };
                self.device.submit_copy_region_and_track(
                    &self.buffer,
                    &dst_bf16.buffer,
                    0,
                    0,
                    elem_count * DType::BF16.size_in_bytes(),
                    false,
                )?;
                dst_bf16.to_dtype(&flat_layout, DType::F32)
            }
            GgmlDType::Q8_1 => {
                let repacked = repack_q8_1_storage_to_q8_0(&self.device, self, elem_count)?;
                repacked.quantized_dequantize_f32(GgmlDType::Q8_0, elem_count)
            }
            GgmlDType::Q8K => {
                let dst = unsafe { self.device.alloc_uninit(&flat_shape, DType::F32)? };
                let spirv = quantized_dequant_spirv(qdtype)?;
                let params = VulkanDequantizeParams {
                    m: 0,
                    k: 0,
                    stride_a: 0,
                    stride_b: 0,
                    nel: elem_count.try_into()?,
                };
                let num_blocks = elem_count / qdtype.block_size();
                let blocks_per_workgroup = quantized_dequant_blocks_per_workgroup(qdtype)?;
                let workgroups = (num_blocks as u32).div_ceil(blocks_per_workgroup);
                self.device.run_compute(
                    spirv,
                    &[
                        VulkanBinding::Storage(&self.buffer),
                        VulkanBinding::Storage(&dst.buffer),
                    ],
                    Some(any_as_bytes(&params)),
                    workgroups,
                )?;
                Ok(dst)
            }
            GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K => {
                let dst_f16 = unsafe { self.device.alloc_uninit(&flat_shape, DType::F16)? };
                let spirv = quantized_dequant_spirv(qdtype)?;
                let params = VulkanDequantizeParams {
                    m: 0,
                    k: 0,
                    stride_a: 0,
                    stride_b: 0,
                    nel: elem_count.try_into()?,
                };
                let num_blocks = elem_count / qdtype.block_size();
                let blocks_per_workgroup = quantized_dequant_blocks_per_workgroup(qdtype)?;
                let workgroups = (num_blocks as u32).div_ceil(blocks_per_workgroup);
                self.device.run_compute(
                    spirv,
                    &[
                        VulkanBinding::Storage(&self.buffer),
                        VulkanBinding::Storage(&dst_f16.buffer),
                    ],
                    Some(any_as_bytes(&params)),
                    workgroups,
                )?;
                dst_f16.to_dtype(&flat_layout, DType::F32)
            }
        }
    }

    fn run_rank2_matmul_f32_via_batched_matvec(
        &self,
        rhs_t: &Self,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        let dst_shape = Shape::from(vec![1, m, n]);
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::F32)? };
        let bindings = [
            VulkanBinding::Storage(&rhs_t.buffer),
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let dmmv_workgroup = vulkan_dmmv_workgroup(&self.device, GgmlDType::F32, n, k, false);
        let spirv_name = vulkan_dmmv_shader_name(
            &self.device,
            GgmlDType::F32,
            "mul_mat_vec_f32_f32_f32".to_string(),
            dmmv_workgroup,
        );
        let spirv = candle_vulkan_kernels::spirv(&spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        let block_size = vulkan_dmmv_block_size(&self.device, GgmlDType::F32, dmmv_workgroup);
        let rows_per_group = 1u32;
        let use_dmmv_subgroups = vulkan_dmmv_use_subgroups(&self.device, GgmlDType::F32);
        let required_subgroup_size = if use_dmmv_subgroups {
            Some(vulkan_dmmv_subgroup_size(&self.device))
        } else {
            None
        };
        let workgroups = (n.div_ceil(rows_per_group as usize).try_into()?, 1, 1);
        let mut row_idx = 0usize;
        while row_idx < m {
            // Dense f32 Conv/MatMul traffic is dominated by host-side per-dispatch
            // overhead on the current Vulkan path. Processing more rows per dispatch
            // amortizes descriptor-set allocation and command recording without
            // changing shader semantics or quantized-path limits.
            let row_chunk = usize::min(VULKAN_DENSE_MUL_MAT_VEC_MAX_ROWS, m - row_idx);
            let spec = [
                (0, block_size),
                (1, rows_per_group),
                (2, row_chunk.try_into()?),
            ];
            let params = VulkanMatVecParams {
                ncols: k.try_into()?,
                stride_a: k.try_into()?,
                stride_b: k.try_into()?,
                stride_d: n.try_into()?,
                batch_stride_a: 0,
                batch_stride_b: k.try_into()?,
                batch_stride_d: n.try_into()?,
                fusion_flags: 0,
                base_work_group_y: row_idx.try_into()?,
                ne02: 1,
                ne12: 1,
                broadcast2: 1,
                broadcast3: 1,
            };
            self.device.run_compute_specialized_with_options(
                spirv,
                &bindings,
                Some(any_as_bytes(&params)),
                workgroups,
                Some(&spec),
                use_dmmv_subgroups,
                required_subgroup_size,
            )?;
            row_idx += row_chunk;
        }
        Ok(dst)
    }

    fn run_batched_matmul_f32_via_rank2_batches(
        &self,
        rhs_t: &Self,
        b: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Self> {
        let dst_shape = Shape::from(vec![b, m, n]);
        let mut dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::F32)? };
        let lhs_batch_shape = Shape::from((m, k));
        let rhs_batch_shape = Shape::from((n, k));
        let batch_out_layout = Layout::contiguous((1, m, n));
        for batch_idx in 0..b {
            let lhs_batch_layout = Layout::contiguous_with_offset((m, k), batch_idx * m * k);
            let rhs_batch_layout = Layout::contiguous_with_offset((n, k), batch_idx * n * k);
            let mut lhs_batch = unsafe { self.device.alloc_uninit(&lhs_batch_shape, DType::F32)? };
            let mut rhs_batch = unsafe { self.device.alloc_uninit(&rhs_batch_shape, DType::F32)? };
            self.copy_strided_src(&mut lhs_batch, 0, &lhs_batch_layout)?;
            rhs_t.copy_strided_src(&mut rhs_batch, 0, &rhs_batch_layout)?;
            let batch_out =
                lhs_batch.run_rank2_matmul_f32_via_batched_matvec(&rhs_batch, m, n, k)?;
            batch_out.copy_strided_src(&mut dst, batch_idx * m * n, &batch_out_layout)?;
        }
        Ok(dst)
    }

    fn run_bf16_to_dtype_via_get_rows(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype != DType::BF16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan to_dtype").bt());
        }
        if dtype != DType::F16 && dtype != DType::F32 {
            return Err(unsupported("to_dtype bf16"));
        }
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("to_dtype bf16 strided"));
        }

        let elem_count = layout.shape().elem_count();
        let cols = layout.dims().last().copied().unwrap_or(1).max(1);
        let rows = elem_count.div_ceil(cols);
        let rows_u32: u32 = rows.try_into()?;
        let cols_u32: u32 = cols.try_into()?;

        let ids = CpuStorage::U32((0..rows_u32).collect());
        let ids = self.device.storage_from_cpu_storage(&ids)?;
        let ids_layout = Layout::contiguous(Shape::from(rows));
        let src_layout = Layout::contiguous(Shape::from((rows, cols)));
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), dtype)? };
        let params = GgmlBinaryParams {
            ne: (rows * cols).try_into()?,
            ne00: cols_u32,
            ne01: rows_u32,
            ne02: 1,
            ne03: 1,
            nb00: 1,
            nb01: cols_u32,
            nb02: 0,
            nb03: 0,
            ne10: rows_u32,
            ne11: 1,
            ne12: 1,
            ne13: 1,
            nb10: 1,
            nb11: 0,
            nb12: 0,
            nb13: 0,
            ne20: cols_u32,
            ne21: rows_u32,
            ne22: 1,
            ne23: 1,
            nb20: 1,
            nb21: cols_u32,
            nb22: 0,
            nb23: 0,
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&ids.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv_name = match dtype {
            DType::F16 => "get_rows_bf16",
            DType::F32 => "get_rows_bf16_f32",
            _ => unreachable!(),
        };
        let spirv = candle_vulkan_kernels::spirv(spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        self.device.run_compute_3d(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (cols_u32.div_ceil(512), rows_u32, 1),
        )?;
        let _ = ids_layout;
        let _ = src_layout;
        Ok(dst)
    }

    fn run_unary_head(&self, layout: &Layout, spirv: &[u32]) -> Result<Self> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut materialized, 0, layout)?;
            let contiguous_layout = Layout::contiguous(layout.shape());
            return materialized.run_unary_head(&contiguous_layout, spirv);
        }
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let params = GgmlHeadParams {
            kx: count.try_into()?,
            ky: 1,
            param1: 0.0,
            param2: 0.0,
            param3: 0.0,
            param4: 0.0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let workgroups = ggml_linear_workgroups(count)?;
        self.device
            .run_compute_3d(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(dst)
    }

    fn run_unary_generic_with_params(
        &self,
        layout: &Layout,
        spirv: &[u32],
        param1: f32,
        param2: f32,
    ) -> Result<Self> {
        self.run_unary_generic_with_params_dtype(layout, spirv, param1, param2, self.dtype)
    }

    fn run_unary_generic_with_params_dtype(
        &self,
        layout: &Layout,
        spirv: &[u32],
        param1: f32,
        param2: f32,
        dst_dtype: DType,
    ) -> Result<Self> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            // Several shaders dispatched through this helper (scale, powf,
            // clamp) index linearly, and the ggml unary header only carries
            // 16-bit misalign offsets, so strided or offset views are first
            // materialized contiguously on the GPU instead of either producing
            // wrong results or dropping the op to the CPU.
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            let contiguous = Layout::contiguous(layout.shape().clone());
            return tmp.run_unary_generic_with_params_dtype(
                &contiguous,
                spirv,
                param1,
                param2,
                dst_dtype,
            );
        }
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), dst_dtype)? };
        let (src_dims, src_strides) = dims4_ggml(layout)?;
        let dst_dims = src_dims;
        let dst_strides = contiguous_strides_ggml(dst_dims);
        let (ne0_012mp, ne0_012l) = fastdiv_values(src_dims[2] * src_dims[1] * src_dims[0]);
        let (ne0_01mp, ne0_01l) = fastdiv_values(src_dims[1] * src_dims[0]);
        let (ne0_0mp, ne0_0l) = fastdiv_values(src_dims[0]);
        let (ne1_012mp, ne1_012l) = fastdiv_values(dst_dims[2] * dst_dims[1] * dst_dims[0]);
        let (ne1_01mp, ne1_01l) = fastdiv_values(dst_dims[1] * dst_dims[0]);
        let (ne1_0mp, ne1_0l) = fastdiv_values(dst_dims[0]);
        let params = GgmlUnaryParams {
            ne: count.try_into()?,
            ne00: src_dims[0],
            ne01: src_dims[1],
            ne02: src_dims[2],
            ne03: src_dims[3],
            nb00: src_strides[0],
            nb01: src_strides[1],
            nb02: src_strides[2],
            nb03: src_strides[3],
            ne10: dst_dims[0],
            ne11: dst_dims[1],
            ne12: dst_dims[2],
            ne13: dst_dims[3],
            nb10: dst_strides[0],
            nb11: dst_strides[1],
            nb12: dst_strides[2],
            nb13: dst_strides[3],
            misalign_offsets: 0,
            param1,
            param2,
            ne0_012mp,
            ne0_012l,
            ne0_01mp,
            ne0_01l,
            ne0_0mp,
            ne0_0l,
            ne1_012mp,
            ne1_012l,
            ne1_01mp,
            ne1_01l,
            ne1_0mp,
            ne1_0l,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let workgroups = ggml_linear_workgroups(count)?;
        self.device
            .run_compute_3d(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(dst)
    }

    fn run_unary_generic(&self, layout: &Layout, spirv: &[u32]) -> Result<Self> {
        self.run_unary_generic_with_params(layout, spirv, 0.0, 0.0)
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan cmp").bt());
        }
        if rhs.dtype != self.dtype {
            return Err(Error::UnsupportedDTypeForOp(rhs.dtype, "vulkan cmp").bt());
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
        let (lhs_tmp, lhs_layout) = if lhs_layout.start_offset() == 0 {
            (None, lhs_layout.clone())
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(lhs_layout.shape(), self.dtype)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut tmp, 0, lhs_layout)?;
            (Some(tmp), Layout::contiguous(lhs_layout.shape().clone()))
        };
        let lhs = lhs_tmp.as_ref().unwrap_or(self);
        let (rhs_tmp, rhs_layout) = if rhs_layout.start_offset() == 0 {
            (None, rhs_layout.clone())
        } else {
            let mut tmp = unsafe { rhs.device.alloc_uninit(rhs_layout.shape(), rhs.dtype)? };
            <Self as BackendStorage>::copy_strided_src(rhs, &mut tmp, 0, rhs_layout)?;
            (Some(tmp), Layout::contiguous(rhs_layout.shape().clone()))
        };
        let rhs = rhs_tmp.as_ref().unwrap_or(rhs);
        let (lhs_dims, lhs_strides) = dims4_ggml(&lhs_layout)?;
        let (rhs_dims, rhs_strides) = dims4_ggml(&rhs_layout)?;
        let count = lhs_layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(lhs_layout.shape(), DType::U8)? };
        let params = GgmlBinaryParams {
            ne: count.try_into()?,
            ne00: lhs_dims[0],
            ne01: lhs_dims[1],
            ne02: lhs_dims[2],
            ne03: lhs_dims[3],
            nb00: lhs_strides[0],
            nb01: lhs_strides[1],
            nb02: lhs_strides[2],
            nb03: lhs_strides[3],
            ne10: rhs_dims[0],
            ne11: rhs_dims[1],
            ne12: rhs_dims[2],
            ne13: rhs_dims[3],
            nb10: rhs_strides[0],
            nb11: rhs_strides[1],
            nb12: rhs_strides[2],
            nb13: rhs_strides[3],
            ne20: lhs_dims[0],
            ne21: lhs_dims[1],
            ne22: lhs_dims[2],
            ne23: lhs_dims[3],
            nb20: 1,
            nb21: lhs_dims[0],
            nb22: lhs_dims[0] * lhs_dims[1],
            nb23: lhs_dims[0] * lhs_dims[1] * lhs_dims[2],
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            param3: cmp_opcode(op),
        };
        let bindings = [
            VulkanBinding::Storage(&lhs.buffer),
            VulkanBinding::Storage(&rhs.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv_name = match self.dtype {
            DType::F16 => "cmp_f16",
            DType::F32 => "cmp_f32",
            DType::U8 => "cmp_u8",
            DType::U32 => "cmp_u32",
            DType::I64 => "cmp_i64",
            _ => return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan cmp").bt()),
        };
        let spirv = candle_vulkan_kernels::spirv(spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        let workgroups = ggml_linear_workgroups(count)?;
        self.device
            .run_compute_3d(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan where_cond").bt());
        }
        if !matches!(
            t.dtype,
            DType::F32 | DType::F16 | DType::U8 | DType::U32 | DType::I64
        ) {
            return Err(Error::UnsupportedDTypeForOp(t.dtype, "vulkan where_cond").bt());
        }
        if f.dtype != t.dtype {
            return Err(Error::UnsupportedDTypeForOp(f.dtype, "vulkan where_cond").bt());
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
        let (dims, cond_strides) = dims4_ggml(layout)?;
        let (_, true_strides) = dims4_ggml(t_l)?;
        let (_, false_strides) = dims4_ggml(f_l)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), t.dtype)? };
        let params = VulkanWhereU8Params {
            ne: count.try_into()?,
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            ne3: dims[3],
            offset_cond: layout.start_offset().try_into()?,
            offset_true: t_l.start_offset().try_into()?,
            offset_false: f_l.start_offset().try_into()?,
            cond_nb0: cond_strides[0],
            cond_nb1: cond_strides[1],
            cond_nb2: cond_strides[2],
            cond_nb3: cond_strides[3],
            true_nb0: true_strides[0],
            true_nb1: true_strides[1],
            true_nb2: true_strides[2],
            true_nb3: true_strides[3],
            false_nb0: false_strides[0],
            false_nb1: false_strides[1],
            false_nb2: false_strides[2],
            false_nb3: false_strides[3],
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&t.buffer),
            VulkanBinding::Storage(&f.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv_name = match t.dtype {
            DType::F16 => "where_u8_f16",
            DType::F32 => "where_u8_f32",
            DType::U8 => "where_u8_u8",
            DType::U32 => "where_u8_u32",
            DType::I64 => "where_u8_i64",
            _ => return Err(Error::UnsupportedDTypeForOp(t.dtype, "vulkan where_cond").bt()),
        };
        let spirv = candle_vulkan_kernels::spirv(spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        let workgroups = ggml_linear_workgroups(count)?;
        self.device
            .run_compute_3d(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(dst)
    }

    fn run_binary_named(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: &'static str,
    ) -> Result<Self> {
        let int_dtype = matches!(self.dtype, DType::U8 | DType::U32 | DType::I64);
        if self.dtype != DType::F32 && self.dtype != DType::F16 && !int_dtype {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan binary").bt());
        }
        if rhs.dtype != self.dtype {
            return Err(Error::UnsupportedDTypeForOp(rhs.dtype, "vulkan binary").bt());
        }
        self.device
            .same_device(&rhs.device)
            .then_some(())
            .ok_or_else(|| {
                Error::DeviceMismatchBinaryOp {
                    lhs: self.device.location(),
                    rhs: rhs.device.location(),
                    op,
                }
                .bt()
            })?;
        if lhs_layout.start_offset() != 0 || rhs_layout.start_offset() != 0 {
            return Err(unsupported("binary offset"));
        }
        let (lhs_dims, lhs_strides) = dims4_ggml(lhs_layout)?;
        let (rhs_dims, rhs_strides) = dims4_ggml(rhs_layout)?;
        let count = lhs_layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(lhs_layout.shape(), self.dtype)? };
        let params = GgmlBinaryParams {
            ne: count.try_into()?,
            ne00: lhs_dims[0],
            ne01: lhs_dims[1],
            ne02: lhs_dims[2],
            ne03: lhs_dims[3],
            nb00: lhs_strides[0],
            nb01: lhs_strides[1],
            nb02: lhs_strides[2],
            nb03: lhs_strides[3],
            ne10: rhs_dims[0],
            ne11: rhs_dims[1],
            ne12: rhs_dims[2],
            ne13: rhs_dims[3],
            nb10: rhs_strides[0],
            nb11: rhs_strides[1],
            nb12: rhs_strides[2],
            nb13: rhs_strides[3],
            ne20: lhs_dims[0],
            ne21: lhs_dims[1],
            ne22: lhs_dims[2],
            ne23: lhs_dims[3],
            nb20: 1,
            nb21: lhs_dims[0],
            nb22: lhs_dims[0] * lhs_dims[1],
            nb23: lhs_dims[0] * lhs_dims[1] * lhs_dims[2],
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            param3: if int_dtype { binary_int_opcode(op)? } else { 0 },
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&rhs.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = binary_spirv(op, self.dtype)?;
        let workgroups = ggml_linear_workgroups(count)?;
        self.device
            .run_compute_3d(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(dst)
    }

    fn run_binary_min_max_f32(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: &'static str,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, op).bt());
        }
        if rhs.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(rhs.dtype, op).bt());
        }

        let out_layout = Layout::contiguous(lhs_layout.shape());
        let sum = self.run_binary_named(rhs, lhs_layout, rhs_layout, "add")?;
        let diff = self.run_binary_named(rhs, lhs_layout, rhs_layout, "sub")?;
        let (abs_spirv, _) = unary_spirv("abs", DType::F32)?;
        let abs = diff.run_unary_head(&out_layout, abs_spirv)?;
        let combined = match op {
            "maximum" => sum.run_binary_named(&abs, &out_layout, &out_layout, "add")?,
            "minimum" => sum.run_binary_named(&abs, &out_layout, &out_layout, "sub")?,
            _ => return Err(unsupported("binary min/max")),
        };
        let scale_spirv = candle_vulkan_kernels::spirv("scale_f32")
            .ok_or_else(|| Error::Msg("vulkan shader scale_f32 not generated".into()).bt())?;
        combined.run_unary_generic_with_params(&out_layout, scale_spirv, 0.5, 0.0)
    }

    fn run_copy_into(
        &self,
        layout: &Layout,
        dst: &Self,
        dst_offset: usize,
        spirv: &[u32],
    ) -> Result<()> {
        if layout.start_offset() > u16::MAX as usize || dst_offset > u16::MAX as usize {
            return self.run_copy_into_via_regions(layout, dst, dst_offset);
        }
        let count = layout.shape().elem_count();
        let (src_dims, src_strides) = dims4_ggml(layout)?;
        let dst_dims = src_dims;
        let dst_strides = contiguous_strides_ggml(dst_dims);
        let (ne0_012mp, ne0_012l) = fastdiv_values(src_dims[0] * src_dims[1] * src_dims[2]);
        let (ne0_01mp, ne0_01l) = fastdiv_values(src_dims[0] * src_dims[1]);
        let (ne0_0mp, ne0_0l) = fastdiv_values(src_dims[0]);
        let (ne1_012mp, ne1_012l) = fastdiv_values(dst_dims[0] * dst_dims[1] * dst_dims[2]);
        let (ne1_01mp, ne1_01l) = fastdiv_values(dst_dims[0] * dst_dims[1]);
        let (ne1_0mp, ne1_0l) = fastdiv_values(dst_dims[0]);
        let params = GgmlUnaryParams {
            ne: count.try_into()?,
            ne00: src_dims[0],
            ne01: src_dims[1],
            ne02: src_dims[2],
            ne03: src_dims[3],
            nb00: src_strides[0],
            nb01: src_strides[1],
            nb02: src_strides[2],
            nb03: src_strides[3],
            ne10: dst_dims[0],
            ne11: dst_dims[1],
            ne12: dst_dims[2],
            ne13: dst_dims[3],
            nb10: dst_strides[0],
            nb11: dst_strides[1],
            nb12: dst_strides[2],
            nb13: dst_strides[3],
            misalign_offsets: ((layout.start_offset() as u32) << 16) | dst_offset as u32,
            param1: 0.0,
            param2: 0.0,
            ne0_012mp,
            ne0_012l,
            ne0_01mp,
            ne0_01l,
            ne0_0mp,
            ne0_0l,
            ne1_012mp,
            ne1_012l,
            ne1_01mp,
            ne1_01l,
            ne1_0mp,
            ne1_0l,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let workgroups = ggml_linear_workgroups(count)?;
        self.device
            .run_compute_3d(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(())
    }

    fn run_copy_into_via_regions(
        &self,
        layout: &Layout,
        dst: &Self,
        dst_offset: usize,
    ) -> Result<()> {
        let elem_size = self.dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan copy").bt());
        }
        match layout.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => {
                let src_offset = start_offset
                    .checked_mul(elem_size)
                    .ok_or_else(|| Error::msg("vulkan copy src offset overflow"))?;
                let dst_offset = dst_offset
                    .checked_mul(elem_size)
                    .ok_or_else(|| Error::msg("vulkan copy dst offset overflow"))?;
                let size = len
                    .checked_mul(elem_size)
                    .ok_or_else(|| Error::msg("vulkan copy size overflow"))?;
                self.device.submit_copy_region_and_track(
                    &self.buffer,
                    &dst.buffer,
                    src_offset,
                    dst_offset,
                    size,
                    false,
                )?;
            }
            crate::StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            } => {
                if block_len == 0 {
                    return Ok(());
                }
                let block_size = block_len
                    .checked_mul(elem_size)
                    .ok_or_else(|| Error::msg("vulkan copy block size overflow"))?;
                let mut dst_elem_index = dst_offset;
                let mut regions = Vec::with_capacity(block_start_index.len());
                for src_index in block_start_index {
                    let src_offset = src_index
                        .checked_mul(elem_size)
                        .ok_or_else(|| Error::msg("vulkan copy src offset overflow"))?;
                    let dst_offset = dst_elem_index
                        .checked_mul(elem_size)
                        .ok_or_else(|| Error::msg("vulkan copy dst offset overflow"))?;
                    regions.push(
                        vk::BufferCopy::default()
                            .src_offset(src_offset as u64)
                            .dst_offset(dst_offset as u64)
                            .size(block_size as u64),
                    );
                    dst_elem_index = dst_elem_index
                        .checked_add(block_len)
                        .ok_or_else(|| Error::msg("vulkan copy dst element overflow"))?;
                }
                self.device.submit_copy_regions_and_track(
                    &self.buffer,
                    &dst.buffer,
                    &regions,
                    false,
                )?;
            }
        }
        Ok(())
    }

    fn run_sum_rows(&self, layout: &Layout) -> Result<Self> {
        if layout.start_offset() > u16::MAX as usize {
            return Err(unsupported("sum_rows offset"));
        }
        let rank = layout.dims().len();
        let (src_dims, src_strides) = dims4_ggml(layout)?;
        let mut dst_dims_candle = layout.dims().to_vec();
        dst_dims_candle[rank - 1] = 1;
        let dst_shape = Shape::from(dst_dims_candle);
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };

        let mut dst_dims = src_dims;
        dst_dims[0] = 1;
        let dst_strides = contiguous_strides_ggml(dst_dims);
        let rows = src_dims[1] * src_dims[2] * src_dims[3];
        let (ne0_12mp, ne0_12l) = fastdiv_values(src_dims[1] * src_dims[2]);
        let (ne0_1mp, ne0_1l) = fastdiv_values(src_dims[1]);
        let params = GgmlSumRowsParams {
            n_cols: src_dims[0],
            ne01: src_dims[1],
            ne02: src_dims[2],
            nb01: src_strides[1],
            nb02: src_strides[2],
            nb03: src_strides[3],
            nb11: dst_strides[1],
            nb12: dst_strides[2],
            nb13: dst_strides[3],
            weight: 1.0,
            misalign_offsets: (layout.start_offset() as u32) << 16,
            ne0_12mp,
            ne0_12l,
            ne0_1mp,
            ne0_1l,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("sum_rows_f32")
            .ok_or_else(|| Error::Msg("vulkan shader sum_rows_f32 not generated".into()).bt())?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (rows, 1, 1),
            Some(&[(0, self.device.inner.subgroup_size)]),
        )?;
        Ok(dst)
    }

    fn run_argmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("argmax strided"));
        }
        let rank = layout.dims().len();
        let kx = *layout
            .dims()
            .last()
            .ok_or_else(|| unsupported("argmax scalar"))?;
        let mut dst_dims_candle = layout.dims().to_vec();
        dst_dims_candle[rank - 1] = 1;
        let dst_shape = Shape::from(dst_dims_candle);
        let rows = dst_shape.elem_count();
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::U32)? };
        let params = GgmlHeadParams {
            kx: kx.try_into()?,
            ky: rows.try_into()?,
            param1: 0.0,
            param2: 0.0,
            param3: 0.0,
            param4: 0.0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("argmax_f32")
            .ok_or_else(|| Error::Msg("vulkan shader argmax_f32 not generated".into()).bt())?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (rows.try_into()?, 1, 1),
            Some(&[(0, self.device.inner.subgroup_size)]),
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
                let (spirv, kind) = unary_spirv("neg", DType::F32)?;
                let neg = match kind {
                    VulkanUnaryKind::Head => src.run_unary_head(src_layout, spirv)?,
                    VulkanUnaryKind::Generic => src.run_unary_generic(src_layout, spirv)?,
                };
                let neg_layout = Layout::contiguous(src_layout.shape());
                let ids = neg.run_argmax_last_dim(&neg_layout)?;
                if op == ReduceOp::ArgMin {
                    Ok(ids)
                } else {
                    <Self as BackendStorage>::gather(src, src_layout, &ids, &ids_layout, rank - 1)
                }
            }
            ReduceOp::Sum => Err(unsupported("reduce extrema")),
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
        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
            return Err(unsupported("reduce multi-dim arg"));
        }
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
        current_storage.ok_or_else(|| unsupported("reduce multi-dim empty"))
    }

    pub(crate) fn argsort_last_dim_f32(
        &self,
        layout: &Layout,
        asc: bool,
        last_dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan argsort").bt());
        }
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("argsort strided"));
        }
        if last_dim == 0 || layout.dims().last().copied() != Some(last_dim) {
            return Err(unsupported("argsort last-dim"));
        }
        let ncols_padded = next_power_of_two_u32(last_dim, "argsort")?;
        if ncols_padded != last_dim as u32 && !self.device.inner.robust_buffer_access {
            return Err(unsupported(
                "argsort non-power-of-two without robust buffers",
            ));
        }
        let count = layout.shape().elem_count();
        let nrows = count / last_dim;
        let nrows_u32: u32 = nrows.try_into()?;
        if nrows_u32 == 0 {
            return Err(unsupported("argsort empty rows"));
        }
        let workgroups_y = nrows_u32
            .min(self.device.inner.max_workgroup_count_y)
            .max(1);
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), DType::U32)? };
        let ncols_padded_log2 = ncols_padded.trailing_zeros();
        let pipeline_idx = ncols_padded_log2.min(VULKAN_ARGSORT_NUM_PIPELINES - 1);
        let use_small = ncols_padded_log2 <= self.device.inner.max_workgroup_size_log2;
        if !use_small && !self.device.inner.vulkan_memory_model {
            return Err(unsupported("argsort large requires vulkan memory model"));
        }
        let base_params = VulkanArgsortParams {
            ncols: last_dim.try_into()?,
            ncols_padded,
            ncols_padded_log2,
            nrows: nrows_u32,
            order: if asc { 0 } else { 1 },
            outer_start: 0,
            outer_end: 0,
            inner_start: 0,
            inner_end: 0,
        };

        if use_small {
            let bindings = [
                VulkanBinding::Storage(&self.buffer),
                VulkanBinding::Storage(&dst.buffer),
                VulkanBinding::Storage(&dst.buffer),
            ];
            let spirv = candle_vulkan_kernels::spirv("argsort_f32")
                .ok_or_else(|| Error::Msg("vulkan shader argsort_f32 not generated".into()).bt())?;
            self.device.run_compute_specialized(
                spirv,
                &bindings,
                Some(any_as_bytes(&base_params)),
                (1, workgroups_y, 1),
                Some(&[(0, ncols_padded), (1, ncols_padded_log2)]),
            )?;
            return Ok(dst);
        }

        let tmp_count = (ncols_padded as usize)
            .checked_mul(nrows)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| Error::Msg("vulkan backend op argsort tmp overflow".into()).bt())?;
        let tmp_size = byte_len(DType::I32, tmp_count, "vulkan argsort tmp")?;
        let tmp = self
            .device
            .create_buffer(tmp_size, "candle-vulkan-argsort-large-tmp")?;
        self.run_argsort_large_pass(
            &self.buffer,
            &tmp,
            &dst.buffer,
            base_params,
            workgroups_y,
            pipeline_idx,
            0,
            ncols_padded_log2.min(self.device.inner.max_workgroup_size_log2),
            0,
            100,
        )?;
        for outer in self.device.inner.max_workgroup_size_log2..ncols_padded_log2 {
            let mut inner = 0;
            while inner <= outer {
                let inner_start = inner;
                let (pass_idx, inner_end) = if outer - inner_start < pipeline_idx {
                    inner = outer + 1;
                    (pipeline_idx, 100)
                } else {
                    inner += 1;
                    (pipeline_idx.saturating_sub(2), inner_start + 1)
                };
                self.run_argsort_large_pass(
                    &self.buffer,
                    &tmp,
                    &dst.buffer,
                    base_params,
                    workgroups_y,
                    pass_idx,
                    outer,
                    outer + 1,
                    inner_start,
                    inner_end,
                )?;
            }
        }
        Ok(dst)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_argsort_large_pass(
        &self,
        src: &Arc<VulkanBuffer>,
        tmp: &Arc<VulkanBuffer>,
        dst: &Arc<VulkanBuffer>,
        base_params: VulkanArgsortParams,
        workgroups_y: u32,
        pass_idx: u32,
        outer_start: u32,
        outer_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> Result<()> {
        let block_size = 1u32 << pass_idx.min(self.device.inner.max_workgroup_size_log2);
        let elems_per_wg = block_size
            .checked_mul(VULKAN_ARGSORT_WG_UNROLL_FACTOR)
            .ok_or_else(|| {
                Error::Msg("vulkan backend op argsort workgroup overflow".into()).bt()
            })?;
        let workgroups_x = base_params.ncols_padded.div_ceil(elems_per_wg).max(1);
        let params = VulkanArgsortParams {
            outer_start,
            outer_end,
            inner_start,
            inner_end,
            ..base_params
        };
        let bindings = [
            VulkanBinding::Storage(src),
            VulkanBinding::Storage(tmp),
            VulkanBinding::Storage(dst),
        ];
        let spirv = candle_vulkan_kernels::spirv("argsort_large_f32").ok_or_else(|| {
            Error::Msg("vulkan shader argsort_large_f32 not generated".into()).bt()
        })?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (workgroups_x, workgroups_y, 1),
            Some(&[(0, block_size), (1, VULKAN_ARGSORT_WG_UNROLL_FACTOR)]),
        )
    }

    fn run_cumsum_last_dim(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan cumsum").bt());
        }
        if !layout.is_contiguous() || layout.start_offset() > u16::MAX as usize {
            return Err(unsupported("cumsum strided"));
        }
        let (src_dims, src_strides) = dims4_ggml(layout)?;
        let rows = src_dims[1] * src_dims[2] * src_dims[3];
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let dst_strides = contiguous_strides_ggml(src_dims);
        let (ne0_12mp, ne0_12l) = fastdiv_values(src_dims[1] * src_dims[2]);
        let (ne0_1mp, ne0_1l) = fastdiv_values(src_dims[1]);
        let params = GgmlSumRowsParams {
            n_cols: src_dims[0],
            ne01: src_dims[1],
            ne02: src_dims[2],
            nb01: src_strides[1],
            nb02: src_strides[2],
            nb03: src_strides[3],
            nb11: dst_strides[1],
            nb12: dst_strides[2],
            nb13: dst_strides[3],
            weight: 1.0,
            misalign_offsets: (layout.start_offset() as u32) << 16,
            ne0_12mp,
            ne0_12l,
            ne0_1mp,
            ne0_1l,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("cumsum_f32")
            .ok_or_else(|| Error::Msg("vulkan shader cumsum_f32 not generated".into()).bt())?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (rows, 1, 1),
            Some(&[(0, 128), (1, self.device.inner.subgroup_size), (2, 4)]),
        )?;
        Ok(dst)
    }

    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("softmax strided"));
        }
        if self.dtype == DType::F16 {
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape().clone());
            let out_f32 = src_f32.softmax_last_dim(&src_f32_layout)?;
            return out_f32.to_dtype(&src_f32_layout, DType::F16);
        }
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan softmax").bt());
        }
        let rank = layout.dims().len();
        if rank == 0 {
            return Err(unsupported("softmax scalar"));
        }
        let (dims, _) = dims4_ggml(layout)?;
        let count = layout.shape().elem_count();
        let rows = count / layout.dims()[rank - 1];
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let params = GgmlSoftmaxParams {
            kx: dims[0],
            ky: 0,
            ne00: dims[0],
            ne01: dims[1],
            ne02: dims[2],
            ne12: 1,
            ne13: 1,
            nb11: 0,
            nb12: 0,
            nb13: 0,
            scale: 1.0,
            max_bias: 0.0,
            m0: 0.0,
            m1: 0.0,
            n_head_log2: 0,
            nrows_x: rows.try_into()?,
            has_sinks: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("soft_max_f32")
            .ok_or_else(|| Error::Msg("vulkan shader soft_max_f32 not generated".into()).bt())?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (rows.try_into()?, 1, 1),
            Some(&[(0, self.device.inner.subgroup_size)]),
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
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan rope").bt());
        }
        if pos.dtype != DType::I32 {
            return Err(Error::UnsupportedDTypeForOp(pos.dtype, "vulkan rope positions").bt());
        }
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("rope strided"));
        }
        if !pos_layout.is_contiguous() || pos_layout.start_offset() != 0 {
            return Err(unsupported("rope positions strided"));
        }
        let (dims, strides) = dims4_ggml(layout)?;
        let count = layout.shape().elem_count();
        let rows = count / layout.dims().last().copied().unwrap_or(1).max(1);
        let rows_u32: u32 = rows.try_into()?;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let dst_strides = contiguous_strides_ggml(dims);
        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        let params = VulkanRopeParams {
            rope_mode: mode,
            nrows: rows_u32,
            n_dims: n_dims.try_into()?,
            freq_scale: 1.0,
            freq_base,
            ext_factor: 0.0,
            attn_factor: 1.0,
            corr_dims: [0.0, 0.0],
            theta_scale,
            has_ff: 0,
            sections: [0, 0, 0, 0],
            is_imrope: 0,
            is_back: 0,
            set_rows_stride: 0,
            ne00: dims[0],
            ne01: dims[1],
            ne02: dims[2],
            nb01: strides[1],
            nb02: strides[2],
            nb03: strides[3],
            nb11: dst_strides[1],
            nb12: dst_strides[2],
            nb13: dst_strides[3],
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&pos.buffer),
            VulkanBinding::Storage(&pos.buffer),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&pos.buffer),
        ];
        let spirv_name = match self.dtype {
            DType::F32 => "rope_norm_f32",
            DType::F16 => "rope_norm_f16",
            _ => unreachable!(),
        };
        let spirv = candle_vulkan_kernels::spirv(spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        let wg_y = ((n_dims / 2) as u32).div_ceil(256).max(1);
        let groups_x = rows_u32.min(32768);
        let groups_z = rows_u32.div_ceil(32768).max(1);
        self.device.run_compute_3d(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (groups_x, wg_y, groups_z),
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
        if self.dtype == DType::F16 || alpha.dtype == DType::F16 {
            if self.dtype != DType::F16 || alpha.dtype != DType::F16 {
                return Err(unsupported("rms_norm mixed dtype"));
            }
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let alpha_f32 = alpha.to_dtype(alpha_layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape().clone());
            let alpha_f32_layout = Layout::contiguous(alpha_layout.shape().clone());
            let out_f32 = src_f32.rms_norm(&src_f32_layout, &alpha_f32, &alpha_f32_layout, eps)?;
            return out_f32.to_dtype(&src_f32_layout, DType::F16);
        }
        if self.dtype != DType::F32 || alpha.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan rms_norm").bt());
        }
        if !layout.is_contiguous()
            || layout.start_offset() != 0
            || !alpha_layout.is_contiguous()
            || alpha_layout.start_offset() != 0
        {
            let src_shape = layout.shape().clone();
            let alpha_shape = alpha_layout.shape().clone();
            let mut src = unsafe { self.device.alloc_uninit(&src_shape, self.dtype)? };
            let mut alpha_tmp = unsafe { alpha.device.alloc_uninit(&alpha_shape, alpha.dtype)? };
            self.copy_strided_src(&mut src, 0, layout)?;
            alpha.copy_strided_src(&mut alpha_tmp, 0, alpha_layout)?;
            let src_layout = Layout::contiguous(src_shape);
            let alpha_layout = Layout::contiguous(alpha_shape);
            return src.rms_norm(&src_layout, &alpha_tmp, &alpha_layout, eps);
        }
        let (src_dims, src_strides) = dims4_ggml(layout)?;
        let (alpha_dims, alpha_strides) = dims4_ggml(alpha_layout)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let dst_strides = contiguous_strides_ggml(src_dims);
        let params = GgmlBinaryParams {
            ne: count.try_into()?,
            ne00: src_dims[0],
            ne01: src_dims[1],
            ne02: src_dims[2],
            ne03: src_dims[3],
            nb00: src_strides[0],
            nb01: src_strides[1],
            nb02: src_strides[2],
            nb03: src_strides[3],
            ne10: alpha_dims[0],
            ne11: alpha_dims[1],
            ne12: alpha_dims[2],
            ne13: alpha_dims[3],
            nb10: alpha_strides[0],
            nb11: alpha_strides[1],
            nb12: alpha_strides[2],
            nb13: alpha_strides[3],
            ne20: src_dims[0],
            ne21: src_dims[1],
            ne22: src_dims[2],
            ne23: src_dims[3],
            nb20: dst_strides[0],
            nb21: dst_strides[1],
            nb22: dst_strides[2],
            nb23: dst_strides[3],
            misalign_offsets: 0,
            param1: eps,
            param2: 0.0,
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&alpha.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("rms_norm_f32")
            .ok_or_else(|| Error::Msg("vulkan shader rms_norm_f32 not generated".into()).bt())?;
        let rows = count / layout.dims()[layout.dims().len() - 1];
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (rows.try_into()?, 1, 1),
            Some(&[(1, 1)]),
        )?;
        Ok(dst)
    }

    pub fn sigmoid(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan sigmoid").bt());
        }
        let (spirv, kind) = unary_spirv("sigmoid", self.dtype)?;
        match kind {
            VulkanUnaryKind::Head => self.run_unary_head(layout, spirv),
            VulkanUnaryKind::Generic => self.run_unary_generic(layout, spirv),
        }
    }

    fn run_index_select_f32(
        &self,
        ids: &Self,
        src_l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan index_select").bt());
        }
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype, "vulkan index_select ids").bt());
        }
        if !src_l.is_contiguous() {
            return Err(unsupported("index_select strided"));
        }
        if src_l.start_offset() > u16::MAX as usize || ids_l.start_offset() > u8::MAX as usize {
            return Err(unsupported("index_select offset"));
        }
        let ids_len = match ids_l.dims() {
            [ids_len] => *ids_len,
            _ => return Err(unsupported("index_select ids rank")),
        };
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim = src_l.dims()[dim];
        let mut dst_dims = src_l.dims().to_vec();
        dst_dims[dim] = ids_len;
        let dst_shape = Shape::from(dst_dims);
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };
        let params = GgmlBinaryParams {
            ne: (left_size * ids_len * right_size).try_into()?,
            ne00: right_size.try_into()?,
            ne01: src_dim.try_into()?,
            ne02: left_size.try_into()?,
            ne03: 1,
            nb00: 1,
            nb01: right_size.try_into()?,
            nb02: (src_dim * right_size).try_into()?,
            nb03: (src_dim * right_size * left_size).try_into()?,
            ne10: ids_len.try_into()?,
            ne11: left_size.try_into()?,
            ne12: 1,
            ne13: 1,
            nb10: ids_l.stride()[0].try_into()?,
            nb11: 0,
            nb12: 0,
            nb13: 0,
            ne20: right_size.try_into()?,
            ne21: ids_len.try_into()?,
            ne22: left_size.try_into()?,
            ne23: 1,
            nb20: 1,
            nb21: right_size.try_into()?,
            nb22: (ids_len * right_size).try_into()?,
            nb23: (left_size * ids_len * right_size).try_into()?,
            misalign_offsets: ((src_l.start_offset() as u32) << 16)
                | ((ids_l.start_offset() as u32) << 8),
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&ids.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv_name = match self.dtype {
            DType::F32 => "get_rows_f32_f32",
            DType::F16 => "get_rows_f16",
            _ => unreachable!(),
        };
        let spirv = candle_vulkan_kernels::spirv(spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        self.device.run_compute_3d(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (
                right_size.div_ceil(512).try_into()?,
                ids_len.try_into()?,
                left_size.try_into()?,
            ),
        )?;
        Ok(dst)
    }

    fn run_gather_last_dim_f32(&self, ids: &Self, src_l: &Layout, ids_l: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan gather").bt());
        }
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype, "vulkan gather ids").bt());
        }
        if !src_l.is_contiguous() {
            return Err(unsupported("gather strided src"));
        }
        if !ids_l.is_contiguous() {
            return Err(unsupported("gather strided ids"));
        }
        if src_l.start_offset() > u16::MAX as usize || ids_l.start_offset() > u8::MAX as usize {
            return Err(unsupported("gather offset"));
        }
        let rank = src_l.dims().len();
        if rank == 0 || ids_l.dims().len() != rank {
            return Err(unsupported("gather rank"));
        }
        let ids_dim = ids_l.dims()[rank - 1];
        let left_size: usize = ids_l.dims()[..rank - 1].iter().product();
        let src_dim = src_l.dims()[rank - 1];
        let dst_shape = ids_l.shape().clone();
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };
        let params = GgmlBinaryParams {
            ne: (left_size * ids_dim).try_into()?,
            ne00: 1,
            ne01: src_dim.try_into()?,
            ne02: left_size.try_into()?,
            ne03: 1,
            nb00: 1,
            nb01: 1,
            nb02: src_dim.try_into()?,
            nb03: (src_dim * left_size).try_into()?,
            ne10: ids_dim.try_into()?,
            ne11: left_size.try_into()?,
            ne12: 1,
            ne13: 1,
            nb10: 1,
            nb11: ids_dim.try_into()?,
            nb12: 0,
            nb13: 0,
            ne20: 1,
            ne21: ids_dim.try_into()?,
            ne22: left_size.try_into()?,
            ne23: 1,
            nb20: 1,
            nb21: 1,
            nb22: ids_dim.try_into()?,
            nb23: (left_size * ids_dim).try_into()?,
            misalign_offsets: ((src_l.start_offset() as u32) << 16)
                | ((ids_l.start_offset() as u32) << 8),
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&ids.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv_name = match self.dtype {
            DType::F32 => "get_rows_f32_f32",
            DType::F16 => "get_rows_f16",
            _ => unreachable!(),
        };
        let spirv = candle_vulkan_kernels::spirv(spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        self.device.run_compute_3d(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (1, ids_dim.try_into()?, left_size.try_into()?),
        )?;
        Ok(dst)
    }

    fn run_scatter_set_last_dim_f32(
        &mut self,
        dst_l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
    ) -> Result<()> {
        if (self.dtype != DType::F32 && self.dtype != DType::F16) || self.dtype != src.dtype {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan scatter_set").bt());
        }
        if self.dtype == DType::F16 {
            let mut dst_f32 = self.to_dtype(dst_l, DType::F32)?;
            let src_f32 = src.to_dtype(src_l, DType::F32)?;
            let dst_f32_layout = Layout::contiguous(dst_l.shape().clone());
            let src_f32_layout = Layout::contiguous(src_l.shape().clone());
            dst_f32.run_scatter_set_last_dim_f32(
                &dst_f32_layout,
                ids,
                ids_l,
                &src_f32,
                &src_f32_layout,
            )?;
            *self = dst_f32.to_dtype(&dst_f32_layout, DType::F16)?;
            return Ok(());
        }
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype, "vulkan scatter_set ids").bt());
        }
        if !dst_l.is_contiguous() {
            return Err(unsupported("scatter_set strided dst"));
        }
        if !src_l.is_contiguous() {
            return Err(unsupported("scatter_set strided src"));
        }
        if !ids_l.is_contiguous() {
            return Err(unsupported("scatter_set strided ids"));
        }
        if src_l.start_offset() > u16::MAX as usize
            || ids_l.start_offset() > u8::MAX as usize
            || dst_l.start_offset() > u8::MAX as usize
        {
            return Err(unsupported("scatter_set offset"));
        }
        let rank = dst_l.dims().len();
        if rank == 0 || ids_l.dims().len() != rank || src_l.dims().len() != rank {
            return Err(unsupported("scatter_set rank"));
        }
        let ids_dim = ids_l.dims()[rank - 1];
        let left_size: usize = ids_l.dims()[..rank - 1].iter().product();
        let dst_dim = dst_l.dims()[rank - 1];
        let params = GgmlBinaryParams {
            ne: (left_size * ids_dim).try_into()?,
            ne00: 1,
            ne01: ids_dim.try_into()?,
            ne02: left_size.try_into()?,
            ne03: 1,
            nb00: 1,
            nb01: 1,
            nb02: ids_dim.try_into()?,
            nb03: (ids_dim * left_size).try_into()?,
            ne10: ids_dim.try_into()?,
            ne11: left_size.try_into()?,
            ne12: 1,
            ne13: 1,
            nb10: 1,
            nb11: ids_dim.try_into()?,
            nb12: 0,
            nb13: 0,
            ne20: 1,
            ne21: dst_dim.try_into()?,
            ne22: left_size.try_into()?,
            ne23: 1,
            nb20: 1,
            nb21: 1,
            nb22: dst_dim.try_into()?,
            nb23: (dst_dim * left_size).try_into()?,
            misalign_offsets: ((src_l.start_offset() as u32) << 16)
                | ((ids_l.start_offset() as u32) << 8)
                | dst_l.start_offset() as u32,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&src.buffer),
            VulkanBinding::Storage(&ids.buffer),
            VulkanBinding::Storage(&self.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("set_rows_f32_i32").ok_or_else(|| {
            Error::Msg("vulkan shader set_rows_f32_i32 not generated".into()).bt()
        })?;
        let rows: u32 = (left_size * ids_dim).try_into()?;
        self.device.run_compute(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            rows.div_ceil(512),
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
        if self.dtype != DType::F32 || self.dtype != src.dtype {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan scatter_add").bt());
        }
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype, "vulkan scatter_add ids").bt());
        }
        if !dst_l.is_contiguous() {
            return Err(unsupported("scatter_add strided dst"));
        }
        if !src_l.is_contiguous() {
            return Err(unsupported("scatter_add strided src"));
        }
        if !ids_l.is_contiguous() {
            return Err(unsupported("scatter_add strided ids"));
        }
        if src_l.start_offset() > u16::MAX as usize
            || ids_l.start_offset() > u8::MAX as usize
            || dst_l.start_offset() > u8::MAX as usize
        {
            return Err(unsupported("scatter_add offset"));
        }
        let rank = dst_l.dims().len();
        if rank == 0 || src_l.dims().len() != rank {
            return Err(unsupported("scatter_add rank"));
        }
        let ids_dim = src_l.dims()[rank - 1];
        let left_size: usize = src_l.dims()[..rank - 1].iter().product();
        // `scatter_add` passes ids with the same shape as `src`; `index_add`
        // passes one rank-1 id row shared by every leading row. The shader
        // reads ids through the `nb11` row stride, so the broadcast case is
        // simply a zero row stride.
        let ids_row_stride: usize = if ids_l.dims().len() == rank {
            if ids_l.dims() != src_l.dims() {
                return Err(unsupported("scatter_add ids shape"));
            }
            ids_dim
        } else if ids_l.dims() == [ids_dim] {
            0
        } else {
            return Err(unsupported("scatter_add rank"));
        };
        let dst_dim = dst_l.dims()[rank - 1];
        let params = GgmlBinaryParams {
            ne: (left_size * ids_dim).try_into()?,
            ne00: 1,
            ne01: ids_dim.try_into()?,
            ne02: left_size.try_into()?,
            ne03: 1,
            nb00: 1,
            nb01: 1,
            nb02: ids_dim.try_into()?,
            nb03: (ids_dim * left_size).try_into()?,
            ne10: ids_dim.try_into()?,
            ne11: left_size.try_into()?,
            ne12: 1,
            ne13: 1,
            nb10: 1,
            nb11: ids_row_stride.try_into()?,
            nb12: 0,
            nb13: 0,
            ne20: 1,
            ne21: dst_dim.try_into()?,
            ne22: left_size.try_into()?,
            ne23: 1,
            nb20: 1,
            nb21: 1,
            nb22: dst_dim.try_into()?,
            nb23: (dst_dim * left_size).try_into()?,
            misalign_offsets: ((src_l.start_offset() as u32) << 16)
                | ((ids_l.start_offset() as u32) << 8)
                | dst_l.start_offset() as u32,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&src.buffer),
            VulkanBinding::Storage(&ids.buffer),
            VulkanBinding::Storage(&self.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("set_rows_add_f32_i32").ok_or_else(|| {
            Error::Msg("vulkan shader set_rows_add_f32_i32 not generated".into()).bt()
        })?;
        let rows: u32 = (left_size * ids_dim).try_into()?;
        self.device.run_compute(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            rows.div_ceil(512),
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
            return Err(unsupported("matmul mixed dtype"));
        }
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan matmul").bt());
        }
        let rank = lhs_l.dims().len();
        if rank != rhs_l.dims().len() || !(2..=4).contains(&rank) {
            return Err(unsupported("matmul rank"));
        }
        if b != lhs_l.dims()[..rank - 2].iter().product::<usize>() {
            return Err(unsupported("matmul batch"));
        }
        if self.dtype == DType::F16 {
            // Keep f16 storage compatibility while executing numerically stable f32 matmul.
            let lhs_f32 = self.to_dtype(lhs_l, DType::F32)?;
            let rhs_f32 = rhs.to_dtype(rhs_l, DType::F32)?;
            let lhs_f32_l = Layout::contiguous(lhs_l.shape().clone());
            let rhs_f32_l = Layout::contiguous(rhs_l.shape().clone());
            let out_f32 = lhs_f32.run_matmul_f32(&rhs_f32, (b, m, n, k), &lhs_f32_l, &rhs_f32_l)?;
            let out_l = Layout::contiguous(Shape::from(vec![b, m, n]));
            return out_f32.to_dtype(&out_l, DType::F16);
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

        let rhs_t_src_layout = rhs_l.transpose(rank - 2, rank - 1)?;
        let rhs_t_contiguous =
            if rhs_t_src_layout.is_contiguous() && rhs_t_src_layout.start_offset() == 0 {
                None
            } else {
                let rhs_t_shape = rhs_t_src_layout.shape().clone();
                let mut tmp = unsafe { rhs.device.alloc_uninit(&rhs_t_shape, rhs.dtype)? };
                rhs.copy_strided_src(&mut tmp, 0, &rhs_t_src_layout)?;
                Some(tmp)
            };
        let (rhs_t, rhs_t_layout) = if let Some(rhs_t_contiguous) = rhs_t_contiguous.as_ref() {
            (
                rhs_t_contiguous,
                Layout::contiguous(rhs_t_src_layout.shape().clone()),
            )
        } else {
            (rhs, rhs_t_src_layout.clone())
        };

        let lhs_stride = lhs_layout.stride();
        let dst_shape = Shape::from(vec![b, m, n]);
        if rank == 2 && self.dtype == DType::F32 && m == 1 && m <= VULKAN_MUL_MAT_VEC_MAX_COLS {
            let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::F32)? };
            let params = VulkanMatVecParams {
                ncols: k.try_into()?,
                stride_a: k.try_into()?,
                stride_b: lhs_stride[rank - 2].try_into()?,
                stride_d: n.try_into()?,
                batch_stride_a: 0,
                batch_stride_b: k.try_into()?,
                batch_stride_d: n.try_into()?,
                fusion_flags: 0,
                base_work_group_y: 0,
                ne02: 1,
                ne12: 1,
                broadcast2: 1,
                broadcast3: 1,
            };
            let bindings = [
                VulkanBinding::Storage(&rhs_t.buffer),
                VulkanBinding::Storage(&lhs.buffer),
                VulkanBinding::Storage(&dst.buffer),
                VulkanBinding::Storage(&dst.buffer),
                VulkanBinding::Storage(&dst.buffer),
            ];
            let dmmv_workgroup = vulkan_dmmv_workgroup(&self.device, GgmlDType::F32, n, k, false);
            let spirv_name = vulkan_dmmv_shader_name(
                &self.device,
                GgmlDType::F32,
                "mul_mat_vec_f32_f32_f32".to_string(),
                dmmv_workgroup,
            );
            let spirv = candle_vulkan_kernels::spirv(&spirv_name).ok_or_else(|| {
                Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt()
            })?;
            let block_size = vulkan_dmmv_block_size(&self.device, GgmlDType::F32, dmmv_workgroup);
            let rows_per_group = 1u32;
            let spec = [(0, block_size), (1, rows_per_group), (2, m.try_into()?)];
            let use_dmmv_subgroups = vulkan_dmmv_use_subgroups(&self.device, GgmlDType::F32);
            let required_subgroup_size = if use_dmmv_subgroups {
                Some(vulkan_dmmv_subgroup_size(&self.device))
            } else {
                None
            };
            self.device.run_compute_specialized_with_options(
                spirv,
                &bindings,
                Some(any_as_bytes(&params)),
                (n.div_ceil(rows_per_group as usize).try_into()?, 1, 1),
                Some(&spec),
                use_dmmv_subgroups,
                required_subgroup_size,
            )?;
            drop(lhs_contiguous);
            return Ok(dst);
        }
        if rank == 2
            && self.dtype == DType::F32
            && m > 1
            && !vulkan_dense_gemm_prefers_tiled(m, n, k)
        {
            let dst = lhs.run_rank2_matmul_f32_via_batched_matvec(rhs_t, m, n, k)?;
            drop(lhs_contiguous);
            return Ok(dst);
        }
        if rank > 2
            && self.dtype == DType::F32
            && lhs_l.dims()[..rank - 2] == rhs_l.dims()[..rank - 2]
            && !vulkan_dense_gemm_prefers_tiled(m, n, k)
        {
            let dst = lhs.run_batched_matmul_f32_via_rank2_batches(rhs_t, b, m, n, k)?;
            drop(lhs_contiguous);
            return Ok(dst);
        }
        let rhs_t_stride = rhs_t_layout.stride();
        let bs02 = if rank >= 3 {
            lhs_layout.dims()[rank - 3]
        } else {
            1
        };
        let lhs_stride_batch_inner = if rank >= 3 {
            lhs_stride[rank - 3]
        } else {
            m * k
        };
        let rhs_stride_batch_inner = if rank >= 3 {
            rhs_t_stride[rank - 3]
        } else {
            n * k
        };

        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };
        let params = VulkanMatmulParams {
            m: n.try_into()?,
            n: m.try_into()?,
            k: k.try_into()?,
            stride_a: k.try_into()?,
            stride_b: lhs_stride[rank - 2].try_into()?,
            stride_d: n.try_into()?,
            batch_stride_a: rhs_stride_batch_inner.try_into()?,
            batch_stride_b: lhs_stride_batch_inner.try_into()?,
            batch_stride_d: (m * n).try_into()?,
            base_work_group_z: 0,
            num_batches: b.try_into()?,
            k_split: k.try_into()?,
            ne02: bs02.try_into()?,
            ne12: bs02.try_into()?,
            broadcast2: 1,
            broadcast3: 1,
            padded_n: m.try_into()?,
        };
        let bindings = [
            VulkanBinding::Storage(&rhs_t.buffer),
            VulkanBinding::Storage(&lhs.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv_name = match self.dtype {
            // ggml shader naming: the un-suffixed `matmul_f32_f32` variant
            // stages tiles as f16, which loses ~1e-3 precision against the
            // CPU/CUDA f32 reference. `_fp32` keeps full f32 staging and
            // accumulation, matching Candle's parity tolerances.
            DType::F32 => "matmul_f32_f32_fp32",
            _ => unreachable!(),
        };
        let spirv = candle_vulkan_kernels::spirv(spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        // ggml `m_warptile` layout: {BLOCK_SIZE, BM, BN, (BK), WM, WN, WMITER,
        // TM, TN, (TK), WARP}. The thread count must satisfy
        // `BLOCK_SIZE / WARP == (BM / WM) * (BN / WN)` so the warp grid covers
        // the whole BM x BN output tile; with BM=BN=64 and WM=WN=32 that means
        // four warps, i.e. BLOCK_SIZE = 4 * WARP.
        let warp = self.device.inner.subgroup_size.max(1);
        let spec = [
            (0, warp * 4),
            (1, 64),
            (2, 64),
            (4, 32),
            (5, 32),
            (6, 2),
            (7, 4),
            (8, 2),
            (10, warp),
        ];
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (
                n.div_ceil(64).try_into()?,
                m.div_ceil(64).try_into()?,
                b.try_into()?,
            ),
            Some(&spec),
        )?;
        drop(lhs_contiguous);
        Ok(dst)
    }

    fn run_im2col_conv1d_f32(
        &self,
        layout: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan im2col conv1d").bt());
        }
        let src_stride = layout.stride();
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
        let mut input_contiguous = None;
        let input = if input_l.is_contiguous() && input_l.start_offset() == 0 {
            self
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(input_l.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, &input_l)?;
            input_contiguous = Some(tmp);
            input_contiguous.as_ref().unwrap()
        };
        let l_out = params.l_out();
        let chw = params.c_in * params.k_size;
        let out_shape = Shape::from((params.b_size * l_out, chw));
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let push = VulkanIm2ColParams {
            dst_addr: [0, 0],
            batch_offset: (params.c_in * params.l_in).try_into()?,
            offset_delta: params.l_in.try_into()?,
            ic: params.c_in.try_into()?,
            iw: params.l_in.try_into()?,
            ih: 1,
            ow: l_out.try_into()?,
            oh: 1,
            kw: params.k_size.try_into()?,
            kh: 1,
            oh_batch: params.b_size.try_into()?,
            chw: chw.try_into()?,
            s0: params.stride.try_into()?,
            s1: 1,
            p0: params.padding.try_into()?,
            p1: 0,
            d0: params.dilation.try_into()?,
            d1: 1,
            batch_ic: (params.b_size * params.c_in).try_into()?,
        };
        let bindings = [
            VulkanBinding::Storage(&input.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("im2col_f32")
            .ok_or_else(|| Error::Msg("vulkan shader im2col_f32 not generated".into()).bt())?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&push)),
            (
                chw.div_ceil(512).try_into()?,
                l_out.try_into()?,
                params.b_size.try_into()?,
            ),
            Some(&[(0, 32)]),
        )?;
        drop(input_contiguous);
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan conv1d").bt());
        }
        let col = self.run_im2col_conv1d_f32(layout, params)?;
        let b = params.b_size;
        let n = params.c_out;
        let l_out = params.l_out();
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b * m, k));

        let mut kernel_contiguous = None;
        let (kernel_matmul, kernel_l_mm) = if kernel_l.is_contiguous()
            && kernel_l.start_offset() == 0
        {
            (
                kernel,
                Layout::contiguous_with_offset((n, k), 0).transpose(0, 1)?,
            )
        } else {
            let mut tmp = unsafe { kernel.device.alloc_uninit(kernel_l.shape(), kernel.dtype)? };
            kernel.copy_strided_src(&mut tmp, 0, kernel_l)?;
            kernel_contiguous = Some(tmp);
            (
                kernel_contiguous.as_ref().unwrap(),
                Layout::contiguous_with_offset((n, k), 0).transpose(0, 1)?,
            )
        };

        let res = col.matmul(kernel_matmul, (1, b * m, n, k), &col_l, &kernel_l_mm)?;
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut out = unsafe {
            self.device
                .alloc_uninit(&Shape::from(params.out_dims()), res.dtype)?
        };
        res.copy_strided_src(&mut out, 0, &res_l)?;
        drop(kernel_contiguous);
        Ok(out)
    }

    fn run_im2col_f32(&self, layout: &Layout, params: &crate::conv::ParamsConv2D) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan im2col").bt());
        }
        let mut input_contiguous = None;
        let input = if layout.is_contiguous() && layout.start_offset() == 0 {
            self
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            input_contiguous = Some(tmp);
            input_contiguous.as_ref().unwrap()
        };
        let h_out = params.out_h();
        let w_out = params.out_w();
        let chw = params.c_in * params.k_h * params.k_w;
        let out_shape = Shape::from((params.b_size * h_out * w_out, chw));
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let push = VulkanIm2ColParams {
            dst_addr: [0, 0],
            batch_offset: (params.c_in * params.i_h * params.i_w).try_into()?,
            offset_delta: (params.i_h * params.i_w).try_into()?,
            ic: params.c_in.try_into()?,
            iw: params.i_w.try_into()?,
            ih: params.i_h.try_into()?,
            ow: w_out.try_into()?,
            oh: h_out.try_into()?,
            kw: params.k_w.try_into()?,
            kh: params.k_h.try_into()?,
            oh_batch: (params.b_size * h_out).try_into()?,
            chw: chw.try_into()?,
            s0: params.stride.try_into()?,
            s1: params.stride.try_into()?,
            p0: params.padding.try_into()?,
            p1: params.padding.try_into()?,
            d0: params.dilation.try_into()?,
            d1: params.dilation.try_into()?,
            batch_ic: (params.b_size * params.c_in).try_into()?,
        };
        let bindings = [
            VulkanBinding::Storage(&input.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("im2col_f32")
            .ok_or_else(|| Error::Msg("vulkan shader im2col_f32 not generated".into()).bt())?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&push)),
            (
                chw.div_ceil(512).try_into()?,
                w_out.try_into()?,
                (params.b_size * h_out).try_into()?,
            ),
            Some(&[(0, 32)]),
        )?;
        drop(input_contiguous);
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan conv2d").bt());
        }
        let col = self.run_im2col_f32(layout, params)?;
        let h_out = params.out_h();
        let w_out = params.out_w();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_h * params.k_w * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b * m, k));

        let mut kernel_contiguous = None;
        let (kernel_matmul, kernel_l_mm) = if kernel_l.is_contiguous()
            && kernel_l.start_offset() == 0
        {
            (
                kernel,
                Layout::contiguous_with_offset((n, k), 0).transpose(0, 1)?,
            )
        } else {
            let mut tmp = unsafe { kernel.device.alloc_uninit(kernel_l.shape(), kernel.dtype)? };
            kernel.copy_strided_src(&mut tmp, 0, kernel_l)?;
            kernel_contiguous = Some(tmp);
            (
                kernel_contiguous.as_ref().unwrap(),
                Layout::contiguous_with_offset((n, k), 0).transpose(0, 1)?,
            )
        };

        let res = col.matmul(kernel_matmul, (1, b * m, n, k), &col_l, &kernel_l_mm)?;
        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut out = unsafe { self.device.alloc_uninit(res_l.shape(), res.dtype)? };
        res.copy_strided_src(&mut out, 0, &res_l)?;
        drop(kernel_contiguous);
        Ok(out)
    }

    fn run_conv_transpose1d_f32(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan conv_transpose1d").bt());
        }

        let mut input_contiguous = None;
        let (input, input_l) = if layout.is_contiguous() && layout.start_offset() == 0 {
            (self, layout.clone())
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            input_contiguous = Some(tmp);
            (
                input_contiguous.as_ref().unwrap(),
                Layout::contiguous(layout.shape().clone()),
            )
        };

        let mut kernel_contiguous = None;
        let (kernel, kernel_l) = if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
            (kernel, kernel_l.clone())
        } else {
            let mut tmp = unsafe { kernel.device.alloc_uninit(kernel_l.shape(), kernel.dtype)? };
            kernel.copy_strided_src(&mut tmp, 0, kernel_l)?;
            kernel_contiguous = Some(tmp);
            (
                kernel_contiguous.as_ref().unwrap(),
                Layout::contiguous(kernel_l.shape().clone()),
            )
        };

        let l_out = params.l_out();
        let out_shape = Shape::from(params.out_dims());
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let input_stride = input_l.stride();
        let kernel_stride = kernel_l.stride();
        let (owmp, owl) = fastdiv_values(l_out.try_into()?);
        let (owohmp, owohl) = fastdiv_values(l_out.try_into()?);
        let push = VulkanConv2dParams {
            cout: params.c_out.try_into()?,
            cin: params.c_in.try_into()?,
            n: params.b_size.try_into()?,
            w: params.l_in.try_into()?,
            h: 1,
            ow: l_out.try_into()?,
            oh: 1,
            nb01: params.k_size.try_into()?,
            nb02: kernel_stride[1].try_into()?,
            nb03: kernel_stride[0].try_into()?,
            nb11: params.l_in.try_into()?,
            nb12: input_stride[1].try_into()?,
            nb13: input_stride[0].try_into()?,
            nb1: l_out.try_into()?,
            nb2: l_out.try_into()?,
            nb3: (params.c_out * l_out).try_into()?,
            owmp,
            owl,
            owohmp,
            owohl,
        };
        let bindings = [
            VulkanBinding::Storage(&kernel.buffer),
            VulkanBinding::Storage(&input.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("conv_transpose_2d_f32_unroll")
            .or_else(|| candle_vulkan_kernels::spirv("conv_transpose_2d_f32"))
            .ok_or_else(|| {
                Error::Msg("vulkan shader conv_transpose_2d_f32 not generated".into()).bt()
            })?;
        let bs_k = 128u32;
        let bs_crs = 16u32;
        let bs_npq = 128u32;
        let npq = params.b_size * l_out;
        let nb_npq = (npq as u32).div_ceil(bs_npq);
        let wg_y = nb_npq.min(512);
        let wg_z = nb_npq.div_ceil(512).max(1);
        let spec = [
            (0, 256),
            (1, bs_k),
            (2, bs_crs),
            (3, bs_npq),
            (4, 8),
            (5, 0),
            (6, 4),
            (7, params.stride.try_into()?),
            (8, 1),
            (9, params.padding.try_into()?),
            (10, 0),
            (11, params.dilation.try_into()?),
            (12, 1),
            (13, params.k_size.try_into()?),
            (14, 1),
        ];
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&push)),
            (params.c_out.div_ceil(bs_k as usize).try_into()?, wg_y, wg_z),
            Some(&spec),
        )?;
        drop(input_contiguous);
        drop(kernel_contiguous);
        Ok(dst)
    }

    fn run_conv_transpose2d_f32(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || kernel.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan conv_transpose2d").bt());
        }

        let mut input_contiguous = None;
        let (input, input_l) = if layout.is_contiguous() && layout.start_offset() == 0 {
            (self, layout.clone())
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            input_contiguous = Some(tmp);
            (
                input_contiguous.as_ref().unwrap(),
                Layout::contiguous(layout.shape().clone()),
            )
        };

        let mut kernel_contiguous = None;
        let (kernel, kernel_l) = if kernel_l.is_contiguous() && kernel_l.start_offset() == 0 {
            (kernel, kernel_l.clone())
        } else {
            let mut tmp = unsafe { kernel.device.alloc_uninit(kernel_l.shape(), kernel.dtype)? };
            kernel.copy_strided_src(&mut tmp, 0, kernel_l)?;
            kernel_contiguous = Some(tmp);
            (
                kernel_contiguous.as_ref().unwrap(),
                Layout::contiguous(kernel_l.shape().clone()),
            )
        };

        let out_h = params.out_h();
        let out_w = params.out_w();
        let out_shape = Shape::from(params.out_dims());
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let input_stride = input_l.stride();
        let kernel_stride = kernel_l.stride();
        let (owmp, owl) = fastdiv_values(out_w.try_into()?);
        let (owohmp, owohl) = fastdiv_values((out_w * out_h).try_into()?);
        let push = VulkanConv2dParams {
            cout: params.c_out.try_into()?,
            cin: params.c_in.try_into()?,
            n: params.b_size.try_into()?,
            w: params.i_w.try_into()?,
            h: params.i_h.try_into()?,
            ow: out_w.try_into()?,
            oh: out_h.try_into()?,
            nb01: kernel_stride[2].try_into()?,
            nb02: kernel_stride[1].try_into()?,
            nb03: kernel_stride[0].try_into()?,
            nb11: input_stride[2].try_into()?,
            nb12: input_stride[1].try_into()?,
            nb13: input_stride[0].try_into()?,
            nb1: out_w.try_into()?,
            nb2: (out_h * out_w).try_into()?,
            nb3: (params.c_out * out_h * out_w).try_into()?,
            owmp,
            owl,
            owohmp,
            owohl,
        };
        let bindings = [
            VulkanBinding::Storage(&kernel.buffer),
            VulkanBinding::Storage(&input.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("conv_transpose_2d_f32_unroll")
            .or_else(|| candle_vulkan_kernels::spirv("conv_transpose_2d_f32"))
            .ok_or_else(|| {
                Error::Msg("vulkan shader conv_transpose_2d_f32 not generated".into()).bt()
            })?;
        let bs_k = 128u32;
        let bs_crs = 16u32;
        let bs_npq = 128u32;
        let npq = params.b_size * out_h * out_w;
        let nb_npq = (npq as u32).div_ceil(bs_npq);
        let wg_y = nb_npq.min(512);
        let wg_z = nb_npq.div_ceil(512).max(1);
        let spec = [
            (0, 256),
            (1, bs_k),
            (2, bs_crs),
            (3, bs_npq),
            (4, 8),
            (5, 0),
            (6, 4),
            (7, params.stride.try_into()?),
            (8, params.stride.try_into()?),
            (9, params.padding.try_into()?),
            (10, params.padding.try_into()?),
            (11, params.dilation.try_into()?),
            (12, params.dilation.try_into()?),
            (13, params.k_w.try_into()?),
            (14, params.k_h.try_into()?),
        ];
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&push)),
            (params.c_out.div_ceil(bs_k as usize).try_into()?, wg_y, wg_z),
            Some(&spec),
        )?;
        drop(input_contiguous);
        drop(kernel_contiguous);
        Ok(dst)
    }

    fn run_upsample_nearest1d_f32(&self, layout: &Layout, out_l: usize) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan upsample_nearest1d").bt());
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan upsample2d").bt());
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

    fn run_pool2d_f32(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        max_pool: bool,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan pool2d").bt());
        }
        let (b, c, h, w) = layout.shape().dims4()?;
        let out_h = (h - kernel_size.0) / stride.0 + 1;
        let out_w = (w - kernel_size.1) / stride.1 + 1;
        let mut input_contiguous = None;
        let input = if layout.is_contiguous() && layout.start_offset() == 0 {
            self
        } else {
            let mut tmp = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
            self.copy_strided_src(&mut tmp, 0, layout)?;
            input_contiguous = Some(tmp);
            input_contiguous.as_ref().unwrap()
        };
        let out_shape = Shape::from(vec![b, c, out_h, out_w]);
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };
        let push = VulkanPool2dParams {
            iw: w.try_into()?,
            ih: h.try_into()?,
            ow: out_w.try_into()?,
            oh: out_h.try_into()?,
            oc: c.try_into()?,
            pelements: out_shape.elem_count().try_into()?,
            op: if max_pool { 0 } else { 1 },
            k0: kernel_size.0.try_into()?,
            k1: kernel_size.1.try_into()?,
            s0: stride.0.try_into()?,
            s1: stride.1.try_into()?,
            p0: 0,
            p1: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&input.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv = candle_vulkan_kernels::spirv("pool2d_f32")
            .ok_or_else(|| Error::Msg("vulkan shader pool2d_f32 not generated".into()).bt())?;
        self.device.run_compute(
            spirv,
            &bindings,
            Some(any_as_bytes(&push)),
            (out_shape.elem_count() as u32).div_ceil(512),
        )?;
        drop(input_contiguous);
        Ok(dst)
    }

    pub(crate) fn quantized_index_select_f32(
        &self,
        qdtype: GgmlDType,
        src_shape: &Shape,
        ids: &Self,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(
                ids.dtype,
                "vulkan quantized index_select ids",
            )
            .bt());
        }
        if !ids_l.is_contiguous() {
            return Err(unsupported("quantized index_select ids strided"));
        }
        let ids_len = match ids_l.dims() {
            [ids_len] => *ids_len,
            _ => return Err(unsupported("quantized index_select ids rank")),
        };
        let dims = src_shape.dims();
        if dim >= dims.len() {
            crate::bail!("index_select dim {dim} out of range for {src_shape:?}")
        }
        if qdtype == GgmlDType::Q8_1 {
            let repacked = repack_q8_1_storage_to_q8_0(&self.device, self, src_shape.elem_count())?;
            return repacked.quantized_index_select_f32(
                GgmlDType::Q8_0,
                src_shape,
                ids,
                ids_l,
                dim,
            );
        }
        let block_size = qdtype.block_size();
        let right_size: usize = dims[dim + 1..].iter().product();
        if !right_size.is_multiple_of(block_size) {
            crate::bail!(
                "vulkan quantized index_select requires block-aligned rows, got right_size={right_size}, block_size={block_size}"
            )
        }
        let left_size: usize = dims[..dim].iter().product();
        let src_dim = dims[dim];
        let mut dst_dims = dims.to_vec();
        dst_dims[dim] = ids_len;
        let dst_shape = Shape::from(dst_dims);
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, DType::F32)? };
        let params = GgmlBinaryParams {
            ne: (left_size * ids_len * right_size).try_into()?,
            ne00: right_size.try_into()?,
            ne01: src_dim.try_into()?,
            ne02: left_size.try_into()?,
            ne03: 1,
            nb00: 1,
            nb01: (right_size / block_size).try_into()?,
            nb02: (src_dim * right_size / block_size).try_into()?,
            nb03: (src_dim * right_size * left_size / block_size).try_into()?,
            ne10: ids_len.try_into()?,
            ne11: left_size.try_into()?,
            ne12: 1,
            ne13: 1,
            nb10: ids_l.stride()[0].try_into()?,
            nb11: 0,
            nb12: 0,
            nb13: 0,
            ne20: right_size.try_into()?,
            ne21: ids_len.try_into()?,
            ne22: left_size.try_into()?,
            ne23: 1,
            nb20: 1,
            nb21: right_size.try_into()?,
            nb22: (ids_len * right_size).try_into()?,
            nb23: (left_size * ids_len * right_size).try_into()?,
            misalign_offsets: 0,
            param1: 0.0,
            param2: 0.0,
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(&ids.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let spirv_name = format!("get_rows_{}_f32", vulkan_quantized_stem(qdtype)?);
        let spirv = candle_vulkan_kernels::spirv(&spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        self.device.run_compute_3d(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (
                right_size.div_ceil(1024).try_into()?,
                ids_len.try_into()?,
                left_size.try_into()?,
            ),
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
        if storage.dtype != DType::F32 && storage.dtype != DType::F16 {
            return Err(
                Error::UnsupportedDTypeForOp(storage.dtype, "vulkan quantized matmul").bt(),
            );
        }
        let rank = layout.dims().len();
        if !(2..=4).contains(&rank) {
            return Err(unsupported("quantized matmul rank"));
        }
        let (n, k) = qshape.dims2()?;
        let input_m = layout.dims()[rank - 2];
        let last_k = layout.dims()[rank - 1];
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with quantized weights {qshape:?}")
        }
        if qdtype == GgmlDType::Q8_1 {
            let repacked = repack_q8_1_storage_to_q8_0(&self.device, self, qshape.elem_count())?;
            return repacked.quantized_matmul(GgmlDType::Q8_0, qshape, storage, layout);
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
        if input_m <= VULKAN_MUL_MAT_VEC_MAX_COLS
            && rank == 2
            && vulkan_supports_quantized_matvec_weight(qdtype)
        {
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
        let src_elem_count = src_layout.shape().elem_count();
        let q8_1_rhs_spirv_name = if src.dtype == DType::F32
            && self.device.inner.integer_dot_product
            && src_elem_count % 4 == 0
        {
            vulkan_q8_1_rhs_matmul_shader_name(qdtype)?
        } else {
            None
        };
        let use_q8_1_rhs = q8_1_rhs_spirv_name.is_some();

        let dst = unsafe { storage.device.alloc_uninit(&dst_shape, DType::F32)? };
        let params = VulkanMatmulParams {
            m: n.try_into()?,
            n: input_m.try_into()?,
            k: k.try_into()?,
            stride_a: k.try_into()?,
            stride_b: src_stride[rank - 2].try_into()?,
            stride_d: n.try_into()?,
            batch_stride_a: 0,
            batch_stride_b: src_stride_batch_inner.try_into()?,
            batch_stride_d: (input_m * n).try_into()?,
            base_work_group_z: 0,
            num_batches: batch_count.try_into()?,
            k_split: k.try_into()?,
            ne02: 1,
            ne12: batch_inner.try_into()?,
            broadcast2: batch_inner.try_into()?,
            broadcast3: batch_outer.try_into()?,
            padded_n: input_m.try_into()?,
        };
        let q8_1_rhs = if use_q8_1_rhs {
            Some(quantize_f32_storage_to_q8_1_x4(
                &self.device,
                src,
                src_elem_count,
            )?)
        } else {
            None
        };
        let rhs_binding = q8_1_rhs.as_ref().unwrap_or(&src.buffer);
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(rhs_binding),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let rhs_suffix = if use_q8_1_rhs {
            "q8_1"
        } else {
            match src.dtype {
                DType::F32 => "f32",
                DType::F16 => "f16",
                other => {
                    return Err(Error::UnsupportedDTypeForOp(other, "vulkan quantized matmul").bt())
                }
            }
        };
        let spirv_name = if let Some(spirv_name) = q8_1_rhs_spirv_name {
            spirv_name
        } else {
            format!("matmul_{}_{}", vulkan_quantized_stem(qdtype)?, rhs_suffix)
        };
        let spirv = candle_vulkan_kernels::spirv(&spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        let spec = if use_q8_1_rhs {
            let subgroup = self.device.inner.subgroup_size.max(8);
            let subgroup32 = self.device.inner.subgroup_size.max(32);
            let small = input_m <= 32 || n <= 32;
            if vulkan_is_k_quant(qdtype) && small {
                [
                    (0, subgroup32),
                    (1, 32),
                    (2, 32),
                    (4, 32),
                    (5, 32),
                    (6, 1),
                    (7, 2),
                    (8, 1),
                    (10, subgroup),
                ]
            } else if vulkan_is_k_quant(qdtype) {
                [
                    (0, 128),
                    (1, 64),
                    (2, 64),
                    (4, subgroup),
                    (5, 32),
                    (6, 1),
                    (7, 2),
                    (8, 2),
                    (10, subgroup),
                ]
            } else if small {
                [
                    (0, subgroup32),
                    (1, 32),
                    (2, 32),
                    (4, 32),
                    (5, 32),
                    (6, 2),
                    (7, 2),
                    (8, 1),
                    (10, subgroup),
                ]
            } else {
                [
                    (0, 128),
                    (1, 64),
                    (2, 64),
                    (4, subgroup),
                    (5, 32),
                    (6, 2),
                    (7, 2),
                    (8, 2),
                    (10, subgroup),
                ]
            }
        } else {
            [
                (0, 64),
                (1, 64),
                (2, 64),
                (4, 32),
                (5, 32),
                (6, 2),
                (7, 4),
                (8, 2),
                (10, self.device.inner.subgroup_size.max(1)),
            ]
        };
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (n.try_into()?, input_m.try_into()?, batch_count.try_into()?),
            Some(&spec),
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
        let rank = layout.dims().len();
        let (n, k) = qshape.dims2()?;
        let input_m = layout.dims()[rank - 2];
        if input_m == 0 || input_m > VULKAN_MUL_MAT_VEC_MAX_COLS {
            crate::bail!(
                "vulkan quantized matvec expects 1 <= input_m <= {VULKAN_MUL_MAT_VEC_MAX_COLS}, got {input_m}"
            );
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
        if input_m > 1 && rank != 2 {
            crate::bail!(
                "vulkan quantized matvec multi-column expects rank-2 input, got rank {rank}"
            );
        }
        let batch_n = input_m > 1;
        let batch_count = if batch_n {
            input_m
        } else {
            batch_inner * batch_outer
        };
        let dispatch_y = if batch_n { 1 } else { batch_count };
        let src_elem_count = src_layout.shape().elem_count();
        let q8_1_rhs_shader = if src.dtype == DType::F32
            && self.device.inner.integer_dot_product
            && src_elem_count % 4 == 0
        {
            vulkan_q8_1_rhs_matvec_shader_name(&self.device, qdtype, input_m, k, false)?
        } else {
            None
        };
        let use_q8_1_rhs = q8_1_rhs_shader.is_some();
        let mut dst_dims = src_layout.dims().to_vec();
        dst_dims.pop();
        dst_dims.push(n);
        let dst_shape = Shape::from(dst_dims);
        let dst = unsafe { storage.device.alloc_uninit(&dst_shape, DType::F32)? };
        let params = VulkanMatVecParams {
            ncols: k.try_into()?,
            stride_a: k.try_into()?,
            stride_b: k.try_into()?,
            stride_d: n.try_into()?,
            batch_stride_a: if batch_n {
                0
            } else {
                qshape.elem_count().try_into()?
            },
            batch_stride_b: k.try_into()?,
            batch_stride_d: n.try_into()?,
            fusion_flags: 0,
            base_work_group_y: 0,
            ne02: 1,
            ne12: if batch_n { 1 } else { batch_inner.try_into()? },
            broadcast2: if batch_n { 1 } else { batch_inner.try_into()? },
            broadcast3: if batch_n { 1 } else { batch_outer.try_into()? },
        };
        let q8_1_rhs = if use_q8_1_rhs {
            Some(quantize_f32_storage_to_q8_1_x4(
                &self.device,
                src,
                src_elem_count,
            )?)
        } else {
            None
        };
        let rhs_binding = q8_1_rhs.as_ref().unwrap_or(&src.buffer);
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(rhs_binding),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&dst.buffer),
        ];
        let dmmv_workgroup = q8_1_rhs_shader
            .as_ref()
            .map(|(_, workgroup)| *workgroup)
            .unwrap_or_else(|| vulkan_dmmv_workgroup(&self.device, qdtype, n, k, false));
        let (spirv_name, rows_per_group, block_size) = if use_q8_1_rhs {
            (
                q8_1_rhs_shader.as_ref().unwrap().0.clone(),
                vulkan_dmmv_rows_per_group(&self.device, qdtype, true)?,
                vulkan_dmmv_block_size(&self.device, qdtype, dmmv_workgroup),
            )
        } else {
            (
                vulkan_dmmv_shader_name(
                    &self.device,
                    qdtype,
                    format!(
                        "mul_mat_vec_{}_{}",
                        vulkan_quantized_stem(qdtype)?,
                        match src.dtype {
                            DType::F32 => "f32_f32",
                            DType::F16 => "f16_f32",
                            other => {
                                return Err(Error::UnsupportedDTypeForOp(
                                    other,
                                    "vulkan quantized matvec",
                                )
                                .bt());
                            }
                        }
                    ),
                    dmmv_workgroup,
                ),
                vulkan_quantized_vec_rows(qdtype)?,
                vulkan_dmmv_block_size(&self.device, qdtype, dmmv_workgroup),
            )
        };
        let spirv = candle_vulkan_kernels::spirv(&spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        let spec = [
            (0, block_size),
            (1, rows_per_group),
            (2, input_m.try_into()?),
        ];
        let use_dmmv_subgroups = vulkan_dmmv_use_subgroups(&self.device, qdtype);
        let required_subgroup_size = if use_dmmv_subgroups {
            Some(if vulkan_is_k_quant(qdtype) {
                vulkan_dmmv_subgroup_size16(&self.device)
            } else {
                vulkan_dmmv_subgroup_size(&self.device)
            })
        } else {
            None
        };
        self.device.run_compute_specialized_with_options(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (
                n.div_ceil(rows_per_group as usize).try_into()?,
                dispatch_y.try_into()?,
                1,
            ),
            Some(&spec),
            use_dmmv_subgroups,
            required_subgroup_size,
        )?;
        drop(src_contiguous);
        Ok((dst, dst_shape))
    }

    pub(crate) fn quantized_indexed_moe_f32(
        &self,
        qdtype: GgmlDType,
        qshape: &Shape,
        storage: &Self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
    ) -> Result<(Self, Shape)> {
        if storage.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(
                storage.dtype,
                "vulkan quantized indexed_moe",
            )
            .bt());
        }
        if ids.dtype != DType::U32 && ids.dtype != DType::I32 {
            return Err(Error::UnsupportedDTypeForOp(
                ids.dtype,
                "vulkan quantized indexed_moe ids",
            )
            .bt());
        }
        let (num_experts, n, k) = qshape.dims3()?;
        let rank = layout.dims().len();
        if rank != 3 {
            return Err(unsupported("quantized indexed_moe rank"));
        }
        let (batch, input_dim1, input_k) = layout.shape().dims3()?;
        let (ids_batch, topk) = ids_l.shape().dims2()?;
        if batch != ids_batch {
            crate::bail!("indexed_moe_forward batch mismatch: input={batch}, ids={ids_batch}");
        }
        if input_k != k {
            crate::bail!("indexed_moe_forward last dim mismatch: input={input_k}, weight={k}");
        }
        if input_dim1 != 1 && input_dim1 != topk {
            crate::bail!(
                "indexed_moe_forward expects input dim-1 to be 1 or topk ({topk}), got {input_dim1}"
            );
        }
        if !vulkan_supports_quantized_matvec_weight(qdtype) {
            return Err(unsupported("quantized indexed_moe dtype"));
        }
        let mut src_contiguous = None;
        let (src, _src_layout) = if layout.is_contiguous() && layout.start_offset() == 0 {
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
        let mut ids_contiguous = None;
        let (ids_src, ids_layout) = if ids_l.is_contiguous() && ids_l.start_offset() == 0 {
            (ids, ids_l.clone())
        } else {
            let mut tmp = unsafe { ids.device.alloc_uninit(ids_l.shape(), ids.dtype)? };
            ids.copy_strided_src(&mut tmp, 0, ids_l)?;
            ids_contiguous = Some(tmp);
            (
                ids_contiguous.as_ref().unwrap(),
                Layout::contiguous(ids_l.shape().clone()),
            )
        };
        let ids_stride = ids_layout.stride();
        if ids_stride[1] != 1 {
            return Err(unsupported("quantized indexed_moe ids inner stride"));
        }
        let src_elem_count = layout.shape().elem_count();
        // Q8_0 indexed MoE currently matches the native f32 rhs path more closely than
        // the q8_1 rhs MMVQ variant on Vulkan, so keep it on the direct GPU path.
        let q8_1_rhs_shader = if qdtype != GgmlDType::Q8_0
            && self.device.inner.integer_dot_product
            && src_elem_count.is_multiple_of(4)
        {
            vulkan_q8_1_rhs_matvec_shader_name(&self.device, qdtype, batch, k, true)?
        } else {
            None
        };
        let use_q8_1_rhs = q8_1_rhs_shader.is_some();
        let dst_shape = Shape::from((batch, topk, n));
        let dst = unsafe { storage.device.alloc_uninit(&dst_shape, DType::F32)? };
        let params = VulkanMatVecIdParams {
            ncols: k.try_into()?,
            stride_a: k.try_into()?,
            stride_b: k.try_into()?,
            stride_d: n.try_into()?,
            batch_stride_a: (n * k).try_into()?,
            batch_stride_b: (input_dim1 * k).try_into()?,
            batch_stride_d: (topk * n).try_into()?,
            fusion_flags: 0,
            nei0: topk.try_into()?,
            ne11: input_dim1.try_into()?,
            expert_i1: 0,
            nbi1: ids_stride[0].try_into()?,
        };
        let q8_1_rhs = if use_q8_1_rhs {
            Some(quantize_f32_storage_to_q8_1_x4(
                &self.device,
                src,
                src_elem_count,
            )?)
        } else {
            None
        };
        let rhs_binding = q8_1_rhs.as_ref().unwrap_or(&src.buffer);
        let bindings = [
            VulkanBinding::Storage(&self.buffer),
            VulkanBinding::Storage(rhs_binding),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&dst.buffer),
            VulkanBinding::Storage(&ids_src.buffer),
        ];
        let dmmv_workgroup = q8_1_rhs_shader
            .as_ref()
            .map(|(_, workgroup)| *workgroup)
            .unwrap_or_else(|| vulkan_dmmv_workgroup(&self.device, qdtype, n, k, false));
        let (spirv_name, rows_per_group, block_size) = if use_q8_1_rhs {
            (
                q8_1_rhs_shader.as_ref().unwrap().0.clone(),
                vulkan_dmmv_rows_per_group(&self.device, qdtype, true)?,
                vulkan_dmmv_block_size(&self.device, qdtype, dmmv_workgroup),
            )
        } else {
            (
                vulkan_dmmv_shader_name(
                    &self.device,
                    qdtype,
                    format!("mul_mat_vec_id_{}_f32", vulkan_quantized_stem(qdtype)?),
                    dmmv_workgroup,
                ),
                vulkan_quantized_vec_rows(qdtype)?,
                vulkan_dmmv_block_size(&self.device, qdtype, dmmv_workgroup),
            )
        };
        let spirv = candle_vulkan_kernels::spirv(&spirv_name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {spirv_name} not generated")).bt())?;
        let spec = [(0, block_size), (1, rows_per_group), (2, 1)];
        let groups_x_max = self.device.inner.max_workgroup_count_x.max(1);
        let groups_x = (n as u32).div_ceil(rows_per_group).min(groups_x_max);
        let groups_z = (n as u32).div_ceil(groups_x);
        let use_dmmv_subgroups = vulkan_dmmv_use_subgroups(&self.device, qdtype);
        let required_subgroup_size = if use_dmmv_subgroups {
            Some(if vulkan_is_k_quant(qdtype) {
                vulkan_dmmv_subgroup_size16(&self.device)
            } else {
                vulkan_dmmv_subgroup_size(&self.device)
            })
        } else {
            None
        };
        for expert_i1 in 0..batch {
            let params = VulkanMatVecIdParams {
                expert_i1: expert_i1.try_into()?,
                ..params
            };
            self.device.run_compute_specialized_with_options(
                spirv,
                &bindings,
                Some(any_as_bytes(&params)),
                (groups_x, topk.try_into()?, groups_z),
                Some(&spec),
                use_dmmv_subgroups,
                required_subgroup_size,
            )?;
        }
        drop(ids_contiguous);
        drop(src_contiguous);
        let _ = num_experts;
        Ok((dst, dst_shape))
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        if let Ok(mut allocation) = self.allocation.lock() {
            if let Some(allocation) = allocation.take() {
                if let Ok(mut deferred) = self.device.inner.deferred_buffer_frees.lock() {
                    deferred.push(VulkanDeferredBuffer {
                        buffer: self.buffer,
                        allocation,
                    });
                }
            }
        }
    }
}

impl Drop for VulkanInner {
    fn drop(&mut self) {
        unsafe {
            let destroy_submission_resources = |resources: VulkanSubmissionResources| {
                self.device.destroy_fence(resources.fence, None);
                self.device
                    .destroy_command_pool(resources.command_pool, None);
                self.device
                    .destroy_descriptor_pool(resources.descriptor_pool, None);
            };

            let mut submitted_active_batches = Vec::new();
            let mut flush_active_batch_for_drop =
                |slot: &Mutex<Option<VulkanActiveBatch>>, queue: vk::Queue| {
                    if let Ok(mut slot) = slot.lock() {
                        if let Some(batch) = slot.take() {
                            if batch.has_commands()
                                && self
                                    .device
                                    .end_command_buffer(batch.resources.command_buffer)
                                    .is_ok()
                            {
                                let submit_info = vk::SubmitInfo::default().command_buffers(
                                    std::slice::from_ref(&batch.resources.command_buffer),
                                );
                                if self
                                    .device
                                    .queue_submit(
                                        queue,
                                        std::slice::from_ref(&submit_info),
                                        batch.resources.fence,
                                    )
                                    .is_ok()
                                {
                                    submitted_active_batches.push(VulkanPendingSubmission {
                                        resources: batch.resources,
                                        queue_kind: batch.queue_kind,
                                        dispatch_count: batch.dispatch_count,
                                        copy_count: batch.copy_count,
                                        transfer_bytes: batch.transfer_bytes,
                                        compute_bytes: batch.compute_bytes,
                                        retained_buffers: batch.retained_buffers,
                                    });
                                    return;
                                }
                            }
                            destroy_submission_resources(batch.resources);
                        }
                    }
                };

            flush_active_batch_for_drop(&self.active_compute_batch, self.queue);
            if let Some(transfer_queue) = self.transfer_queue {
                flush_active_batch_for_drop(&self.active_transfer_batch, transfer_queue);
            }
            if let Ok(mut pending) = self.pending_submissions.lock() {
                pending.extend(submitted_active_batches);
            }

            let _ = self.device.device_wait_idle();

            if let Ok(mut pending) = self.pending_submissions.lock() {
                for submission in pending.drain(..) {
                    destroy_submission_resources(submission.resources);
                }
            }
            if let Ok(mut reusable) = self.reusable_compute_submissions.lock() {
                for resources in reusable.drain(..) {
                    destroy_submission_resources(resources);
                }
            }
            if let Ok(mut reusable) = self.reusable_transfer_submissions.lock() {
                for resources in reusable.drain(..) {
                    destroy_submission_resources(resources);
                }
            }
            if let Ok(mut cache) = self.pipeline_cache.lock() {
                for (_, cached) in cache.drain() {
                    self.device.destroy_pipeline(cached.pipeline, None);
                    self.device
                        .destroy_pipeline_layout(cached.pipeline_layout, None);
                    self.device
                        .destroy_descriptor_set_layout(cached.descriptor_set_layout, None);
                    self.device.destroy_shader_module(cached.shader, None);
                }
            }
            if let Ok(mut deferred) = self.deferred_buffer_frees.lock() {
                if let Ok(mut allocator) = self.allocator.lock() {
                    if let Some(allocator) = allocator.as_mut() {
                        for deferred in deferred.drain(..) {
                            self.device.destroy_buffer(deferred.buffer, None);
                            let _ = allocator.free(deferred.allocation);
                        }
                    }
                }
            }
            if let Ok(mut allocator) = self.allocator.lock() {
                let _ = allocator.take();
            }
            if self.driver_pipeline_cache != vk::PipelineCache::null() {
                self.device
                    .destroy_pipeline_cache(self.driver_pipeline_cache, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        if !layout.is_contiguous()
            || layout.start_offset() != 0
            || layout.shape().elem_count() != self.count
        {
            let cpu = self.to_cpu_storage()?;
            let out_cpu = <CpuStorage as BackendStorage>::try_clone(&cpu, layout)?;
            return self.device.storage_from_cpu_storage(&out_cpu);
        }
        let bytes = self.device.read_buffer(&self.buffer)?;
        let buffer = self
            .device
            .create_buffer(bytes.len(), "candle-vulkan-clone")?;
        self.device.write_buffer(&buffer, &bytes)?;
        Ok(Self {
            buffer,
            device: self.device.clone(),
            count: self.count,
            dtype: self.dtype,
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let size = byte_len(self.dtype, self.count, "vulkan download")?;
        let mut bytes = self.device.read_buffer(&self.buffer)?;
        bytes.truncate(size);
        bytes_to_cpu_storage(self.dtype, self.count, &bytes)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let gpu = || -> Result<Self> {
            if self.dtype != DType::F32 {
                return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan affine").bt());
            }
            let spirv = candle_vulkan_kernels::spirv("scale_f32")
                .ok_or_else(|| Error::Msg("vulkan shader scale_f32 not generated".into()).bt())?;
            // Strided/offset views are materialized on the GPU inside the
            // generic unary dispatch helper.
            self.run_unary_generic_with_params(layout, spirv, mul as f32, add as f32)
        };
        match gpu() {
            Ok(out) => Ok(out),
            Err(err)
                if should_cpu_fallback(&err) || matches!(err, Error::UnsupportedDTypeForOp(..)) =>
            {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::affine(&src_cpu, layout, mul, add)?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let gpu = || -> Result<Self> {
            if self.dtype != DType::F32 {
                return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan powf").bt());
            }
            let spirv = candle_vulkan_kernels::spirv("powf_f32")
                .ok_or_else(|| Error::Msg("vulkan shader powf_f32 not generated".into()).bt())?;
            self.run_unary_generic_with_params(layout, spirv, e as f32, 0.0)
        };
        match gpu() {
            Ok(out) => Ok(out),
            Err(err)
                if should_cpu_fallback(&err) || matches!(err, Error::UnsupportedDTypeForOp(..)) =>
            {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::powf(&src_cpu, layout, e)?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let cpu_fallback = || -> Result<Self> {
            let src_cpu = self.to_cpu_storage()?;
            let out_cpu = <CpuStorage as BackendStorage>::elu(&src_cpu, layout, alpha)?;
            self.device.storage_from_cpu_storage(&out_cpu)
        };
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return cpu_fallback();
        }
        if alpha != 1.0 {
            return cpu_fallback();
        }
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            _ => unreachable!(),
        };
        let name = format!("elu_{suffix}");
        let spirv = candle_vulkan_kernels::spirv(&name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated")).bt())?;
        match self.run_unary_head(layout, spirv) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => cpu_fallback(),
            Err(err) => Err(err),
        }
    }
    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan reduce").bt());
        }
        let rank = layout.dims().len();
        if rank == 0 {
            crate::bail!("vulkan backend op reduce does not support rank-0 tensors")
        }
        if reduce_dims.is_empty() {
            return self.try_clone(layout);
        }
        for &dim in reduce_dims {
            if dim >= rank {
                crate::bail!("vulkan backend op reduce got out-of-range dim {dim} for rank {rank}")
            }
        }
        if self.dtype == DType::F16 {
            let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), DType::F16)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut materialized, 0, layout)?;
            let contiguous_layout = Layout::contiguous(layout.shape());
            let materialized_f32 = materialized.to_dtype(&contiguous_layout, DType::F32)?;
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
        self.run_sum_rows(layout)
    }
    fn cumsum_last_dim(&self, layout: &Layout) -> Result<Self> {
        match self.run_cumsum_last_dim(layout) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::cumsum_last_dim(&src_cpu, layout)?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn clamp(&self, layout: &Layout, min: f32, max: f32) -> Result<Self> {
        let gpu = || -> Result<Self> {
            if self.dtype != DType::F32 {
                return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan clamp").bt());
            }
            let spirv = candle_vulkan_kernels::spirv("clamp_f32")
                .ok_or_else(|| Error::Msg("vulkan shader clamp_f32 not generated".into()).bt())?;
            self.run_unary_generic_with_params(layout, spirv, min, max)
        };
        match gpu() {
            Ok(out) => Ok(out),
            Err(err)
                if should_cpu_fallback(&err) || matches!(err, Error::UnsupportedDTypeForOp(..)) =>
            {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::clamp(&src_cpu, layout, min, max)?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        match self.run_cmp_u8(rhs, lhs_l, rhs_l, op) {
            Ok(out) => Ok(out),
            Err(err)
                if should_cpu_fallback(&err) || matches!(err, Error::UnsupportedDTypeForOp(..)) =>
            {
                let lhs_cpu = self.to_cpu_storage()?;
                let rhs_cpu = rhs.to_cpu_storage()?;
                let out_cpu =
                    <CpuStorage as BackendStorage>::cmp(&lhs_cpu, op, &rhs_cpu, lhs_l, rhs_l)?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype == DType::BF16 && (dtype == DType::F16 || dtype == DType::F32) {
            return match self.run_bf16_to_dtype_via_get_rows(layout, dtype) {
                Ok(out) => Ok(out),
                Err(err) if should_cpu_fallback(&err) => {
                    let src_cpu = self.to_cpu_storage()?;
                    let out_cpu =
                        <CpuStorage as BackendStorage>::to_dtype(&src_cpu, layout, dtype)?;
                    self.device.storage_from_cpu_storage(&out_cpu)
                }
                Err(err) => Err(err),
            };
        }
        if self.dtype == DType::BF16 {
            // No direct bf16 -> integer kernel exists; decompose into the
            // native bf16 -> f32 path followed by the native f32 -> dst cast,
            // keeping the whole chain on the GPU.
            let f32_storage = self.to_dtype(layout, DType::F32)?;
            let contiguous = Layout::contiguous(layout.shape().clone());
            return f32_storage.to_dtype(&contiguous, dtype);
        }
        let spirv = copy_spirv(self.dtype, dtype)?;
        match self.run_unary_generic_with_params_dtype(layout, spirv, 0.0, 0.0, dtype) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::to_dtype(&src_cpu, layout, dtype)?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan unary").bt());
        }
        if self.dtype == DType::F16 && matches!(B::NAME, "sin" | "cos" | "sqr" | "sqrt") {
            let mut materialized = unsafe { self.device.alloc_uninit(layout.shape(), DType::F16)? };
            <Self as BackendStorage>::copy_strided_src(self, &mut materialized, 0, layout)?;
            let contiguous_layout = Layout::contiguous(layout.shape());
            let materialized_f32 = materialized.to_dtype(&contiguous_layout, DType::F32)?;
            let (spirv, kind) = unary_spirv(B::NAME, DType::F32)?;
            let out_f32 = match kind {
                VulkanUnaryKind::Head => materialized_f32.run_unary_head(&contiguous_layout, spirv),
                VulkanUnaryKind::Generic => {
                    materialized_f32.run_unary_generic(&contiguous_layout, spirv)
                }
            }?;
            return out_f32.to_dtype(&contiguous_layout, DType::F16);
        }
        let (spirv, kind) = unary_spirv(B::NAME, self.dtype)?;
        match match kind {
            VulkanUnaryKind::Head => self.run_unary_head(layout, spirv),
            VulkanUnaryKind::Generic => self.run_unary_generic(layout, spirv),
        } {
            Ok(out) => Ok(out),
            Err(err) => Err(err),
        }
    }
    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        match match B::NAME {
            "maximum" | "minimum" if matches!(self.dtype, DType::U8 | DType::U32 | DType::I64) => {
                self.run_binary_named(rhs, lhs_layout, rhs_layout, B::NAME)
            }
            "maximum" | "minimum" => {
                if self.dtype == DType::F16 && rhs.dtype == DType::F16 {
                    let mut lhs_mat =
                        unsafe { self.device.alloc_uninit(lhs_layout.shape(), DType::F16)? };
                    <Self as BackendStorage>::copy_strided_src(self, &mut lhs_mat, 0, lhs_layout)?;
                    let lhs_contiguous = Layout::contiguous(lhs_layout.shape());
                    let lhs_f32 = lhs_mat.to_dtype(&lhs_contiguous, DType::F32)?;

                    let mut rhs_mat =
                        unsafe { self.device.alloc_uninit(rhs_layout.shape(), DType::F16)? };
                    <Self as BackendStorage>::copy_strided_src(rhs, &mut rhs_mat, 0, rhs_layout)?;
                    let rhs_contiguous = Layout::contiguous(rhs_layout.shape());
                    let rhs_f32 = rhs_mat.to_dtype(&rhs_contiguous, DType::F32)?;

                    let out_f32 = lhs_f32.run_binary_min_max_f32(
                        &rhs_f32,
                        &lhs_contiguous,
                        &rhs_contiguous,
                        B::NAME,
                    )?;
                    out_f32.to_dtype(&lhs_contiguous, DType::F16)
                } else {
                    self.run_binary_min_max_f32(rhs, lhs_layout, rhs_layout, B::NAME)
                }
            }
            _ => self.run_binary_named(rhs, lhs_layout, rhs_layout, B::NAME),
        } {
            Ok(out) => Ok(out),
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
        match self.run_where_u8_cond(layout, t, t_l, f, f_l) {
            Ok(out) => Ok(out),
            Err(err)
                if should_cpu_fallback(&err) || matches!(err, Error::UnsupportedDTypeForOp(..)) =>
            {
                let cond_cpu = self.to_cpu_storage()?;
                let t_cpu = t.to_cpu_storage()?;
                let f_cpu = f.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::where_cond(
                    &cond_cpu, layout, &t_cpu, t_l, &f_cpu, f_l,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        match self.run_conv1d_f32(layout, kernel, kernel_l, params) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let kernel_cpu = kernel.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::conv1d(
                    &src_cpu,
                    layout,
                    &kernel_cpu,
                    kernel_l,
                    params,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        match self.run_conv_transpose1d_f32(layout, kernel, kernel_l, params) {
            Ok(out) => Ok(out),
            Err(err)
                if should_cpu_fallback(&err) || matches!(err, Error::UnsupportedDTypeForOp(..)) =>
            {
                let src_cpu = self.to_cpu_storage()?;
                let kernel_cpu = kernel.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::conv_transpose1d(
                    &src_cpu,
                    layout,
                    &kernel_cpu,
                    kernel_l,
                    params,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        match self.run_conv2d_f32(layout, kernel, kernel_l, params) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let kernel_cpu = kernel.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::conv2d(
                    &src_cpu,
                    layout,
                    &kernel_cpu,
                    kernel_l,
                    params,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        match self.run_conv_transpose2d_f32(layout, kernel, kernel_l, params) {
            Ok(out) => Ok(out),
            Err(err)
                if should_cpu_fallback(&err) || matches!(err, Error::UnsupportedDTypeForOp(..)) =>
            {
                let src_cpu = self.to_cpu_storage()?;
                let kernel_cpu = kernel.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::conv_transpose2d(
                    &src_cpu,
                    layout,
                    &kernel_cpu,
                    kernel_l,
                    params,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        match self.run_pool2d_f32(layout, kernel_size, stride, false) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::avg_pool2d(
                    &src_cpu,
                    layout,
                    kernel_size,
                    stride,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        match self.run_pool2d_f32(layout, kernel_size, stride, true) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::max_pool2d(
                    &src_cpu,
                    layout,
                    kernel_size,
                    stride,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn upsample_nearest1d(&self, layout: &Layout, out_l: usize) -> Result<Self> {
        match self.run_upsample_nearest1d_f32(layout, out_l) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu =
                    <CpuStorage as BackendStorage>::upsample_nearest1d(&src_cpu, layout, out_l)?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn upsample_nearest2d(&self, layout: &Layout, out_h: usize, out_w: usize) -> Result<Self> {
        let (_, _, h, w) = layout.shape().dims4()?;
        match self.run_upsample2d_f32(
            layout,
            out_h,
            out_w,
            nearest_interp_weights(h, out_h),
            nearest_interp_weights(w, out_w),
        ) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::upsample_nearest2d(
                    &src_cpu, layout, out_h, out_w,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
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
        let (_, _, h, w) = layout.shape().dims4()?;
        match self.run_upsample2d_f32(
            layout,
            out_h,
            out_w,
            bilinear_interp_weights(h, out_h, align_corners, scale_h),
            bilinear_interp_weights(w, out_w, align_corners, scale_w),
        ) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::upsample_bilinear2d(
                    &src_cpu,
                    layout,
                    out_h,
                    out_w,
                    align_corners,
                    scale_h,
                    scale_w,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
    }
    fn gather(&self, src_l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if dim + 1 == src_l.dims().len() {
            match self.run_gather_last_dim_f32(ids, src_l, ids_l) {
                Ok(out) => return Ok(out),
                Err(err) if should_cpu_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }
        let src_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let out_cpu =
            <CpuStorage as BackendStorage>::gather(&src_cpu, src_l, &ids_cpu, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out_cpu)
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
        if dim + 1 == dst_l.dims().len() {
            match self.run_scatter_set_last_dim_f32(dst_l, ids, ids_l, src, src_l) {
                Ok(()) => return Ok(()),
                Err(err) if should_cpu_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }
        let mut dst_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        <CpuStorage as BackendStorage>::scatter_set(
            &mut dst_cpu,
            dst_l,
            &ids_cpu,
            ids_l,
            &src_cpu,
            src_l,
            dim,
        )?;
        *self = self.device.storage_from_cpu_storage(&dst_cpu)?;
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
        if dim + 1 == dst_l.dims().len() {
            match self.run_scatter_add_last_dim_f32(dst_l, ids, ids_l, src, src_l) {
                Ok(()) => return Ok(()),
                Err(err) if should_cpu_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }
        let mut dst_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        <CpuStorage as BackendStorage>::scatter_add_set(
            &mut dst_cpu,
            dst_l,
            &ids_cpu,
            ids_l,
            &src_cpu,
            src_l,
            dim,
        )?;
        *self = self.device.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }
    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        match self.run_index_select_f32(ids, src_l, ids_l, dim) {
            Ok(out) => Ok(out),
            Err(err) if should_cpu_fallback(&err) => {
                let src_cpu = self.to_cpu_storage()?;
                let ids_cpu = ids.to_cpu_storage()?;
                let out_cpu = <CpuStorage as BackendStorage>::index_select(
                    &src_cpu, &ids_cpu, src_l, ids_l, dim,
                )?;
                self.device.storage_from_cpu_storage(&out_cpu)
            }
            Err(err) => Err(err),
        }
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
            match out.run_scatter_add_last_dim_f32(dst_l, ids, ids_l, src, src_l) {
                Ok(()) => return Ok(out),
                Err(err) if should_cpu_fallback(&err) => {}
                Err(err) => return Err(err),
            }
        }
        let dst_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        let out_cpu = <CpuStorage as BackendStorage>::index_add(
            &dst_cpu, dst_l, &ids_cpu, ids_l, &src_cpu, src_l, dim,
        )?;
        self.device.storage_from_cpu_storage(&out_cpu)
    }
    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        match self.run_matmul_f32(rhs, bmnk, lhs_l, rhs_l) {
            Ok(out) => Ok(out),
            Err(err) => Err(err),
        }
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
            match self.dtype {
                DType::F32 | DType::F16 | DType::U8 | DType::U32 | DType::I64 => {
                    let spirv = copy_spirv(self.dtype, self.dtype)?;
                    return self.run_copy_into(src_l, dst, dst_offset, spirv);
                }
                _ => {
                    let src_cpu = self.to_cpu_storage()?;
                    let mut dst_cpu = dst.to_cpu_storage()?;
                    src_cpu.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
                    *dst = dst.device.storage_from_cpu_storage(&dst_cpu)?;
                    return Ok(());
                }
            }
        }
        let elem_size = self.dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan copy").bt());
        }
        let src_offset = src_l.start_offset() * elem_size;
        let dst_offset = dst_offset * elem_size;
        let size = src_l.shape().elem_count() * elem_size;
        self.device.submit_copy_region_and_track(
            &self.buffer,
            &dst.buffer,
            src_offset,
            dst_offset,
            size,
            false,
        )?;
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
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let elem_size = self.dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan copy2d").bt());
        }
        let mut regions = Vec::with_capacity(d1);
        for i1 in 0..d1 {
            let src_idx = (i1 * src_stride1 + src_offset) * elem_size;
            let dst_idx = (i1 * dst_stride1 + dst_offset) * elem_size;
            let len = d2 * elem_size;
            regions.push(
                vk::BufferCopy::default()
                    .src_offset(src_idx as u64)
                    .dst_offset(dst_idx as u64)
                    .size(len as u64),
            );
        }
        self.device
            .submit_copy_regions_and_track(&self.buffer, &dst.buffer, &regions, false)?;
        Ok(())
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let (start, end) = match layout.contiguous_offsets() {
            Some(v) => v,
            None => {
                let mut cpu = self.to_cpu_storage()?;
                <CpuStorage as BackendStorage>::const_set(&mut cpu, scalar, layout)?;
                *self = self.device.storage_from_cpu_storage(&cpu)?;
                return Ok(());
            }
        };
        let scalar = scalar_bytes(scalar, self.dtype, "vulkan const_set")?;
        let mut bytes = self.device.read_buffer(&self.buffer)?;
        let start = start * scalar.len();
        let end = end * scalar.len();
        for chunk in bytes[start..end].chunks_exact_mut(scalar.len()) {
            chunk.copy_from_slice(&scalar);
        }
        self.device.write_buffer(&self.buffer, &bytes)?;
        Ok(())
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(Error::wrap)?;
        let app_name = CString::new("candle-vulkan").map_err(Error::wrap)?;
        let subgroup_size_control_ext = c"VK_EXT_subgroup_size_control";
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance =
            unsafe { entry.create_instance(&instance_info, None) }.map_err(Error::wrap)?;
        let mut init_guard = VulkanInitGuard::new(instance.clone());
        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.map_err(Error::wrap)?;
        if physical_devices.is_empty() {
            crate::bail!("no vulkan physical device found")
        }
        let requested_name = std::env::var("CANDLE_VULKAN_DEVICE_NAME")
            .ok()
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty());
        let selected_device = if let Some(requested_name) = requested_name {
            let requested_name = requested_name.to_ascii_lowercase();
            physical_devices
                .iter()
                .find_map(|physical_device| {
                    let properties = unsafe { instance.get_physical_device_properties(*physical_device) };
                    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                        .to_string_lossy()
                        .into_owned();
                    device_name
                        .to_ascii_lowercase()
                        .contains(&requested_name)
                        .then_some((*physical_device, properties.device_type, device_name))
                })
                .ok_or_else(|| {
                    Error::msg(format!(
                        "no vulkan physical device matching CANDLE_VULKAN_DEVICE_NAME={requested_name:?}"
                    ))
                })?
        } else {
            let mut preferred = physical_devices.iter().filter_map(|physical_device| {
                let properties =
                    unsafe { instance.get_physical_device_properties(*physical_device) };
                (properties.device_type != vk::PhysicalDeviceType::CPU).then(|| {
                    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                        .to_string_lossy()
                        .into_owned();
                    (*physical_device, properties.device_type, device_name)
                })
            });
            preferred
                .nth(ordinal)
                .or_else(|| {
                    physical_devices.iter().find_map(|physical_device| {
                        let properties =
                            unsafe { instance.get_physical_device_properties(*physical_device) };
                        (properties.device_type != vk::PhysicalDeviceType::CPU).then(|| {
                            let device_name =
                                unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                                    .to_string_lossy()
                                    .into_owned();
                            (*physical_device, properties.device_type, device_name)
                        })
                    })
                })
                .or_else(|| {
                    physical_devices.first().map(|physical_device| {
                        let properties =
                            unsafe { instance.get_physical_device_properties(*physical_device) };
                        let device_name =
                            unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                                .to_string_lossy()
                                .into_owned();
                        (*physical_device, properties.device_type, device_name)
                    })
                })
                .ok_or_else(|| Error::msg("no vulkan physical device found"))?
        };
        let (physical_device, physical_device_type, physical_device_name) = selected_device;
        let device_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map_err(Error::wrap)?
        };
        let has_subgroup_size_control_ext = device_extensions.iter().any(|ext| unsafe {
            CStr::from_ptr(ext.extension_name.as_ptr()) == subgroup_size_control_ext
        });
        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut subgroup_size_control_properties =
            vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT::default();
        let mut integer_dot_properties =
            vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
        let mut physical_device_properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_properties)
            .push_next(&mut subgroup_size_control_properties)
            .push_next(&mut integer_dot_properties);
        unsafe {
            instance
                .get_physical_device_properties2(physical_device, &mut physical_device_properties);
        }
        let max_workgroup_invocations = physical_device_properties
            .properties
            .limits
            .max_compute_work_group_invocations
            .max(1);
        let max_workgroup_size_log2 = floor_log2_u32(max_workgroup_invocations);
        let max_workgroup_count_x = physical_device_properties
            .properties
            .limits
            .max_compute_work_group_count[0]
            .max(1);
        let max_workgroup_count_y = physical_device_properties
            .properties
            .limits
            .max_compute_work_group_count[1]
            .max(1);
        let max_push_constants_size = physical_device_properties
            .properties
            .limits
            .max_push_constants_size
            .max(1);
        let vendor_id = physical_device_properties.properties.vendor_id;
        let subgroup_arithmetic = subgroup_properties
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::ARITHMETIC)
            && subgroup_properties
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE);
        let subgroup_size = subgroup_properties.subgroup_size.max(1);
        let mut vulkan11_features = vk::PhysicalDeviceVulkan11Features::default();
        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut integer_dot_features = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut subgroup_size_control_features =
            vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT::default();
        let mut physical_features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut vulkan11_features)
            .push_next(&mut vulkan12_features)
            .push_next(&mut subgroup_size_control_features)
            .push_next(&mut integer_dot_features);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut physical_features2);
        }
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, props)| {
                let flags = props.queue_flags;
                (flags.contains(vk::QueueFlags::COMPUTE)
                    && !flags.contains(vk::QueueFlags::GRAPHICS))
                .then_some(index as u32)
            })
            .or_else(|| {
                queue_families
                    .iter()
                    .enumerate()
                    .find_map(|(index, props)| {
                        props
                            .queue_flags
                            .contains(vk::QueueFlags::COMPUTE)
                            .then_some(index as u32)
                    })
            })
            .ok_or_else(|| Error::msg("no vulkan compute queue family found"))?;
        let transfer_queue_family_index = queue_families
            .iter()
            .enumerate()
            .find_map(|(index, props)| {
                let flags = props.queue_flags;
                (flags.contains(vk::QueueFlags::TRANSFER)
                    && !flags.contains(vk::QueueFlags::COMPUTE)
                    && !flags.contains(vk::QueueFlags::GRAPHICS))
                .then_some(index as u32)
            })
            .or_else(|| {
                queue_families
                    .iter()
                    .enumerate()
                    .find_map(|(index, props)| {
                        (props.queue_flags.contains(vk::QueueFlags::TRANSFER)
                            && index as u32 != queue_family_index)
                            .then_some(index as u32)
                    })
            })
            .filter(|idx| *idx != queue_family_index);
        let robust_buffer_access = physical_features2.features.robust_buffer_access == vk::TRUE;
        let shader_int64 = physical_features2.features.shader_int64 == vk::TRUE;
        let vulkan_memory_model = vulkan12_features.vulkan_memory_model == vk::TRUE;
        let integer_dot_product = integer_dot_features.shader_integer_dot_product == vk::TRUE
            && integer_dot_properties.integer_dot_product4x8_bit_packed_signed_accelerated
                == vk::TRUE;
        let subgroup_size_control = has_subgroup_size_control_ext
            && subgroup_size_control_features.subgroup_size_control == vk::TRUE;
        let compute_full_subgroups = subgroup_size_control
            && subgroup_size_control_features.compute_full_subgroups == vk::TRUE;
        let subgroup_min_size = subgroup_size_control_properties.min_subgroup_size.max(1);
        let subgroup_max_size = subgroup_size_control_properties.max_subgroup_size.max(1);
        let mut enabled_features = vk::PhysicalDeviceFeatures::default();
        if robust_buffer_access {
            enabled_features.robust_buffer_access = vk::TRUE;
        }
        // 64-bit integers are needed by the Candle-owned I64 cast kernels.
        if shader_int64 {
            enabled_features.shader_int64 = vk::TRUE;
        }
        let mut enabled_vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
        // The ggml shader corpus reads 16-bit storage buffers throughout.
        if vulkan11_features.storage_buffer16_bit_access == vk::TRUE {
            enabled_vulkan11.storage_buffer16_bit_access = vk::TRUE;
        }
        if vulkan11_features.uniform_and_storage_buffer16_bit_access == vk::TRUE {
            enabled_vulkan11.uniform_and_storage_buffer16_bit_access = vk::TRUE;
        }
        let mut enabled_vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        if vulkan_memory_model {
            enabled_vulkan12.vulkan_memory_model = vk::TRUE;
        }
        // 8-bit storage backs both the quantized blocks and the U8 cast family.
        if vulkan12_features.storage_buffer8_bit_access == vk::TRUE {
            enabled_vulkan12.storage_buffer8_bit_access = vk::TRUE;
        }
        if vulkan12_features.uniform_and_storage_buffer8_bit_access == vk::TRUE {
            enabled_vulkan12.uniform_and_storage_buffer8_bit_access = vk::TRUE;
        }
        if vulkan12_features.shader_int8 == vk::TRUE {
            enabled_vulkan12.shader_int8 = vk::TRUE;
        }
        if vulkan12_features.shader_float16 == vk::TRUE {
            enabled_vulkan12.shader_float16 = vk::TRUE;
        }
        let mut enabled_integer_dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        if integer_dot_product {
            enabled_integer_dot.shader_integer_dot_product = vk::TRUE;
        }
        let mut enabled_subgroup_size_control =
            vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT::default();
        if subgroup_size_control {
            enabled_subgroup_size_control.subgroup_size_control = vk::TRUE;
            if compute_full_subgroups {
                enabled_subgroup_size_control.compute_full_subgroups = vk::TRUE;
            }
        }
        let priorities = [1.0f32];
        let mut queue_infos = Vec::with_capacity(2);
        queue_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities),
        );
        if let Some(transfer_family) = transfer_queue_family_index {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_family)
                    .queue_priorities(&priorities),
            );
        }
        let mut enabled_extension_names = Vec::new();
        if has_subgroup_size_control_ext {
            enabled_extension_names.push(subgroup_size_control_ext.as_ptr());
        }
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&enabled_features)
            .enabled_extension_names(&enabled_extension_names);
        device_info = device_info.push_next(&mut enabled_vulkan11);
        device_info = device_info.push_next(&mut enabled_vulkan12);
        if subgroup_size_control {
            device_info = device_info.push_next(&mut enabled_subgroup_size_control);
        }
        if integer_dot_product {
            device_info = device_info.push_next(&mut enabled_integer_dot);
        }
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(Error::wrap)?;
        init_guard.attach_device(device.clone());
        let driver_pipeline_cache =
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
                .map_err(Error::wrap)?;
        init_guard.attach_pipeline_cache(driver_pipeline_cache);
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let transfer_queue =
            transfer_queue_family_index.map(|family| unsafe { device.get_device_queue(family, 0) });
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(Error::wrap)?;
        init_guard.disarm();
        Ok(Self {
            inner: Arc::new(VulkanInner {
                ordinal,
                physical_device_name,
                physical_device_type,
                _entry: entry,
                instance,
                physical_device,
                vendor_id,
                integer_dot_product,
                subgroup_arithmetic,
                subgroup_size,
                subgroup_size_control,
                compute_full_subgroups,
                subgroup_min_size,
                subgroup_max_size,
                max_workgroup_size_log2,
                max_workgroup_count_x,
                max_workgroup_count_y,
                max_push_constants_size,
                robust_buffer_access,
                vulkan_memory_model,
                device,
                driver_pipeline_cache,
                queue_family_index,
                queue,
                transfer_queue_family_index,
                transfer_queue,
                allocator: Mutex::new(Some(allocator)),
                seed_value: RwLock::new(299_792_458),
                pipeline_cache: Mutex::new(HashMap::new()),
                pending_submissions: Mutex::new(Vec::new()),
                active_compute_batch: Mutex::new(None),
                active_transfer_batch: Mutex::new(None),
                reusable_compute_submissions: Mutex::new(Vec::new()),
                reusable_transfer_submissions: Mutex::new(Vec::new()),
                deferred_buffer_frees: Mutex::new(Vec::new()),
            }),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.inner.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let size = byte_len(dtype, count, "vulkan zeros")?;
        let buffer = self.create_buffer(size, "candle-vulkan-zeros")?;
        self.write_buffer(&buffer, &vec![0u8; size])?;
        Ok(VulkanStorage {
            buffer,
            device: self.clone(),
            count,
            dtype,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let size = byte_len(dtype, count, "vulkan alloc_uninit")?;
        let buffer = self.create_buffer(size, "candle-vulkan-alloc-uninit")?;
        Ok(VulkanStorage {
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
        let normalized_storage;
        let storage = match storage {
            CpuStorage::BF16(values) => {
                normalized_storage = CpuStorage::F16(
                    values
                        .iter()
                        .map(|value| half::f16::from_f32(value.to_f32()))
                        .collect(),
                );
                &normalized_storage
            }
            _ => storage,
        };
        let (dtype, count, bytes) = cpu_storage_to_bytes(storage)?;
        let buffer = self.create_buffer(bytes.len(), "candle-vulkan-upload")?;
        self.write_buffer(&buffer, &bytes)?;
        Ok(VulkanStorage {
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
        let cpu = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, min, max)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let mut guard = self
            .inner
            .seed_value
            .write()
            .map_err(|_| Error::msg("vulkan seed lock poisoned"))?;
        *guard = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let guard = self
            .inner
            .seed_value
            .read()
            .map_err(|_| Error::msg("vulkan seed lock poisoned"))?;
        Ok(*guard)
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize_pending()
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::Module;

    fn exact_q8_1_x4_bytes(xs: &[f32]) -> Vec<u8> {
        assert_eq!(xs.len() % 128, 0);
        let mut out = Vec::new();
        for x4 in xs.chunks_exact(128) {
            let mut ds = [0u16; 8];
            let mut qs = [0i32; 32];
            for block_idx in 0..4 {
                let block = &x4[block_idx * 32..(block_idx + 1) * 32];
                let mut amax = 0f32;
                for &x in block {
                    amax = amax.max(x.abs());
                }
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let mut qbytes = [0u8; 32];
                let mut sum = 0i32;
                for i in 0..16 {
                    let q0 = f32::round(block[i] * id) as i8;
                    let q1 = f32::round(block[i + 16] * id) as i8;
                    qbytes[i] = q0 as u8;
                    qbytes[i + 16] = q1 as u8;
                    sum += q0 as i32 + q1 as i32;
                }
                ds[block_idx * 2] = half::f16::from_f32(d).to_bits();
                ds[block_idx * 2 + 1] = half::f16::from_f32(sum as f32 * d).to_bits();
                for i in 0..8 {
                    let base = i * 4;
                    qs[block_idx * 8 + i] = i32::from_le_bytes([
                        qbytes[base],
                        qbytes[base + 1],
                        qbytes[base + 2],
                        qbytes[base + 3],
                    ]);
                }
            }
            for bits in ds {
                out.extend_from_slice(&bits.to_le_bytes());
            }
            for packed in qs {
                out.extend_from_slice(&packed.to_le_bytes());
            }
        }
        out
    }

    #[test]
    #[ignore]
    fn vulkan_conv1d_im2col_matches_reference() -> Result<()> {
        let device = VulkanDevice::new(0)?;
        let src_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let src = device.storage_from_cpu_storage(&CpuStorage::F32(src_vals))?;
        let src_l = Layout::contiguous((1, 1, 4));
        let params = crate::conv::ParamsConv1D {
            b_size: 1,
            l_in: 4,
            c_out: 1,
            c_in: 1,
            k_size: 3,
            padding: 0,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };

        let col = src.run_im2col_conv1d_f32(&src_l, &params)?;
        let col_cpu = col.to_cpu_storage()?;
        let got = match col_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from im2col: {other:?}"),
        };
        let expected = vec![1.0f32, 2.0, 3.0, 2.0, 3.0, 4.0];
        assert_eq!(got, expected);

        let params_pad = crate::conv::ParamsConv1D {
            padding: 1,
            ..params
        };
        let col_pad = src.run_im2col_conv1d_f32(&src_l, &params_pad)?;
        let col_pad_cpu = col_pad.to_cpu_storage()?;
        let got_pad = match col_pad_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from im2col (pad): {other:?}"),
        };
        let expected_pad = vec![
            0.0f32, 1.0, 2.0, //
            1.0, 2.0, 3.0, //
            2.0, 3.0, 4.0, //
            3.0, 4.0, 0.0,
        ];
        assert_eq!(got_pad, expected_pad);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_conv1d_matmul_stage_matches_reference() -> Result<()> {
        let device = VulkanDevice::new(0)?;
        let src_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let kernel_vals = vec![1.0f32, 0.0, 1.0];
        let src = device.storage_from_cpu_storage(&CpuStorage::F32(src_vals))?;
        let kernel = device.storage_from_cpu_storage(&CpuStorage::F32(kernel_vals))?;
        let src_l = Layout::contiguous((1, 1, 4));
        let _kernel_l = Layout::contiguous((1, 1, 3));
        let params = crate::conv::ParamsConv1D {
            b_size: 1,
            l_in: 4,
            c_out: 1,
            c_in: 1,
            k_size: 3,
            padding: 0,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };

        let col = src.run_im2col_conv1d_f32(&src_l, &params)?;
        let b = params.b_size;
        let n = params.c_out;
        let l_out = params.l_out();
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b * m, k));
        let kernel_l_mm = Layout::contiguous_with_offset((n, k), 0).transpose(0, 1)?;
        let res = col.matmul(&kernel, (1, b * m, n, k), &col_l, &kernel_l_mm)?;

        let res_cpu = res.to_cpu_storage()?;
        let got_res = match res_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv1d matmul stage: {other:?}"),
        };
        assert_eq!(got_res, vec![4.0f32, 6.0]);

        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut out = unsafe { device.alloc_uninit(&Shape::from(params.out_dims()), DType::F32)? };
        res.copy_strided_src(&mut out, 0, &res_l)?;
        let out_cpu = out.to_cpu_storage()?;
        let got_out = match out_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv1d out stage: {other:?}"),
        };
        assert_eq!(got_out, vec![4.0f32, 6.0]);

        let params_pad = crate::conv::ParamsConv1D {
            padding: 1,
            ..params
        };
        let col_pad = src.run_im2col_conv1d_f32(&src_l, &params_pad)?;
        let b = params_pad.b_size;
        let n = params_pad.c_out;
        let l_out = params_pad.l_out();
        let k = params_pad.k_size * params_pad.c_in;
        let m = l_out;
        let col_pad_l = Layout::contiguous((b * m, k));
        let res_pad = col_pad.matmul(&kernel, (1, b * m, n, k), &col_pad_l, &kernel_l_mm)?;
        let res_pad_cpu = res_pad.to_cpu_storage()?;
        let got_res_pad = match res_pad_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv1d matmul stage (pad): {other:?}"),
        };
        assert_eq!(got_res_pad, vec![2.0f32, 4.0, 6.0, 3.0]);

        let res_pad_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut out_pad =
            unsafe { device.alloc_uninit(&Shape::from(params_pad.out_dims()), DType::F32)? };
        res_pad.copy_strided_src(&mut out_pad, 0, &res_pad_l)?;
        let out_pad_cpu = out_pad.to_cpu_storage()?;
        let got_out_pad = match out_pad_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv1d out stage (pad): {other:?}"),
        };
        assert_eq!(got_out_pad, vec![2.0f32, 4.0, 6.0, 3.0]);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_conv1d_large_multi_channel_stage_probe() -> Result<()> {
        fn deterministic_f32_data(len: usize, seed: u64) -> Vec<f32> {
            let mut state = seed | 1;
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let bits = ((state >> 41) as u32) | 0x3f80_0000;
                    (f32::from_bits(bits) - 1.0) * 2.0 - 1.0
                })
                .collect()
        }

        fn max_abs_diff(actual: &[f32], expected: &[f32]) -> (usize, f32, f32, f32) {
            let mut best_idx = 0usize;
            let mut best_actual = 0f32;
            let mut best_expected = 0f32;
            let mut best_diff = 0f32;
            for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let diff = (a - e).abs();
                if diff > best_diff {
                    best_idx = idx;
                    best_actual = *a;
                    best_expected = *e;
                    best_diff = diff;
                }
            }
            (best_idx, best_actual, best_expected, best_diff)
        }

        fn conv1d_im2col_reference(
            input: &[f32],
            c_in: usize,
            l_in: usize,
            k_size: usize,
            padding: usize,
        ) -> Vec<f32> {
            let l_out = l_in;
            let chw = c_in * k_size;
            let mut out = vec![0f32; l_out * chw];
            for ow in 0..l_out {
                for ic in 0..c_in {
                    for kx in 0..k_size {
                        let src_x = ow as isize + kx as isize - padding as isize;
                        let val = if (0..l_in as isize).contains(&src_x) {
                            input[ic * l_in + src_x as usize]
                        } else {
                            0.0
                        };
                        let chw_idx = ic * k_size + kx;
                        out[ow * chw + chw_idx] = val;
                    }
                }
            }
            out
        }

        let cpu = crate::Device::Cpu;
        let device = VulkanDevice::new(0)?;
        let c_in = 80usize;
        let c_out = 384usize;
        let l_in = 3000usize;
        let k_size = 3usize;
        let padding = 1usize;

        let src_vals = deterministic_f32_data(c_in * l_in, 0x5151);
        let kernel_vals = deterministic_f32_data(c_out * c_in * k_size, 0xA11CE);

        let src = device.storage_from_cpu_storage(&CpuStorage::F32(src_vals.clone()))?;
        let kernel = device.storage_from_cpu_storage(&CpuStorage::F32(kernel_vals.clone()))?;
        let src_l = Layout::contiguous((1, c_in, l_in));
        let params = crate::conv::ParamsConv1D {
            b_size: 1,
            l_in,
            c_out,
            c_in,
            k_size,
            padding,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };

        let col = src.run_im2col_conv1d_f32(&src_l, &params)?;
        let col_cpu = match col.to_cpu_storage()? {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from large conv1d im2col: {other:?}"),
        };
        let expected_col = conv1d_im2col_reference(&src_vals, c_in, l_in, k_size, padding);
        let (idx, got, expected, diff) = max_abs_diff(&col_cpu, &expected_col);
        assert!(
            diff <= 1e-6,
            "large conv1d im2col mismatch at idx {idx}: got {got}, expected {expected}, diff {diff}"
        );

        let b = params.b_size;
        let n = params.c_out;
        let l_out = params.l_out();
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b * m, k));
        let kernel_l_mm = Layout::contiguous_with_offset((n, k), 0).transpose(0, 1)?;
        let res = col.matmul(&kernel, (1, b * m, n, k), &col_l, &kernel_l_mm)?;
        let res_cpu = match res.to_cpu_storage()? {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from large conv1d matmul: {other:?}"),
        };

        let col_ref = crate::Tensor::from_slice(&expected_col, (l_out, k), &cpu)?;
        let kernel_ref = crate::Tensor::from_slice(&kernel_vals, (c_out, c_in, k_size), &cpu)?;
        let kernel_ref = kernel_ref.reshape((c_out, k))?;
        let expected_res = col_ref.matmul(&kernel_ref.t()?)?;
        let expected_res = expected_res.flatten_all()?.to_vec1::<f32>()?;
        let (idx, got, expected, diff) = max_abs_diff(&res_cpu, &expected_res);
        assert!(
            diff <= 1e-3,
            "large conv1d matmul mismatch at idx {idx}: got {got}, expected {expected}, diff {diff}"
        );

        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut out = unsafe { device.alloc_uninit(&Shape::from(params.out_dims()), DType::F32)? };
        res.copy_strided_src(&mut out, 0, &res_l)?;
        let out_cpu = match out.to_cpu_storage()? {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from large conv1d reorder: {other:?}"),
        };
        let expected_out = expected_res
            .chunks_exact(n)
            .enumerate()
            .flat_map(|(ow, row)| (0..n).map(move |oc| (oc, ow, row[oc])))
            .fold(vec![0f32; n * l_out], |mut acc, (oc, ow, value)| {
                acc[oc * l_out + ow] = value;
                acc
            });
        let (idx, got, expected, diff) = max_abs_diff(&out_cpu, &expected_out);
        assert!(
            diff <= 1e-3,
            "large conv1d reorder mismatch at idx {idx}: got {got}, expected {expected}, diff {diff}"
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_conv2d_native_matches_reference() -> Result<()> {
        let device = VulkanDevice::new(0)?;
        let src_vals = vec![
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let kernel_vals = vec![
            1.0f32, 0.0, //
            0.0, 1.0,
        ];
        let src = device.storage_from_cpu_storage(&CpuStorage::F32(src_vals))?;
        let kernel = device.storage_from_cpu_storage(&CpuStorage::F32(kernel_vals))?;
        let src_l = Layout::contiguous((1, 1, 3, 3));
        let kernel_l = Layout::contiguous((1, 1, 2, 2));
        let params = crate::conv::ParamsConv2D {
            b_size: 1,
            i_h: 3,
            i_w: 3,
            k_h: 2,
            k_w: 2,
            c_out: 1,
            c_in: 1,
            padding: 0,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let out = src.run_conv2d_f32(&src_l, &kernel, &kernel_l, &params)?;
        let out_cpu = out.to_cpu_storage()?;
        let got = match out_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv2d out stage: {other:?}"),
        };
        assert_eq!(got, vec![6.0f32, 8.0, 12.0, 14.0]);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_conv2d_im2col_matches_reference() -> Result<()> {
        let device = VulkanDevice::new(0)?;
        let src_vals = vec![
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let src = device.storage_from_cpu_storage(&CpuStorage::F32(src_vals))?;
        let src_l = Layout::contiguous((1, 1, 3, 3));
        let params = crate::conv::ParamsConv2D {
            b_size: 1,
            i_h: 3,
            i_w: 3,
            k_h: 2,
            k_w: 2,
            c_out: 1,
            c_in: 1,
            padding: 0,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let col = src.run_im2col_f32(&src_l, &params)?;
        let col_cpu = col.to_cpu_storage()?;
        let got = match col_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv2d im2col: {other:?}"),
        };
        let expected = vec![
            1.0f32, 2.0, 4.0, 5.0, //
            2.0, 3.0, 5.0, 6.0, //
            4.0, 5.0, 7.0, 8.0, //
            5.0, 6.0, 8.0, 9.0,
        ];
        assert_eq!(got, expected);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_conv2d_matmul_stage_matches_reference() -> Result<()> {
        let device = VulkanDevice::new(0)?;
        let src_vals = vec![
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        let kernel_vals = vec![
            1.0f32, 0.0, //
            0.0, 1.0,
        ];
        let src = device.storage_from_cpu_storage(&CpuStorage::F32(src_vals))?;
        let kernel = device.storage_from_cpu_storage(&CpuStorage::F32(kernel_vals))?;
        let src_l = Layout::contiguous((1, 1, 3, 3));
        let _kernel_l = Layout::contiguous((1, 1, 2, 2));
        let params = crate::conv::ParamsConv2D {
            b_size: 1,
            i_h: 3,
            i_w: 3,
            k_h: 2,
            k_w: 2,
            c_out: 1,
            c_in: 1,
            padding: 0,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };

        let col = src.run_im2col_f32(&src_l, &params)?;
        let h_out = params.out_h();
        let w_out = params.out_w();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_h * params.k_w * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b * m, k));
        let kernel_l_mm = Layout::contiguous_with_offset((n, k), 0).transpose(0, 1)?;
        let res = col.matmul(&kernel, (1, b * m, n, k), &col_l, &kernel_l_mm)?;

        let res_cpu = res.to_cpu_storage()?;
        let got_res = match res_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv2d matmul stage: {other:?}"),
        };
        assert_eq!(got_res, vec![6.0f32, 8.0, 12.0, 14.0]);

        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut out = unsafe { device.alloc_uninit(res_l.shape(), DType::F32)? };
        res.copy_strided_src(&mut out, 0, &res_l)?;
        let out_cpu = out.to_cpu_storage()?;
        let got_out = match out_cpu {
            CpuStorage::F32(v) => v,
            other => crate::bail!("unexpected dtype from conv2d reordered out: {other:?}"),
        };
        assert_eq!(got_out, vec![6.0f32, 8.0, 12.0, 14.0]);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_conv1d_after_matmul_regression_probe() -> Result<()> {
        let device = crate::Device::Vulkan(VulkanDevice::new(0)?);

        let lhs = crate::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)?;
        let rhs =
            crate::Tensor::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2), &device)?;
        let _ = lhs.matmul(&rhs)?.to_vec2::<f32>()?;

        let lhs_b = crate::Tensor::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, //
                2.0, 0.0, 1.0, 3.0, 4.0, 5.0,
            ],
            (2, 2, 3),
            &device,
        )?;
        let rhs_b = crate::Tensor::from_slice(
            &[
                7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, //
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
            (2, 3, 2),
            &device,
        )?;
        let _ = lhs_b.matmul(&rhs_b)?.to_vec3::<f32>()?;

        let input = crate::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device)?;
        let kernel = crate::Tensor::from_slice(&[1.0f32, 0.0, 1.0], (1, 1, 3), &device)?;
        let got = input.conv1d(&kernel, 0, 1, 1, 1)?.to_vec3::<f32>()?;
        assert_eq!(got, [[[4.0f32, 6.0]]]);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_conv1d_after_gather_scatter_probe() -> Result<()> {
        let device = crate::Device::Vulkan(VulkanDevice::new(0)?);

        let xs = crate::Tensor::from_slice(&[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], (3, 2), &device)?;
        let ids = crate::Tensor::from_slice(&[0u32, 1, 2, 0], (2, 2), &device)?;
        let _ = xs.gather(&ids, 0)?.to_vec2::<f32>()?;
        let base = crate::Tensor::zeros((3, 2), DType::F32, &device)?;
        let src = crate::Tensor::from_slice(&[7.0f32, 70.0, 8.0, 80.0], (2, 2), &device)?;
        let _ = base.scatter(&ids, &src, 0)?.to_vec2::<f32>()?;

        let xs_f16 = crate::Tensor::from_slice(
            &[
                half::f16::from_f32(1.0),
                half::f16::from_f32(10.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(20.0),
                half::f16::from_f32(3.0),
                half::f16::from_f32(30.0),
            ],
            (3, 2),
            &device,
        )?;
        let src_f16 = crate::Tensor::from_slice(
            &[
                half::f16::from_f32(7.0),
                half::f16::from_f32(70.0),
                half::f16::from_f32(8.0),
                half::f16::from_f32(80.0),
            ],
            (2, 2),
            &device,
        )?;
        let _ = xs_f16.gather(&ids, 0)?.to_vec2::<half::f16>()?;
        let base_f16 = crate::Tensor::zeros((3, 2), DType::F16, &device)?;
        let _ = base_f16
            .scatter(&ids, &src_f16, 0)?
            .to_vec2::<half::f16>()?;

        let xs_t = xs.t()?;
        let _ = xs_t.contiguous()?.to_vec2::<f32>()?;
        let xs_u32 = crate::Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (3, 2), &device)?;
        let _ = xs_u32.t()?.contiguous()?.to_vec2::<u32>()?;
        let _ = xs_f16.t()?.contiguous()?.to_vec2::<half::f16>()?;
        device.synchronize()?;

        let input = crate::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device)?;
        let kernel = crate::Tensor::from_slice(&[1.0f32, 0.0, 1.0], (1, 1, 3), &device)?;
        let got0 = input.conv1d(&kernel, 0, 1, 1, 1)?.to_vec3::<f32>()?;
        let got1 = input.conv1d(&kernel, 1, 1, 1, 1)?.to_vec3::<f32>()?;
        assert_eq!(got0, [[[4.0f32, 6.0]]]);
        assert_eq!(got1, [[[2.0f32, 4.0, 6.0, 3.0]]]);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_quantize_q8_1_x4_matches_cpu_pack() -> Result<()> {
        let device = VulkanDevice::new(0)?;
        let xs = (0..128)
            .map(|i| (i as f32 - 64.0) / 7.0)
            .collect::<Vec<_>>();
        let src = device.storage_from_cpu_storage(&CpuStorage::F32(xs.clone()))?;
        let packed = quantize_f32_storage_to_q8_1_x4(&device, &src, xs.len())?;
        let gpu = device.read_buffer(&packed)?;
        let cpu = exact_q8_1_x4_bytes(&xs);
        assert_eq!(gpu, cpu);
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_q8_1_qmatmul_matches_cpu_reference() -> Result<()> {
        let cpu = crate::Device::Cpu;
        let vk = crate::Device::Vulkan(VulkanDevice::new(0)?);
        let k = 256;
        let lhs_vals = (0..k).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
        let rhs_vals = (0..(k * 4))
            .map(|v| (v as f32 - 384.0) / 64.0)
            .collect::<Vec<_>>();

        let lhs_cpu = crate::Tensor::from_slice(&lhs_vals, (1, k), &cpu)?;
        let rhs_cpu = crate::Tensor::from_slice(&rhs_vals, (k, 4), &cpu)?;
        let lhs_vk = crate::Tensor::from_slice(&lhs_vals, (1, k), &vk)?;
        let rhs_vk = crate::Tensor::from_slice(&rhs_vals, (k, 4), &vk)?;

        let q_cpu = crate::quantized::QTensor::quantize(&rhs_cpu.t()?, GgmlDType::Q8_1)?;
        let q_vk = crate::quantized::QTensor::quantize(&rhs_vk.t()?, GgmlDType::Q8_1)?;
        let q_vk_direct = crate::quantized::QTensor::quantize(&rhs_vk.t()?, GgmlDType::Q8_1)?;

        assert_eq!(q_cpu.data()?.as_ref(), q_vk.data()?.as_ref());
        assert_eq!(q_vk.dequantize(&vk)?.shape().dims(), &[4, k]);

        let expected_qmm = crate::quantized::QMatMul::from_qtensor(q_cpu)?;
        let actual_qmm = crate::quantized::QMatMul::from_qtensor(q_vk)?;
        assert!(matches!(
            expected_qmm,
            crate::quantized::QMatMul::QTensor(_)
        ));
        assert!(matches!(actual_qmm, crate::quantized::QMatMul::QTensor(_)));

        let expected = expected_qmm.forward(&lhs_cpu)?.to_vec2::<f32>()?;
        let lhs_vk_storage = lhs_vk.storage();
        let direct = match &*lhs_vk_storage {
            crate::Storage::Vulkan(storage) => {
                let (out, shape) = <crate::quantized::QTensor as crate::CustomOp1>::vulkan_fwd(
                    &q_vk_direct,
                    storage,
                    lhs_vk.layout(),
                )?;
                crate::tensor::from_storage(
                    crate::Storage::Vulkan(out),
                    shape,
                    crate::op::BackpropOp::none(),
                    false,
                )
            }
            other => panic!("expected Vulkan storage, got {other:?}"),
        };
        drop(lhs_vk_storage);
        let direct = direct.to_vec2::<f32>()?;
        assert_eq!(direct.len(), expected.len());
        for (row_idx, (direct_row, expected_row)) in direct.iter().zip(expected.iter()).enumerate()
        {
            for (col_idx, (actual, expected)) in
                direct_row.iter().zip(expected_row.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1e-6,
                    "direct q8_1 vulkan_fwd mismatch at ({row_idx}, {col_idx}): got {actual}, expected {expected}"
                );
            }
        }
        let actual = actual_qmm.forward(&lhs_vk)?.to_vec2::<f32>()?;

        assert_eq!(actual.len(), expected.len());
        for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate()
        {
            for (col_idx, (actual, expected)) in
                actual_row.iter().zip(expected_row.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1e-6,
                    "q8_1 qmatmul mismatch at ({row_idx}, {col_idx}): got {actual}, expected {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_q8_1_qmatmul_matches_cpu_reference_no_warmup() -> Result<()> {
        let cpu = crate::Device::Cpu;
        let vk = crate::Device::Vulkan(VulkanDevice::new(0)?);
        let k = 256;
        let lhs_vals = (0..k).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
        let rhs_vals = (0..(k * 4))
            .map(|v| (v as f32 - 384.0) / 64.0)
            .collect::<Vec<_>>();

        let lhs_cpu = crate::Tensor::from_slice(&lhs_vals, (1, k), &cpu)?;
        let rhs_cpu = crate::Tensor::from_slice(&rhs_vals, (k, 4), &cpu)?;
        let lhs_vk = crate::Tensor::from_slice(&lhs_vals, (1, k), &vk)?;
        let rhs_vk = crate::Tensor::from_slice(&rhs_vals, (k, 4), &vk)?;

        let q_cpu = crate::quantized::QTensor::quantize(&rhs_cpu.t()?, GgmlDType::Q8_1)?;
        let q_vk = crate::quantized::QTensor::quantize(&rhs_vk.t()?, GgmlDType::Q8_1)?;

        let expected_qmm = crate::quantized::QMatMul::from_qtensor(q_cpu)?;
        let actual_qmm = crate::quantized::QMatMul::from_qtensor(q_vk)?;
        assert!(matches!(
            expected_qmm,
            crate::quantized::QMatMul::QTensor(_)
        ));
        assert!(matches!(actual_qmm, crate::quantized::QMatMul::QTensor(_)));

        let expected = expected_qmm.forward(&lhs_cpu)?.to_vec2::<f32>()?;
        let actual = actual_qmm.forward(&lhs_vk)?.to_vec2::<f32>()?;

        assert_eq!(actual.len(), expected.len());
        for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate()
        {
            for (col_idx, (actual, expected)) in
                actual_row.iter().zip(expected_row.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1e-6,
                    "q8_1 qmatmul no-warmup mismatch at ({row_idx}, {col_idx}): got {actual}, expected {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn vulkan_q8_0_indexed_moe_runtime_profile() -> Result<()> {
        let cpu = crate::Device::Cpu;
        let vk_dev = VulkanDevice::new(0)?;
        let vk = crate::Device::Vulkan(vk_dev.clone());
        let k = 256;
        let batch = 2;
        let topk = 2;

        let moe_w_vals = (0..(2 * 3 * k))
            .map(|v| (v as f32 - 3.0 * k as f32) / 128.0)
            .collect::<Vec<_>>();
        let moe_x_vals = (0..(batch * k))
            .map(|v| (v as f32 - k as f32 / 2.0) / 16.0)
            .collect::<Vec<_>>();

        let moe_w_cpu = crate::Tensor::from_slice(&moe_w_vals, (2, 3, k), &cpu)?;
        let moe_x_cpu = crate::Tensor::from_slice(&moe_x_vals, (batch, k), &cpu)?;
        let moe_ids_cpu = crate::Tensor::from_slice(&[0u32, 1, 1, 0], (batch, topk), &cpu)?;
        let q_moe_cpu = crate::quantized::QTensor::quantize(&moe_w_cpu, GgmlDType::Q8_0)?;

        let plain_expected = q_moe_cpu.indexed_moe_forward(&moe_x_cpu, &moe_ids_cpu)?;
        let x_q8_1 =
            crate::quantized::QTensor::quantize(&moe_x_cpu, GgmlDType::Q8_1)?.dequantize(&cpu)?;
        let q8_1_expected = {
            let weights = q_moe_cpu.dequantize(&cpu)?;
            let weight_vals = weights.to_vec3::<f32>()?;
            let x_vals = x_q8_1.to_vec2::<f32>()?;
            let id_vals = moe_ids_cpu.to_vec2::<u32>()?;
            let mut out = vec![0f32; batch * topk * 3];
            for batch_idx in 0..batch {
                for (topk_idx, expert_id) in id_vals[batch_idx].iter().take(topk).enumerate() {
                    let expert = *expert_id as usize;
                    let out_base = (batch_idx * topk + topk_idx) * 3;
                    for row in 0..3 {
                        let mut acc = 0f32;
                        for col in 0..k {
                            acc += x_vals[batch_idx][col] * weight_vals[expert][row][col];
                        }
                        out[out_base + row] = acc;
                    }
                }
            }
            crate::Tensor::from_vec(out, (batch, topk, 3), &cpu)?
        };

        let moe_w_vk = crate::Tensor::from_slice(&moe_w_vals, (2, 3, k), &vk)?;
        let moe_x_vk = crate::Tensor::from_slice(&moe_x_vals, (batch, k), &vk)?;
        let moe_ids_vk = crate::Tensor::from_slice(&[0u32, 1, 1, 0], (batch, topk), &vk)?;
        let q_moe_vk = crate::quantized::QTensor::quantize(&moe_w_vk, GgmlDType::Q8_0)?;
        let actual = q_moe_vk.indexed_moe_forward(&moe_x_vk, &moe_ids_vk)?;

        let use_q8_1_rhs = false;

        eprintln!(
            "vendor_id={:#x} integer_dot={} use_q8_1_rhs={}",
            vk_dev.inner.vendor_id, vk_dev.inner.integer_dot_product, use_q8_1_rhs
        );

        let actual = actual.to_vec3::<f32>()?;
        let plain_expected = plain_expected.to_vec3::<f32>()?;
        let q8_1_expected = q8_1_expected.to_vec3::<f32>()?;
        eprintln!(
            "actual[0][0][0]={:.6} plain[0][0][0]={:.6} q8_1[0][0][0]={:.6}",
            actual[0][0][0], plain_expected[0][0][0], q8_1_expected[0][0][0]
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn q8_1_from_data_roundtrip_matches_cpu_quantized() -> Result<()> {
        let cpu = crate::Device::Cpu;
        let k = 256;
        let lhs_vals = (0..k).map(|v| v as f32 / 32.0).collect::<Vec<_>>();
        let rhs_vals = (0..(k * 4))
            .map(|v| (v as f32 - 384.0) / 64.0)
            .collect::<Vec<_>>();

        let lhs = crate::Tensor::from_slice(&lhs_vals, (1, k), &cpu)?;
        let rhs = crate::Tensor::from_slice(&rhs_vals, (k, 4), &cpu)?;
        let q = crate::quantized::QTensor::quantize(&rhs.t()?, GgmlDType::Q8_1)?;
        let bytes = q.data()?.into_owned();
        let reparsed = GgmlDType::Q8_1.from_data(std::borrow::Cow::Owned(bytes));

        let expected = crate::quantized::QMatMul::from_qtensor(q)?.forward(&lhs)?;
        let lhs = lhs.to_vec2::<f32>()?;
        let lhs_row = &lhs[0];
        let mut dst = vec![0f32; 4];
        reparsed.matmul_t((1, k, 4), lhs_row, &mut dst)?;
        let actual = crate::Tensor::from_vec(dst, (1, 4), &cpu)?;

        for (row_idx, (actual_row, expected_row)) in actual
            .to_vec2::<f32>()?
            .iter()
            .zip(expected.to_vec2::<f32>()?.iter())
            .enumerate()
        {
            for (col_idx, (actual, expected)) in
                actual_row.iter().zip(expected_row.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1e-6,
                    "q8_1 from_data mismatch at ({row_idx}, {col_idx}): got {actual}, expected {expected}"
                );
            }
        }
        Ok(())
    }
}
