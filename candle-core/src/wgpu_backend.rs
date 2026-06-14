use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, Mul, ReduceOp, UnaryOpT};
use crate::quantized::GgmlDType;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, WithDType};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

const WG_SIZE: u32 = 256;

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
    seed_value: RwLock<u64>,
    pipeline_cache: Mutex<HashMap<WgpuPipelineCacheKey, Arc<WgpuCachedPipeline>>>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum WgpuBindingKindKey {
    Storage { read_only: bool },
    Uniform,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct WgpuPipelineCacheKey {
    shader: String,
    entries: Vec<(u32, WgpuBindingKindKey)>,
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
        let shader = candle_wgpu_kernels::rand_uniform_shader(wgpu_kernel_dtype(kernel_dtype)?, WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader rand_uniform.wgsl not embedded".into()).bt())?;
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
        let shader = candle_wgpu_kernels::rand_normal_shader(wgpu_kernel_dtype(kernel_dtype)?, WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader rand_normal.wgsl not embedded".into()).bt())?;
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

    fn create_storage_buffer(&self, size: usize, label: &'static str) -> wgpu::Buffer {
        self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: wgpu_copy_size(size) as u64,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn read_buffer(&self, buffer: &wgpu::Buffer, size: usize) -> Result<Vec<u8>> {
        let copy_size = wgpu_copy_size(size);
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
        self.inner.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.synchronize()?;
        rx.recv().map_err(Error::wrap)?.map_err(Error::wrap)?;
        let mut data = slice.get_mapped_range().to_vec();
        data.truncate(size);
        staging.unmap();
        Ok(data)
    }

    fn run_compute(
        &self,
        shader: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        bindings: &[wgpu::BindGroupEntry<'_>],
        workgroups: u32,
        label: &'static str,
    ) -> Result<()> {
        self.run_compute_xyz(shader, entries, bindings, (workgroups, 1, 1), label)
    }

    fn run_compute_xyz(
        &self,
        shader: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
        bindings: &[wgpu::BindGroupEntry<'_>],
        workgroups: (u32, u32, u32),
        label: &'static str,
    ) -> Result<()> {
        let cache_key = WgpuPipelineCacheKey {
            shader: shader.to_owned(),
            entries: entries
                .iter()
                .map(|entry| {
                    let kind = match entry.ty {
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only },
                            ..
                        } => WgpuBindingKindKey::Storage { read_only },
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            ..
                        } => WgpuBindingKindKey::Uniform,
                        _ => unreachable!("unsupported wgpu binding type for compute cache"),
                    };
                    (entry.binding, kind)
                })
                .collect(),
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
                    cache.insert(cache_key, cached.clone());
                    cached
                }
            };
        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &cached.bind_group_layout,
                entries: bindings,
            });
        let mut encoder = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.inner.queue.submit([encoder.finish()]);
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
    let max_per_dim = max_per_dim.max(1);
    let wg_y = std::cmp::max(1, total_wg.div_ceil(max_per_dim));
    let wg_x = total_wg.div_ceil(wg_y);
    (wg_x, wg_y)
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
    if matches!(dtype, DType::U8 | DType::U32 | DType::I64) {
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
    let logical_idx = gid.x;
    let coords = decompose_idx(logical_idx);
    let pred = cond_is_true(params.offset_cond + cond_index(coords));
    let t = on_true[params.offset_true + true_index(coords)];
    let f = on_false[params.offset_false + false_index(coords)];
    dst[params.offset_dst + logical_idx] = select(f, t, pred);
}}"#
    );
    let u8_main = format!(
        r#"@compute @workgroup_size({WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let base = gid.x * 4u;
    if (base >= params.ne) {{ return; }}
    var out_word: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {{
        let logical_idx = base + lane;
        if (logical_idx >= params.ne) {{ break; }}
        let coords = decompose_idx(logical_idx);
        let pred = cond_is_true(params.offset_cond + cond_index(coords));
        let t_idx = params.offset_true + true_index(coords);
        let f_idx = params.offset_false + false_index(coords);
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
    let logical_idx = gid.x;
    let coords = decompose_idx(logical_idx);
    let pred = cond_is_true(params.offset_cond + cond_index(coords));
    let t_idx = params.offset_true + true_index(coords);
    let f_idx = params.offset_false + false_index(coords);
    let lo = select(on_false[2u * f_idx], on_true[2u * t_idx], pred);
    let hi = select(on_false[2u * f_idx + 1u], on_true[2u * t_idx + 1u], pred);
    dst[2u * (params.offset_dst + logical_idx)] = lo;
    dst[2u * (params.offset_dst + logical_idx) + 1u] = hi;
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

fn buffer_binding<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
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
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-unary-params"),
                size: std::mem::size_of::<UnaryParams>() as u64,
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
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        self.device
            .run_compute(shader, &entries, &bindings, workgroups, label)?;
        Ok(dst)
    }

    fn run_scale(&self, layout: &Layout, scale: f32, bias: f32) -> Result<Self> {
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let params = ScaleParams {
            offset_src: layout.start_offset().try_into()?,
            offset_dst: 0,
            stride_src1: strides[1],
            stride_src2: strides[2],
            stride_src3: strides[3],
            stride_dst1: dims[0],
            stride_dst2: dims[0] * dims[1],
            stride_dst3: dims[0] * dims[1] * dims[2],
            ne: count.try_into()?,
            ne0: dims[0],
            ne1: dims[1],
            ne2: dims[2],
            scale,
            bias,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-scale-params"),
                size: std::mem::size_of::<ScaleParams>() as u64,
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
        let shader = candle_wgpu_kernels::scale_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader scale.wgsl not embedded".into()).bt())?;
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-scale",
        )?;
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
            offset_cond: layout.start_offset().try_into()?,
            offset_true: t_l.start_offset().try_into()?,
            offset_false: f_l.start_offset().try_into()?,
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
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &t.buffer),
            buffer_binding(2, &f.buffer),
            buffer_binding(3, &dst.buffer),
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
    let base = gid.x * 4u;
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
    dst[gid.x] = w;
"#
            ),
            DType::F16 => format!(
                r#"
    let base = gid.x * 2u;
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
    dst[gid.x] = pack2x16float(vec2<f32>(v0, v1));
"#
            ),
            DType::BF16 => format!(
                r#"
    let base = gid.x * 2u;
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
    dst[gid.x] = w;
"#
            ),
            DType::F32 => format!(
                r#"
    let i = gid.x;
    if (i >= params.ne) {{ return; }}
    {load}
    dst[i] = bitcast<u32>({conv_f32});
"#
            ),
            DType::U32 => format!(
                r#"
    let i = gid.x;
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
    let i = gid.x;
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
    let base = gid.x * 2u;
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
    dst[gid.x] = w;
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
    _pad0: u32,
    _pad1: u32,
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
                label: Some("candle-wgpu-emulated-cast-params"),
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
        let groups_of = match dst_dtype {
            DType::U8 => 4,
            DType::F16 | DType::BF16 => 2,
            _ => 1,
        };
        let work_items = ne.div_ceil(groups_of);
        let workgroups: u32 = work_items.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-emulated-cast",
        )?;
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
        let params = CopyParams {
            ne: count.try_into()?,
            offset_src: layout.start_offset().try_into()?,
            offset_dst: dst_offset.try_into()?,
            stride_src0: src_strides[0],
            stride_src1: src_strides[1],
            stride_src2: src_strides[2],
            stride_src3: src_strides[3],
            stride_dst0: 1,
            stride_dst1: 0,
            stride_dst2: 0,
            stride_dst3: 0,
            src_ne0: src_dims[0],
            src_ne1: src_dims[1],
            src_ne2: src_dims[2],
            dst_ne0: src_dims[0],
            dst_ne1: src_dims[1],
            dst_ne2: src_dims[2],
        };
        let body = match self.dtype {
            DType::BF16 | DType::I16 => {
                // One thread per packed 16-bit output word: gathers two strided
                // source halfwords, so concurrent threads never share a word.
                r#"
    let word_idx = params.offset_dst / 2u + gid.x;
    let word_base = word_idx * 2u;
    if (word_base >= params.offset_dst + params.ne) { return; }
    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 2u; lane = lane + 1u) {
        let elem_idx = word_base + lane;
        var bits: u32;
        if (elem_idx >= params.offset_dst && elem_idx < params.offset_dst + params.ne) {
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
                // Bytes past params.ne in the final word are merged from the
                // existing destination so neighbors in dst are not clobbered.
                r#"
    let base = gid.x * 4u;
    if (base >= params.ne) { return; }
    let word_idx = params.offset_dst / 4u + gid.x;
    var w: u32 = 0u;
    for (var lane: u32 = 0u; lane < 4u; lane = lane + 1u) {
        let logical_idx = base + lane;
        var b: u32;
        if (logical_idx < params.ne) {
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
    let e = params.offset_src + src_index(gid.x);
    let d = params.offset_dst + gid.x;
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
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-emulated-copy-params"),
                size: std::mem::size_of::<CopyParams>() as u64,
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
        let workgroups: u32 = work_items.try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-emulated-copy",
        )?;
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

        let max_workgroups = self
            .device
            .inner
            .limits
            .max_compute_workgroups_per_dimension as usize;
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
            let chunk = if chunk_linear {
                (count - processed).min(max_elems_per_dispatch)
            } else {
                if count > max_elems_per_dispatch {
                    return Err(unsupported("copy too large for non-contiguous layout"));
                }
                count - processed
            };
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
                }
            } else {
                CopyParams {
                    ne: count.try_into()?,
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
            if !chunk_linear {
                break;
            }
        }
        Ok(())
    }

    fn run_argmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        if !layout.is_contiguous() {
            return Err(unsupported("argmax strided"));
        }
        let rank = layout.dims().len();
        let ne0 = *layout
            .dims()
            .last()
            .ok_or_else(|| unsupported("argmax scalar"))?;
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
        last_dim: usize,
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
        if !layout.is_contiguous() {
            return Err(unsupported("argsort strided"));
        }
        if last_dim == 0 || layout.dims().last().copied() != Some(last_dim) {
            return Err(unsupported("argsort last-dim"));
        }
        let workgroup_size = next_power_of_two_u32(last_dim.min(WG_SIZE as usize), "argsort")?;
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let nrows = count / last_dim;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), DType::U32)? };
        let dst_strides = contiguous_strides(dims);
        let npr = last_dim.div_ceil(workgroup_size as usize);
        let top_k = if npr == 1 {
            last_dim.try_into()?
        } else {
            workgroup_size
        };
        let params = ArgsortParams {
            offset_src: layout.start_offset().try_into()?,
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
        if !layout.is_contiguous() {
            return Err(unsupported("cumsum strided"));
        }
        let ne0 = *layout
            .dims()
            .last()
            .ok_or_else(|| unsupported("cumsum scalar"))?;
        let rows = layout.shape().elem_count() / ne0;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let params = CumsumParams {
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
            return Err(unsupported("softmax strided"));
        }
        if self.dtype == DType::F16 {
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape().clone());
            let out_f32 = src_f32.softmax_last_dim(&src_f32_layout)?;
            return out_f32.to_dtype(&src_f32_layout, DType::F16);
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
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &dst.buffer),
            buffer_binding(2, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::softmax_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader soft_max.wgsl not embedded".into()).bt())?;
        let rows = count / layout.dims()[layout.dims().len() - 1];
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.try_into()?,
            "candle-wgpu-softmax",
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu rope").bt());
        }
        if wgpu_f16_emulates_f32(&self.device, self.dtype) {
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let src_l = Layout::contiguous(layout.shape().clone());
            let out_f32 = src_f32.ggml_rope(
                &src_l,
                pos,
                pos_layout,
                n_dims,
                freq_base,
                mode,
            )?;
            return out_f32.to_dtype(&src_l, DType::F16);
        }
        if pos.dtype != DType::I32 {
            return Err(Error::UnsupportedDTypeForOp(pos.dtype, "wgpu rope positions").bt());
        }
        if !layout.is_contiguous() || !pos_layout.is_contiguous() {
            return Err(unsupported("rope strided"));
        }
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let dst_strides = contiguous_strides(dims);
        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        let params = WgpuRopeParams {
            offset_src0: layout.start_offset().try_into()?,
            offset_src1: pos_layout.start_offset().try_into()?,
            offset_src2: 0,
            offset_dst: 0,
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
        let shader = candle_wgpu_kernels::rope_shader(wgpu_kernel_dtype(self.dtype)?, WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader rope.wgsl not embedded".into()).bt())?;
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
        if self.dtype == DType::F16 || alpha.dtype == DType::F16 {
            if self.dtype != DType::F16 || alpha.dtype != DType::F16 {
                return Err(unsupported("rms_norm mixed dtype"));
            }
            let src_f32 = self.to_dtype(layout, DType::F32)?;
            let alpha_f32 = alpha.to_dtype(alpha_layout, DType::F32)?;
            let src_f32_layout = Layout::contiguous(layout.shape().clone());
            let alpha_f32_layout = Layout::contiguous(alpha_layout.shape().clone());
            let out_f32 =
                src_f32.rms_norm(&src_f32_layout, &alpha_f32, &alpha_f32_layout, eps)?;
            let copy = copy_shader(DType::F32, DType::F16)?;
            return out_f32.run_copy_to_dtype(&src_f32_layout, DType::F16, &copy);
        }
        if self.dtype != DType::F32 || alpha.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu rms_norm").bt());
        }
        if !layout.is_contiguous() || !alpha_layout.is_contiguous() {
            return Err(unsupported("rms_norm strided"));
        }
        let (dims, strides) = dims4(layout)?;
        let (alpha_dims, alpha_strides) = dims4(alpha_layout)?;
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
        let dst_strides = contiguous_strides(dims);
        let params = RmsNormMulParams {
            offset_rn_src: layout.start_offset().try_into()?,
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
        let bindings = [
            buffer_binding(0, &self.buffer),
            buffer_binding(1, &alpha.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader = candle_wgpu_kernels::rms_norm_mul_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader rms_norm_mul.wgsl not embedded".into()).bt())?;
        let rows = count / layout.dims()[layout.dims().len() - 1];
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.try_into()?,
            "candle-wgpu-rms-norm",
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
            let mut dst_dims = src_l.dims().to_vec();
            dst_dims[dim] = match ids_l.dims() {
                [ids_len] => *ids_len,
                _ => return Err(unsupported("index_select ids rank")),
            };
            return out_f32.to_dtype(&Layout::contiguous(Shape::from(dst_dims)), src.dtype);
        }
        let ids_len = match ids_l.dims() {
            [ids_len] => *ids_len,
            _ => return Err(unsupported("index_select ids rank")),
        };
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
        if rank == 0 || ids_l.dims().len() != rank {
            return Err(unsupported("gather rank"));
        }
        let ids_dim = ids_l.dims()[rank - 1];
        let left_size: usize = ids_l.dims()[..rank - 1].iter().product();
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
        if rank == 0 || ids_l.dims().len() != rank || src_l.dims().len() != rank {
            return Err(unsupported("scatter_set rank"));
        }
        let ids_dim = ids_l.dims()[rank - 1];
        let left_size: usize = ids_l.dims()[..rank - 1].iter().product();
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
        if rank == 0 || src_l.dims().len() != rank {
            return Err(unsupported("scatter_add rank"));
        }
        let ids_dim = src_l.dims()[rank - 1];
        let left_size: usize = src_l.dims()[..rank - 1].iter().product();
        // `scatter_add` passes ids shaped like `src`; `index_add` passes one
        // rank-1 id row shared by every leading row, which maps to a zero ids
        // row stride in the shader.
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
            return Err(unsupported("matmul mixed dtype"));
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
            let lhs_l = if matches!(self.dtype, DType::F16 | DType::BF16) {
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
        if rank != rhs_l.dims().len() || !(2..=4).contains(&rank) {
            return Err(unsupported("matmul rank"));
        }
        if b != lhs_l.dims()[..rank - 2].iter().product::<usize>() {
            return Err(unsupported("matmul batch"));
        }
        if matches!(self.dtype, DType::F16 | DType::BF16)
            || wgpu_f16_emulates_f32(&self.device, self.dtype)
        {
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
        let rhs_t_stride = rhs_t_layout.stride();
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
        let rhs_stride_batch_inner = if rank >= 3 {
            rhs_t_stride[rank - 3]
        } else {
            n * k
        };
        let rhs_stride_batch_outer = if rank >= 4 {
            rhs_t_stride[rank - 4]
        } else {
            b * n * k
        };

        let dst_shape = Shape::from(vec![b, m, n]);
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };
        let params = MulMatParams {
            offset_src0: 0,
            offset_src1: lhs_layout.start_offset().try_into()?,
            offset_dst: 0,
            m: n.try_into()?,
            n: m.try_into()?,
            k: k.try_into()?,
            stride_01: k.try_into()?,
            stride_11: lhs_stride[rank - 2].try_into()?,
            stride_02: rhs_stride_batch_inner.try_into()?,
            stride_12: lhs_stride_batch_inner.try_into()?,
            stride_03: rhs_stride_batch_outer.try_into()?,
            stride_13: lhs_stride_batch_outer.try_into()?,
            bs02: bs02.try_into()?,
            bs03: bs03.try_into()?,
            broadcast2: 1,
            broadcast3: 1,
        };
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-matmul-params"),
                size: std::mem::size_of::<MulMatParams>() as u64,
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
            buffer_binding(0, &rhs_t.buffer),
            buffer_binding(1, &lhs.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let shader_storage;
        let matmul_label: &'static str;
        let shader: &str = match self.dtype {
            DType::F32 => {
                matmul_label = "candle-wgpu-matmul";
                shader_storage = candle_wgpu_kernels::matmul_f32_shader().ok_or_else(|| {
                    Error::Msg("wgpu shader mul_mat.wgsl not embedded".into()).bt()
                })?;
                &shader_storage
            }
            DType::F16 => {
                matmul_label = "candle-wgpu-matmul";
                shader_storage = candle_wgpu_kernels::matmul_f16_shader().ok_or_else(|| {
                    Error::Msg("wgpu shader mul_mat.wgsl not embedded".into()).bt()
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
        let workgroups = (b * m * n).try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            shader,
            &entries,
            &bindings,
            workgroups,
            matmul_label,
        )?;
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
        let (input_t, input_mm_l) = if input_t_view_l.is_contiguous() && input_t_view_l.start_offset() == 0
        {
            (
                &input_f32,
                Layout::contiguous((params.b_size * params.l_in, params.c_in)),
            )
        } else {
            let mut tmp =
                unsafe { self.device.alloc_uninit(input_t_view_l.shape(), DType::F32)? };
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
        let cols_l = Layout::contiguous((
            params.b_size,
            params.l_in,
            params.c_out,
            params.k_size,
        ));
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
        let input_spatial = params
            .i_h
            .checked_mul(params.i_w)
            .ok_or_else(|| Error::Msg("wgpu conv_transpose2d input_spatial overflow".into()).bt())?;
        let kernel_spatial = params
            .k_h
            .checked_mul(params.k_w)
            .ok_or_else(|| Error::Msg("wgpu conv_transpose2d kernel_spatial overflow".into()).bt())?;
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
                let mut tmp =
                    unsafe { self.device.alloc_uninit(input_hw_view_l.shape(), DType::F32)? };
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
        let cols_l = Layout::contiguous((
            params.b_size,
            input_spatial,
            params.c_out,
            kernel_spatial,
        ));
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
            buffer_binding(1, &ids.buffer),
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
        if !(2..=4).contains(&rank) {
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
        let (wg_x, wg_y) = compute_2d_workgroups(
            total_wg,
            storage
                .device
                .inner
                .limits
                .max_compute_workgroups_per_dimension,
        );
        storage.device.run_compute_xyz(
            &shader,
            &entries,
            &bindings,
            (wg_x, wg_y, 1),
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
        let (wg_x, wg_y) = compute_2d_workgroups(
            total_wg,
            storage
                .device
                .inner
                .limits
                .max_compute_workgroups_per_dimension,
        );
        storage.device.run_compute_xyz(
            &shader,
            &entries,
            &bindings,
            (wg_x, wg_y, 1),
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
                return Err(unsupported("int reduce dim out of range"));
            }
        }
        if reduce_dims.len() > 1 {
            if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                return Err(unsupported("int reduce multi-dim arg"));
            }
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
            return current.ok_or_else(|| unsupported("int reduce multi-dim empty"));
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
        let kx = *layout
            .dims()
            .last()
            .ok_or_else(|| unsupported("int reduce scalar"))?;
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
            let buffer = self.device.create_storage_buffer(size, "candle-wgpu-clone");
            let mut encoder =
                self.device
                    .inner
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("candle-wgpu-clone"),
                    });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, wgpu_copy_size(size) as u64);
            self.device.inner.queue.submit([encoder.finish()]);
            return Ok(Self {
                buffer: Arc::new(buffer),
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
            DType::BF16 => self.bf16_unary_via_f32(layout, |src, src_l| {
                src.run_scale(src_l, mul as f32, add as f32)
            }),
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
            let shader = custom_unary_wgsl(&format!(
                "select({a} * (exp(x) - 1.0), x, x > 0.0)"
            ));
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
            && (matches!(B::NAME, "erf" | "recip")
                || !wgpu_shader_f16_enabled(&self.device))
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
            DType::F32 | DType::F16 | DType::U8 | DType::U32 | DType::I64
        ) {
            if self.dtype == DType::BF16 {
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "op").bt());
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
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-binary-params"),
                size: std::mem::size_of::<BinaryParams>() as u64,
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
            buffer_binding(1, &rhs.buffer),
            buffer_binding(2, &dst.buffer),
            buffer_binding(3, &param_buffer),
        ];
        let work_items = match self.dtype {
            // One thread per packed four-byte output word.
            DType::U8 => count.div_ceil(4),
            _ => count,
        };
        let workgroups = (work_items as u32).div_ceil(WG_SIZE);
        match self.device.run_compute(
            &binary_shader(B::NAME, self.dtype)?,
            &entries,
            &bindings,
            workgroups,
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
        self.gpu_resident_via_f32(layout, &out_shape, "wgpu upsample_nearest2d", |src, src_l| {
            src.run_upsample2d_f32(
                src_l,
                out_h,
                out_w,
                nearest_interp_weights(h, out_h),
                nearest_interp_weights(w, out_w),
            )
        })
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
        self.gpu_resident_via_f32(layout, &out_shape, "wgpu upsample_bilinear2d", |src, src_l| {
            src.run_upsample2d_f32(
                src_l,
                out_h,
                out_w,
                bilinear_interp_weights(h, out_h, align_corners, scale_h),
                bilinear_interp_weights(w, out_w, align_corners, scale_w),
            )
        })
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
            let mut compact =
                unsafe { storage.device.alloc_uninit(permuted_l.shape(), storage.dtype)? };
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
            let mut compact =
                unsafe { storage.device.alloc_uninit(permuted_l.shape(), storage.dtype)? };
            storage.copy_strided_src(&mut compact, 0, &permuted_l)?;
            Ok((compact, Layout::contiguous(permuted_l.shape())))
        };
        let dst_perm_l = dst_l.permute(&perm)?;
        let mut dst_p =
            unsafe { self.device.alloc_uninit(dst_perm_l.shape(), self.dtype)? };
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
            let mut compact =
                unsafe { storage.device.alloc_uninit(permuted_l.shape(), storage.dtype)? };
            storage.copy_strided_src(&mut compact, 0, &permuted_l)?;
            Ok((compact, Layout::contiguous(permuted_l.shape())))
        };
        let dst_perm_l = dst_l.permute(&perm)?;
        let mut dst_p =
            unsafe { self.device.alloc_uninit(dst_perm_l.shape(), self.dtype)? };
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
            let mut compact =
                unsafe { storage.device.alloc_uninit(permuted_l.shape(), storage.dtype)? };
            storage.copy_strided_src(&mut compact, 0, &permuted_l)?;
            Ok((compact, Layout::contiguous(permuted_l.shape())))
        };
        let dst_perm_l = dst_l.permute(&perm)?;
        let mut dst_p =
            unsafe { self.device.alloc_uninit(dst_perm_l.shape(), self.dtype)? };
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
                src_f32.copy_strided_src(&mut dst_f32, dst_offset, &Layout::contiguous(src_l.shape()))?;
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
        instance_desc.backends = wgpu::Backends::all();
        let backends = instance_desc.backends;
        let instance = wgpu::Instance::new(instance_desc);
        let mut adapters = pollster::block_on(instance.enumerate_adapters(backends));
        if adapters.is_empty() {
            crate::bail!("no wgpu adapters found")
        }
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
        let required_features = wgpu::Features::SHADER_F64
            | (adapter_features & wgpu::Features::SHADER_F16);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("candle-wgpu"),
            required_features,
            required_limits: adapter_limits.clone(),
            ..Default::default()
        }))
        .map_err(Error::wrap)?;
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
                seed_value: RwLock::new(299_792_458),
                pipeline_cache: Mutex::new(HashMap::new()),
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
        let buffer = self.create_storage_buffer(size, "candle-wgpu-zeros");
        self.inner
            .queue
            .write_buffer(&buffer, 0, &vec![0u8; wgpu_copy_size(size)]);
        Ok(WgpuStorage {
            buffer: Arc::new(buffer),
            device: self.clone(),
            count,
            dtype,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let size = byte_len(dtype, count, "wgpu alloc_uninit")?;
        let buffer = self.create_storage_buffer(size, "candle-wgpu-alloc-uninit");
        Ok(WgpuStorage {
            buffer: Arc::new(buffer),
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
        let buffer = self.create_storage_buffer(bytes.len(), "candle-wgpu-upload");
        self.inner
            .queue
            .write_buffer(&buffer, 0, &wgpu_padded_write_bytes(&bytes));
        Ok(WgpuStorage {
            buffer: Arc::new(buffer),
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
        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(Error::wrap)?;
        Ok(())
    }
}
