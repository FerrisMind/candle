use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, WithDType};
use std::sync::Arc;

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
    device: wgpu::Device,
    queue: wgpu::Queue,
    features: wgpu::Features,
}

#[derive(Debug, Clone)]
pub struct WgpuStorage {
    buffer: Arc<wgpu::Buffer>,
    device: WgpuDevice,
    count: usize,
    dtype: DType,
}

fn unsupported(op: &'static str) -> Error {
    Error::Msg(format!("wgpu backend op {op} not implemented").into()).bt()
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
        rx.recv()
            .map_err(|e| Error::wrap(e))?
            .map_err(|e| Error::wrap(e))?;
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
        let module = self
            .inner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });
        let bind_group_layout =
            self.inner
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries,
                });
        let pipeline_layout =
            self.inner
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let pipeline =
            self.inner
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
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
        .ok_or_else(|| Error::Msg(format!("wgpu backend op {op} dimension overflow").into()).bt())?
        .try_into()
        .map_err(Error::wrap)
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

fn unary_shader(op: &str, dtype: DType) -> Result<String> {
    if op == "recip" {
        if dtype != DType::F32 {
            return Err(unsupported("unary recip f16"));
        }
        return Ok(custom_unary_wgsl("1.0 / x"));
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
        _ => return Err(Error::Msg(format!("wgpu backend op {op} not implemented").into()).bt()),
    };
    Ok(candle_wgpu_kernels::shader(
        candle_wgpu_kernels::ShaderOp::Unary(op),
        wgpu_kernel_dtype(dtype)?,
        WG_SIZE,
    ))
}

fn binary_shader(op: &str, dtype: DType) -> Result<String> {
    if op == "maximum" {
        if dtype != DType::F32 {
            return Err(unsupported("binary maximum f16"));
        }
        return Ok(custom_binary_wgsl("max(a, b)"));
    }
    if op == "minimum" {
        if dtype != DType::F32 {
            return Err(unsupported("binary minimum f16"));
        }
        return Ok(custom_binary_wgsl("min(a, b)"));
    }
    let op = match op {
        "add" => candle_wgpu_kernels::BinaryOp::Add,
        "div" => candle_wgpu_kernels::BinaryOp::Div,
        "mul" => candle_wgpu_kernels::BinaryOp::Mul,
        "sub" => candle_wgpu_kernels::BinaryOp::Sub,
        _ => return Err(Error::Msg(format!("wgpu backend op {op} not implemented").into()).bt()),
    };
    Ok(candle_wgpu_kernels::shader(
        candle_wgpu_kernels::ShaderOp::Binary(op),
        wgpu_kernel_dtype(dtype)?,
        WG_SIZE,
    ))
}

fn copy_shader(src: DType, dst: DType) -> Result<String> {
    let defines = match (src, dst) {
        (DType::F32, DType::F32) => ["SRC_F32", "DST_F32"],
        (DType::F32, DType::F16) => ["SRC_F32", "DST_F16"],
        (DType::F32, DType::I32) => ["SRC_F32", "DST_I32"],
        (DType::F16, DType::F32) => ["SRC_F16", "DST_F32"],
        (DType::F16, DType::F16) => ["SRC_F16", "DST_F16"],
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

fn custom_binary_wgsl(expr: &str) -> String {
    format!(
        r#"
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

@group(0) @binding(0) var<storage, read_write> src0: array<f32>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
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
    )
}

fn custom_unary_wgsl(expr: &str) -> String {
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
    dst[params.offset_dst + gid.x] = {expr};
}}
"#
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

    fn run_fill_inplace(&self, layout: &Layout, value: f32) -> Result<()> {
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
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

    fn run_copy_to_dtype(&self, layout: &Layout, dtype: DType, shader: &str) -> Result<Self> {
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), dtype)? };
        self.run_copy_into(layout, &dst, 0, shader)?;
        Ok(dst)
    }

    fn run_copy_into(
        &self,
        layout: &Layout,
        dst: &Self,
        dst_offset: usize,
        shader: &str,
    ) -> Result<()> {
        let (src_dims, src_strides) = dims4(layout)?;
        let dst_strides = contiguous_strides(src_dims);
        let count = layout.shape().elem_count();
        let params = CopyParams {
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
            .run_compute(shader, &entries, &bindings, workgroups, "candle-wgpu-copy")?;
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

    pub(crate) fn argsort_last_dim_f32(
        &self,
        layout: &Layout,
        asc: bool,
        last_dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu argsort").bt());
        }
        if !layout.is_contiguous() {
            return Err(unsupported("argsort strided"));
        }
        if last_dim == 0 || layout.dims().last().copied() != Some(last_dim) {
            return Err(unsupported("argsort last-dim"));
        }
        let workgroup_size = next_power_of_two_u32(last_dim, "argsort")?;
        if workgroup_size > WG_SIZE {
            return Err(unsupported("argsort last-dim > 256"));
        }
        let (dims, strides) = dims4(layout)?;
        let count = layout.shape().elem_count();
        let nrows = count / last_dim;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), DType::U32)? };
        let dst_strides = contiguous_strides(dims);
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
            top_k: last_dim.try_into()?,
            npr: 1,
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
        let shader = candle_wgpu_kernels::argsort_shader(workgroup_size, asc)
            .ok_or_else(|| Error::Msg("wgpu shader argsort.wgsl not embedded".into()).bt())?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            nrows.try_into()?,
            "candle-wgpu-argsort",
        )?;
        Ok(dst)
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
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu softmax").bt());
        }
        if !layout.is_contiguous() {
            return Err(unsupported("softmax strided"));
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

    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_layout: &Layout,
        eps: f32,
    ) -> Result<Self> {
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
        if self.dtype == DType::F16
            && !self
                .device
                .inner
                .features
                .contains(wgpu::Features::SHADER_F16)
        {
            return Err(unsupported("sigmoid f16"));
        }
        let shader = unary_shader("sigmoid", self.dtype)?;
        self.run_unary_like(layout, &shader, "candle-wgpu-sigmoid")
    }

    fn run_index_select_f32(
        &self,
        ids: &Self,
        src_l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu index_select").bt());
        }
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype, "wgpu index_select ids").bt());
        }
        if !src_l.is_contiguous() {
            return Err(unsupported("index_select strided"));
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
        let dst = unsafe { self.device.alloc_uninit(&dst_shape, self.dtype)? };
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
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-get-rows-params"),
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
        let shader = candle_wgpu_kernels::get_rows_f32_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader get_rows.wgsl not embedded".into()).bt())?;
        let rows: u32 = (left_size * ids_len).try_into()?;
        let workgroups = rows.div_ceil(WG_SIZE);
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-get-rows",
        )?;
        debug_assert_eq!(dst.count, dst_el);
        Ok(dst)
    }

    fn run_gather_last_dim_f32(&self, ids: &Self, src_l: &Layout, ids_l: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu gather").bt());
        }
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype, "wgpu gather ids").bt());
        }
        if !src_l.is_contiguous() {
            return Err(unsupported("gather strided src"));
        }
        if !ids_l.is_contiguous() {
            return Err(unsupported("gather strided ids"));
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
        let param_buffer = self
            .device
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("candle-wgpu-gather-params"),
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
        let shader = candle_wgpu_kernels::get_rows_f32_shader(WG_SIZE)
            .ok_or_else(|| Error::Msg("wgpu shader get_rows.wgsl not embedded".into()).bt())?;
        let rows: u32 = (left_size * ids_dim).try_into()?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows.div_ceil(WG_SIZE),
            "candle-wgpu-gather",
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
        if self.dtype != DType::F32 || src.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu scatter_set").bt());
        }
        if ids.dtype != DType::U32 {
            return Err(Error::UnsupportedDTypeForOp(ids.dtype, "wgpu scatter_set ids").bt());
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
        let shader = candle_wgpu_kernels::set_rows_f32_shader(WG_SIZE)
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

    fn run_matmul_f32(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu matmul").bt());
        }
        let rank = lhs_l.dims().len();
        if rank != rhs_l.dims().len() || rank < 2 || rank > 4 {
            return Err(unsupported("matmul rank"));
        }
        if b != lhs_l.dims()[..rank - 2].iter().product::<usize>() {
            return Err(unsupported("matmul batch"));
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
        let rhs_t_shape = rhs_t_src_layout.shape().clone();
        let mut rhs_t = unsafe { rhs.device.alloc_uninit(&rhs_t_shape, rhs.dtype)? };
        rhs.copy_strided_src(&mut rhs_t, 0, &rhs_t_src_layout)?;

        let lhs_stride = lhs_layout.stride();
        let rhs_t_stride = rhs_t_shape.stride_contiguous();
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
        let shader = candle_wgpu_kernels::matmul_f32_shader()
            .ok_or_else(|| Error::Msg("wgpu shader mul_mat.wgsl not embedded".into()).bt())?;
        let workgroups = (b * m * n).try_into().map(|v: u32| v.div_ceil(WG_SIZE))?;
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-matmul",
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
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        if !layout.is_contiguous()
            || layout.start_offset() != 0
            || layout.shape().elem_count() != self.count
        {
            return Err(unsupported("try_clone_strided"));
        }
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
        Ok(Self {
            buffer: Arc::new(buffer),
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
        let size = byte_len(self.dtype, self.count, "wgpu download")?;
        let bytes = self.device.read_buffer(&self.buffer, size)?;
        bytes_to_cpu_storage(self.dtype, self.count, &bytes)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu affine").bt());
        }
        self.run_scale(layout, mul as f32, add as f32)
    }
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu powf").bt());
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
            return Err(unsupported("elu f16"));
        }
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu elu").bt());
        }
        if alpha != 1.0 {
            return Err(unsupported("elu alpha"));
        }
        let shader = unary_shader("elu", self.dtype)?;
        self.run_unary_like(layout, &shader, "candle-wgpu-elu")
    }
    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu reduce").bt());
        }
        let rank = layout.dims().len();
        if rank == 0 {
            return Err(unsupported("reduce scalar"));
        }
        if reduce_dims != [rank - 1] {
            return Err(unsupported("reduce non-last-dim"));
        }
        if op == ReduceOp::ArgMax {
            return self.run_argmax_last_dim(layout);
        }
        if op != ReduceOp::Sum {
            return Err(unsupported("reduce"));
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
        self.device.run_compute(
            &shader,
            &entries,
            &bindings,
            rows as u32,
            "candle-wgpu-sum-rows",
        )?;
        Ok(dst)
    }
    fn cumsum_last_dim(&self, layout: &Layout) -> Result<Self> {
        self.run_cumsum_last_dim(layout)
    }
    fn clamp(&self, layout: &Layout, min: f32, max: f32) -> Result<Self> {
        self.run_clamp(layout, min, max)
    }
    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(unsupported("cmp"))
    }
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match (self.dtype, dtype) {
            (DType::F32, DType::F16) | (DType::F16, DType::F32) | (DType::F16, DType::F16)
                if !self
                    .device
                    .inner
                    .features
                    .contains(wgpu::Features::SHADER_F16) =>
            {
                Err(unsupported("to_dtype f16"))
            }
            (DType::F32, DType::F32)
            | (DType::F32, DType::F16)
            | (DType::F32, DType::I32)
            | (DType::F16, DType::F32)
            | (DType::F16, DType::F16) => {
                let shader = copy_shader(self.dtype, dtype)?;
                self.run_copy_to_dtype(layout, dtype, &shader)
            }
            _ => Err(unsupported("to_dtype")),
        }
    }
    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype == DType::F16
            && !self
                .device
                .inner
                .features
                .contains(wgpu::Features::SHADER_F16)
        {
            return Err(unsupported("unary f16"));
        }
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu unary").bt());
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
            return Err(unsupported("binary f16"));
        }
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu binary").bt());
        }
        if rhs.dtype != self.dtype {
            return Err(Error::UnsupportedDTypeForOp(rhs.dtype, "wgpu binary").bt());
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
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        self.device.run_compute(
            &binary_shader(B::NAME, self.dtype)?,
            &entries,
            &bindings,
            workgroups,
            "candle-wgpu-binary",
        )?;
        Ok(dst)
    }
    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(unsupported("where"))
    }
    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.run_conv1d_f32(layout, kernel, kernel_l, params)
    }
    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(unsupported("conv_transpose1d"))
    }
    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.run_conv2d_f32(layout, kernel, kernel_l, params)
    }
    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(unsupported("conv_transpose2d"))
    }
    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.run_pool2d_im2col_f32(layout, kernel_size, stride, false)
    }
    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.run_pool2d_im2col_f32(layout, kernel_size, stride, true)
    }
    fn upsample_nearest1d(&self, layout: &Layout, out_l: usize) -> Result<Self> {
        self.run_upsample_nearest1d_f32(layout, out_l)
    }
    fn upsample_nearest2d(&self, layout: &Layout, out_h: usize, out_w: usize) -> Result<Self> {
        let (_, _, h, w) = layout.shape().dims4()?;
        self.run_upsample2d_f32(
            layout,
            out_h,
            out_w,
            nearest_interp_weights(h, out_h),
            nearest_interp_weights(w, out_w),
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
        let (_, _, h, w) = layout.shape().dims4()?;
        self.run_upsample2d_f32(
            layout,
            out_h,
            out_w,
            bilinear_interp_weights(h, out_h, align_corners, scale_h),
            bilinear_interp_weights(w, out_w, align_corners, scale_w),
        )
    }
    fn gather(&self, src_l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if dim + 1 != src_l.dims().len() {
            return Err(unsupported("gather non-last-dim"));
        }
        self.run_gather_last_dim_f32(ids, src_l, ids_l)
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
        if dim + 1 != dst_l.dims().len() {
            return Err(unsupported("scatter_set non-last-dim"));
        }
        self.run_scatter_set_last_dim_f32(dst_l, ids, ids_l, src, src_l)
    }
    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(unsupported("scatter_add"))
    }
    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        self.run_index_select_f32(ids, src_l, ids_l, dim)
    }
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(unsupported("index_add"))
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
            if self.dtype == DType::F16
                && !self
                    .device
                    .inner
                    .features
                    .contains(wgpu::Features::SHADER_F16)
            {
                return Err(unsupported("copy_strided_src_f16"));
            }
            match self.dtype {
                DType::F32 | DType::F16 => {
                    let shader = copy_shader(self.dtype, self.dtype)?;
                    return self.run_copy_into(src_l, dst, dst_offset, &shader);
                }
                _ => return Err(unsupported("copy_strided_src_non_contiguous")),
            }
        }
        let elem_size = self.dtype.size_in_bytes();
        if elem_size == 0 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu copy").bt());
        }
        let src_offset = src_l.start_offset() * elem_size;
        let size = src_l.shape().elem_count() * elem_size;
        let dst_offset = dst_offset * elem_size;
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
            dst_offset as u64,
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
            (DType::F32, crate::scalar::Scalar::F32(value)) => {
                return self.run_fill_inplace(layout, value);
            }
            (DType::F16, crate::scalar::Scalar::F16(value)) => {
                if !self
                    .device
                    .inner
                    .features
                    .contains(wgpu::Features::SHADER_F16)
                {
                    return Err(unsupported("const_set_f16"));
                }
                return self.run_fill_inplace(layout, value.to_f32());
            }
            _ if !layout.is_contiguous() => {
                return Err(unsupported("const_set_non_contiguous"));
            }
            _ => {}
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| unsupported("const_set_non_contiguous"))?;
        let scalar = scalar_bytes(scalar, self.dtype, "wgpu const_set")?;
        let mut bytes = Vec::with_capacity((end - start) * scalar.len());
        for _ in start..end {
            bytes.extend_from_slice(&scalar);
        }
        self.device
            .inner
            .queue
            .write_buffer(&self.buffer, (start * scalar.len()) as u64, &bytes);
        Ok(())
    }
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| Error::wrap(e))?;
        let adapter_features = adapter.features();
        let required_features = adapter_features & wgpu::Features::SHADER_F16;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("candle-wgpu"),
            required_features,
            ..Default::default()
        }))
        .map_err(|e| Error::wrap(e))?;
        Ok(Self {
            inner: Arc::new(WgpuInner {
                ordinal,
                device,
                queue,
                features: required_features,
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

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(unsupported("rand_uniform"))
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(unsupported("rand_normal"))
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(unsupported("set_seed"))
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(unsupported("get_current_seed"))
    }

    fn synchronize(&self) -> Result<()> {
        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::wrap(e))?;
        Ok(())
    }
}
