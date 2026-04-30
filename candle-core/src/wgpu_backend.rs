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

impl WgpuDevice {
    pub fn transfer_to_device(&self, storage: &WgpuStorage) -> Result<WgpuStorage> {
        let cpu = storage.to_cpu_storage()?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn create_storage_buffer(&self, size: usize, label: &'static str) -> wgpu::Buffer {
        self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn read_buffer(&self, buffer: &wgpu::Buffer, size: usize) -> Result<Vec<u8>> {
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle-wgpu-readback"),
            size: size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("candle-wgpu-readback"),
                });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
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
        let data = slice.get_mapped_range().to_vec();
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
    for (idx, dim) in layout.dims().iter().enumerate() {
        dims[idx] = (*dim).try_into()?;
    }
    for (idx, stride) in layout.stride().iter().enumerate() {
        strides[idx] = (*stride).try_into()?;
    }
    Ok((dims, strides))
}

fn unary_shader(op: &str) -> Result<String> {
    if op == "recip" {
        return Ok(custom_unary_wgsl("1.0 / x"));
    }
    let op = match op {
        "abs" => candle_wgpu_kernels::UnaryOp::Abs,
        "ceil" => candle_wgpu_kernels::UnaryOp::Ceil,
        "cos" => candle_wgpu_kernels::UnaryOp::Cos,
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
        candle_wgpu_kernels::DType::F32,
        WG_SIZE,
    ))
}

fn binary_shader(op: &str) -> Result<String> {
    if op == "maximum" {
        return Ok(custom_binary_wgsl("max(a, b)"));
    }
    if op == "minimum" {
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
        candle_wgpu_kernels::DType::F32,
        WG_SIZE,
    ))
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

@group(0) @binding(0) var<storage, read> src0: array<f32>;
@group(0) @binding(1) var<storage, read> src1: array<f32>;
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
@group(0) @binding(0) var<storage, read> src: array<f32>;
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
            storage_entry(0, true),
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
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, size as u64);
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
        let shader = custom_unary_wgsl(&format!("x * {:?} + {:?}", mul as f32, add as f32));
        self.run_unary_like(layout, &shader, "candle-wgpu-affine")
    }
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu powf").bt());
        }
        let shader = custom_unary_wgsl(&format!("pow(x, {:?})", e as f32));
        self.run_unary_like(layout, &shader, "candle-wgpu-powf")
    }
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu elu").bt());
        }
        let shader = custom_unary_wgsl(&format!(
            "select({:?} * (exp(x) - 1.0), x, x > 0.0)",
            alpha as f32
        ));
        self.run_unary_like(layout, &shader, "candle-wgpu-elu")
    }
    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(unsupported("reduce"))
    }
    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(unsupported("cmp"))
    }
    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(unsupported("to_dtype"))
    }
    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu unary").bt());
        }
        let shader = unary_shader(B::NAME)?;
        self.run_unary_like(layout, &shader, "candle-wgpu-unary")
    }
    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "wgpu binary").bt());
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
        let workgroups = (count as u32).div_ceil(WG_SIZE);
        self.device.run_compute(
            &binary_shader(B::NAME)?,
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
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(unsupported("conv1d"))
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
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(unsupported("conv2d"))
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
    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(unsupported("avg_pool2d"))
    }
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(unsupported("max_pool2d"))
    }
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(unsupported("upsample_nearest1d"))
    }
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(unsupported("upsample_nearest2d"))
    }
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Err(unsupported("upsample_bilinear2d"))
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(unsupported("gather"))
    }
    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(unsupported("scatter_set"))
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
    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(unsupported("index_select"))
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
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(unsupported("matmul"))
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
            return Err(unsupported("copy_strided_src_non_contiguous"));
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("candle-wgpu"),
            ..Default::default()
        }))
        .map_err(|e| Error::wrap(e))?;
        Ok(Self {
            inner: Arc::new(WgpuInner {
                ordinal,
                device,
                queue,
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
        self.inner.queue.write_buffer(&buffer, 0, &vec![0u8; size]);
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
        self.inner.queue.write_buffer(&buffer, 0, &bytes);
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
