use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, WithDType};
use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

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
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: vk::Queue,
    allocator: Mutex<Option<Allocator>>,
}

impl std::fmt::Debug for VulkanInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanInner")
            .field("ordinal", &self.ordinal)
            .field("physical_device", &self.physical_device)
            .field("queue_family_index", &self.queue_family_index)
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
    Error::Msg(format!("vulkan backend op {op} not implemented").into()).bt()
}

pub fn shader_source(name: &str) -> Option<&'static str> {
    candle_vulkan_kernels::get(name).map(|module| module.source())
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

impl VulkanDevice {
    pub fn transfer_to_device(&self, storage: &VulkanStorage) -> Result<VulkanStorage> {
        let cpu = storage.to_cpu_storage()?;
        self.storage_from_cpu_storage(&cpu)
    }

    fn create_buffer(&self, size: usize, name: &'static str) -> Result<Arc<VulkanBuffer>> {
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            unsafe { self.inner.device.create_buffer(&info, None) }.map_err(Error::wrap)?;
        let requirements = unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };
        let mut allocator = self
            .inner
            .allocator
            .lock()
            .map_err(|e| Error::wrap(e.to_string()))?;
        let allocator = allocator
            .as_mut()
            .ok_or_else(|| Error::msg("vulkan allocator already dropped"))?;
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(Error::wrap)?;
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

    fn write_buffer(&self, buffer: &VulkanBuffer, bytes: &[u8]) -> Result<()> {
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
            let ptr = self
                .inner
                .device
                .map_memory(
                    allocation.memory(),
                    allocation.offset(),
                    buffer.size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(Error::wrap)?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            let range = vk::MappedMemoryRange::default()
                .memory(allocation.memory())
                .offset(allocation.offset())
                .size(buffer.size as u64);
            self.inner
                .device
                .flush_mapped_memory_ranges(&[range])
                .map_err(Error::wrap)?;
            self.inner.device.unmap_memory(allocation.memory());
        }
        Ok(())
    }

    fn read_buffer(&self, buffer: &VulkanBuffer) -> Result<Vec<u8>> {
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
            let ptr = self
                .inner
                .device
                .map_memory(
                    allocation.memory(),
                    allocation.offset(),
                    buffer.size as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(Error::wrap)?;
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), buffer.size).to_vec();
            self.inner.device.unmap_memory(allocation.memory());
            Ok(bytes)
        }
    }

    fn run_compute(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: u32,
    ) -> Result<()> {
        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv);
            let shader = self
                .inner
                .device
                .create_shader_module(&shader_info, None)
                .map_err(Error::wrap)?;
            let result = self.run_compute_with_shader(shader, bindings, push_constants, workgroups);
            self.inner.device.destroy_shader_module(shader, None);
            result
        }
    }

    unsafe fn run_compute_with_shader(
        &self,
        shader: vk::ShaderModule,
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: u32,
    ) -> Result<()> {
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
            .collect::<Vec<_>>();
        let set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let set_layout = self
            .inner
            .device
            .create_descriptor_set_layout(&set_layout_info, None)
            .map_err(Error::wrap)?;
        let push_constant_ranges = push_constants
            .map(|bytes| {
                vec![vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(bytes.len() as u32)]
            })
            .unwrap_or_default();
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&set_layout))
            .push_constant_ranges(&push_constant_ranges);
        let pipeline_layout = self
            .inner
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(Error::wrap)?;
        let entry = CString::new("main").map_err(Error::wrap)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipelines = self
            .inner
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, e)| Error::wrap(e))?;
        let pipeline = pipelines[0];

        let mut storage_count = 0;
        for binding in bindings {
            match binding {
                VulkanBinding::Storage(_) => storage_count += 1,
            }
        }
        let mut pool_sizes = Vec::new();
        if storage_count > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(storage_count),
            );
        }
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = self
            .inner
            .device
            .create_descriptor_pool(&pool_info, None)
            .map_err(Error::wrap)?;
        let set_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&set_layout));
        let descriptor_set = self
            .inner
            .device
            .allocate_descriptor_sets(&set_alloc_info)
            .map_err(Error::wrap)?[0];
        let buffer_infos = bindings
            .iter()
            .map(|binding| {
                let buffer = binding.buffer();
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.size as u64)
            })
            .collect::<Vec<_>>();
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
            .collect::<Vec<_>>();
        self.inner.device.update_descriptor_sets(&writes, &[]);

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.inner.queue_family_index);
        let command_pool = self
            .inner
            .device
            .create_command_pool(&command_pool_info, None)
            .map_err(Error::wrap)?;
        let command_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = self
            .inner
            .device
            .allocate_command_buffers(&command_alloc)
            .map_err(Error::wrap)?[0];
        let begin_info = vk::CommandBufferBeginInfo::default();
        self.inner
            .device
            .begin_command_buffer(command_buffer, &begin_info)
            .map_err(Error::wrap)?;
        self.inner.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline,
        );
        self.inner.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            std::slice::from_ref(&descriptor_set),
            &[],
        );
        if let Some(bytes) = push_constants {
            self.inner.device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
        self.inner
            .device
            .cmd_dispatch(command_buffer, workgroups, 1, 1);
        self.inner
            .device
            .end_command_buffer(command_buffer)
            .map_err(Error::wrap)?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        let fence = self
            .inner
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(Error::wrap)?;
        self.inner
            .device
            .queue_submit(self.inner.queue, std::slice::from_ref(&submit_info), fence)
            .map_err(Error::wrap)?;
        self.inner
            .device
            .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
            .map_err(Error::wrap)?;
        self.inner.device.destroy_fence(fence, None);
        self.inner.device.destroy_command_pool(command_pool, None);
        self.inner
            .device
            .destroy_descriptor_pool(descriptor_pool, None);
        self.inner.device.destroy_pipeline(pipeline, None);
        self.inner
            .device
            .destroy_pipeline_layout(pipeline_layout, None);
        self.inner
            .device
            .destroy_descriptor_set_layout(set_layout, None);
        Ok(())
    }
}

enum VulkanBinding<'a> {
    Storage(&'a VulkanBuffer),
}

impl VulkanBinding<'_> {
    fn buffer(&self) -> &VulkanBuffer {
        match self {
            Self::Storage(buffer) => buffer,
        }
    }

    fn descriptor_type(&self) -> vk::DescriptorType {
        match self {
            Self::Storage(_) => vk::DescriptorType::STORAGE_BUFFER,
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

fn unary_spirv(op: &str) -> Result<(&'static [u32], VulkanUnaryKind)> {
    let name = match op {
        "abs" => "abs_f32",
        "ceil" => "ceil_f32",
        "cos" => "cos_f32",
        "exp" => "exp_f32",
        "floor" => "floor_f32",
        "gelu" => "gelu_f32",
        "gelu_erf" => "gelu_erf_f32",
        "log" => "log_f32",
        "neg" => "neg_f32",
        "relu" => "relu_f32",
        "round" => "round_f32",
        "sign" => "sgn_f32",
        "silu" => "silu_f32",
        "sin" => "sin_f32",
        "sqr" => "sqr_f32",
        "sqrt" => "sqrt_f32",
        "tanh" => "tanh_f32",
        _ => return Err(unsupported("unary")),
    };
    let kind = match op {
        "cos" | "log" | "sin" | "sqr" | "sqrt" => VulkanUnaryKind::Generic,
        _ => VulkanUnaryKind::Head,
    };
    let spirv = candle_vulkan_kernels::spirv(name)
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated").into()).bt())?;
    Ok((spirv, kind))
}

fn binary_spirv(op: &str) -> Result<&'static [u32]> {
    let name = match op {
        "add" => "add_f32_f32_f32",
        "div" => "div_f32_f32_f32",
        "mul" => "mul_f32_f32_f32",
        "sub" => "sub_f32_f32_f32",
        _ => return Err(unsupported("binary")),
    };
    candle_vulkan_kernels::spirv(name)
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated").into()).bt())
}

impl VulkanStorage {
    fn run_unary_head(&self, layout: &Layout, spirv: &[u32]) -> Result<Self> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("unary strided"));
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let workgroups = (count as u32).div_ceil(512);
        self.device
            .run_compute(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(dst)
    }

    fn run_unary_generic(&self, layout: &Layout, spirv: &[u32]) -> Result<Self> {
        if layout.start_offset() != 0 {
            return Err(unsupported("unary offset"));
        }
        let count = layout.shape().elem_count();
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let workgroups = (count as u32).div_ceil(512);
        self.device
            .run_compute(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(dst)
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.inner.device.device_wait_idle();
            self.device.inner.device.destroy_buffer(self.buffer, None);
            if let Ok(mut allocation) = self.allocation.lock() {
                if let Some(allocation) = allocation.take() {
                    if let Ok(mut allocator) = self.device.inner.allocator.lock() {
                        if let Some(allocator) = allocator.as_mut() {
                            let _ = allocator.free(allocation);
                        }
                    }
                }
            }
        }
    }
}

impl Drop for VulkanInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Ok(mut allocator) = self.allocator.lock() {
                let _ = allocator.take();
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
            return Err(unsupported("try_clone_strided"));
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
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan affine").bt());
        }
        let _ = (layout, mul, add);
        Err(unsupported("affine"))
    }
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan powf").bt());
        }
        let _ = (layout, e);
        Err(unsupported("powf"))
    }
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan elu").bt());
        }
        if alpha != 1.0 {
            return Err(unsupported("elu alpha"));
        }
        let spirv = candle_vulkan_kernels::spirv("elu_f32")
            .ok_or_else(|| Error::Msg("vulkan shader elu_f32 not generated".into()).bt())?;
        self.run_unary_head(layout, spirv)
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan unary").bt());
        }
        let (spirv, kind) = unary_spirv(B::NAME)?;
        match kind {
            VulkanUnaryKind::Head => self.run_unary_head(layout, spirv),
            VulkanUnaryKind::Generic => self.run_unary_generic(layout, spirv),
        }
    }
    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan binary").bt());
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
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(rhs.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let spirv = binary_spirv(B::NAME)?;
        let workgroups = (count as u32).div_ceil(512);
        self.device
            .run_compute(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan copy").bt());
        }
        let src_offset = src_l.start_offset() * elem_size;
        let dst_offset = dst_offset * elem_size;
        let size = src_l.shape().elem_count() * elem_size;
        let bytes = self.device.read_buffer(&self.buffer)?;
        let mut dst_bytes = dst.device.read_buffer(&dst.buffer)?;
        dst_bytes[dst_offset..dst_offset + size]
            .copy_from_slice(&bytes[src_offset..src_offset + size]);
        dst.device.write_buffer(&dst.buffer, &dst_bytes)?;
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
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan copy2d").bt());
        }
        let bytes = self.device.read_buffer(&self.buffer)?;
        let mut dst_bytes = dst.device.read_buffer(&dst.buffer)?;
        for i1 in 0..d1 {
            let src_idx = (i1 * src_stride1 + src_offset) * elem_size;
            let dst_idx = (i1 * dst_stride1 + dst_offset) * elem_size;
            let len = d2 * elem_size;
            dst_bytes[dst_idx..dst_idx + len].copy_from_slice(&bytes[src_idx..src_idx + len]);
        }
        dst.device.write_buffer(&dst.buffer, &dst_bytes)?;
        Ok(())
    }

    fn const_set(&mut self, scalar: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| unsupported("const_set_non_contiguous"))?;
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
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance =
            unsafe { entry.create_instance(&instance_info, None) }.map_err(Error::wrap)?;
        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.map_err(Error::wrap)?;
        let physical_device = *physical_devices
            .get(ordinal)
            .or_else(|| physical_devices.first())
            .ok_or_else(|| Error::msg("no vulkan physical device found"))?;
        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find_map(|(index, props)| {
                    props
                        .queue_flags
                        .contains(vk::QueueFlags::COMPUTE)
                        .then_some(index as u32)
                })
                .ok_or_else(|| Error::msg("no vulkan compute queue family found"))?
        };
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(Error::wrap)?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(Error::wrap)?;
        Ok(Self {
            inner: Arc::new(VulkanInner {
                ordinal,
                _entry: entry,
                instance,
                physical_device,
                device,
                queue_family_index,
                queue,
                allocator: Mutex::new(Some(allocator)),
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
        unsafe { self.inner.device.queue_wait_idle(self.inner.queue) }.map_err(Error::wrap)?;
        Ok(())
    }
}
