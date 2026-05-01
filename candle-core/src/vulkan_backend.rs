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
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    subgroup_size: u32,
    robust_buffer_access: bool,
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
            .field("subgroup_size", &self.subgroup_size)
            .field("robust_buffer_access", &self.robust_buffer_access)
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
        self.run_compute_3d(spirv, bindings, push_constants, (workgroups, 1, 1))
    }

    fn run_compute_3d(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        self.run_compute_specialized(spirv, bindings, push_constants, workgroups, None)
    }

    fn run_compute_specialized(
        &self,
        spirv: &[u32],
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: (u32, u32, u32),
        specialization_u32: Option<&[(u32, u32)]>,
    ) -> Result<()> {
        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::default().code(spirv);
            let shader = self
                .inner
                .device
                .create_shader_module(&shader_info, None)
                .map_err(Error::wrap)?;
            let result = self.run_compute_with_shader(
                shader,
                bindings,
                push_constants,
                workgroups,
                specialization_u32,
            );
            self.inner.device.destroy_shader_module(shader, None);
            result
        }
    }

    unsafe fn run_compute_with_shader(
        &self,
        shader: vk::ShaderModule,
        bindings: &[VulkanBinding<'_>],
        push_constants: Option<&[u8]>,
        workgroups: (u32, u32, u32),
        specialization_u32: Option<&[(u32, u32)]>,
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
        let mut spec_entries = Vec::new();
        let mut spec_data = Vec::new();
        let spec_info;
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
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
            .cmd_dispatch(command_buffer, workgroups.0, workgroups.1, workgroups.2);
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

fn next_power_of_two_u32(value: usize, op: &'static str) -> Result<u32> {
    value
        .checked_next_power_of_two()
        .ok_or_else(|| {
            Error::Msg(format!("vulkan backend op {op} dimension overflow").into()).bt()
        })?
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
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated").into()).bt())?;
    Ok((spirv, kind))
}

fn binary_spirv(op: &str, dtype: DType) -> Result<&'static [u32]> {
    let suffix = match dtype {
        DType::F32 => "f32_f32_f32",
        DType::F16 => "f16_f16_f16",
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
        .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated").into()).bt())
}

fn copy_spirv(src: DType, dst: DType) -> Result<&'static [u32]> {
    let name = match (src, dst) {
        (DType::F32, DType::F32) => "cpy_f32_f32",
        (DType::F32, DType::I32) => "cpy_f32_i32",
        (DType::I32, DType::F32) => "cpy_i32_f32",
        (DType::F32, DType::F16) => "cpy_f32_f16",
        (DType::F16, DType::F32) => "cpy_f16_f32",
        (DType::F16, DType::F16) => "cpy_f16_f16",
        _ => return Err(unsupported("to_dtype")),
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
        if layout.start_offset() != 0 {
            return Err(unsupported("unary offset"));
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let workgroups = (count as u32).div_ceil(512);
        self.device
            .run_compute(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
        Ok(dst)
    }

    fn run_unary_generic(&self, layout: &Layout, spirv: &[u32]) -> Result<Self> {
        self.run_unary_generic_with_params(layout, spirv, 0.0, 0.0)
    }

    fn run_binary_named(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: &'static str,
    ) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
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
            param3: 0,
        };
        let bindings = [
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(rhs.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let spirv = binary_spirv(op, self.dtype)?;
        let workgroups = (count as u32).div_ceil(512);
        self.device
            .run_compute(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
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
            return Err(unsupported("copy_strided_src_offset"));
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let workgroups = (count as u32).div_ceil(512);
        self.device
            .run_compute(spirv, &bindings, Some(any_as_bytes(&params)), workgroups)?;
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
        if ncols_padded > 1024 {
            return Err(unsupported("argsort last-dim > 1024"));
        }
        if ncols_padded != last_dim as u32 && !self.device.inner.robust_buffer_access {
            return Err(unsupported(
                "argsort non-power-of-two without robust buffers",
            ));
        }
        let count = layout.shape().elem_count();
        let nrows = count / last_dim;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), DType::U32)? };
        let ncols_padded_log2 = ncols_padded.trailing_zeros();
        let params = VulkanArgsortParams {
            ncols: last_dim.try_into()?,
            ncols_padded,
            ncols_padded_log2,
            nrows: nrows.try_into()?,
            order: if asc { 0 } else { 1 },
            outer_start: 0,
            outer_end: 0,
            inner_start: 0,
            inner_end: 0,
        };
        let bindings = [
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let spirv = candle_vulkan_kernels::spirv("argsort_f32")
            .ok_or_else(|| Error::Msg("vulkan shader argsort_f32 not generated".into()).bt())?;
        self.device.run_compute_specialized(
            spirv,
            &bindings,
            Some(any_as_bytes(&params)),
            (1, nrows.try_into()?, 1),
            Some(&[(0, ncols_padded), (1, ncols_padded_log2)]),
        )?;
        Ok(dst)
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan softmax").bt());
        }
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("softmax strided"));
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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

    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &Self,
        alpha_layout: &Layout,
        eps: f32,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || alpha.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan rms_norm").bt());
        }
        if !layout.is_contiguous()
            || layout.start_offset() != 0
            || !alpha_layout.is_contiguous()
            || alpha_layout.start_offset() != 0
        {
            return Err(unsupported("rms_norm strided"));
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(alpha.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
        if self.dtype != DType::F32 {
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(ids.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let spirv = candle_vulkan_kernels::spirv("get_rows_f32_f32").ok_or_else(|| {
            Error::Msg("vulkan shader get_rows_f32_f32 not generated".into()).bt()
        })?;
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
        if self.dtype != DType::F32 {
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
            VulkanBinding::Storage(self.buffer.as_ref()),
            VulkanBinding::Storage(ids.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let spirv = candle_vulkan_kernels::spirv("get_rows_f32_f32").ok_or_else(|| {
            Error::Msg("vulkan shader get_rows_f32_f32 not generated".into()).bt()
        })?;
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
        if self.dtype != DType::F32 || src.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan scatter_set").bt());
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
            VulkanBinding::Storage(src.buffer.as_ref()),
            VulkanBinding::Storage(ids.buffer.as_ref()),
            VulkanBinding::Storage(self.buffer.as_ref()),
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

    fn run_matmul_f32(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 || rhs.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan matmul").bt());
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

        let dst_shape = Shape::from(vec![b, m, n]);
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
            VulkanBinding::Storage(rhs_t.buffer.as_ref()),
            VulkanBinding::Storage(lhs.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
        ];
        let spirv = candle_vulkan_kernels::spirv("matmul_f32_f32")
            .ok_or_else(|| Error::Msg("vulkan shader matmul_f32_f32 not generated".into()).bt())?;
        let spec = [
            (0, 64),
            (1, 64),
            (2, 64),
            (4, 32),
            (5, 32),
            (6, 2),
            (7, 4),
            (8, 2),
            (10, self.device.inner.subgroup_size.max(1)),
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
            VulkanBinding::Storage(input.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
            VulkanBinding::Storage(input.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
            VulkanBinding::Storage(kernel.buffer.as_ref()),
            VulkanBinding::Storage(input.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
            VulkanBinding::Storage(kernel.buffer.as_ref()),
            VulkanBinding::Storage(input.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
            VulkanBinding::Storage(input.buffer.as_ref()),
            VulkanBinding::Storage(dst.buffer.as_ref()),
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
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Err(unsupported("affine strided"));
        }
        let spirv = candle_vulkan_kernels::spirv("scale_f32")
            .ok_or_else(|| Error::Msg("vulkan shader scale_f32 not generated".into()).bt())?;
        self.run_unary_generic_with_params(layout, spirv, mul as f32, add as f32)
    }
    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan powf").bt());
        }
        let spirv = candle_vulkan_kernels::spirv("powf_f32")
            .ok_or_else(|| Error::Msg("vulkan shader powf_f32 not generated".into()).bt())?;
        self.run_unary_generic_with_params(layout, spirv, e as f32, 0.0)
    }
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan elu").bt());
        }
        if alpha != 1.0 {
            return Err(unsupported("elu alpha"));
        }
        let suffix = match self.dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            _ => unreachable!(),
        };
        let name = format!("elu_{suffix}");
        let spirv = candle_vulkan_kernels::spirv(&name)
            .ok_or_else(|| Error::Msg(format!("vulkan shader {name} not generated").into()).bt())?;
        self.run_unary_head(layout, spirv)
    }
    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan reduce").bt());
        }
        let rank = layout.dims().len();
        if rank == 0 {
            return Err(unsupported("reduce scalar"));
        }
        if reduce_dims.len() > 1 {
            return self.run_reduce_multi_dim(op, layout, reduce_dims);
        }
        if reduce_dims.is_empty() {
            return Err(unsupported("reduce empty dims"));
        }
        let dim = reduce_dims[0];
        if dim >= rank {
            return Err(unsupported("reduce dim"));
        }
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
        self.run_cumsum_last_dim(layout)
    }
    fn clamp(&self, layout: &Layout, min: f32, max: f32) -> Result<Self> {
        if self.dtype != DType::F32 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan clamp").bt());
        }
        let spirv = candle_vulkan_kernels::spirv("clamp_f32")
            .ok_or_else(|| Error::Msg("vulkan shader clamp_f32 not generated".into()).bt())?;
        self.run_unary_generic_with_params(layout, spirv, min, max)
    }
    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(unsupported("cmp"))
    }
    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let spirv = copy_spirv(self.dtype, dtype)?;
        self.run_unary_generic_with_params_dtype(layout, spirv, 0.0, 0.0, dtype)
    }
    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype != DType::F32 && self.dtype != DType::F16 {
            return Err(Error::UnsupportedDTypeForOp(self.dtype, "vulkan unary").bt());
        }
        let (spirv, kind) = unary_spirv(B::NAME, self.dtype)?;
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
        match B::NAME {
            "maximum" | "minimum" => {
                self.run_binary_min_max_f32(rhs, lhs_layout, rhs_layout, B::NAME)
            }
            _ => self.run_binary_named(rhs, lhs_layout, rhs_layout, B::NAME),
        }
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
        self.run_conv2d_f32(layout, kernel, kernel_l, params)
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
        self.run_pool2d_f32(layout, kernel_size, stride, false)
    }
    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.run_pool2d_f32(layout, kernel_size, stride, true)
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
            match self.dtype {
                DType::F32 | DType::F16 => {
                    let spirv = copy_spirv(self.dtype, self.dtype)?;
                    return self.run_copy_into(src_l, dst, dst_offset, spirv);
                }
                _ => return Err(unsupported("copy_strided_src_non_contiguous")),
            }
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
        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut physical_device_properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_properties);
        unsafe {
            instance
                .get_physical_device_properties2(physical_device, &mut physical_device_properties);
        }
        let subgroup_size = subgroup_properties.subgroup_size.max(1);
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
        let physical_features = unsafe { instance.get_physical_device_features(physical_device) };
        let robust_buffer_access = physical_features.robust_buffer_access == vk::TRUE;
        let mut enabled_features = vk::PhysicalDeviceFeatures::default();
        if robust_buffer_access {
            enabled_features.robust_buffer_access = vk::TRUE;
        }
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&enabled_features);
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
                subgroup_size,
                robust_buffer_access,
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
