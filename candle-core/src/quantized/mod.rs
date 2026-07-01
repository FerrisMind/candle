use crate::{
    backend::{BackendDevice, BackendStorage},
    CpuStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod imatrix_file;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

fn decode_t_vec<T>(data: &[u8]) -> Vec<T> {
    let size = std::mem::size_of::<T>();
    assert!(size > 0, "zero-sized quantized types are not supported");
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let mut out = Vec::with_capacity(data.len() / size);
    let mut ptr = data.as_ptr();
    for _ in 0..(data.len() / size) {
        // Quantized blocks are serialized as packed bytes. Reconstruct them into aligned Rust
        // values without assuming that the input byte pointer is aligned.
        out.push(unsafe { std::ptr::read_unaligned(ptr as *const T) });
        ptr = unsafe { ptr.add(size) };
    }
    out
}

fn is_backend_not_implemented_msg(msg: &str, backend: &str) -> bool {
    let backend_rank_limit = msg.contains(backend) && msg.contains("supports up to rank-4 tensors");
    let backend_overflow = msg.contains(backend)
        && (msg.contains("dimension overflow")
            || msg.contains("tmp overflow")
            || msg.contains("workgroup overflow"));
    (msg.contains(backend) && msg.contains("backend op") && msg.contains("not implemented"))
        || msg.contains(&format!("no {backend} implementation for"))
        || (msg.contains(backend) && msg.contains("shader") && msg.contains("not generated"))
        || (msg.contains("backend op") && msg.contains("not implemented"))
        || backend_rank_limit
        || backend_overflow
}

fn should_quantized_backend_fallback(err: &crate::Error, backend: &str) -> bool {
    match err {
        crate::Error::UnsupportedDTypeForOp(..) => true,
        crate::Error::Msg(msg) => is_backend_not_implemented_msg(msg, backend),
        crate::Error::WithBacktrace { inner, .. }
        | crate::Error::WithPath { inner, .. }
        | crate::Error::Context { inner, .. } => should_quantized_backend_fallback(inner, backend),
        _ => false,
    }
}

fn load_quantized_metal_from_bytes<T: k_quants::GgmlType + Send + Sync + 'static>(
    device: &crate::MetalDevice,
    data: &[u8],
) -> Result<QStorage> {
    let values = decode_t_vec::<T>(data);
    metal::load_quantized(device, &values)
}

fn load_quantized_cuda_from_bytes<T: k_quants::GgmlType + Send + Sync + 'static>(
    device: &crate::CudaDevice,
    data: &[u8],
) -> Result<QStorage> {
    let values = decode_t_vec::<T>(data);
    cuda::load_quantized(device, &values)
}

fn cpu_storage_to_f32_vec(storage: &crate::CpuStorage) -> Result<Vec<f32>> {
    match storage.dtype() {
        DType::F32 => Ok(storage.as_slice::<f32>()?.to_vec()),
        DType::F16 => Ok(storage
            .as_slice::<f16>()?
            .iter()
            .map(|v| v.to_f32())
            .collect()),
        DType::BF16 => Ok(storage
            .as_slice::<bf16>()?
            .iter()
            .map(|v| v.to_f32())
            .collect()),
        dtype => crate::bail!("expected f32/f16/bf16 cpu storage, got {dtype:?}"),
    }
}

fn dense_qmatmul_shape_spec(
    self_shape: &Shape,
    layout: &crate::Layout,
) -> Result<(Shape, usize, usize, usize, usize)> {
    let (n, k) = self_shape.dims2()?;
    let rank = layout.dims().len();
    if rank < 2 {
        crate::bail!("input tensor has only one dimension {layout:?}")
    }
    let input_m = layout.dims()[rank - 2];
    let last_k = layout.dims()[rank - 1];
    if last_k != k {
        crate::bail!("input tensor {layout:?} incompatible with {self_shape:?}")
    }
    let batching = layout.dims()[..rank - 2].iter().product();
    let mut dst_dims = layout.dims().to_vec();
    dst_dims.pop();
    dst_dims.push(n);
    Ok((Shape::from(dst_dims), batching, input_m, n, k))
}

fn dense_qmatmul_rhs_layout(self_shape: &Shape, layout: &crate::Layout) -> Result<crate::Layout> {
    let (n, k) = self_shape.dims2()?;
    let rank = layout.dims().len();
    let rhs = crate::Layout::contiguous(self_shape.clone()).transpose(0, 1)?;
    if rank == 2 {
        return Ok(rhs);
    }
    let shape = Shape::from(&layout.dims()[..rank - 2]).extend(&[k, n]);
    rhs.broadcast_as(shape)
}

fn decode_block_q8_1_data(data: &[u8]) -> Vec<BlockQ8_1> {
    debug_assert_eq!(data.len() % std::mem::size_of::<BlockQ8_1>(), 0);
    data.chunks_exact(std::mem::size_of::<BlockQ8_1>())
        .map(|chunk| {
            let mut qs = [0i8; k_quants::QK8_1];
            for (dst, src) in qs.iter_mut().zip(&chunk[4..4 + k_quants::QK8_1]) {
                *dst = *src as i8;
            }
            BlockQ8_1 {
                d: f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])),
                s: f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])),
                qs,
            }
        })
        .collect()
}

fn decode_block_q8k_data(data: &[u8]) -> Vec<BlockQ8K> {
    debug_assert_eq!(data.len() % std::mem::size_of::<BlockQ8K>(), 0);
    data.chunks_exact(std::mem::size_of::<BlockQ8K>())
        .map(|chunk| {
            let mut qs = [0i8; k_quants::QK_K];
            for (dst, src) in qs.iter_mut().zip(&chunk[4..4 + k_quants::QK_K]) {
                *dst = *src as i8;
            }
            let mut bsums = [0i16; k_quants::QK_K / 16];
            let bsums_bytes = &chunk[4 + k_quants::QK_K..];
            for (dst, raw) in bsums.iter_mut().zip(bsums_bytes.chunks_exact(2)) {
                *dst = i16::from_le_bytes([raw[0], raw[1]]);
            }
            BlockQ8K {
                d: f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                qs,
                bsums,
            }
        })
        .collect()
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
}

pub struct QWgpuStorage {
    dtype: GgmlDType,
    len_bytes: usize,
    storage: crate::WgpuStorage,
}

impl QWgpuStorage {
    fn from_bytes(
        device: &crate::WgpuDevice,
        dtype: GgmlDType,
        data: Cow<'_, [u8]>,
    ) -> Result<Self> {
        let raw = data.into_owned();
        let len_bytes = raw.len();
        let mut padded = raw;
        padded.resize(len_bytes + 4, 0);
        let storage = device.storage_from_cpu_storage(&crate::CpuStorage::U8(padded))?;
        Ok(Self {
            dtype,
            len_bytes,
            storage,
        })
    }

    fn zeros(device: &crate::WgpuDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let len_bytes = elem_count.div_ceil(dtype.block_size()) * dtype.type_size();
        let padded = vec![0u8; len_bytes + 4];
        let storage = device.storage_from_cpu_storage(&crate::CpuStorage::U8(padded))?;
        Ok(Self {
            dtype,
            len_bytes,
            storage,
        })
    }

    fn device(&self) -> &crate::WgpuDevice {
        self.storage.device()
    }

    fn data(&self) -> Result<Vec<u8>> {
        let bytes = self.storage.to_cpu_storage()?;
        let mut bytes = bytes.as_slice::<u8>()?.to_vec();
        bytes.truncate(self.len_bytes);
        Ok(bytes)
    }

    fn to_cpu_quantized(&self) -> Result<Box<dyn QuantizedType>> {
        Ok(self.dtype.from_data(Cow::Owned(self.data()?)))
    }

    fn quantize_from_cpu(
        &mut self,
        src: &crate::CpuStorage,
        imatrix: Option<(&[f32], usize)>,
    ) -> Result<()> {
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        match (&mut qcpu, imatrix) {
            (QStorage::Cpu(storage), None) => storage.from_float(src.as_slice::<f32>()?),
            (QStorage::Cpu(storage), Some((weights, n_per_row))) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, weights, n_per_row)
            }
            _ => unreachable!(),
        }
        let data = qcpu.data()?.into_owned();
        self.len_bytes = data.len();
        let mut padded = data;
        padded.resize(self.len_bytes + 4, 0);
        self.storage = self
            .device()
            .storage_from_cpu_storage(&crate::CpuStorage::U8(padded))?;
        Ok(())
    }

    fn quantize(&mut self, src: &crate::WgpuStorage) -> Result<()> {
        let src = src.to_cpu_storage()?;
        self.quantize_from_cpu(&src, None)
    }

    fn quantize_imatrix(
        &mut self,
        src: &crate::WgpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let src = src.to_cpu_storage()?;
        self.quantize_from_cpu(&src, Some((imatrix_weights, n_per_row)))
    }

    fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        self.quantize_from_cpu(src, None)
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        self.quantize_from_cpu(src, Some((imatrix_weights, n_per_row)))
    }

    fn dequantize(&self, elem_count: usize) -> Result<crate::WgpuStorage> {
        // GGUF norm/bias tensors are stored as raw F32/F16/BF16 blobs under
        // the GGML dtype enum; decode them with a dedicated device-side path
        // instead of the block-quantized get-rows family.
        if matches!(
            self.dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            return match self
                .storage
                .quantized_raw_float_dequantize_f32(self.dtype, elem_count)
            {
                Ok(out) => Ok(out),
                Err(err) if should_quantized_backend_fallback(&err, "wgpu") => {
                    crate::storage::record_wgpu_cpu_fallback(&err);
                    let cpu = self.to_cpu_quantized()?.dequantize(elem_count)?;
                    self.device().storage_from_cpu_storage(&cpu)
                }
                Err(err) => Err(err),
            };
        }
        let block_size = self.dtype.block_size();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "wgpu quantized dequantize expects element count divisible by block size {}, got {elem_count}",
                block_size
            )
        }
        let num_blocks = elem_count / block_size;
        let ids = (0..num_blocks)
            .map(|idx| u32::try_from(idx).map_err(crate::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        let ids_storage = self
            .device()
            .storage_from_cpu_storage(&crate::CpuStorage::U32(ids))?;
        let ids_layout = crate::Layout::contiguous(Shape::from(num_blocks));
        let flat_shape = Shape::from((num_blocks, block_size));
        match self.storage.quantized_index_select_f32(
            self.dtype,
            &flat_shape,
            &ids_storage,
            &ids_layout,
            0,
        ) {
            Ok(out) => Ok(out),
            Err(err) if should_quantized_backend_fallback(&err, "wgpu") => {
                crate::storage::record_wgpu_cpu_fallback(&err);
                let cpu = self.to_cpu_quantized()?.dequantize(elem_count)?;
                self.device().storage_from_cpu_storage(&cpu)
            }
            Err(err) => Err(err),
        }
    }

    fn q8k_fwd_via_dense_gpu(
        &self,
        self_shape: &Shape,
        storage: &crate::WgpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::WgpuStorage, Shape)> {
        let (dst_shape, batching, input_m, n, k) = dense_qmatmul_shape_spec(self_shape, layout)?;
        let weights = self.dequantize(self_shape.elem_count())?;
        if storage.dtype() == DType::F32 {
            let rhs_layout = dense_qmatmul_rhs_layout(self_shape, layout)?;
            let out = storage.matmul(&weights, (batching, input_m, n, k), layout, &rhs_layout)?;
            Ok((out, dst_shape))
        } else {
            let lhs = storage.to_dtype(layout, DType::F32)?;
            let lhs_layout = crate::Layout::contiguous(layout.shape().clone());
            let rhs_layout = dense_qmatmul_rhs_layout(self_shape, &lhs_layout)?;
            let out = lhs.matmul(
                &weights,
                (batching, input_m, n, k),
                &lhs_layout,
                &rhs_layout,
            )?;
            Ok((out, dst_shape))
        }
    }

    fn q8k_index_select_f32_via_dense_gpu(
        &self,
        self_shape: &Shape,
        ids: &crate::WgpuStorage,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<crate::WgpuStorage> {
        let weights = self.dequantize(self_shape.elem_count())?;
        let weights_l = crate::Layout::contiguous(self_shape.clone());
        weights.index_select(ids, &weights_l, ids_l, dim)
    }

    fn fwd_cpu_fallback(
        &self,
        self_shape: &Shape,
        storage: &crate::WgpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::WgpuStorage, Shape)> {
        let src_cpu = if layout.is_contiguous() && layout.start_offset() == 0 {
            storage.to_cpu_storage()?
        } else {
            let mut tmp = unsafe {
                storage
                    .device()
                    .alloc_uninit(layout.shape(), storage.dtype())?
            };
            storage.copy_strided_src(&mut tmp, 0, layout)?;
            tmp.to_cpu_storage()?
        };
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {self_shape:?}")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let lhs = cpu_storage_to_f32_vec(&src_cpu)?;
        let mut dst = vec![0f32; dst_shape.elem_count()];
        self.to_cpu_quantized()?
            .matmul_t((dst_shape.elem_count() / n, k, n), &lhs, &mut dst)?;
        let storage = self
            .device()
            .storage_from_cpu_storage(&crate::CpuStorage::F32(dst))?;
        Ok((storage, dst_shape))
    }

    fn index_select_f32_cpu_fallback(
        &self,
        self_shape: &Shape,
        ids: &crate::WgpuStorage,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<crate::WgpuStorage> {
        let ids_cpu = if ids_l.is_contiguous() && ids_l.start_offset() == 0 {
            ids.to_cpu_storage()?
        } else {
            let mut tmp = unsafe { ids.device().alloc_uninit(ids_l.shape(), ids.dtype())? };
            ids.copy_strided_src(&mut tmp, 0, ids_l)?;
            tmp.to_cpu_storage()?
        };
        let ids = ids_cpu.as_slice::<u32>()?;
        let src = cpu_storage_to_f32_vec(
            &self
                .to_cpu_quantized()?
                .dequantize(self_shape.elem_count())?,
        )?;
        let dims = self_shape.dims();
        if dim >= dims.len() {
            crate::bail!("index_select dim {dim} out of range for {self_shape:?}")
        }
        let left_size: usize = dims[..dim].iter().product();
        let src_dim = dims[dim];
        let right_size: usize = dims[dim + 1..].iter().product();
        let mut dst_dims = dims.to_vec();
        dst_dims[dim] = ids.len();
        let mut dst = vec![0f32; dst_dims.iter().product()];
        for left_idx in 0..left_size {
            let src_left_base = left_idx * src_dim * right_size;
            let dst_left_base = left_idx * ids.len() * right_size;
            for (dst_row, &src_row) in ids.iter().enumerate() {
                let src_row = src_row as usize;
                if src_row >= src_dim {
                    crate::bail!("index_select id {src_row} out of range for dim size {src_dim}")
                }
                let src_offset = src_left_base + src_row * right_size;
                let dst_offset = dst_left_base + dst_row * right_size;
                dst[dst_offset..dst_offset + right_size]
                    .copy_from_slice(&src[src_offset..src_offset + right_size]);
            }
        }
        self.device()
            .storage_from_cpu_storage(&crate::CpuStorage::F32(dst))
    }

    fn fwd(
        &self,
        self_shape: &Shape,
        storage: &crate::WgpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::WgpuStorage, Shape)> {
        if self.dtype == GgmlDType::Q8K {
            return self.q8k_fwd_via_dense_gpu(self_shape, storage, layout);
        }
        match self
            .storage
            .quantized_matmul(self.dtype, self_shape, storage, layout)
        {
            Ok(out) => Ok(out),
            Err(err) if should_quantized_backend_fallback(&err, "wgpu") => {
                crate::storage::record_wgpu_cpu_fallback(&err);
                self.fwd_cpu_fallback(self_shape, storage, layout)
            }
            Err(err) => Err(err),
        }
    }

    fn index_select_f32(
        &self,
        self_shape: &Shape,
        ids: &crate::WgpuStorage,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<crate::WgpuStorage> {
        if self.dtype == GgmlDType::Q8K {
            return self.q8k_index_select_f32_via_dense_gpu(self_shape, ids, ids_l, dim);
        }
        match self
            .storage
            .quantized_index_select_f32(self.dtype, self_shape, ids, ids_l, dim)
        {
            Ok(out) => Ok(out),
            Err(err) if should_quantized_backend_fallback(&err, "wgpu") => {
                crate::storage::record_wgpu_cpu_fallback(&err);
                self.index_select_f32_cpu_fallback(self_shape, ids, ids_l, dim)
            }
            Err(err) => Err(err),
        }
    }
}

pub struct QVulkanStorage {
    dtype: GgmlDType,
    len_bytes: usize,
    storage: crate::VulkanStorage,
}

impl QVulkanStorage {
    fn from_bytes(
        device: &crate::VulkanDevice,
        dtype: GgmlDType,
        data: Cow<'_, [u8]>,
    ) -> Result<Self> {
        let raw = data.into_owned();
        let len_bytes = raw.len();
        let mut padded = raw;
        padded.resize(len_bytes + 4, 0);
        let storage = device.storage_from_cpu_storage(&crate::CpuStorage::U8(padded))?;
        Ok(Self {
            dtype,
            len_bytes,
            storage,
        })
    }

    fn zeros(device: &crate::VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let len_bytes = elem_count.div_ceil(dtype.block_size()) * dtype.type_size();
        let padded = vec![0u8; len_bytes + 4];
        let storage = device.storage_from_cpu_storage(&crate::CpuStorage::U8(padded))?;
        Ok(Self {
            dtype,
            len_bytes,
            storage,
        })
    }

    fn device(&self) -> &crate::VulkanDevice {
        self.storage.device()
    }

    fn data(&self) -> Result<Vec<u8>> {
        let bytes = self.storage.to_cpu_storage()?;
        let mut bytes = bytes.as_slice::<u8>()?.to_vec();
        bytes.truncate(self.len_bytes);
        Ok(bytes)
    }

    fn to_cpu_quantized(&self) -> Result<Box<dyn QuantizedType>> {
        Ok(self.dtype.from_data(Cow::Owned(self.data()?)))
    }

    fn quantize_from_cpu(
        &mut self,
        src: &crate::CpuStorage,
        imatrix: Option<(&[f32], usize)>,
    ) -> Result<()> {
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        match (&mut qcpu, imatrix) {
            (QStorage::Cpu(storage), None) => storage.from_float(src.as_slice::<f32>()?),
            (QStorage::Cpu(storage), Some((weights, n_per_row))) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, weights, n_per_row)
            }
            _ => unreachable!(),
        }
        let data = qcpu.data()?.into_owned();
        self.len_bytes = data.len();
        let mut padded = data;
        padded.resize(self.len_bytes + 4, 0);
        self.storage = self
            .device()
            .storage_from_cpu_storage(&crate::CpuStorage::U8(padded))?;
        Ok(())
    }

    fn quantize(&mut self, src: &crate::VulkanStorage) -> Result<()> {
        let src = src.to_cpu_storage()?;
        self.quantize_from_cpu(&src, None)
    }

    fn quantize_imatrix(
        &mut self,
        src: &crate::VulkanStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let src = src.to_cpu_storage()?;
        self.quantize_from_cpu(&src, Some((imatrix_weights, n_per_row)))
    }

    fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        self.quantize_from_cpu(src, None)
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        self.quantize_from_cpu(src, Some((imatrix_weights, n_per_row)))
    }

    fn dequantize(&self, elem_count: usize) -> Result<crate::VulkanStorage> {
        match self
            .storage
            .quantized_dequantize_f32(self.dtype, elem_count)
        {
            Ok(out) => Ok(out),
            Err(err) if should_quantized_backend_fallback(&err, "vulkan") => {
                crate::storage::record_vulkan_cpu_fallback(&err);
                let cpu = self.to_cpu_quantized()?.dequantize(elem_count)?;
                self.device().storage_from_cpu_storage(&cpu)
            }
            Err(err) => Err(err),
        }
    }

    fn q8k_fwd_via_dense_gpu(
        &self,
        self_shape: &Shape,
        storage: &crate::VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::VulkanStorage, Shape)> {
        let (dst_shape, batching, input_m, n, k) = dense_qmatmul_shape_spec(self_shape, layout)?;
        let weights = self.dequantize(self_shape.elem_count())?;
        if storage.dtype() == DType::F32 {
            let rhs_layout = dense_qmatmul_rhs_layout(self_shape, layout)?;
            let out = storage.matmul(&weights, (batching, input_m, n, k), layout, &rhs_layout)?;
            Ok((out, dst_shape))
        } else {
            let lhs = storage.to_dtype(layout, DType::F32)?;
            let lhs_layout = crate::Layout::contiguous(layout.shape().clone());
            let rhs_layout = dense_qmatmul_rhs_layout(self_shape, &lhs_layout)?;
            let out = lhs.matmul(
                &weights,
                (batching, input_m, n, k),
                &lhs_layout,
                &rhs_layout,
            )?;
            Ok((out, dst_shape))
        }
    }

    fn q8k_index_select_f32_via_dense_gpu(
        &self,
        self_shape: &Shape,
        ids: &crate::VulkanStorage,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<crate::VulkanStorage> {
        let weights = self.dequantize(self_shape.elem_count())?;
        let weights_l = crate::Layout::contiguous(self_shape.clone());
        weights.index_select(ids, &weights_l, ids_l, dim)
    }

    fn fwd_cpu_fallback(
        &self,
        self_shape: &Shape,
        storage: &crate::VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::VulkanStorage, Shape)> {
        storage.device().synchronize()?;
        let src_cpu = if layout.is_contiguous() && layout.start_offset() == 0 {
            storage.to_cpu_storage()?
        } else {
            let mut tmp = unsafe {
                storage
                    .device()
                    .alloc_uninit(layout.shape(), storage.dtype())?
            };
            storage.copy_strided_src(&mut tmp, 0, layout)?;
            tmp.to_cpu_storage()?
        };
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {self_shape:?}")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let lhs = cpu_storage_to_f32_vec(&src_cpu)?;
        let mut dst = vec![0f32; dst_shape.elem_count()];
        self.to_cpu_quantized()?
            .matmul_t((dst_shape.elem_count() / n, k, n), &lhs, &mut dst)?;
        let storage = self
            .device()
            .storage_from_cpu_storage(&crate::CpuStorage::F32(dst))?;
        self.device().synchronize()?;
        Ok((storage, dst_shape))
    }

    fn index_select_f32_cpu_fallback(
        &self,
        self_shape: &Shape,
        ids: &crate::VulkanStorage,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<crate::VulkanStorage> {
        ids.device().synchronize()?;
        let ids_cpu = if ids_l.is_contiguous() && ids_l.start_offset() == 0 {
            ids.to_cpu_storage()?
        } else {
            let mut tmp = unsafe { ids.device().alloc_uninit(ids_l.shape(), ids.dtype())? };
            ids.copy_strided_src(&mut tmp, 0, ids_l)?;
            tmp.to_cpu_storage()?
        };
        let ids = ids_cpu.as_slice::<u32>()?;
        let src = cpu_storage_to_f32_vec(
            &self
                .to_cpu_quantized()?
                .dequantize(self_shape.elem_count())?,
        )?;
        let dims = self_shape.dims();
        if dim >= dims.len() {
            crate::bail!("index_select dim {dim} out of range for {self_shape:?}")
        }
        let left_size: usize = dims[..dim].iter().product();
        let src_dim = dims[dim];
        let right_size: usize = dims[dim + 1..].iter().product();
        let mut dst_dims = dims.to_vec();
        dst_dims[dim] = ids.len();
        let mut dst = vec![0f32; dst_dims.iter().product()];
        for left_idx in 0..left_size {
            let src_left_base = left_idx * src_dim * right_size;
            let dst_left_base = left_idx * ids.len() * right_size;
            for (dst_row, &src_row) in ids.iter().enumerate() {
                let src_row = src_row as usize;
                if src_row >= src_dim {
                    crate::bail!("index_select id {src_row} out of range for dim size {src_dim}")
                }
                let src_offset = src_left_base + src_row * right_size;
                let dst_offset = dst_left_base + dst_row * right_size;
                dst[dst_offset..dst_offset + right_size]
                    .copy_from_slice(&src[src_offset..src_offset + right_size]);
            }
        }
        let storage = self
            .device()
            .storage_from_cpu_storage(&crate::CpuStorage::F32(dst))?;
        self.device().synchronize()?;
        Ok(storage)
    }

    fn fwd(
        &self,
        self_shape: &Shape,
        storage: &crate::VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::VulkanStorage, Shape)> {
        if self.dtype == GgmlDType::Q8K {
            return self.q8k_fwd_via_dense_gpu(self_shape, storage, layout);
        }
        match self
            .storage
            .quantized_matmul(self.dtype, self_shape, storage, layout)
        {
            Ok(out) => Ok(out),
            Err(err) if should_quantized_backend_fallback(&err, "vulkan") => {
                crate::storage::record_vulkan_cpu_fallback(&err);
                self.fwd_cpu_fallback(self_shape, storage, layout)
            }
            Err(err) => Err(err),
        }
    }

    fn index_select_f32(
        &self,
        self_shape: &Shape,
        ids: &crate::VulkanStorage,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<crate::VulkanStorage> {
        if self.dtype == GgmlDType::Q8K {
            return self.q8k_index_select_f32_via_dense_gpu(self_shape, ids, ids_l, dim);
        }
        match self
            .storage
            .quantized_index_select_f32(self.dtype, self_shape, ids, ids_l, dim)
        {
            Ok(out) => Ok(out),
            Err(err) if should_quantized_backend_fallback(&err, "vulkan") => {
                crate::storage::record_vulkan_cpu_fallback(&err);
                self.index_select_f32_cpu_fallback(self_shape, ids, ids_l, dim)
            }
            Err(err) => Err(err),
        }
    }

    fn indexed_moe_forward_f32(
        &self,
        self_shape: &Shape,
        input: &crate::VulkanStorage,
        input_l: &crate::Layout,
        ids: &crate::VulkanStorage,
        ids_l: &crate::Layout,
    ) -> Result<(crate::VulkanStorage, Shape)> {
        self.storage
            .quantized_indexed_moe_f32(self.dtype, self_shape, input, input_l, ids, ids_l)
    }
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
            Device::Wgpu(wgpu) => Ok(QStorage::Wgpu(QWgpuStorage::zeros(
                wgpu, elem_count, dtype,
            )?)),
            Device::Vulkan(vulkan) => Ok(QStorage::Vulkan(QVulkanStorage::zeros(
                vulkan, elem_count, dtype,
            )?)),
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Wgpu(QWgpuStorage),
    Vulkan(QVulkanStorage),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        let data: &[u8] = &data;
        match device {
            Device::Cpu => Ok(Self::Cpu(dtype.from_data(Cow::Borrowed(data)))),
            Device::Metal(d) => match dtype {
                GgmlDType::F32 => load_quantized_metal_from_bytes::<f32>(d, data),
                GgmlDType::F16 => load_quantized_metal_from_bytes::<f16>(d, data),
                GgmlDType::Q4_0 => load_quantized_metal_from_bytes::<BlockQ4_0>(d, data),
                GgmlDType::Q4_1 => load_quantized_metal_from_bytes::<BlockQ4_1>(d, data),
                GgmlDType::Q5_0 => load_quantized_metal_from_bytes::<BlockQ5_0>(d, data),
                GgmlDType::Q5_1 => load_quantized_metal_from_bytes::<BlockQ5_1>(d, data),
                GgmlDType::Q8_0 => load_quantized_metal_from_bytes::<BlockQ8_0>(d, data),
                GgmlDType::Q8_1 => load_quantized_metal_from_bytes::<BlockQ8_1>(d, data),
                GgmlDType::Q2K => load_quantized_metal_from_bytes::<BlockQ2K>(d, data),
                GgmlDType::Q3K => load_quantized_metal_from_bytes::<BlockQ3K>(d, data),
                GgmlDType::Q4K => load_quantized_metal_from_bytes::<BlockQ4K>(d, data),
                GgmlDType::Q5K => load_quantized_metal_from_bytes::<BlockQ5K>(d, data),
                GgmlDType::Q6K => load_quantized_metal_from_bytes::<BlockQ6K>(d, data),
                GgmlDType::Q8K => load_quantized_metal_from_bytes::<BlockQ8K>(d, data),
                GgmlDType::BF16 => load_quantized_metal_from_bytes::<bf16>(d, data),
            },
            Device::Cuda(d) => match dtype {
                GgmlDType::F32 => load_quantized_cuda_from_bytes::<f32>(d, data),
                GgmlDType::F16 => load_quantized_cuda_from_bytes::<f16>(d, data),
                GgmlDType::Q4_0 => load_quantized_cuda_from_bytes::<BlockQ4_0>(d, data),
                GgmlDType::Q4_1 => load_quantized_cuda_from_bytes::<BlockQ4_1>(d, data),
                GgmlDType::Q5_0 => load_quantized_cuda_from_bytes::<BlockQ5_0>(d, data),
                GgmlDType::Q5_1 => load_quantized_cuda_from_bytes::<BlockQ5_1>(d, data),
                GgmlDType::Q8_0 => load_quantized_cuda_from_bytes::<BlockQ8_0>(d, data),
                GgmlDType::Q8_1 => load_quantized_cuda_from_bytes::<BlockQ8_1>(d, data),
                GgmlDType::Q2K => load_quantized_cuda_from_bytes::<BlockQ2K>(d, data),
                GgmlDType::Q3K => load_quantized_cuda_from_bytes::<BlockQ3K>(d, data),
                GgmlDType::Q4K => load_quantized_cuda_from_bytes::<BlockQ4K>(d, data),
                GgmlDType::Q5K => load_quantized_cuda_from_bytes::<BlockQ5K>(d, data),
                GgmlDType::Q6K => load_quantized_cuda_from_bytes::<BlockQ6K>(d, data),
                GgmlDType::Q8K => load_quantized_cuda_from_bytes::<BlockQ8K>(d, data),
                GgmlDType::BF16 => load_quantized_cuda_from_bytes::<bf16>(d, data),
            },
            Device::Wgpu(d) => Ok(Self::Wgpu(QWgpuStorage::from_bytes(
                d,
                dtype,
                Cow::Borrowed(data),
            )?)),
            Device::Vulkan(d) => Ok(Self::Vulkan(QVulkanStorage::from_bytes(
                d,
                dtype,
                Cow::Borrowed(data),
            )?)),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Wgpu(storage) => storage.dtype.block_size(),
            QStorage::Vulkan(storage) => storage.dtype.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Wgpu(storage) => storage.dtype,
            QStorage::Vulkan(storage) => storage.dtype,
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Wgpu(storage) => Device::Wgpu(storage.device().clone()),
            QStorage::Vulkan(storage) => Device::Vulkan(storage.device().clone()),
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Wgpu(storage) => storage.len_bytes,
            QStorage::Vulkan(storage) => storage.len_bytes,
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Wgpu(storage), Storage::Wgpu(src)) => storage.quantize(src)?,
            (QStorage::Vulkan(storage), Storage::Vulkan(src)) => storage.quantize(src)?,
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Wgpu(storage), Storage::Wgpu(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Vulkan(storage), Storage::Vulkan(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cuda(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Wgpu(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Vulkan(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Metal(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Cuda(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            _ => crate::bail!("Invalid quantize source storage locations: not on cpu"),
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Wgpu(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Vulkan(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Wgpu(storage) => Ok(Storage::Wgpu(storage.dequantize(elem_count)?)),
            QStorage::Vulkan(storage) => Ok(Storage::Vulkan(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Wgpu(storage) => Ok(Cow::Owned(storage.data()?)),
            QStorage::Vulkan(storage) => Ok(Cow::Owned(storage.data()?)),
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::Wgpu(_) | QStorage::Vulkan(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr_with_guard(stream),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::Wgpu(_) | QStorage::Vulkan(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
        }
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
        let data: &[u8] = &data;
        match self {
            Self::F32 => Box::new(decode_t_vec::<f32>(data)),
            Self::F16 => Box::new(decode_t_vec::<f16>(data)),
            Self::Q4_0 => Box::new(decode_t_vec::<BlockQ4_0>(data)),
            Self::Q4_1 => Box::new(decode_t_vec::<BlockQ4_1>(data)),
            Self::Q5_0 => Box::new(decode_t_vec::<BlockQ5_0>(data)),
            Self::Q5_1 => Box::new(decode_t_vec::<BlockQ5_1>(data)),
            Self::Q8_0 => Box::new(decode_t_vec::<BlockQ8_0>(data)),
            Self::Q8_1 => Box::new(decode_block_q8_1_data(data)),
            Self::Q2K => Box::new(decode_t_vec::<BlockQ2K>(data)),
            Self::Q3K => Box::new(decode_t_vec::<BlockQ3K>(data)),
            Self::Q4K => Box::new(decode_t_vec::<BlockQ4K>(data)),
            Self::Q5K => Box::new(decode_t_vec::<BlockQ5K>(data)),
            Self::Q6K => Box::new(decode_t_vec::<BlockQ6K>(data)),
            Self::Q8K => Box::new(decode_block_q8k_data(data)),
            Self::BF16 => Box::new(decode_t_vec::<bf16>(data)),
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

fn indexed_moe_forward_dense(weights: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (num_experts, n, k) = weights.dims3()?;
    let (batch, input_dim1, input_k, x3) = match x.rank() {
        2 => {
            let (batch, k) = x.dims2()?;
            (batch, 1, k, x.unsqueeze(1)?)
        }
        3 => {
            let (batch, slots, k) = x.dims3()?;
            (batch, slots, k, x.clone())
        }
        rank => crate::bail!("indexed_moe_forward expects rank-2/3 input, got rank {rank}"),
    };
    let (ids_batch, topk) = ids.dims2()?;
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
    let ids = if ids.dtype() == DType::U32 {
        ids.clone()
    } else {
        ids.to_dtype(DType::U32)?
    };
    let max_id = ids.max_all()?.to_scalar::<u32>()? as usize;
    if max_id >= num_experts {
        crate::bail!("indexed_moe_forward id {max_id} out of range for {num_experts} experts");
    }
    let weights = if weights.dtype() == x3.dtype() {
        weights.clone()
    } else {
        weights.to_dtype(x3.dtype())?
    };
    let selected = weights
        .index_select(&ids.flatten_all()?, 0)?
        .reshape((batch * topk, n, k))?;
    let rows = if input_dim1 == topk {
        x3
    } else {
        x3.broadcast_as((batch, topk, k))?
    };
    let out = rows
        .reshape((batch * topk, 1, k))?
        .matmul(&selected.transpose(1, 2)?)?
        .reshape((batch, topk, n))?;
    if out.dtype() == x.dtype() {
        Ok(out)
    } else {
        out.to_dtype(x.dtype())
    }
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self { storage, shape })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, hidden_size) = self.shape.dims2()?;
        let flat_ids = ids.flatten_all()?;
        let gathered = self.index_select_rows0_f32(&flat_ids)?;
        let mut out_dims = ids.dims().to_vec();
        out_dims.push(hidden_size);
        gathered.reshape(out_dims)
    }

    fn index_select_rows0_f32(&self, ids: &Tensor) -> Result<Tensor> {
        if ids.rank() != 1 {
            crate::bail!(
                "quantized index_select expects rank-1 ids, got {:?}",
                ids.shape()
            );
        }
        let device = self.device();
        let ids = ids.to_dtype(DType::U32)?.to_device(&device)?;
        let ids_len = ids.dim(0)?;
        let mut out_dims = self.shape.dims().to_vec();
        out_dims[0] = ids_len;
        let out_shape = Shape::from(out_dims);
        let out = {
            let ids_storage = ids.storage();
            match (&self.storage, &*ids_storage) {
                (QStorage::Wgpu(storage), Storage::Wgpu(ids_storage)) => {
                    let out =
                        storage.index_select_f32(&self.shape, ids_storage, ids.layout(), 0)?;
                    Some(Ok(crate::tensor::from_storage(
                        Storage::Wgpu(out),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    )))
                }
                (QStorage::Vulkan(storage), Storage::Vulkan(ids_storage)) => {
                    let out =
                        storage.index_select_f32(&self.shape, ids_storage, ids.layout(), 0)?;
                    Some(Ok(crate::tensor::from_storage(
                        Storage::Vulkan(out),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    )))
                }
                _ => None,
            }
        };
        match out {
            Some(out) => out,
            None => {
                let weights = self.dequantize(&device)?;
                weights.index_select(&ids, 0)
            }
        }
    }

    fn indexed_moe_forward_quantized_gpu(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let (num_experts, n, k) = self.shape.dims3()?;
        let output_dtype = x.dtype();
        let output_device = x.device().clone();
        let device = self.device();
        let x = x.to_device(&device)?;
        let ids = ids.to_dtype(DType::U32)?.to_device(&device)?;
        let (batch, input_dim1, input_k, x3) = match x.rank() {
            2 => {
                let (batch, k) = x.dims2()?;
                (batch, 1, k, x.unsqueeze(1)?)
            }
            3 => {
                let (batch, slots, k) = x.dims3()?;
                (batch, slots, k, x.clone())
            }
            rank => crate::bail!("indexed_moe_forward expects rank-2/3 input, got rank {rank}"),
        };
        let (ids_batch, topk) = ids.dims2()?;
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
        let flat_ids = ids.flatten_all()?;
        let max_id = flat_ids.to_vec1::<u32>()?.into_iter().max().unwrap_or(0) as usize;
        if max_id >= num_experts {
            crate::bail!("indexed_moe_forward id {max_id} out of range for {num_experts} experts");
        }
        let x3_f32 = if x3.dtype() == DType::F32 {
            x3.clone()
        } else {
            x3.to_dtype(DType::F32)?
        };
        if let (
            QStorage::Vulkan(storage),
            Storage::Vulkan(x_storage),
            Storage::Vulkan(ids_storage),
        ) = (&self.storage, &*x3_f32.storage(), &*ids.storage())
        {
            match storage.indexed_moe_forward_f32(
                &self.shape,
                x_storage,
                x3_f32.layout(),
                ids_storage,
                ids.layout(),
            ) {
                Ok((storage, out_shape)) => {
                    let out = crate::tensor::from_storage(
                        Storage::Vulkan(storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    );
                    let out = if out.dtype() == output_dtype {
                        out
                    } else {
                        out.to_dtype(output_dtype)?
                    };
                    return out.to_device(&output_device);
                }
                Err(err) if should_quantized_backend_fallback(&err, "vulkan") => {}
                Err(err) => return Err(err),
            }
        }
        let selected = self
            .index_select_rows0_f32(&flat_ids)?
            .reshape((batch * topk, n, k))?;
        let rows = if input_dim1 == topk {
            x3
        } else {
            x3.broadcast_as((batch, topk, k))?
        };
        let rows = if rows.dtype() == DType::F32 {
            rows
        } else {
            rows.to_dtype(DType::F32)?
        };
        let out = rows
            .reshape((batch * topk, 1, k))?
            .matmul(&selected.transpose(1, 2)?)?
            .reshape((batch, topk, n))?;
        let out = if out.dtype() == output_dtype {
            out
        } else {
            out.to_dtype(output_dtype)?
        };
        out.to_device(&output_device)
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match &self.storage {
            QStorage::Cuda(s) => match (&*x.storage(), &*ids.storage()) {
                (Storage::Cuda(x_storage), Storage::Cuda(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Cuda(storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                _ => {
                    let weights = self.dequantize(x.device())?;
                    indexed_moe_forward_dense(&weights, x, ids)
                }
            },
            QStorage::Wgpu(_) | QStorage::Vulkan(_) => {
                self.indexed_moe_forward_quantized_gpu(x, ids)
            }
            _ => {
                let weights = self.dequantize(x.device())?;
                indexed_moe_forward_dense(&weights, x, ids)
            }
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::Wgpu(_) | QStorage::Vulkan(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        self.storage.device_ptr_with_guard(stream)
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            Self::Tensor(w) | Self::TensorF16(w) => indexed_moe_forward_dense(w, x, ids),
        }
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let run = |self_storage: &dyn QuantizedType| match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];
                self_storage.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F32(dst_storage), dst_shape.clone()))
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self_storage.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F16(dst_storage), dst_shape.clone()))
            }
            _ => crate::bail!("Expected f32/f16"),
        };
        match &self.storage {
            QStorage::Cpu(storage) => run(storage.as_ref()),
            QStorage::Wgpu(storage) => {
                let cpu_storage = storage.to_cpu_quantized()?;
                run(cpu_storage.as_ref())
            }
            QStorage::Vulkan(storage) => {
                let cpu_storage = storage.to_cpu_quantized()?;
                run(cpu_storage.as_ref())
            }
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        }
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn wgpu_fwd(
        &self,
        storage: &crate::WgpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::WgpuStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Wgpu(wgpu) => wgpu,
            _ => unreachable!("Cannot call wgpu matmul on non wgpu QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn vulkan_fwd(
        &self,
        storage: &crate::VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::VulkanStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Vulkan(vulkan) => vulkan,
            _ => unreachable!("Cannot call vulkan matmul on non vulkan QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => {
                let out = {
                    let xs_storage = xs.storage();
                    match (&t.storage, &*xs_storage) {
                        (QStorage::Wgpu(wgpu), Storage::Wgpu(storage)) => {
                            let (storage, shape) = wgpu.fwd(&t.shape, storage, xs.layout())?;
                            let out = crate::tensor::from_storage(
                                Storage::Wgpu(storage),
                                shape,
                                crate::op::BackpropOp::none(),
                                false,
                            );
                            Some(Ok(out))
                        }
                        (QStorage::Vulkan(vulkan), Storage::Vulkan(storage)) => {
                            let (storage, shape) = vulkan.fwd(&t.shape, storage, xs.layout())?;
                            let out = crate::tensor::from_storage(
                                Storage::Vulkan(storage),
                                shape,
                                crate::op::BackpropOp::none(),
                                false,
                            );
                            Some(Ok(out))
                        }
                        _ => None,
                    }
                };
                match out {
                    Some(out) => out,
                    None => xs.apply_op1_no_bwd(t.as_ref()),
                }
            }
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_data_accepts_misaligned_quantized_bytes() -> Result<()> {
        let zeros = GgmlDType::Q4_0.cpu_zeros(k_quants::QK4_0);
        let bytes =
            unsafe { std::slice::from_raw_parts(zeros.as_ptr(), zeros.storage_size_in_bytes()) };
        let mut misaligned = vec![0x7f];
        misaligned.extend_from_slice(bytes);
        let reparsed = GgmlDType::Q4_0.from_data(Cow::Borrowed(&misaligned[1..]));
        let dequantized = reparsed
            .dequantize(k_quants::QK4_0)?
            .as_slice::<f32>()?
            .to_vec();
        assert_eq!(dequantized, vec![0.0; k_quants::QK4_0]);
        Ok(())
    }
}
