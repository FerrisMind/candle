//! Rotary Embeddings
//!
use candle::{CpuStorage, Layout, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Interleaved variant of rotary embeddings.
/// The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
/// The resulting y0 and y1 are also interleaved with:
///   y0 = x0*cos - x1*sin
///   y1 = x0*sin + x1*cos
#[derive(Debug, Clone)]
struct RotaryEmbI;

impl candle::CustomOp3 for RotaryEmbI {
    fn name(&self) -> &'static str {
        "rotary-emb-int"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (src, dst))| {
                    for i_over_2 in 0..t * d / 2 {
                        let i = 2 * i_over_2;
                        let rope_i = if unbatched_rope {
                            let b_i = bh_i / h;
                            i_over_2 + b_i * t * d / 2
                        } else {
                            i_over_2
                        };
                        dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                        dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_i"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, (b * h) as u32, (t * d) as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope_i");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope-i {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_i_f32",
            candle::DType::F16 => "rope_i_f16",
            candle::DType::BF16 => "rope_i_bf16",
            dtype => candle::bail!("rope-i is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, src.dtype())
            .with_label("rope_i")
            .build()?;
        candle_metal_kernels::call_rope_i(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b * h,
            t * d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

fn rope_check_cs(cs: &Tensor, b_sz: usize) -> Result<(usize, usize)> {
    match *cs.dims() {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                candle::bail!("inconsistent batch size in rope {b_sz} {cs:?}",)
            }
            Ok((t, d))
        }
        _ => candle::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs:?}"),
    }
}

pub fn rope_i(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    // Avoid per-call CPU fallback on WGPU/Vulkan for the custom-op path.
    // Use a rank-4 tensor decomposition so execution stays on the backend
    // without hitting the current rank-5 fallback paths.
    if xs.device().is_wgpu() || xs.device().is_vulkan() {
        let even_ids = Tensor::arange_step(0u32, n_embd as u32, 2u32, xs.device())?;
        let odd_ids = Tensor::arange_step(1u32, n_embd as u32, 2u32, xs.device())?;
        let x0 = xs.index_select(&even_ids, D::Minus1)?;
        let x1 = xs.index_select(&odd_ids, D::Minus1)?;
        let cos = match cos.dims() {
            [_t, _d] => cos.unsqueeze(0)?.unsqueeze(0)?,
            [_b, _t, _d] => cos.unsqueeze(1)?,
            dims => candle::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {dims:?}"),
        };
        let sin = match sin.dims() {
            [_t, _d] => sin.unsqueeze(0)?.unsqueeze(0)?,
            [_b, _t, _d] => sin.unsqueeze(1)?,
            dims => candle::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {dims:?}"),
        };
        let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
        let ids_shape = y0.dims().to_vec();
        let even_ids = even_ids
            .reshape((1, 1, 1, n_embd / 2))?
            .broadcast_as(ids_shape.clone())?
            .contiguous()?;
        let odd_ids = odd_ids
            .reshape((1, 1, 1, n_embd / 2))?
            .broadcast_as(ids_shape)?
            .contiguous()?;
        let out = xs.zeros_like()?;
        out.scatter_set(&even_ids, &y0, D::Minus1)?;
        out.scatter_set(&odd_ids, &y1, D::Minus1)?;
        return Ok(out);
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbI)
}

pub fn rope_i_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
    let rope = rope.flatten_from(D::Minus2)?;
    Ok(rope)
}

/// Contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmb;

impl candle::CustomOp3 for RotaryEmb {
    fn name(&self) -> &'static str {
        "rotary-emb"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i1 = i_t * d + i_d;
                            let i2 = i1 + d / 2;
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                let b_i = bh_i / h;
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                            dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, (b * h) as u32, (t * d) as u32, d as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_f32",
            candle::DType::F16 => "rope_f16",
            candle::DType::BF16 => "rope_bf16",
            dtype => candle::bail!("rope is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, src.dtype())
            .with_label("rope")
            .build()?;
        candle_metal_kernels::call_rope(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b * h,
            t * d,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

pub fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    // Avoid per-call CPU fallback on WGPU/Vulkan for the custom-op path.
    // Use the explicit tensor decomposition so execution stays on the backend.
    if xs.device().is_wgpu() || xs.device().is_vulkan() {
        return rope_slow(xs, cos, sin);
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmb)
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

pub fn rope_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _h, seq_len, _n_embd) = x.dims4()?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?;
    let cos = cos.narrow(0, 0, seq_len)?;
    let sin = sin.narrow(0, 0, seq_len)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

#[derive(Debug, Clone)]
#[cfg_attr(not(any(feature = "wgpu", feature = "vulkan")), allow(dead_code))]
struct RotaryEmbGgml {
    n_dims: usize,
    freq_base: f32,
    mode: u32,
}

impl candle::CustomOp2 for RotaryEmbGgml {
    fn name(&self) -> &'static str {
        "rotary-emb-ggml"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("ggml rope has no cpu impl")
    }

    #[cfg(feature = "wgpu")]
    fn wgpu_fwd(
        &self,
        s1: &candle::WgpuStorage,
        l1: &Layout,
        s2: &candle::WgpuStorage,
        l2: &Layout,
    ) -> Result<(candle::WgpuStorage, Shape)> {
        let storage = s1.ggml_rope(l1, s2, l2, self.n_dims, self.freq_base, self.mode)?;
        Ok((storage, l1.shape().clone()))
    }

    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(
        &self,
        s1: &candle::VulkanStorage,
        l1: &Layout,
        s2: &candle::VulkanStorage,
        l2: &Layout,
    ) -> Result<(candle::VulkanStorage, Shape)> {
        let storage = s1.ggml_rope(l1, s2, l2, self.n_dims, self.freq_base, self.mode)?;
        Ok((storage, l1.shape().clone()))
    }
}

pub fn rope_ggml(
    xs: &Tensor,
    positions: &Tensor,
    n_dims: usize,
    freq_base: f32,
    mode: u32,
) -> Result<Tensor> {
    if !positions.is_contiguous() {
        candle::bail!("positions has to be contiguous in rope_ggml")
    }
    xs.apply_op2_no_bwd(
        positions,
        &RotaryEmbGgml {
            n_dims,
            freq_base,
            mode,
        },
    )
}

/// T (seqlen)/H (num-heads)/D (head-dim) contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmbThd;

impl candle::CustomOp3 for RotaryEmbThd {
    fn name(&self) -> &'static str {
        "rotary-emb"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * h * d)
                .zip(dst.par_chunks_mut(t * h * d))
                .enumerate()
                .for_each(|(b_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            for i_h in 0..h {
                                let i1 = i_t * h * d + i_h * d + i_d;
                                let i2 = i1 + d / 2;
                                dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                                dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                            }
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, t, h, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, WrapErr};
        use candle::{CudaDevice, WithDType};

        fn inner<T: DeviceRepr + WithDType>(
            src: &CudaSlice<T>,
            l_src: &Layout,
            cos: &CudaSlice<T>,
            l_cos: &Layout,
            sin: &CudaSlice<T>,
            l_sin: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("src input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("cos input has to be contiguous"),
                Some((o1, o2)) => cos.slice(o1..o2),
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("sin input has to be contiguous"),
                Some((o1, o2)) => sin.slice(o1..o2),
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
                (h * t * d) as u32
            } else {
                0u32
            };
            let el = b * h * t * d;
            let cfg = LaunchConfig::for_num_elems((el / 2) as u32);
            let func = dev.get_or_load_func(&kernel_name::<T>("rope_thd"), &kernels::REDUCE)?;
            // SAFETY: Set later by running the kernel.
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&dst);
            candle::builder_arg!(builder, b as u32, t as u32, h as u32, d as u32, stride_b);
            // SAFETY: ffi.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use candle::backend::BackendStorage;
        use candle::cuda_backend::CudaStorageSlice::{BF16, F16, F32, F64};
        let dev = s1.device();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (BF16(s1), BF16(s2), BF16(s3)) => BF16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F16(s1), F16(s2), F16(s3)) => F16(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F32(s1), F32(s2), F32(s3)) => F32(inner(s1, l1, s2, l2, s3, l3, dev)?),
            (F64(s1), F64(s2), F64(s3)) => F64(inner(s1, l1, s2, l2, s3, l3, dev)?),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rope_thd");
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_thd_f32",
            candle::DType::F16 => "rope_thd_f16",
            candle::DType::BF16 => "rope_thd_bf16",
            dtype => candle::bail!("rope_thd is not implemented for {dtype:?}"),
        };
        let (b, t, h, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device
            .new_buffer_builder()
            .with_size_for(el, src.dtype())
            .with_label("rope_thd")
            .build()?;
        candle_metal_kernels::call_rope_thd(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            b,
            t,
            h,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

pub fn rope_thd(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len, _n_head, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    // Keep WGPU/Vulkan on the tensor-composed path (no custom-op CPU fallthrough).
    // Layout is (B, T, H, D) — transpose to (B, H, T, D), apply rope, transpose back.
    if xs.device().is_wgpu() || xs.device().is_vulkan() {
        let x = xs.transpose(1, 2)?.contiguous()?;
        let y = rope(&x, cos, sin)?;
        return y.transpose(1, 2)?.contiguous();
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbThd)
}
