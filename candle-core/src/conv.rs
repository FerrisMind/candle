//! 1D and 2D Convolutions
//!
use crate::{op::BackpropOp, op::Op, DType, Error, Result, Tensor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    pub(crate) b_size: usize,
    // Maybe we should have a version without l_in as this bit depends on the input and not only on
    // the weights.
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub(crate) cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in + 2 * self.padding - self.dilation * (self.k_size - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose1D {
    pub(crate) b_size: usize,
    pub(crate) l_in: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) k_size: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose1D {
    pub(crate) fn l_out(&self) -> usize {
        (self.l_in - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.k_size - 1)
            + self.output_padding
            + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        let l_out = self.l_out();
        vec![self.b_size, self.c_out, l_out]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudnnFwdAlgo {
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonFused,
    Count,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl ParamsConv2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h + 2 * self.padding - self.dilation * (self.k_h - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w + 2 * self.padding - self.dilation * (self.k_w - 1) - 1) / self.stride + 1
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose2D {
    pub(crate) b_size: usize,
    pub(crate) i_h: usize,
    pub(crate) i_w: usize,
    pub(crate) k_h: usize,
    pub(crate) k_w: usize,
    pub(crate) c_out: usize,
    pub(crate) c_in: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

impl ParamsConvTranspose2D {
    pub(crate) fn out_h(&self) -> usize {
        (self.i_h - 1) * self.stride + self.dilation * (self.k_h - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_w(&self) -> usize {
        (self.i_w - 1) * self.stride + self.dilation * (self.k_w - 1) + self.output_padding + 1
            - 2 * self.padding
    }

    pub(crate) fn out_dims(&self) -> Vec<usize> {
        vec![self.b_size, self.c_out, self.out_h(), self.out_w()]
    }
}

impl Tensor {
    fn conv_transpose1d_wgpu_decomp(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose1D,
    ) -> Result<Option<Self>> {
        if !self.device().is_wgpu() {
            return Ok(None);
        }
        let src_dtype = match self.dtype() {
            DType::F32 | DType::F16 => self.dtype(),
            _ => return Ok(None),
        };
        if src_dtype != kernel.dtype() {
            return Ok(None);
        }

        let src_len = match params.l_in.checked_mul(params.k_size) {
            Some(v) => v,
            None => return Ok(None),
        };
        let l_out = params.l_out();
        if l_out > u32::MAX as usize {
            return Ok(None);
        }

        let input = if src_dtype == DType::F32 {
            self.clone()
        } else {
            self.to_dtype(DType::F32)?
        };
        let kernel = if src_dtype == DType::F32 {
            kernel.clone()
        } else {
            kernel.to_dtype(DType::F32)?
        };

        let input_t = input
            .transpose(1, 2)?
            .contiguous()?
            .reshape((params.b_size * params.l_in, params.c_in))?;
        let kernel_mm = kernel
            .contiguous()?
            .reshape((params.c_in, params.c_out * params.k_size))?;
        let cols = input_t.matmul(&kernel_mm)?.reshape((
            params.b_size,
            params.l_in,
            params.c_out,
            params.k_size,
        ))?;
        let src = cols
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((params.b_size * params.c_out, src_len))?;

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

        let ids = Tensor::from_vec(ids, src_len, self.device())?;
        let mask = Tensor::from_vec(mask, (1, src_len), self.device())?;
        let src = src.broadcast_mul(&mask)?;
        let out = Tensor::zeros(
            (params.b_size * params.c_out, l_out),
            DType::F32,
            self.device(),
        )?
        .index_add(&ids, &src, 1)?
        .reshape((params.b_size, params.c_out, l_out))?;
        let out = if src_dtype == DType::F32 {
            out
        } else {
            out.to_dtype(src_dtype)?
        };
        Ok(Some(out))
    }

    fn conv_transpose2d_wgpu_decomp(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose2D,
    ) -> Result<Option<Self>> {
        if !self.device().is_wgpu() {
            return Ok(None);
        }
        let src_dtype = match self.dtype() {
            DType::F32 | DType::F16 => self.dtype(),
            _ => return Ok(None),
        };
        if src_dtype != kernel.dtype() {
            return Ok(None);
        }

        let input_spatial = match params.i_h.checked_mul(params.i_w) {
            Some(v) => v,
            None => return Ok(None),
        };
        let kernel_spatial = match params.k_h.checked_mul(params.k_w) {
            Some(v) => v,
            None => return Ok(None),
        };
        let src_len = match input_spatial.checked_mul(kernel_spatial) {
            Some(v) => v,
            None => return Ok(None),
        };
        let out_h = params.out_h();
        let out_w = params.out_w();
        let out_spatial = match out_h.checked_mul(out_w) {
            Some(v) => v,
            None => return Ok(None),
        };
        if out_spatial > u32::MAX as usize {
            return Ok(None);
        }

        let input = if src_dtype == DType::F32 {
            self.clone()
        } else {
            self.to_dtype(DType::F32)?
        };
        let kernel = if src_dtype == DType::F32 {
            kernel.clone()
        } else {
            kernel.to_dtype(DType::F32)?
        };

        let input_hw = input
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((params.b_size * input_spatial, params.c_in))?;
        let kernel_mm = kernel
            .contiguous()?
            .reshape((params.c_in, params.c_out * kernel_spatial))?;
        let cols = input_hw.matmul(&kernel_mm)?.reshape((
            params.b_size,
            input_spatial,
            params.c_out,
            params.k_h,
            params.k_w,
        ))?;
        let src = cols
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((params.b_size * params.c_out, src_len))?;

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

        let ids = Tensor::from_vec(ids, src_len, self.device())?;
        let mask = Tensor::from_vec(mask, (1, src_len), self.device())?;
        let src = src.broadcast_mul(&mask)?;
        let out = Tensor::zeros(
            (params.b_size * params.c_out, out_spatial),
            DType::F32,
            self.device(),
        )?
        .index_add(&ids, &src, 1)?
        .reshape((params.b_size, params.c_out, out_h, out_w))?;
        let out = if src_dtype == DType::F32 {
            out
        } else {
            out.to_dtype(src_dtype)?
        };
        Ok(Some(out))
    }

    fn conv1d_single_group(&self, kernel: &Self, params: &ParamsConv1D) -> Result<Self> {
        let storage =
            self.storage()
                .conv1d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv1D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv1d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    /// Applies a 1D convolution over the input tensor.
    pub fn conv1d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k * groups {
            Err(Error::Conv1dInvalidArgs {
                inp_shape: self.shape().clone(),
                k_shape: kernel.shape().clone(),
                padding,
                stride,
                msg: "the number of in-channels on the input doesn't match the kernel size",
            }
            .bt())?
        }

        let params = ParamsConv1D {
            b_size,
            l_in,
            c_out: c_out / groups,
            c_in: c_in / groups,
            k_size,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv_transpose1d_single_group(
        &self,
        kernel: &Self,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        if let Some(out) = self.conv_transpose1d_wgpu_decomp(kernel, params)? {
            let out = out.contiguous()?;
            let storage = out.storage().try_clone(out.layout())?;
            let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose1D {
                arg,
                kernel,
                padding: params.padding,
                output_padding: params.output_padding,
                stride: params.stride,
                dilation: params.dilation,
            });
            let out_dims = params.out_dims();
            return Ok(crate::tensor::from_storage(storage, out_dims, op, false));
        }
        let storage = self.storage().conv_transpose1d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose1D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 1D transposed convolution over the input tensor.
    pub fn conv_transpose1d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let (c_in_k, c_out, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = self.dims3()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        if c_in % groups != 0 {
            crate::bail!("in_channel {c_in} is not divisible by the number of groups")
        }
        let params = ParamsConvTranspose1D {
            b_size,
            l_in,
            k_size,
            c_out,
            c_in: c_in / groups,
            padding,
            output_padding,
            stride,
            dilation,
        };
        if groups == 1 {
            self.conv_transpose1d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv_transpose1d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    fn conv2d_single_group(&self, kernel: &Self, params: &ParamsConv2D) -> Result<Self> {
        let storage =
            self.storage()
                .conv2d(self.layout(), &kernel.storage(), kernel.layout(), params)?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::Conv2D {
            arg,
            kernel,
            padding: params.padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }

    /// Applies a 2D convolution over the input tensor.
    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        self.conv2d_with_algo(kernel, padding, stride, dilation, groups, None)
    }

    pub fn conv2d_with_algo(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        cudnn_fwd_algo: Option<CudnnFwdAlgo>,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_out, c_in_k, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k * groups {
            crate::bail!(
                "in_channel mismatch between input ({c_in}, groups {groups}) and kernel ({c_in_k})"
            )
        }
        let params = ParamsConv2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out: c_out / groups,
            c_in: c_in / groups,
            padding,
            stride,
            dilation,
            cudnn_fwd_algo,
        };
        if groups == 1 {
            self.conv2d_single_group(kernel, &params)
        } else {
            let blocks = self.chunk(groups, 1)?;
            let kernel = kernel.chunk(groups, 0)?;
            let blocks = blocks
                .iter()
                .zip(&kernel)
                .map(|(block, kernel)| block.conv2d_single_group(kernel, &params))
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&blocks, 1)
        }
    }

    /// Applies a 2D transposed convolution over the input tensor.
    pub fn conv_transpose2d(
        &self,
        kernel: &Self,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Result<Self> {
        let (b_size, c_in, i_h, i_w) = self.dims4()?;
        let (c_in_k, c_out, k_h, k_w) = kernel.dims4()?;
        if c_in != c_in_k {
            crate::bail!("in_channel mismatch between input ({c_in}) and kernel ({c_in_k})")
        }
        let params = ParamsConvTranspose2D {
            b_size,
            i_h,
            i_w,
            k_h,
            k_w,
            c_out,
            c_in,
            padding,
            output_padding,
            stride,
            dilation,
        };
        if let Some(out) = self.conv_transpose2d_wgpu_decomp(kernel, &params)? {
            let out = out.contiguous()?;
            let storage = out.storage().try_clone(out.layout())?;
            let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose2D {
                arg,
                kernel,
                padding: params.padding,
                output_padding: params.output_padding,
                stride: params.stride,
                dilation: params.dilation,
            });
            let out_dims = params.out_dims();
            return Ok(crate::tensor::from_storage(storage, out_dims, op, false));
        }
        let storage = self.storage().conv_transpose2d(
            self.layout(),
            &kernel.storage(),
            kernel.layout(),
            &params,
        )?;
        let op = BackpropOp::new2(self, kernel, |arg, kernel| Op::ConvTranspose2D {
            arg,
            kernel,
            padding: params.padding,
            output_padding: params.output_padding,
            stride: params.stride,
            dilation: params.dilation,
        });
        let out_dims = params.out_dims();
        Ok(crate::tensor::from_storage(storage, out_dims, op, false))
    }
}
