#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct WgpuDevice;

#[derive(Debug)]
pub struct WgpuStorage;

impl WgpuStorage {
    pub(crate) fn quantized_index_select_f32(
        &self,
        _: crate::quantized::GgmlDType,
        _: &Shape,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub(crate) fn quantized_matmul(
        &self,
        _: crate::quantized::GgmlDType,
        _: &Shape,
        _: &Self,
        _: &Layout,
    ) -> Result<(Self, Shape)> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub(crate) fn quantized_indexed_moe_f32(
        &self,
        _: crate::quantized::GgmlDType,
        _: &Shape,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
    ) -> Result<(Self, Shape)> {
        Err(Error::NotCompiledWithWgpuSupport)
    }

    pub(crate) fn quantized_raw_float_dequantize_f32(
        &self,
        _: crate::quantized::GgmlDType,
        _: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("{0}")]
    Message(String),
}

macro_rules! fail {
    () => {
        unimplemented!("wgpu support has not been enabled, add `wgpu` feature to enable.")
    };
}

impl crate::backend::BackendStorage for WgpuStorage {
    type Device = WgpuDevice;
    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn dtype(&self) -> DType {
        fail!()
    }
    fn device(&self) -> &Self::Device {
        fail!()
    }
    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
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
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
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
        Err(Error::NotCompiledWithWgpuSupport)
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
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
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
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
}

impl crate::backend::BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }
    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }
    fn zeros_impl(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    unsafe fn alloc_uninit(&self, _: &Shape, _: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn get_current_seed(&self) -> Result<u64> {
        Err(Error::NotCompiledWithWgpuSupport)
    }
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
