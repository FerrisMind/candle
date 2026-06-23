pub(crate) mod affine;
pub(crate) mod binary;
pub(crate) mod broadcast;
pub(crate) mod cat;
pub(crate) mod contiguous;
pub(crate) mod conv_transpose2d;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod qmatmul;
pub(crate) mod random;
pub(crate) mod reduce;
pub(crate) mod unary;
pub(crate) mod vulkan_minimal;
pub(crate) mod where_cond;

use candle_core::{Device, Result};

pub(crate) trait BenchDevice {
    fn sync(&self) -> Result<()>;

    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    fn sync(&self) -> Result<()> {
        self.synchronize()
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu => {
                let cpu_type = if cfg!(feature = "accelerate") {
                    "accelerate"
                } else if cfg!(feature = "mkl") {
                    "mkl"
                } else {
                    "cpu"
                };
                format!("{}_{}", cpu_type, name.into())
            }
            Device::Cuda(_) => format!("cuda_{}", name.into()),
            Device::Metal(_) => format!("metal_{}", name.into()),
            Device::Wgpu(_) => format!("wgpu_{}", name.into()),
            Device::Vulkan(_) => format!("vulkan_{}", name.into()),
        }
    }
}

struct BenchDeviceHandler {
    devices: Vec<Device>,
}

impl BenchDeviceHandler {
    pub fn new() -> Result<Self> {
        let mut devices = Vec::new();
        #[cfg(feature = "vulkan")]
        if let Ok(device) = Device::new_vulkan(0) {
            devices.push(device);
        }

        #[cfg(feature = "wgpu")]
        if let Ok(device) = Device::new_wgpu(0) {
            devices.push(device);
        }

        if cfg!(feature = "metal") {
            devices.push(Device::new_metal(0)?);
        } else if cfg!(feature = "cuda") {
            devices.push(Device::new_cuda(0)?);
        }
        if devices.is_empty() {
            devices.push(Device::Cpu);
        }
        Ok(Self { devices })
    }
}
