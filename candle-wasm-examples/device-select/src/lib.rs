//! Shared CPU / wgpu / auto device resolution for Candle WASM examples.

use serde::{Deserialize, Serialize};

/// User-facing device selection (`cpu` | `wgpu` | `auto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceMode {
    Cpu,
    Wgpu,
    Auto,
}

/// Backend actually used after resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedKind {
    Cpu,
    Wgpu,
}

/// Resolved Candle [`Device`](candle::Device) plus reporting metadata.
pub struct ResolvedDevice {
    pub device: candle::Device,
    pub resolved: ResolvedKind,
    pub adapter_name: Option<String>,
}

impl DeviceMode {
    /// Parse `"cpu"` | `"wgpu"` | `"auto"` (case-sensitive).
    pub fn parse(s: &str) -> Result<Self, candle::Error> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "wgpu" => Ok(Self::Wgpu),
            "auto" => Ok(Self::Auto),
            other => {
                candle::bail!("invalid device mode `{other}`; expected cpu|wgpu|auto")
            }
        }
    }

    /// Resolve to a concrete Candle device.
    ///
    /// - `Cpu` → always CPU.
    /// - `Wgpu` → WebGPU/native wgpu (errors if unavailable or feature off).
    /// - `Auto` → try wgpu; on failure return explicit CPU (not a silent GPU fake).
    pub async fn resolve(self) -> Result<ResolvedDevice, candle::Error> {
        match self {
            DeviceMode::Cpu => Ok(ResolvedDevice {
                device: candle::Device::Cpu,
                resolved: ResolvedKind::Cpu,
                adapter_name: None,
            }),
            DeviceMode::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    let device = candle::Device::new_wgpu_async(0).await?;
                    let adapter_name = match &device {
                        candle::Device::Wgpu(d) => Some(d.adapter_name().to_string()),
                        _ => None,
                    };
                    Ok(ResolvedDevice {
                        device,
                        resolved: ResolvedKind::Wgpu,
                        adapter_name,
                    })
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    candle::bail!("wgpu feature not enabled in this build")
                }
            }
            DeviceMode::Auto => {
                #[cfg(feature = "wgpu")]
                {
                    match candle::Device::new_wgpu_async(0).await {
                        Ok(device) => {
                            let adapter_name = match &device {
                                candle::Device::Wgpu(d) => Some(d.adapter_name().to_string()),
                                _ => None,
                            };
                            Ok(ResolvedDevice {
                                device,
                                resolved: ResolvedKind::Wgpu,
                                adapter_name,
                            })
                        }
                        Err(_) => Ok(ResolvedDevice {
                            device: candle::Device::Cpu,
                            resolved: ResolvedKind::Cpu,
                            adapter_name: None,
                        }),
                    }
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Ok(ResolvedDevice {
                        device: candle::Device::Cpu,
                        resolved: ResolvedKind::Cpu,
                        adapter_name: None,
                    })
                }
            }
        }
    }
}
