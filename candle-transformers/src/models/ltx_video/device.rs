//! Device management and backend selection for LTX-Video
//!
//! This module provides utilities for automatic device detection, backend selection,
//! and backend-specific optimizations.

use candle::{Device, Result};
use std::fmt;

/// Backend type for device selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU backend with MKL or Accelerate optimizations
    Cpu,
    /// NVIDIA CUDA backend
    Cuda(usize), // GPU index
    /// Apple Metal backend
    Metal(usize), // GPU index
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Cpu => write!(f, "CPU"),
            Backend::Cuda(idx) => write!(f, "CUDA:{}", idx),
            Backend::Metal(idx) => write!(f, "Metal:{}", idx),
        }
    }
}

impl Backend {
    /// Auto-detect the best available backend
    ///
    /// Priority: CUDA > Metal > CPU
    pub fn auto_detect() -> Result<Self> {
        // Try CUDA first
        #[cfg(feature = "cuda")]
        {
            if Device::cuda_if_available(0).is_ok() {
                return Ok(Backend::Cuda(0));
            }
        }

        // Try Metal on macOS
        #[cfg(feature = "metal")]
        {
            if Device::new_metal(0).is_ok() {
                return Ok(Backend::Metal(0));
            }
        }

        // Fallback to CPU
        Ok(Backend::Cpu)
    }

    /// Create a device from this backend
    pub fn to_device(&self) -> Result<Device> {
        match self {
            Backend::Cpu => Ok(Device::Cpu),
            Backend::Cuda(_idx) => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(*_idx)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(candle::Error::Msg(
                        "CUDA support not compiled. Rebuild with --features cuda".to_string(),
                    ))
                }
            }
            Backend::Metal(_idx) => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(*_idx)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(candle::Error::Msg(
                        "Metal support not compiled. Rebuild with --features metal".to_string(),
                    ))
                }
            }
        }
    }

    /// Check if this backend is available on the current system
    pub fn is_available(&self) -> bool {
        self.to_device().is_ok()
    }

    /// Get available backends on the current system
    pub fn available_backends() -> Vec<Backend> {
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            vec![Backend::Cpu]
        }

        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            let mut backends = vec![Backend::Cpu];

            #[cfg(feature = "cuda")]
            {
                // Try to detect CUDA devices
                for idx in 0..8 {
                    if Device::new_cuda(idx).is_ok() {
                        backends.push(Backend::Cuda(idx));
                    }
                }
            }

            #[cfg(feature = "metal")]
            {
                // Try to detect Metal devices
                for idx in 0..4 {
                    if Device::new_metal(idx).is_ok() {
                        backends.push(Backend::Metal(idx));
                    }
                }
            }

            backends
        }
    }

    /// Get the name of the backend for display
    pub fn name(&self) -> &str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Cuda(_) => "CUDA",
            Backend::Metal(_) => "Metal",
        }
    }

    /// Check if this backend supports GPU optimizations
    pub fn is_gpu(&self) -> bool {
        matches!(self, Backend::Cuda(_) | Backend::Metal(_))
    }

    /// Get recommended memory buffer size for this backend
    ///
    /// Returns a safety margin factor (e.g., 0.9 means use 90% of available memory)
    pub fn memory_buffer_factor(&self) -> f32 {
        match self {
            Backend::Cpu => 0.95,      // CPU typically has more memory available
            Backend::Cuda(_) => 0.85,  // CUDA needs buffer for kernel overhead
            Backend::Metal(_) => 0.90, // Metal shares memory with system
        }
    }
}

/// Device manager for LTX-Video pipeline
#[derive(Debug)]
pub struct DeviceManager {
    backend: Backend,
    device: Device,
}

impl DeviceManager {
    /// Create a new device manager with automatic backend detection
    pub fn auto() -> Result<Self> {
        let backend = Backend::auto_detect()?;
        let device = backend.to_device()?;

        println!("Auto-detected backend: {}", backend);

        Ok(Self { backend, device })
    }

    /// Create a new device manager with a specific backend
    pub fn new(backend: Backend) -> Result<Self> {
        if !backend.is_available() {
            return Err(candle::Error::Msg(format!(
                "Backend {} is not available on this system",
                backend
            )));
        }

        let device = backend.to_device()?;

        println!("Using backend: {}", backend);

        Ok(Self { backend, device })
    }

    /// Create a new device manager from an existing device
    pub fn from_device(device: Device) -> Self {
        let backend = match &device {
            Device::Cpu => Backend::Cpu,
            Device::Cuda(_cuda_device) => {
                // Extract ordinal from CUDA device if possible
                Backend::Cuda(0) // Default to 0, Candle doesn't expose ordinal easily
            }
            Device::Metal(_metal_device) => {
                Backend::Metal(0) // Default to 0
            }
        };

        Self { backend, device }
    }

    /// Get the backend type
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Print device information
    pub fn print_info(&self) {
        println!("Device Information:");
        println!("  Backend: {}", self.backend);
        println!("  Type: {}", self.backend.name());
        println!(
            "  GPU: {}",
            if self.backend.is_gpu() { "Yes" } else { "No" }
        );

        // Print backend-specific optimizations
        self.print_optimizations();
    }

    /// Print available optimizations for the current backend
    fn print_optimizations(&self) {
        println!("  Optimizations:");

        match self.backend {
            Backend::Cpu => {
                #[cfg(feature = "mkl")]
                println!("    - Intel MKL: Enabled");

                #[cfg(feature = "accelerate")]
                println!("    - Apple Accelerate: Enabled");

                #[cfg(not(any(feature = "mkl", feature = "accelerate")))]
                println!("    - Basic CPU kernels only");
            }
            Backend::Cuda(_) => {
                #[cfg(feature = "cudnn")]
                println!("    - cuDNN: Enabled");

                #[cfg(feature = "flash-attn")]
                println!("    - Flash Attention: Enabled");

                println!("    - CUDA kernels: Enabled");
            }
            Backend::Metal(_) => {
                println!("    - Metal Performance Shaders: Enabled");
                println!("    - Metal kernels: Enabled");
            }
        }
    }

    /// Get estimated available memory for this device
    ///
    /// Returns None if memory information is not available
    pub fn available_memory_gb(&self) -> Option<f32> {
        match &self.device {
            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_device) => {
                // Note: Candle doesn't expose CUDA memory info directly
                // This is a placeholder for potential future implementation
                None
            }
            _ => None,
        }
    }

    /// Check if there's enough memory for the given tensor size
    pub fn check_memory_requirements(&self, estimated_memory_gb: f32) -> Result<()> {
        if let Some(available) = self.available_memory_gb() {
            let buffer_factor = self.backend.memory_buffer_factor();
            let usable = available * buffer_factor;

            if estimated_memory_gb > usable {
                return Err(candle::Error::Msg(format!(
                    "Insufficient memory: need {:.2}GB, but only {:.2}GB available (with {:.0}% buffer)",
                    estimated_memory_gb,
                    usable,
                    buffer_factor * 100.0
                )));
            }
        }

        Ok(())
    }
}

/// Estimate memory requirements for LTX-Video generation
pub struct MemoryEstimator {
    model_memory_gb: f32,
    activation_memory_gb: f32,
}

impl MemoryEstimator {
    /// Create a new memory estimator
    pub fn new() -> Self {
        Self {
            model_memory_gb: 0.0,
            activation_memory_gb: 0.0,
        }
    }

    /// Estimate model weights memory
    pub fn estimate_model_memory(
        &mut self,
        num_parameters: usize,
        dtype_bytes: usize,
    ) -> &mut Self {
        self.model_memory_gb = (num_parameters * dtype_bytes) as f32 / 1e9;
        self
    }

    /// Estimate activation memory for a video generation
    pub fn estimate_activation_memory(
        &mut self,
        batch_size: usize,
        num_frames: usize,
        height: usize,
        width: usize,
        dtype_bytes: usize,
    ) -> &mut Self {
        // Rough estimation based on latent size and intermediate activations
        let latent_frames = num_frames / 4; // temporal compression
        let latent_height = height / 8; // spatial compression
        let latent_width = width / 8;
        let latent_channels = 128;

        // Main latent tensor
        let latent_size =
            batch_size * latent_channels * latent_frames * latent_height * latent_width;

        // Intermediate activations (rough estimate: 3x latent size for transformer)
        let activation_size = latent_size * 3;

        self.activation_memory_gb = (activation_size * dtype_bytes) as f32 / 1e9;
        self
    }

    /// Get total estimated memory
    pub fn total_memory_gb(&self) -> f32 {
        self.model_memory_gb + self.activation_memory_gb
    }

    /// Get model memory estimation
    pub fn model_memory_gb(&self) -> f32 {
        self.model_memory_gb
    }

    /// Get activation memory estimation
    pub fn activation_memory_gb(&self) -> f32 {
        self.activation_memory_gb
    }

    /// Print memory estimation
    pub fn print_estimation(&self) {
        println!("Memory Estimation:");
        println!("  Model weights: {:.2} GB", self.model_memory_gb);
        println!("  Activations: {:.2} GB", self.activation_memory_gb);
        println!("  Total: {:.2} GB", self.total_memory_gb());
    }
}

impl Default for MemoryEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_cpu_available() {
        let backend = Backend::Cpu;
        assert!(backend.is_available());
    }

    #[test]
    fn test_backend_to_string() {
        assert_eq!(Backend::Cpu.to_string(), "CPU");
        assert_eq!(Backend::Cuda(0).to_string(), "CUDA:0");
        assert_eq!(Backend::Metal(1).to_string(), "Metal:1");
    }

    #[test]
    fn test_backend_is_gpu() {
        assert!(!Backend::Cpu.is_gpu());
        assert!(Backend::Cuda(0).is_gpu());
        assert!(Backend::Metal(0).is_gpu());
    }

    #[test]
    fn test_available_backends() {
        let backends = Backend::available_backends();
        assert!(!backends.is_empty());
        assert!(backends.contains(&Backend::Cpu));
    }

    #[test]
    fn test_auto_detect() {
        let backend = Backend::auto_detect();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_device_manager_auto() {
        let dm = DeviceManager::auto();
        assert!(dm.is_ok());
    }

    #[test]
    fn test_memory_estimator() {
        let mut estimator = MemoryEstimator::new();

        // Estimate for 2B model with BF16 (2 bytes per param)
        estimator.estimate_model_memory(2_000_000_000, 2);
        assert!((estimator.model_memory_gb() - 4.0).abs() < 0.1);

        // Estimate activation memory for 121 frames at 768x512
        estimator.estimate_activation_memory(1, 121, 512, 768, 2);
        assert!(estimator.activation_memory_gb() > 0.0);

        let total = estimator.total_memory_gb();
        assert!(total > 4.0);
    }
}
