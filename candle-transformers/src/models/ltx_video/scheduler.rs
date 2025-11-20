//! Rectified Flow Scheduler for LTX-Video
//!
//! This module implements the rectified flow sampling process with
//! support for various timestep schedules and classifier-free guidance.

use candle::{Result, Tensor};
use serde::{Deserialize, Serialize};

/// Timestep schedule mode for the rectified flow scheduler
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum ShiftMode {
    LinearQuadratic {
        threshold: f64,
        linear_steps: Option<usize>,
    },
    SD3 {
        target_terminal: Option<f64>,
    },
    #[default]
    Uniform,
}

/// Configuration for the Rectified Flow Scheduler
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RectifiedFlowConfig {
    pub num_train_timesteps: usize, // 1000
    pub shift_mode: ShiftMode,
    pub shift_terminal: f64, // 0.1
}

impl Default for RectifiedFlowConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift_mode: ShiftMode::Uniform,
            shift_terminal: 0.1,
        }
    }
}

impl RectifiedFlowConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.num_train_timesteps == 0 {
            return Err(candle::Error::Msg(
                "num_train_timesteps must be greater than 0".to_string(),
            ));
        }
        if self.shift_terminal < 0.0 || self.shift_terminal > 1.0 {
            return Err(candle::Error::Msg(
                "shift_terminal must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Rectified Flow Scheduler for diffusion sampling
#[derive(Clone)]
pub struct RectifiedFlowScheduler {
    #[allow(dead_code)]
    config: RectifiedFlowConfig,
    timesteps: Vec<f64>,
    num_inference_steps: usize,
}

impl RectifiedFlowScheduler {
    /// Create a new RectifiedFlowScheduler
    pub fn new(config: RectifiedFlowConfig) -> Self {
        Self {
            config,
            timesteps: Vec::new(),
            num_inference_steps: 50,
        }
    }

    /// Set the number of inference steps and generate timesteps
    pub fn set_timesteps(&mut self, num_steps: usize, timesteps: Option<Vec<f64>>) -> Result<()> {
        self.num_inference_steps = num_steps;
        if let Some(ts) = timesteps {
            self.timesteps = ts;
        } else {
            self.timesteps = self.generate_timesteps(num_steps)?;
        }
        Ok(())
    }

    /// Generate timesteps based on the configured schedule
    fn generate_timesteps(&self, num_steps: usize) -> Result<Vec<f64>> {
        if num_steps == 0 {
            return Err(candle::Error::Msg(
                "num_steps must be greater than 0".to_string(),
            ));
        }

        match &self.config.shift_mode {
            ShiftMode::Uniform => {
                // Uniform schedule from 1.0 to shift_terminal
                let mut timesteps = Vec::with_capacity(num_steps);
                for i in 0..num_steps {
                    let t =
                        1.0 - (i as f64 / num_steps as f64) * (1.0 - self.config.shift_terminal);
                    timesteps.push(t);
                }
                Ok(timesteps)
            }
            ShiftMode::LinearQuadratic {
                threshold,
                linear_steps,
            } => {
                // Linear-quadratic schedule
                // Linear portion from 1.0 to threshold, then quadratic to shift_terminal
                let linear_step_count = linear_steps.unwrap_or(num_steps / 2);
                let mut timesteps = Vec::with_capacity(num_steps);

                // Linear portion
                if linear_step_count > 0 {
                    for i in 0..linear_step_count.min(num_steps) {
                        let t = 1.0 - (i as f64 / linear_step_count as f64) * (1.0 - threshold);
                        timesteps.push(t);
                    }
                }

                // Quadratic portion
                let remaining_steps = num_steps.saturating_sub(linear_step_count);
                if remaining_steps > 0 {
                    for i in 0..remaining_steps {
                        let progress = i as f64 / remaining_steps as f64;
                        let t = threshold
                            - (threshold - self.config.shift_terminal) * progress * progress;
                        timesteps.push(t);
                    }
                }

                Ok(timesteps)
            }
            ShiftMode::SD3 { target_terminal } => {
                // SD3 timestep shifting (resolution-dependent)
                // This is a simplified version; full implementation would consider resolution
                let terminal = target_terminal.unwrap_or(self.config.shift_terminal);
                let mut timesteps = Vec::with_capacity(num_steps);

                for i in 0..num_steps {
                    let t = 1.0 - (i as f64 / num_steps as f64) * (1.0 - terminal);
                    timesteps.push(t);
                }

                Ok(timesteps)
            }
        }
    }

    /// Perform one Euler step of the rectified flow integration
    ///
    /// # Arguments
    /// * `model_output` - Velocity prediction from the model (v_θ)
    /// * `timestep` - Current timestep value
    /// * `sample` - Current latent sample (x_t)
    ///
    /// # Returns
    /// Updated sample for the next timestep
    pub fn step(&self, model_output: &Tensor, timestep: f64, sample: &Tensor) -> Result<Tensor> {
        // Find the index of the current timestep
        let current_idx = self
            .timesteps
            .iter()
            .position(|t| (t - timestep).abs() < 1e-6)
            .ok_or_else(|| {
                candle::Error::Msg(format!("Timestep {} not found in schedule", timestep))
            })?;

        // Calculate dt (timestep delta)
        let dt = if current_idx + 1 < self.timesteps.len() {
            self.timesteps[current_idx] - self.timesteps[current_idx + 1]
        } else {
            // Last step goes to 0
            timestep - self.config.shift_terminal
        };

        // Euler method: x_{t-dt} = x_t - v_θ * dt
        // In rectified flow, the velocity v_θ points from noise to data
        // So we subtract v_θ * dt to move towards data
        let update = model_output.affine(-dt, 1.0)?; // -dt * model_output + sample
        let next_sample = (sample + update)?;

        Ok(next_sample)
    }

    /// Scale noise for forward diffusion process
    ///
    /// Computes: x_t = (1 - t) * x_0 + t * noise
    ///
    /// # Arguments
    /// * `sample` - Clean sample (x_0)
    /// * `timestep` - Timestep value (t ∈ [0, 1])
    /// * `noise` - Random noise
    ///
    /// # Returns
    /// Noisy sample at timestep t
    pub fn scale_noise(&self, sample: &Tensor, timestep: f64, noise: &Tensor) -> Result<Tensor> {
        // Rectified flow interpolation: x_t = (1 - t) * x_0 + t * noise
        let sample_coef = 1.0 - timestep;
        let noise_coef = timestep;

        let scaled_sample = (sample * sample_coef)?;
        let scaled_noise = (noise * noise_coef)?;
        let noisy_sample = (scaled_sample + scaled_noise)?;

        Ok(noisy_sample)
    }

    /// Apply classifier-free guidance to model predictions
    ///
    /// Computes: v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
    ///
    /// # Arguments
    /// * `conditional_pred` - Model prediction with text conditioning
    /// * `unconditional_pred` - Model prediction without text conditioning (negative prompt)
    /// * `guidance_scale` - Guidance strength (1.0 = no guidance, >1.0 = stronger guidance)
    ///
    /// # Returns
    /// Guided prediction
    pub fn apply_guidance(
        &self,
        conditional_pred: &Tensor,
        unconditional_pred: &Tensor,
        guidance_scale: f64,
    ) -> Result<Tensor> {
        if (guidance_scale - 1.0).abs() < 1e-6 {
            // No guidance, return conditional prediction
            return Ok(conditional_pred.clone());
        }

        // v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        let diff = conditional_pred.sub(unconditional_pred)?;
        let scaled_diff = (diff * guidance_scale)?;
        let guided = (unconditional_pred + scaled_diff)?;

        Ok(guided)
    }

    /// Apply timestep shifting for resolution-dependent scheduling
    ///
    /// This implements the SD3 timestep shifting method, which adjusts
    /// the noise schedule based on the target resolution.
    ///
    /// # Arguments
    /// * `resolution` - Target resolution as (frames, height, width)
    ///
    /// # Returns
    /// Shifted timesteps
    #[allow(dead_code)]
    fn apply_timestep_shift(&self, resolution: (usize, usize, usize)) -> Result<Vec<f64>> {
        // Calculate resolution factor (higher resolution = more shifting)
        let (frames, height, width) = resolution;
        let base_resolution = 256.0 * 256.0 * 9.0; // Base: 256x256, 9 frames
        let current_resolution = (width * height * frames) as f64;
        let resolution_factor = (current_resolution / base_resolution).sqrt();

        // Shift timesteps based on resolution
        let shifted_timesteps: Vec<f64> = self
            .timesteps
            .iter()
            .map(|t| {
                // Apply non-linear shifting
                let shift_amount = self.config.shift_terminal * (resolution_factor - 1.0);
                (t - shift_amount).max(self.config.shift_terminal)
            })
            .collect();

        Ok(shifted_timesteps)
    }

    /// Get the current timesteps
    pub fn timesteps(&self) -> &[f64] {
        &self.timesteps
    }

    /// Get the number of inference steps
    pub fn num_inference_steps(&self) -> usize {
        self.num_inference_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_scheduler_creation() {
        let config = RectifiedFlowConfig::default();
        let scheduler = RectifiedFlowScheduler::new(config).unwrap();
        assert_eq!(scheduler.num_inference_steps, 50);
    }

    #[test]
    fn test_config_validation() {
        let config = RectifiedFlowConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_timesteps() {
        let mut config = RectifiedFlowConfig::default();
        config.num_train_timesteps = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_shift_terminal() {
        let mut config = RectifiedFlowConfig::default();
        config.shift_terminal = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_set_timesteps() {
        let config = RectifiedFlowConfig::default();
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();
        scheduler.set_timesteps(25, None).unwrap();
        assert_eq!(scheduler.num_inference_steps(), 25);
    }

    #[test]
    fn test_uniform_timesteps() {
        let config = RectifiedFlowConfig {
            num_train_timesteps: 1000,
            shift_mode: ShiftMode::Uniform,
            shift_terminal: 0.1,
        };
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();
        scheduler.set_timesteps(10, None).unwrap();

        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), 10);
        // First timestep should be close to 1.0
        assert!((timesteps[0] - 1.0).abs() < 1e-6);
        // Last timestep should be close to shift_terminal
        assert!((timesteps[9] - 0.1).abs() < 0.1);
        // Timesteps should be monotonically decreasing
        for i in 0..timesteps.len() - 1 {
            assert!(timesteps[i] > timesteps[i + 1]);
        }
    }

    #[test]
    fn test_linear_quadratic_timesteps() {
        let config = RectifiedFlowConfig {
            num_train_timesteps: 1000,
            shift_mode: ShiftMode::LinearQuadratic {
                threshold: 0.5,
                linear_steps: Some(5),
            },
            shift_terminal: 0.1,
        };
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();
        scheduler.set_timesteps(10, None).unwrap();

        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), 10);
        assert!((timesteps[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sd3_timesteps() {
        let config = RectifiedFlowConfig {
            num_train_timesteps: 1000,
            shift_mode: ShiftMode::SD3 {
                target_terminal: Some(0.05),
            },
            shift_terminal: 0.1,
        };
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();
        scheduler.set_timesteps(8, None).unwrap();

        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), 8);
        assert!((timesteps[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_custom_timesteps() {
        let config = RectifiedFlowConfig::default();
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();

        let custom_timesteps = vec![1.0, 0.75, 0.5, 0.25, 0.1];
        scheduler
            .set_timesteps(5, Some(custom_timesteps.clone()))
            .unwrap();

        let timesteps = scheduler.timesteps();
        assert_eq!(timesteps.len(), 5);
        for (i, t) in timesteps.iter().enumerate() {
            assert!((t - custom_timesteps[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scale_noise() {
        let config = RectifiedFlowConfig::default();
        let scheduler = RectifiedFlowScheduler::new(config).unwrap();
        let device = Device::Cpu;

        // Create test tensors
        let sample = Tensor::ones(&[2, 4, 8, 8], candle::DType::F32, &device).unwrap();
        let noise = Tensor::zeros(&[2, 4, 8, 8], candle::DType::F32, &device).unwrap();

        // Test at t=0 (should return sample)
        let result = scheduler.scale_noise(&sample, 0.0, &noise).unwrap();
        let result_vec = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(result_vec.iter().all(|x| (x - 1.0).abs() < 1e-6));

        // Test at t=1 (should return noise)
        let result = scheduler.scale_noise(&sample, 1.0, &noise).unwrap();
        let result_vec = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(result_vec.iter().all(|x| x.abs() < 1e-6));

        // Test at t=0.5 (should return average)
        let result = scheduler.scale_noise(&sample, 0.5, &noise).unwrap();
        let result_vec = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(result_vec.iter().all(|x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_apply_guidance() {
        let config = RectifiedFlowConfig::default();
        let scheduler = RectifiedFlowScheduler::new(config).unwrap();
        let device = Device::Cpu;

        let cond = Tensor::ones(&[2, 4, 8, 8], candle::DType::F32, &device).unwrap();
        let uncond = Tensor::zeros(&[2, 4, 8, 8], candle::DType::F32, &device).unwrap();

        // Test with guidance_scale = 1.0 (no guidance)
        let result = scheduler.apply_guidance(&cond, &uncond, 1.0).unwrap();
        let result_vec = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(result_vec.iter().all(|x| (x - 1.0).abs() < 1e-6));

        // Test with guidance_scale = 2.0
        // result = uncond + 2.0 * (cond - uncond) = 0 + 2.0 * (1 - 0) = 2.0
        let result = scheduler.apply_guidance(&cond, &uncond, 2.0).unwrap();
        let result_vec = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(result_vec.iter().all(|x| (x - 2.0).abs() < 1e-6));

        // Test with guidance_scale = 0.5
        // result = uncond + 0.5 * (cond - uncond) = 0 + 0.5 * 1 = 0.5
        let result = scheduler.apply_guidance(&cond, &uncond, 0.5).unwrap();
        let result_vec = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(result_vec.iter().all(|x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_euler_step() {
        let config = RectifiedFlowConfig::default();
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();
        scheduler.set_timesteps(10, None).unwrap();

        let device = Device::Cpu;
        let sample = Tensor::ones(&[2, 4, 8, 8], candle::DType::F32, &device).unwrap();
        let model_output = Tensor::ones(&[2, 4, 8, 8], candle::DType::F32, &device)
            .unwrap()
            .affine(0.5, 0.0)
            .unwrap(); // 0.5

        let timestep = scheduler.timesteps()[0];
        let result = scheduler.step(&model_output, timestep, &sample).unwrap();

        // Result shape should match input
        assert_eq!(result.dims(), sample.dims());
    }

    #[test]
    fn test_timestep_shift() {
        let config = RectifiedFlowConfig {
            num_train_timesteps: 1000,
            shift_mode: ShiftMode::SD3 {
                target_terminal: Some(0.05),
            },
            shift_terminal: 0.1,
        };
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();
        scheduler.set_timesteps(8, None).unwrap();

        // Test resolution-dependent shifting
        let low_res = (9, 256, 256);
        let high_res = (25, 512, 512);

        let shifted_low = scheduler.apply_timestep_shift(low_res).unwrap();
        let shifted_high = scheduler.apply_timestep_shift(high_res).unwrap();

        assert_eq!(shifted_low.len(), scheduler.timesteps().len());
        assert_eq!(shifted_high.len(), scheduler.timesteps().len());

        // Higher resolution should have more aggressive shifting
        // (timesteps should be closer to terminal value)
        for i in 0..shifted_low.len() {
            assert!(shifted_high[i] <= shifted_low[i]);
        }
    }

    #[test]
    fn test_zero_steps_error() {
        let config = RectifiedFlowConfig::default();
        let mut scheduler = RectifiedFlowScheduler::new(config).unwrap();
        let result = scheduler.set_timesteps(0, None);
        assert!(result.is_err());
    }
}
