//! EDM helpers that mirror the reference `sgm` sampling code so our Rust
//! pipeline can reuse the same noise schedule, scaling, and guidance logic.
use candle::{bail, DType, Device, IndexOp, Result, Tensor, D};
use std::collections::HashMap;
use std::sync::Arc;

/// Conditioning entries collected by the `conditioner` module.
pub type Conditioning = HashMap<String, Tensor>;

/// Sigma range configuration for the EDM discretizer.
#[derive(Debug, Clone)]
pub struct EdmDiscretizationConfig {
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub rho: f64,
    pub num_steps: usize,
}

impl Default for EdmDiscretizationConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 700.0,
            rho: 7.0,
            num_steps: 30,
        }
    }
}

pub struct EdmDiscretization {
    config: EdmDiscretizationConfig,
}

impl EdmDiscretization {
    pub fn new(config: EdmDiscretizationConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &EdmDiscretizationConfig {
        &self.config
    }

    /// Generates a sigma schedule with an appended zero to match the PyTorch
    /// implementation.
    pub fn sigmas(&self, num_steps: usize, device: &Device) -> Result<Tensor> {
        let steps = num_steps.max(1);
        let mut values = self.base_sigmas(steps);
        values.push(0.0);
        Tensor::from_slice(&values, (values.len(),), device)
    }

    fn base_sigmas(&self, num_steps: usize) -> Vec<f32> {
        let rho = self.config.rho;
        let min_inv = self.config.sigma_min.powf(1.0 / rho);
        let max_inv = self.config.sigma_max.powf(1.0 / rho);
        let span = min_inv - max_inv;

        (0..num_steps)
            .map(|i| {
                let ramp = if num_steps == 1 {
                    0.0
                } else {
                    i as f64 / (num_steps - 1) as f64
                };
                ((max_inv + ramp * span).powf(rho)) as f32
            })
            .collect()
    }
}

/// Guides the sampler so that conditional and unconditional batches can be
/// concatenated and resolved later.
pub trait EdmGuider: Send + Sync {
    fn prepare_inputs(
        &self,
        batch: &Tensor,
        sigma: &Tensor,
        cond: &Conditioning,
        uc: &Conditioning,
    ) -> Result<(Tensor, Tensor, Conditioning)>;

    fn guide(&self, prediction: &Tensor, sigma: &Tensor) -> Result<Tensor>;
}

/// The scaling strategy used by the official SVD code.
pub struct VScalingWithEdmCNoise;

impl VScalingWithEdmCNoise {
    pub fn scale(&self, sigma: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let dtype = sigma.dtype();
        let device = sigma.device();
        let ones = Tensor::ones(sigma.shape().clone(), dtype, device)?;
        let denom = (sigma.sqr()? + &ones)?;
        let denom_sqrt = denom.sqrt()?;
        let c_skip = ones.broadcast_div(&denom)?;
        let c_out = (sigma.neg()? / &denom_sqrt)?;
        let c_in = ones.broadcast_div(&denom_sqrt)?;
        let c_noise = (sigma.log()? * 0.25)?;
        Ok((c_skip, c_out, c_in, c_noise))
    }
}

/// Wraps the scaling logic so the sampler only needs to provide the network.
pub struct Denoiser {
    scaling: VScalingWithEdmCNoise,
}

impl Denoiser {
    pub fn new() -> Self {
        Self {
            scaling: VScalingWithEdmCNoise,
        }
    }

    pub fn denoise<F>(
        &self,
        network: &F,
        input: &Tensor,
        sigma: &Tensor,
        cond: &Conditioning,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, &Tensor, &Conditioning) -> Result<Tensor>,
    {
        let sigma_shape = sigma.shape().clone();
        let sigma_rank = input.rank();
        let sigma_broadcast = append_dims(sigma, sigma_rank)?;
        let (c_skip, c_out, c_in, c_noise) = self.scaling.scale(&sigma_broadcast)?;
        let c_noise = c_noise.reshape(sigma_shape)?;
        let scaled_input = input.mul(&c_in)?;
        let model_output = network(&scaled_input, &c_noise, cond)?;
        let scaled_output = model_output.mul(&c_out)?;
        let skip = input.mul(&c_skip)?;
        let combined = scaled_output.add(&skip)?;
        Ok(combined)
    }
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}

const BASE_CONCAT_KEYS: [&str; 3] = ["vector", "crossattn", "concat"];

/// Mirrors `LinearPredictionGuider` from the reference implementation.
#[derive(Clone, Debug)]
pub struct LinearPredictionGuider {
    scale: Vec<f32>,
    num_frames: usize,
    additional_cond_keys: Vec<String>,
}

impl LinearPredictionGuider {
    pub fn new(
        max_scale: f64,
        num_frames: usize,
        min_scale: f64,
        additional_cond_keys: Vec<String>,
    ) -> Self {
        Self {
            scale: linspace(min_scale, max_scale, num_frames.max(1)),
            num_frames: num_frames.max(1),
            additional_cond_keys: additional_cond_keys
                .into_iter()
                .map(|k| k.to_string())
                .collect(),
        }
    }

    fn should_concat(&self, key: &str) -> bool {
        BASE_CONCAT_KEYS.iter().any(|base| base == &key)
            || self.additional_cond_keys.iter().any(|extra| extra == key)
    }

    fn reshape_temporal(&self, tensor: &Tensor) -> Result<(Tensor, usize, Vec<usize>)> {
        let total = tensor.dim(0)?;
        if total % self.num_frames != 0 {
            bail!(
                "prediction batch ({total}) must be divisible by num_frames ({})",
                self.num_frames
            );
        }
        let batch = total / self.num_frames;
        let rest_dims = tensor.shape().dims()[1..].to_vec();
        let mut shape = Vec::with_capacity(rest_dims.len() + 2);
        shape.push(batch);
        shape.push(self.num_frames);
        shape.extend(&rest_dims);
        let reshaped = tensor.reshape(shape.as_slice())?;
        Ok((reshaped, batch, rest_dims))
    }

    fn scale_tensor(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        Tensor::from_slice(&self.scale, (self.scale.len(),), &Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)
    }
}

impl EdmGuider for LinearPredictionGuider {
    fn prepare_inputs(
        &self,
        batch: &Tensor,
        sigma: &Tensor,
        cond: &Conditioning,
        uc: &Conditioning,
    ) -> Result<(Tensor, Tensor, Conditioning)> {
        let doubled_batch = Tensor::cat(&[batch, batch], 0)?;
        let doubled_sigma = Tensor::cat(&[sigma, sigma], 0)?;
        let mut combined = HashMap::with_capacity(cond.len());
        for (key, tensor) in cond {
            let entry = if self.should_concat(key) {
                if let Some(uc_tensor) = uc.get(key) {
                    Tensor::cat(&[uc_tensor, tensor], 0)?
                } else {
                    tensor.clone()
                }
            } else {
                tensor.clone()
            };
            combined.insert(key.clone(), entry);
        }
        Ok((doubled_batch, doubled_sigma, combined))
    }

    fn guide(&self, prediction: &Tensor, _sigma: &Tensor) -> Result<Tensor> {
        let total = prediction.dim(0)?;
        if total % 2 != 0 {
            bail!("prediction batch must be even after duplication");
        }
        let half = total / 2;
        let x_u = prediction.narrow(0, 0, half)?;
        let x_c = prediction.narrow(0, half, half)?;
        let (x_u, batch, rest) = self.reshape_temporal(&x_u)?;
        let (x_c, _, _) = self.reshape_temporal(&x_c)?;
        let mut scale = self.scale_tensor(x_u.device(), x_u.dtype())?;
        scale = scale.reshape((1, self.num_frames))?;
        scale = scale.expand((batch, self.num_frames))?;
        let scale_rank = x_u.rank();
        scale = append_dims(&scale, scale_rank)?;
        let diff = x_c.sub(&x_u)?;
        let scaled_diff = scale.mul(&diff)?;
        let guided = x_u.add(&scaled_diff)?;
        let mut final_shape = vec![half];
        final_shape.extend(rest.iter().copied());
        guided.reshape(final_shape.as_slice())
    }
}

/// Pure Euler EDM sampler.
pub struct EulerEdmSampler {
    discretization: EdmDiscretization,
    guider: Arc<dyn EdmGuider>,
    s_churn: f64,
    s_tmin: f64,
    s_tmax: f64,
    s_noise: f64,
}

impl EulerEdmSampler {
    pub fn new(config: EdmDiscretizationConfig, guider: Arc<dyn EdmGuider>) -> Self {
        Self {
            discretization: EdmDiscretization::new(config),
            guider,
            s_churn: 0.0,
            s_tmin: 0.0,
            s_tmax: f64::INFINITY,
            s_noise: 1.0,
        }
    }

    pub fn with_churn(mut self, s_churn: f64, s_tmin: f64, s_tmax: f64, s_noise: f64) -> Self {
        self.s_churn = s_churn;
        self.s_tmin = s_tmin;
        self.s_tmax = s_tmax;
        self.s_noise = s_noise;
        self
    }

    pub fn sample<F>(
        &self,
        denoiser: &Denoiser,
        network: &F,
        mut latents: Tensor,
        cond: &Conditioning,
        uc: Option<&Conditioning>,
        num_steps: Option<usize>,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, &Tensor, &Conditioning) -> Result<Tensor>,
    {
        let steps = num_steps
            .unwrap_or(self.discretization.config().num_steps)
            .max(1);
        let device = latents.device().clone();
        let dtype = latents.dtype();
        let sigmas = self.discretization.sigmas(steps, &device)?;
        let sigma0 = sigmas.i((0,))?;
        let init_scale = (sigma0.powf(2.0)? + 1.0)?.sqrt()?;
        latents = latents.mul(&init_scale)?;
        let num_sigmas = sigmas.dim(0)?;
        if num_sigmas < 2 {
            bail!("sampler requires at least one inference step");
        }
        let batch = latents.dim(0)?;
        let s_in = Tensor::ones((batch,), dtype, &device)?;
        let cond_uc = uc.unwrap_or(cond);
        let mut current = latents;

        for index in 0..num_sigmas - 1 {
            let sigma_batch = sigmas.i((index,))?;
            let next_sigma_batch = sigmas.i((index + 1,))?;
            let gamma = self.compute_gamma(sigma_batch.clone(), num_sigmas - 1)?;
            let sigma_tensor = s_in.mul(&sigma_batch)?;
            let next_sigma_tensor = s_in.mul(&next_sigma_batch)?;
            current = self.sampler_step(
                current,
                &sigma_tensor,
                &next_sigma_tensor,
                cond,
                cond_uc,
                gamma,
                denoiser,
                network,
            )?;
        }

        Ok(current)
    }

    #[allow(clippy::too_many_arguments)]
    fn sampler_step<F>(
        &self,
        mut x: Tensor,
        sigma: &Tensor,
        next_sigma: &Tensor,
        cond: &Conditioning,
        uc: &Conditioning,
        gamma: f64,
        denoiser: &Denoiser,
        network: &F,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, &Tensor, &Conditioning) -> Result<Tensor>,
    {
        let sigma_hat = (sigma.clone() * (gamma + 1.0))?;
        let x_rank = x.rank();
        if gamma > 0.0 {
            let noise = x.randn_like(0., 1.)?;
            let sigma_hat_sq = sigma_hat.powf(2.0)?;
            let sigma_sq = sigma.powf(2.0)?;
            let delta = (&sigma_hat_sq - &sigma_sq)?;
            let noise_scale = append_dims(&delta, x_rank)?.sqrt()?;
            let noise_scaled = noise.mul(&noise_scale)?;
            x = x.add(&noise_scaled)?;
        }
        let (prepared_x, prepared_sigma, prepared_cond) =
            self.guider.prepare_inputs(&x, &sigma_hat, cond, uc)?;
        let denoised = denoiser.denoise(network, &prepared_x, &prepared_sigma, &prepared_cond)?;
        let guided = self.guider.guide(&denoised, &prepared_sigma)?;
        let d = to_d(&x, &sigma_hat, &guided)?;
        let step_diff = next_sigma.sub(&sigma_hat)?;
        let dt = append_dims(&step_diff, x_rank)?;
        let dt_mul = dt.mul(&d)?;
        let euler_step = x.add(&dt_mul)?;

        self.possible_correction_step(
            euler_step, x, &d, &dt, next_sigma, denoiser, cond, uc, network,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn possible_correction_step<F>(
        &self,
        euler_step: Tensor,
        _x: Tensor,
        _d: &Tensor,
        _dt: &Tensor,
        _next_sigma: &Tensor,
        _denoiser: &Denoiser,
        _cond: &Conditioning,
        _uc: &Conditioning,
        _network: &F,
    ) -> Result<Tensor>
    where
        F: Fn(&Tensor, &Tensor, &Conditioning) -> Result<Tensor>,
    {
        Ok(euler_step)
    }

    fn compute_gamma(&self, sigma: Tensor, steps: usize) -> Result<f64> {
        let sigma = scalar_to_f64(&sigma)?;
        if steps == 0 {
            return Ok(0.0);
        }
        if self.s_tmin <= sigma && sigma <= self.s_tmax {
            let limit = (self.s_churn / steps as f64).min(2f64.sqrt() - 1.0);
            Ok(limit)
        } else {
            Ok(0.0)
        }
    }
}

fn scalar_to_f64(tensor: &Tensor) -> Result<f64> {
    tensor.to_scalar::<f64>()
}

fn to_d(x: &Tensor, sigma: &Tensor, denoised: &Tensor) -> Result<Tensor> {
    let numerator = (x - denoised)?;
    let x_rank = x.rank();
    let denom = append_dims(sigma, x_rank)?;
    numerator.broadcast_div(&denom)
}

fn append_dims(tensor: &Tensor, target_rank: usize) -> Result<Tensor> {
    let mut result = tensor.clone();
    let rank = tensor.rank();
    if rank > target_rank {
        bail!("target rank ({target_rank}) is smaller than tensor rank ({rank})");
    }
    for _ in rank..target_rank {
        result = result.unsqueeze(D::Minus1)?;
    }
    Ok(result)
}

fn linspace(start: f64, end: f64, steps: usize) -> Vec<f32> {
    if steps == 0 {
        return Vec::new();
    }
    if steps == 1 {
        return vec![start as f32];
    }
    (0..steps)
        .map(|i| {
            let t = i as f64 / (steps - 1) as f64;
            (start + t * (end - start)) as f32
        })
        .collect()
}
