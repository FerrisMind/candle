use candle::safetensors::MmapedSafetensors;
use candle::Result;
use std::{collections::HashMap, fmt, path::Path};

const SAMPLE_KEYS_LIMIT: usize = 8;

/// Heuristic categories for `svd_xt_1_1.safetensors` keys.
///
/// The prefixes mirror the `DiffusionEngine` layout defined in
/// `tp/generative-models/sgm/models/diffusion.py` and the sampling config at
/// `tp/generative-models/scripts/sampling/configs/svd_xt_1_1.yaml`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SvdCheckpointComponent {
    VideoUnet,
    FirstStage,
    Conditioner,
    Sampler,
    Denoiser,
    Other,
}

impl SvdCheckpointComponent {
    pub fn from_key(key: &str) -> Self {
        match key.split('.').next().unwrap_or("") {
            "model" => Self::VideoUnet,
            "first_stage_model" => Self::FirstStage,
            "conditioner" => Self::Conditioner,
            "sampler" => Self::Sampler,
            "denoiser" => Self::Denoiser,
            _ => Self::Other,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::VideoUnet => "UNet",
            Self::FirstStage => "First-stage autoencoder",
            Self::Conditioner => "Conditioner",
            Self::Sampler => "Sampler helpers",
            Self::Denoiser => "Denoiser scaling",
            Self::Other => "Other",
        }
    }
}

impl fmt::Display for SvdCheckpointComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Summary for a single component inside a checkpoint.
#[derive(Clone, Debug, Default)]
pub struct GroupSummary {
    pub count: usize,
    pub sample_keys: Vec<String>,
}

impl GroupSummary {
    fn add_key(&mut self, key: &str) {
        self.count += 1;
        if self.sample_keys.len() < SAMPLE_KEYS_LIMIT {
            self.sample_keys.push(key.to_string());
        }
    }
}

/// Analysis of how the `svd_xt_1_1.safetensors` keys are distributed.
#[derive(Clone, Debug, Default)]
pub struct SvdCheckpointAnalysis {
    pub total_keys: usize,
    pub breakdown: HashMap<SvdCheckpointComponent, GroupSummary>,
}

impl SvdCheckpointAnalysis {
    /// Builds an analysis directly from the safetensors file.
    pub fn analyze_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let safetensors = unsafe { MmapedSafetensors::new(path.as_ref())? };
        Ok(Self::from_names(
            safetensors.tensors().into_iter().map(|(name, _)| name),
        ))
    }

    /// Builds an analysis from an iterator of tensor names.
    pub fn from_names<I, N>(names: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: AsRef<str>,
    {
        let mut analysis = Self::default();
        for name in names {
            analysis.add_key(name.as_ref());
        }
        analysis
    }

    fn add_key(&mut self, key: &str) {
        self.total_keys += 1;
        self.breakdown
            .entry(SvdCheckpointComponent::from_key(key))
            .or_default()
            .add_key(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorizes_known_prefixes() {
        let keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "first_stage_model.encoder.conv.weight",
            "conditioner.embedders.0.open_clip.weight",
            "sampler.discretization.sigmas",
            "denoiser.scaling.gamma",
            "something_else.weight",
        ];
        let analysis = SvdCheckpointAnalysis::from_names(keys.iter().copied());
        assert_eq!(analysis.total_keys, keys.len());
        assert_eq!(
            analysis.breakdown[&SvdCheckpointComponent::VideoUnet].count,
            1
        );
        assert_eq!(
            analysis.breakdown[&SvdCheckpointComponent::FirstStage].count,
            1
        );
        assert_eq!(
            analysis.breakdown[&SvdCheckpointComponent::Conditioner].count,
            1
        );
        assert_eq!(analysis.breakdown[&SvdCheckpointComponent::Other].count, 1);
    }

    #[test]
    fn sample_keys_limit_is_enforced() {
        let names =
            (0..(SAMPLE_KEYS_LIMIT + 3)).map(|i| format!("model.diffusion_model.{i}.weight"));
        let analysis = SvdCheckpointAnalysis::from_names(names);
        let summary = &analysis.breakdown[&SvdCheckpointComponent::VideoUnet];
        assert_eq!(summary.count, SAMPLE_KEYS_LIMIT + 3);
        assert_eq!(summary.sample_keys.len(), SAMPLE_KEYS_LIMIT);
    }
}
