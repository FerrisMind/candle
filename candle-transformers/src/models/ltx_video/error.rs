use candle::Error as CandleError;
use std::fmt;

#[derive(Debug)]
pub enum LtxVideoError {
    InvalidDimension(String, usize),
    InvalidFrameCount(usize),
    ConditioningFrameOutOfBounds(usize, usize),
    UnsupportedDType(String),
    BackendError(String),
    ConfigError(String),
    IoError(std::io::Error),
    CandleError(CandleError),
}

impl fmt::Display for LtxVideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LtxVideoError::InvalidDimension(dim, val) => {
                write!(f, "Invalid dimension: {} must be divisible by {}", dim, val)
            }
            LtxVideoError::InvalidFrameCount(count) => write!(
                f,
                "Invalid frame count: {}. Must follow pattern 8n+1 (e.g., 9, 17, 25, 121)",
                count
            ),
            LtxVideoError::ConditioningFrameOutOfBounds(idx, max) => write!(
                f,
                "Conditioning frame index {} out of bounds. Max frame index is {}",
                idx, max
            ),
            LtxVideoError::UnsupportedDType(dtype) => {
                write!(
                    f,
                    "Unsupported dtype: {}. Only bf16 and fp8 are supported.",
                    dtype
                )
            }
            LtxVideoError::BackendError(msg) => write!(f, "Backend error: {}", msg),
            LtxVideoError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            LtxVideoError::IoError(err) => write!(f, "IO error: {}", err),
            LtxVideoError::CandleError(err) => write!(f, "Candle error: {}", err),
        }
    }
}

impl std::error::Error for LtxVideoError {}

impl From<std::io::Error> for LtxVideoError {
    fn from(err: std::io::Error) -> Self {
        LtxVideoError::IoError(err)
    }
}

impl From<CandleError> for LtxVideoError {
    fn from(err: CandleError) -> Self {
        LtxVideoError::CandleError(err)
    }
}
