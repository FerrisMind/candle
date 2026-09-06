//! `hf-hub` utils

use hf_hub::api::sync::{Api as SyncApi, ApiError};
use std::path::PathBuf;

/// Result alias for hub operations.
pub type HFResult<T> = std::result::Result<T, ApiError>;

/// Blocking hub client
#[derive(Clone, Debug)]
pub struct Api(SyncApi);

impl Api {
    /// Client configured from env (`HF_TOKEN`, `HF_HUB_CACHE`, etc)
    pub fn new() -> HFResult<Self> {
        Ok(Self(SyncApi::new()?))
    }

    pub fn with_cache_dir<P: Into<PathBuf>>(cache_dir: P) -> HFResult<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.into())
            .build()?;
        Ok(Self(api))
    }

    fn repo_of(&self, id: impl AsRef<str>, repo_type: hf_hub::RepoType) -> Repo {
        Repo {
            api: self.0.clone(),
            repo_id: id.as_ref().to_string(),
            repo_type,
            revision: None,
        }
    }

    /// Creates a blocking handle for a model repository
    pub fn model(&self, id: impl AsRef<str>) -> Repo {
        self.repo_of(id, hf_hub::RepoType::Model)
    }

    /// Creates a blocking handle for a dataset repository
    pub fn dataset(&self, id: impl AsRef<str>) -> Repo {
        self.repo_of(id, hf_hub::RepoType::Dataset)
    }
}

/// Repository handle. Optionally pinned to a revision.
#[derive(Clone, Debug)]
pub struct Repo {
    api: SyncApi,
    repo_id: String,
    repo_type: hf_hub::RepoType,
    revision: Option<String>,
}

impl Repo {
    /// Pins every subsequent [`Repo::get`] to `revision`.
    pub fn with_revision<S: AsRef<str>>(mut self, revision: S) -> Self {
        self.revision = Some(revision.as_ref().to_string());
        self
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    fn hf_repo(&self) -> hf_hub::Repo {
        match &self.revision {
            Some(revision) => hf_hub::Repo::with_revision(
                self.repo_id.clone(),
                self.repo_type,
                revision.clone(),
            ),
            None => hf_hub::Repo::new(self.repo_id.clone(), self.repo_type),
        }
    }

    /// Returns the path to `filename`. Downloads if not in cache.
    pub fn get<S: AsRef<str>>(&self, filename: S) -> HFResult<PathBuf> {
        self.api.repo(self.hf_repo()).get(filename.as_ref())
    }
}
