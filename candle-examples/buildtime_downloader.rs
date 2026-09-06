use anyhow::Result;
use hf_hub::api::sync::Api;

pub fn download_model(model_and_revision: &str) -> Result<()> {
    let (model_id, revision) = match model_and_revision.split_once(":") {
        Some((model_id, revision)) => (model_id, revision),
        None => (model_and_revision, "main"),
    };
    let api = Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.into(),
        hf_hub::RepoType::Model,
        revision.into(),
    ));
    let config_filename = repo.get("config.json")?;
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let weights_filename = repo.get("model.safetensors")?;
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_CONFIG={}", config_filename.display());
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_TOKENIZER={}", tokenizer_filename.display());
    println!("cargo::rustc-env=CANDLE_BUILDTIME_MODEL_WEIGHTS={}", weights_filename.display());

    Ok(())
}
