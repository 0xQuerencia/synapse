//! Resolve one immutable Hub revision, copy its complete verified artifacts to
//! a private local directory and print an integrity receipt. Downloaded weights
//! are deliberately outside git. Usage: materialize_model MODEL DIRECTORY [REV].
use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};
use synapse_core::{
    ModelIdentity,
    inference::model_loader::{ModelFiles, download_model_revision},
};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (2..=3).contains(&args.len()),
        "usage: materialize_model MODEL DIRECTORY [REVISION]"
    );
    let target = PathBuf::from(&args[1]);
    if target.exists() {
        let files = ModelFiles::from_directory(&target)?;
        ensure!(
            files.manifest.model_id == args[0],
            "existing directory holds a different model"
        );
        if let Some(revision) = args.get(2) {
            ensure!(
                revision == &files.manifest.revision,
                "existing directory uses another pinned revision"
            );
        }
        println!(
            "{}",
            serde_json::json!({"directory":target,"model_id":files.manifest.model_id,"revision":files.manifest.revision,"manifest_sha256":files.manifest.digest()?,"reused_verified_directory":true})
        );
        return Ok(());
    }
    let files = download_model_revision(&ModelIdentity {
        id: args[0].clone(),
        revision: args.get(2).cloned(),
        sha256: None,
    })
    .await?;
    let parent = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".model-preparation-{}-{:016x}",
        std::process::id(),
        rand::random::<u64>()
    ));
    std::fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        for artifact in &files.manifest.artifacts {
            let source = match artifact.filename.as_str() {
                "config.json" => files.config.clone(),
                "tokenizer.json" => files.tokenizer.clone(),
                _ => files
                    .weights
                    .parent()
                    .context("weights parent missing")?
                    .join(&artifact.filename),
            };
            std::fs::copy(source, staging.join(&artifact.filename))?;
        }
        std::fs::write(
            staging.join("manifest.json"),
            serde_json::to_vec_pretty(&files.manifest)?,
        )?;
        ModelFiles::from_directory(&staging)?;
        ensure!(!target.exists(), "destination appeared during preparation");
        std::fs::rename(&staging, &target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result?;
    println!(
        "{}",
        serde_json::json!({"directory":target,"model_id":files.manifest.model_id,"revision":files.manifest.revision,"manifest_sha256":files.manifest.digest()?,"artifact_count":files.manifest.artifacts.len(),"artifact_bytes":files.manifest.artifacts.iter().map(|a|a.size_bytes).sum::<u64>()})
    );
    Ok(())
}
