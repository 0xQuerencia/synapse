pub fn fixture_files() -> synapse_core::inference::model_loader::ModelFiles {
    synapse_core::inference::model_loader::ModelFiles::from_directory(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama"),
    )
    .expect("committed offline fixture must verify")
}
