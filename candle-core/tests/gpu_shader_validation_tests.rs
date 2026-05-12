#[cfg(feature = "vulkan")]
use std::fs;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "utility test for CI SPIR-V materialization"]
fn vulkan_spirv_modules_can_be_materialized() {
    let out_dir = std::env::var_os("CANDLE_SPIRV_DUMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("candle-vulkan-spirv"));
    fs::create_dir_all(&out_dir).expect("create SPIR-V dump dir");

    for module in candle_vulkan_kernels::spirv_modules() {
        let out_path = out_dir.join(format!("{}.spv", module.name()));
        let mut bytes = Vec::with_capacity(module.words().len() * 4);
        for word in module.words() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        fs::write(out_path, bytes).expect("write SPIR-V module");
    }
}
