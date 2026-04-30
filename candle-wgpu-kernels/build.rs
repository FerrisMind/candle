use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn const_name(path: &Path) -> String {
    path.file_name()
        .unwrap()
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn main() -> std::io::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/shaders");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("src/shaders");
    let mut shaders = Vec::new();
    for entry in fs::read_dir(&shaders_dir)? {
        let path = entry?.path();
        if path.is_file() {
            let name = path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let cst = const_name(&path);
            shaders.push((cst, name));
        }
    }
    shaders.sort_by(|a, b| a.1.cmp(&b.1));

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut source = String::new();
    for (cst, name) in &shaders {
        source.push_str(&format!(
            "pub const {cst}: Module = Module {{ name: {name:?}, source: include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/src/shaders/{name}\")) }};\n"
        ));
    }
    source.push_str("pub const ALL_MODULES: &[Module] = &[\n");
    for (cst, _) in &shaders {
        source.push_str(&format!("    {cst},\n"));
    }
    source.push_str("];\n");
    fs::write(out_dir.join("wgsl.rs"), source)?;
    Ok(())
}
