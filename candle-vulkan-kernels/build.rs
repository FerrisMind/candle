use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn const_name(path: &str) -> String {
    path.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn collect(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, root, out)?;
        } else if path.is_file() {
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if rel.ends_with(".comp") || rel.ends_with(".glsl") {
                out.push((const_name(&rel), rel));
            }
        }
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/shaders");
    println!("cargo::rerun-if-changed=src/candle-shaders");
    println!("cargo::rerun-if-env-changed=GLSLC");
    println!("cargo::rerun-if-env-changed=CXX");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("src/shaders");
    let candle_shaders_dir = manifest_dir.join("src/candle-shaders");
    let mut shaders = Vec::new();
    collect(&shaders_dir, &shaders_dir, &mut shaders)?;
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
    fs::write(out_dir.join("comp.rs"), source)?;
    generate_spirv_with_ggml_generator(&shaders_dir, &candle_shaders_dir, &out_dir)?;
    Ok(())
}

fn generate_spirv_with_ggml_generator(
    shaders_dir: &Path,
    candle_shaders_dir: &Path,
    out_dir: &Path,
) -> std::io::Result<()> {
    let glslc = env::var("GLSLC").unwrap_or_else(|_| "glslc".to_string());
    ensure_glslc_available(&glslc);
    let integer_dot_support = glslc_supports_feature(
        &glslc,
        &shaders_dir.join("feature-tests").join("integer_dot.comp"),
        &out_dir.join("feature-test-integer-dot.spv"),
    );
    let generator = out_dir.join(if cfg!(windows) {
        "vulkan-shaders-gen.exe"
    } else {
        "vulkan-shaders-gen"
    });
    compile_generator(
        &shaders_dir.join("vulkan-shaders-gen.cpp"),
        &generator,
        integer_dot_support,
    )?;

    let spv_dir = out_dir.join("spv");
    fs::create_dir_all(&spv_dir)?;
    let mut shader_sources = Vec::new();
    collect_comp_sources(shaders_dir, &mut shader_sources)?;
    shader_sources.sort();
    for source in shader_sources {
        let stem = source.file_stem().unwrap().to_string_lossy();
        let output = Command::new(&generator)
            .arg("--glslc")
            .arg(&glslc)
            .arg("--source")
            .arg(&source)
            .arg("--output-dir")
            .arg(&spv_dir)
            .arg("--target-hpp")
            .arg(out_dir.join(format!("{stem}.hpp")))
            .arg("--target-cpp")
            .arg(out_dir.join(format!("{stem}.cpp")))
            .output()?;
        if !output.status.success() {
            panic!(
                "failed to run vulkan-shaders-gen for {} with {glslc}: {}{}",
                source.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    generate_candle_spirv_modules(&glslc, shaders_dir, candle_shaders_dir, &spv_dir)?;

    let mut modules = Vec::new();
    collect_spv(&spv_dir, &mut modules)?;
    if modules.is_empty() {
        panic!(
            "vulkan shader generation produced no SPIR-V modules in {} using {glslc}; \
             verify that glslc is installed and shader compilation succeeded",
            spv_dir.display()
        );
    }
    modules.sort_by(|a, b| a.0.cmp(&b.0));
    let mut generated = String::new();
    for (name, out_path) in &modules {
        let const_name = const_name(name);
        let bytes = fs::read(out_path)?;
        let words = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        generated.push_str(&format!(
            "pub const {const_name}: SpirvModule = SpirvModule {{ name: {name:?}, words: &{:?} }};\n",
            words
        ));
    }
    generated.push_str("pub const ALL_SPIRV_MODULES: &[SpirvModule] = &[\n");
    for (name, _) in &modules {
        generated.push_str(&format!("    {},\n", const_name(name)));
    }
    generated.push_str("];\n");
    fs::write(out_dir.join("spv.rs"), generated)?;
    Ok(())
}

fn generate_candle_spirv_modules(
    glslc: &str,
    shaders_dir: &Path,
    candle_shaders_dir: &Path,
    spv_dir: &Path,
) -> std::io::Result<()> {
    let modules: &[(&str, PathBuf, &[&str])] = &[
        (
            "powf_f32",
            candle_shaders_dir.join("powf.comp"),
            &["A_TYPE=float", "D_TYPE=float", "FLOAT_TYPE=float"],
        ),
        (
            "cmp_f32",
            candle_shaders_dir.join("cmp.comp"),
            &[
                "A_TYPE=float",
                "B_TYPE=float",
                "D_TYPE=uint8_t",
                "FLOAT_TYPE=float",
            ],
        ),
        (
            "cmp_f16",
            candle_shaders_dir.join("cmp.comp"),
            &[
                "A_TYPE=float16_t",
                "B_TYPE=float16_t",
                "D_TYPE=uint8_t",
                "FLOAT_TYPE=float16_t",
            ],
        ),
        (
            "cmp_u8",
            candle_shaders_dir.join("cmp.comp"),
            &[
                "A_TYPE=uint8_t",
                "B_TYPE=uint8_t",
                "D_TYPE=uint8_t",
                "FLOAT_TYPE=float",
            ],
        ),
        (
            "cmp_u32",
            candle_shaders_dir.join("cmp.comp"),
            &[
                "A_TYPE=uint",
                "B_TYPE=uint",
                "D_TYPE=uint8_t",
                "FLOAT_TYPE=float",
            ],
        ),
        (
            "cmp_i64",
            candle_shaders_dir.join("cmp.comp"),
            &[
                "A_TYPE=int64_t",
                "B_TYPE=int64_t",
                "D_TYPE=uint8_t",
                "FLOAT_TYPE=float",
            ],
        ),
        (
            "where_u8_f32",
            candle_shaders_dir.join("where_u8.comp"),
            &["T_TYPE=float", "D_TYPE=float"],
        ),
        (
            "where_u8_f16",
            candle_shaders_dir.join("where_u8.comp"),
            &["T_TYPE=float16_t", "D_TYPE=float16_t"],
        ),
        (
            "where_u8_u8",
            candle_shaders_dir.join("where_u8.comp"),
            &["T_TYPE=uint8_t", "D_TYPE=uint8_t"],
        ),
        (
            "where_u8_u32",
            candle_shaders_dir.join("where_u8.comp"),
            &["T_TYPE=uint", "D_TYPE=uint"],
        ),
        (
            "where_u8_i64",
            candle_shaders_dir.join("where_u8.comp"),
            &["T_TYPE=int64_t", "D_TYPE=int64_t"],
        ),
        (
            "set_rows_add_f32_i32",
            candle_shaders_dir.join("set_rows_add.comp"),
            &[
                "A_TYPE=float",
                "B_TYPE=uint",
                "D_TYPE=uint",
                "FLOAT_TYPE=float",
                "D_READ_WRITE",
            ],
        ),
        (
            "convert_f32_u8",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float", "D_TYPE=uint8_t", "DST_SAT_U8"],
        ),
        (
            "convert_f32_u32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float", "D_TYPE=uint", "DST_SAT_U32"],
        ),
        (
            "convert_f32_i64",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float", "D_TYPE=int64_t", "DST_SAT_I64"],
        ),
        (
            "convert_f16_u8",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float16_t", "D_TYPE=uint8_t", "DST_SAT_U8"],
        ),
        (
            "convert_f16_u32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float16_t", "D_TYPE=uint", "DST_SAT_U32"],
        ),
        (
            "convert_f16_i64",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float16_t", "D_TYPE=int64_t", "DST_SAT_I64"],
        ),
        (
            "convert_u8_f32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint8_t", "D_TYPE=float"],
        ),
        (
            "convert_u8_f16",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint8_t", "D_TYPE=float16_t", "DST_VIA_F32"],
        ),
        (
            "convert_u8_u32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint8_t", "D_TYPE=uint"],
        ),
        (
            "convert_u8_i64",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint8_t", "D_TYPE=int64_t"],
        ),
        (
            "convert_u32_f16",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint", "D_TYPE=float16_t", "DST_VIA_F32"],
        ),
        (
            "convert_u32_u8",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint", "D_TYPE=uint8_t"],
        ),
        (
            "convert_u32_i64",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint", "D_TYPE=int64_t"],
        ),
        (
            "convert_i64_f32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int64_t", "D_TYPE=float"],
        ),
        (
            "convert_i64_f16",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int64_t", "D_TYPE=float16_t", "DST_VIA_F32"],
        ),
        (
            "convert_i64_u8",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int64_t", "D_TYPE=uint8_t"],
        ),
        (
            "convert_i64_u32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int64_t", "D_TYPE=uint"],
        ),
        (
            "repack_q8_1_to_q8_0",
            candle_shaders_dir.join("repack_q8_1_to_q8_0.comp"),
            &[],
        ),
        (
            "dequant_q8_k_f32",
            candle_shaders_dir.join("dequant_q8_k_f32.comp"),
            &[],
        ),
    ];
    for (name, source, defines) in modules {
        let output = spv_dir.join(format!("{name}.spv"));
        compile_spirv(glslc, source, &output, shaders_dir, defines)?;
    }
    Ok(())
}

fn compile_spirv(
    glslc: &str,
    source: &Path,
    output: &Path,
    include_dir: &Path,
    defines: &[&str],
) -> std::io::Result<()> {
    let mut command = Command::new(glslc);
    command
        .arg("-fshader-stage=compute")
        .arg("--target-env=vulkan1.2")
        .arg("-O")
        .arg(format!("-I{}", include_dir.display()));
    for define in defines {
        command.arg(format!("-D{define}"));
    }
    let output_result = command.arg(source).arg("-o").arg(output).output()?;
    if !output_result.status.success() {
        panic!(
            "failed to compile Candle Vulkan shader {} with {glslc}: {}{}",
            source.display(),
            String::from_utf8_lossy(&output_result.stdout),
            String::from_utf8_lossy(&output_result.stderr)
        );
    }
    Ok(())
}

fn ensure_glslc_available(glslc: &str) {
    match Command::new(glslc).arg("--version").output() {
        Ok(output) if output.status.success() => {}
        Ok(output) => panic!(
            "glslc command {glslc:?} is not usable (exit status {:?}): {}{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(err) => panic!("failed to execute glslc command {glslc:?}: {err}"),
    }
}

fn glslc_supports_feature(glslc: &str, source: &Path, output: &Path) -> bool {
    matches!(
        Command::new(glslc)
            .arg(source)
            .arg("-o")
            .arg(output)
            .output(),
        Ok(result) if result.status.success()
    )
}

fn compile_generator(
    source: &Path,
    output: &Path,
    integer_dot_support: bool,
) -> std::io::Result<()> {
    let mut compilers = Vec::new();
    if let Ok(cxx) = env::var("CXX") {
        compilers.push(cxx);
    }
    compilers.extend(
        if cfg!(windows) {
            ["g++", "c++", "clang++"]
        } else {
            ["c++", "g++", "clang++"]
        }
        .into_iter()
        .map(String::from),
    );

    let mut errors = Vec::new();
    for compiler in compilers {
        let mut cmd = Command::new(&compiler);
        cmd.arg("-std=c++17").arg(source).arg("-o").arg(output);
        if integer_dot_support {
            cmd.arg("-DGGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT");
        }
        match cmd.output() {
            Ok(output_result) if output_result.status.success() => return Ok(()),
            Ok(output_result) => errors.push(format!(
                "{compiler}: {}{}",
                String::from_utf8_lossy(&output_result.stdout),
                String::from_utf8_lossy(&output_result.stderr)
            )),
            Err(err) => errors.push(format!("{compiler}: {err}")),
        }
    }
    panic!(
        "failed to compile {}:\n{}",
        source.display(),
        errors.join("\n")
    );
}

fn collect_spv(dir: &Path, out: &mut Vec<(String, PathBuf)>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_spv(&path, out)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("spv") {
            let name = path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push((name, path));
        }
    }
    Ok(())
}

fn collect_comp_sources(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_comp_sources(&path, out)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("comp") {
            out.push(path);
        }
    }
    Ok(())
}
