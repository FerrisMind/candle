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
    let coopmat_support = glslc_supports_feature(
        &glslc,
        &shaders_dir.join("feature-tests").join("coopmat.comp"),
        &out_dir.join("feature-test-coopmat.spv"),
    );
    let coopmat2_support = glslc_supports_feature(
        &glslc,
        &shaders_dir.join("feature-tests").join("coopmat2.comp"),
        &out_dir.join("feature-test-coopmat2.spv"),
    );
    println!("cargo:warning=glslc features: integer_dot={integer_dot_support} coopmat={coopmat_support} coopmat2={coopmat2_support}");
    let generator = out_dir.join(if cfg!(windows) {
        "vulkan-shaders-gen.exe"
    } else {
        "vulkan-shaders-gen"
    });
    compile_generator(
        &shaders_dir.join("vulkan-shaders-gen.cpp"),
        &generator,
        GeneratorFeatures {
            integer_dot: integer_dot_support,
            coopmat: coopmat_support,
            coopmat2: coopmat2_support,
        },
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
            "erf_f32",
            candle_shaders_dir.join("erf.comp"),
            &["A_TYPE=float", "D_TYPE=float", "FLOAT_TYPE=float"],
        ),
        (
            "recip_f32",
            candle_shaders_dir.join("recip.comp"),
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
            "sum_rows_int_u32",
            candle_shaders_dir.join("sum_rows_int.comp"),
            &["A_TYPE=uint", "D_TYPE=uint", "ACC_TYPE=uint"],
        ),
        (
            "sum_rows_int_u8",
            candle_shaders_dir.join("sum_rows_int.comp"),
            &["A_TYPE=uint8_t", "D_TYPE=uint8_t", "ACC_TYPE=uint"],
        ),
        (
            "sum_rows_int_i64",
            candle_shaders_dir.join("sum_rows_int.comp"),
            &["A_TYPE=int64_t", "D_TYPE=int64_t", "ACC_TYPE=int64_t"],
        ),
        (
            "reduce_extrema_int_u32",
            candle_shaders_dir.join("reduce_extrema_int.comp"),
            &["A_TYPE=uint", "D_TYPE=uint"],
        ),
        (
            "reduce_extrema_int_u8",
            candle_shaders_dir.join("reduce_extrema_int.comp"),
            &["A_TYPE=uint8_t", "D_TYPE=uint8_t"],
        ),
        (
            "reduce_extrema_int_i64",
            candle_shaders_dir.join("reduce_extrema_int.comp"),
            &["A_TYPE=int64_t", "D_TYPE=int64_t"],
        ),
        (
            "argextrema_int_u32",
            candle_shaders_dir.join("argextrema_int.comp"),
            &["A_TYPE=uint"],
        ),
        (
            "argextrema_int_u8",
            candle_shaders_dir.join("argextrema_int.comp"),
            &["A_TYPE=uint8_t"],
        ),
        (
            "argextrema_int_i64",
            candle_shaders_dir.join("argextrema_int.comp"),
            &["A_TYPE=int64_t"],
        ),
        (
            "binary_int_u8",
            candle_shaders_dir.join("binary_int.comp"),
            &[
                "A_TYPE=uint8_t",
                "B_TYPE=uint8_t",
                "D_TYPE=uint8_t",
                "FLOAT_TYPE=float",
            ],
        ),
        (
            "binary_int_u32",
            candle_shaders_dir.join("binary_int.comp"),
            &[
                "A_TYPE=uint",
                "B_TYPE=uint",
                "D_TYPE=uint",
                "FLOAT_TYPE=float",
            ],
        ),
        (
            "binary_int_i32",
            candle_shaders_dir.join("binary_int.comp"),
            &[
                "A_TYPE=int",
                "B_TYPE=int",
                "D_TYPE=int",
                "FLOAT_TYPE=float",
            ],
        ),
        (
            "binary_int_i64",
            candle_shaders_dir.join("binary_int.comp"),
            &[
                "A_TYPE=int64_t",
                "B_TYPE=int64_t",
                "D_TYPE=int64_t",
                "FLOAT_TYPE=float",
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
            "get_rows_u8",
            candle_shaders_dir.join("get_rows_u8.comp"),
            &[],
        ),
        (
            "get_rows_bf16_raw",
            candle_shaders_dir.join("get_rows_bf16.comp"),
            &[],
        ),
        (
            "argsort_u32",
            candle_shaders_dir.join("argsort_u32.comp"),
            &[],
        ),
        (
            "argsort_large_u32",
            candle_shaders_dir.join("argsort_large_u32.comp"),
            &[],
        ),
        (
            "argsort_i64",
            candle_shaders_dir.join("argsort_i64.comp"),
            &[],
        ),
        (
            "argsort_large_i64",
            candle_shaders_dir.join("argsort_large_i64.comp"),
            &[],
        ),
        (
            "argsort_f64",
            candle_shaders_dir.join("argsort_f64.comp"),
            &[],
        ),
        (
            "argsort_large_f64",
            candle_shaders_dir.join("argsort_large_f64.comp"),
            &[],
        ),
        (
            "get_rows_i64",
            candle_shaders_dir.join("get_rows_i64.comp"),
            &[],
        ),
        (
            "get_rows_f64",
            candle_shaders_dir.join("get_rows_f64.comp"),
            &[],
        ),
        (
            "matmul_f64",
            candle_shaders_dir.join("matmul_f64.comp"),
            &[],
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
        // Native f16 scatter-add (packed half CAS into u32 words). No F32 hub.
        (
            "set_rows_add_f16_i32",
            candle_shaders_dir.join("set_rows_add.comp"),
            &[
                "A_TYPE=float16_t",
                "B_TYPE=uint",
                "D_TYPE=uint",
                "FLOAT_TYPE=float16_t",
                "D_READ_WRITE",
                "SRC_F16",
            ],
        ),
        (
            "set_rows_u32_i32",
            candle_shaders_dir.join("set_rows_u32.comp"),
            &[
                "A_TYPE=uint",
                "B_TYPE=uint",
                "D_TYPE=uint",
                "FLOAT_TYPE=float",
                "D_READ_WRITE",
            ],
        ),
        (
            "fill_raw_u8",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint8_t", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_u32",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_i16",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint16_t", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_i32",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_i64",
            candle_shaders_dir.join("fill_raw.comp"),
            &["RAW_U32X2"],
        ),
        (
            "fill_raw_bf16",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint16_t", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_f16",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint16_t", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_f32",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_f64",
            candle_shaders_dir.join("fill_raw.comp"),
            &["RAW_U32X2"],
        ),
        (
            "fill_raw_f8e4m3",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint8_t", "RAW_CAST=p.value0"],
        ),
        (
            "fill_raw_f8e8m0",
            candle_shaders_dir.join("fill_raw.comp"),
            &["D_TYPE=uint8_t", "RAW_CAST=p.value0"],
        ),
        (
            "rand_uniform_f32",
            candle_shaders_dir.join("rand_uniform.comp"),
            &[],
        ),
        (
            "rand_uniform_f64",
            candle_shaders_dir.join("rand_uniform.comp"),
            &["USE_F64"],
        ),
        (
            "rand_normal_f32",
            candle_shaders_dir.join("rand_normal.comp"),
            &[],
        ),
        (
            "rand_normal_f64",
            candle_shaders_dir.join("rand_normal.comp"),
            &["USE_F64"],
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
            "convert_f32_bf16",
            candle_shaders_dir.join("convert.comp"),
            &[
                "A_TYPE=float",
                "D_TYPE=uint16_t",
                "DST_VIA_F32",
                "DATA_D_BF16",
            ],
        ),
        (
            "convert_f32_f64",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float", "D_TYPE=double", "USE_F64"],
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
            "convert_f16_bf16",
            candle_shaders_dir.join("convert.comp"),
            &[
                "A_TYPE=float16_t",
                "D_TYPE=uint16_t",
                "DST_VIA_F32",
                "DATA_D_BF16",
            ],
        ),
        (
            "convert_bf16_bf16",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint16_t", "D_TYPE=uint16_t"],
        ),
        (
            "convert_bf16_f32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint16_t", "D_TYPE=float", "DST_VIA_F32", "DATA_A_BF16"],
        ),
        (
            "convert_bf16_f16",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint16_t", "D_TYPE=float16_t", "DST_VIA_F32", "DATA_A_BF16"],
        ),
        (
            "convert_u8_u8",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint8_t", "D_TYPE=uint8_t"],
        ),
        (
            "convert_u32_u32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=uint", "D_TYPE=uint"],
        ),
        (
            "convert_i16_i16",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int16_t", "D_TYPE=int16_t"],
        ),
        (
            "convert_f32_i16",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float", "D_TYPE=int16_t"],
        ),
        (
            "convert_i16_f32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int16_t", "D_TYPE=float"],
        ),
        (
            "convert_f32_i32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=float", "D_TYPE=int32_t"],
        ),
        (
            "convert_i32_f32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int32_t", "D_TYPE=float"],
        ),
        (
            "convert_i64_i64",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=int64_t", "D_TYPE=int64_t"],
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
            "convert_u8_bf16",
            candle_shaders_dir.join("convert.comp"),
            &[
                "A_TYPE=uint8_t",
                "D_TYPE=uint16_t",
                "DST_VIA_F32",
                "DATA_D_BF16",
            ],
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
            "convert_u32_bf16",
            candle_shaders_dir.join("convert.comp"),
            &[
                "A_TYPE=uint",
                "D_TYPE=uint16_t",
                "DST_VIA_F32",
                "DATA_D_BF16",
            ],
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
            "convert_i64_bf16",
            candle_shaders_dir.join("convert.comp"),
            &[
                "A_TYPE=int64_t",
                "D_TYPE=uint16_t",
                "DST_VIA_F32",
                "DATA_D_BF16",
            ],
        ),
        (
            "convert_f64_f32",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=double", "D_TYPE=float", "USE_F64"],
        ),
        (
            "convert_f64_f64",
            candle_shaders_dir.join("convert.comp"),
            &["A_TYPE=double", "D_TYPE=double", "USE_F64"],
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
        (
            "layernorm",
            candle_shaders_dir.join("layernorm.comp"),
            &[],
        ),
        (
            "flash_attn",
            candle_shaders_dir.join("flash_attn.comp"),
            &[],
        ),
        (
            "flash_attn_f16",
            candle_shaders_dir.join("flash_attn.comp"),
            &["INPUT_F16"],
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

#[derive(Clone, Copy)]
struct GeneratorFeatures {
    integer_dot: bool,
    coopmat: bool,
    coopmat2: bool,
}

impl GeneratorFeatures {
    fn cpp_defines(self) -> Vec<&'static str> {
        let mut d = Vec::new();
        if self.integer_dot {
            d.push("GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT");
        }
        if self.coopmat {
            d.push("GGML_VULKAN_COOPMAT_GLSLC_SUPPORT");
        }
        if self.coopmat2 {
            d.push("GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT");
        }
        d
    }
}

fn compile_generator(
    source: &Path,
    output: &Path,
    features: GeneratorFeatures,
) -> std::io::Result<()> {
    let mut errors = Vec::new();

    // Prefer an explicit CXX when provided.
    if let Ok(cxx) = env::var("CXX") {
        match try_compile_with_unix_style(&cxx, source, output, features) {
            Ok(()) => return Ok(()),
            Err(err) => errors.push(err),
        }
    }

    // Unix-style compilers (also available via MinGW/LLVM on Windows).
    for compiler in ["c++", "g++", "clang++"] {
        match try_compile_with_unix_style(compiler, source, output, features) {
            Ok(()) => return Ok(()),
            Err(err) => errors.push(err),
        }
    }

    // MSVC on Windows: build through a developer command prompt so INCLUDE/LIB
    // are set. Vulkan-shaders-gen is a host tool and does not need CUDA.
    #[cfg(windows)]
    {
        match try_compile_with_msvc(source, output, features) {
            Ok(()) => return Ok(()),
            Err(err) => errors.push(err),
        }
    }

    panic!(
        "failed to compile {}:\n{}",
        source.display(),
        errors.join("\n")
    );
}

fn try_compile_with_unix_style(
    compiler: &str,
    source: &Path,
    output: &Path,
    features: GeneratorFeatures,
) -> Result<(), String> {
    let mut cmd = Command::new(compiler);
    cmd.arg("-std=c++17").arg(source).arg("-o").arg(output);
    for def in features.cpp_defines() {
        cmd.arg(format!("-D{def}"));
    }
    match cmd.output() {
        Ok(output_result) if output_result.status.success() => Ok(()),
        Ok(output_result) => Err(format!(
            "{compiler}: {}{}",
            String::from_utf8_lossy(&output_result.stdout),
            String::from_utf8_lossy(&output_result.stderr)
        )),
        Err(err) => Err(format!("{compiler}: {err}")),
    }
}

#[cfg(windows)]
fn try_compile_with_msvc(
    source: &Path,
    output: &Path,
    features: GeneratorFeatures,
) -> Result<(), String> {
    let vcvars = find_vcvars64().ok_or_else(|| {
        "cl/msvc: could not locate VC\\Auxiliary\\Build\\vcvars64.bat (Visual Studio)".to_string()
    })?;
    let define: String = features
        .cpp_defines()
        .into_iter()
        .map(|d| format!(" /D{d}"))
        .collect();

    // MSVC's cl.exe historically fails to open source files when the path
    // contains non-ASCII characters (e.g. Cyrillic user profile dirs) under
    // the system ANSI code page. Compile from %TEMP% (ASCII) and copy the
    // resulting binary back to OUT_DIR.
    let temp_root = env::temp_dir().join(format!(
        "candle-vulkan-shaders-gen-{}",
        std::process::id()
    ));
    fs::create_dir_all(&temp_root).map_err(|err| format!("cl/msvc: mkdir temp: {err}"))?;
    let temp_src = temp_root.join("vulkan-shaders-gen.cpp");
    let temp_exe = temp_root.join("vulkan-shaders-gen.exe");
    let temp_obj = temp_root.join("vulkan-shaders-gen.obj");
    let bat_path = temp_root.join("build.bat");
    fs::copy(source, &temp_src).map_err(|err| format!("cl/msvc: copy source: {err}"))?;

    let bat = format!(
        "@echo off\r\n\
         chcp 65001 >nul\r\n\
         call \"{vcvars}\"\r\n\
         if errorlevel 1 exit /b 1\r\n\
         cl /nologo /std:c++17 /EHsc /O2{define} \"{source}\" /Fe:\"{output}\" /Fo:\"{obj}\" /link /SUBSYSTEM:CONSOLE\r\n\
         exit /b %ERRORLEVEL%\r\n",
        vcvars = vcvars.display(),
        define = define,
        source = temp_src.display(),
        output = temp_exe.display(),
        obj = temp_obj.display(),
    );
    fs::write(&bat_path, bat).map_err(|err| format!("cl/msvc: write bat: {err}"))?;
    let output_result = Command::new("cmd")
        .args(["/C", &bat_path.to_string_lossy()])
        .output()
        .map_err(|err| format!("cl/msvc: failed to spawn cmd: {err}"))?;

    let compile_ok = output_result.status.success() && temp_exe.is_file();
    if compile_ok {
        if let Some(parent) = output.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::copy(&temp_exe, output).map_err(|err| format!("cl/msvc: copy exe: {err}"))?;
    }
    let _ = fs::remove_dir_all(&temp_root);

    if compile_ok {
        Ok(())
    } else {
        Err(format!(
            "cl/msvc: {}{}",
            String::from_utf8_lossy(&output_result.stdout),
            String::from_utf8_lossy(&output_result.stderr)
        ))
    }
}

#[cfg(windows)]
fn find_vcvars64() -> Option<PathBuf> {
    if let Ok(path) = env::var("VCVARS64") {
        let p = PathBuf::from(path);
        if p.is_file() {
            return Some(p);
        }
    }
    let candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.is_file() {
            return Some(p);
        }
    }
    // Fall back to vswhere if present.
    let vswhere = PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if vswhere.is_file() {
        if let Ok(out) = Command::new(&vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-find",
                r"VC\Auxiliary\Build\vcvars64.bat",
            ])
            .output()
        {
            if out.status.success() {
                let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !path.is_empty() {
                    let p = PathBuf::from(path);
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
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
