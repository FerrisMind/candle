mod wgsl {
    use super::Module;

    include!(concat!(env!("OUT_DIR"), "/wgsl.rs"));
}

#[derive(Debug, Clone, Copy)]
pub struct Module {
    name: &'static str,
    source: &'static str,
}

impl Module {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn source(&self) -> &'static str {
        self.source
    }
}

pub use wgsl::*;

pub fn get(name: &str) -> Option<Module> {
    ALL_MODULES
        .iter()
        .copied()
        .find(|module| module.name == name)
}

#[derive(Debug, Clone, Copy)]
pub enum DType {
    F32,
    F16,
}

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum QuantizedDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

// Larger tiles improve dense F32 throughput vs CUDA/cuBLAS; keep WM*WN*TM*TN
// within typical register/shared-memory budgets on mid-range GPUs.
const MUL_MAT_TILE_M: u32 = 8;
const MUL_MAT_TILE_N: u32 = 8;
const MUL_MAT_WG_SIZE_M: u32 = 8;
const MUL_MAT_WG_SIZE_N: u32 = 8;
const MUL_MAT_REG_TILE_K_FLOAT: u32 = 16;
const QUANT_MUL_MAT_TILE_M: u32 = MUL_MAT_TILE_M;
const QUANT_MUL_MAT_TILE_N: u32 = MUL_MAT_TILE_N;
const QUANT_MUL_MAT_WG_SIZE_M: u32 = MUL_MAT_WG_SIZE_M;
const QUANT_MUL_MAT_WG_SIZE_N: u32 = MUL_MAT_WG_SIZE_N;
const QUANT_MUL_MAT_REG_TILE_K_QUANT: u32 = 32;
const QUANT_MUL_MAT_VEC_WG_SIZE: u32 = 256;
const QUANT_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG: u32 = 4;
const QUANT_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG: u32 = 4;
const QUANT_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG: u32 = 4;

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Abs,
    Ceil,
    Clamp,
    Cos,
    Elu,
    Exp,
    Floor,
    Gelu,
    GeluErf,
    GeluQuick,
    Hardsigmoid,
    Hardswish,
    Fill,
    Log,
    Neg,
    Relu,
    Round,
    Sgn,
    Sigmoid,
    Silu,
    Sin,
    Sqrt,
    Square,
    Step,
    Tanh,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Div,
    Mul,
    Sub,
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderOp {
    Unary(UnaryOp),
    Binary(BinaryOp),
}

impl UnaryOp {
    fn define(self) -> &'static str {
        match self {
            Self::Abs => "ABS",
            Self::Ceil => "CEIL",
            Self::Clamp => "CLAMP",
            Self::Cos => "COS",
            Self::Elu => "ELU",
            Self::Exp => "EXP",
            Self::Floor => "FLOOR",
            Self::Gelu => "GELU",
            Self::GeluErf => "GELU_ERF",
            Self::GeluQuick => "GELU_QUICK",
            Self::Hardsigmoid => "HARDSIGMOID",
            Self::Hardswish => "HARDSWISH",
            Self::Fill => "FILL",
            Self::Log => "LOG",
            Self::Neg => "NEG",
            Self::Relu => "RELU",
            Self::Round => "ROUND",
            Self::Sgn => "SGN",
            Self::Sigmoid => "SIGMOID",
            Self::Silu => "SILU",
            Self::Sin => "SIN",
            Self::Sqrt => "SQRT",
            Self::Square => "SQR",
            Self::Step => "STEP",
            Self::Tanh => "TANH",
        }
    }
}

impl BinaryOp {
    fn define(self) -> &'static str {
        match self {
            Self::Add => "OP_ADD",
            Self::Div => "OP_DIV",
            Self::Mul => "OP_MUL",
            Self::Sub => "OP_SUB",
        }
    }
}

pub fn shader(op: ShaderOp, dtype: DType, workgroup_size: u32) -> String {
    let source = match op {
        ShaderOp::Unary(_) => UNARY_WGSL.source(),
        ShaderOp::Binary(_) => BINARY_WGSL.source(),
    };
    let mut defines = vec!["WG_SIZE".to_string()];
    let mut replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    match dtype {
        DType::F32 => {
            defines.push("TYPE_F32".to_string());
            replacements.push(("TYPE".to_string(), "f32".to_string()));
            replacements.push(("DataType".to_string(), "f32".to_string()));
        }
        DType::F16 => {
            defines.push("TYPE_F16".to_string());
            replacements.push(("TYPE".to_string(), "f16".to_string()));
            replacements.push(("DataType".to_string(), "f16".to_string()));
        }
    }
    match op {
        ShaderOp::Unary(op) => defines.push(op.define().to_string()),
        ShaderOp::Binary(op) => defines.push(op.define().to_string()),
    }
    preprocess(source, &defines, &replacements, dtype)
}

pub fn scale_shader(workgroup_size: u32) -> Option<String> {
    let source = get("scale.wgsl")?.source();
    let defines = vec!["WG_SIZE".to_string()];
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn rand_uniform_shader(dtype: DType, workgroup_size: u32) -> Option<String> {
    let source = get("rand_uniform.wgsl")?.source().replace(
        "#include \"rand_common.wgsl\"",
        get("rand_common.wgsl")?.source(),
    );
    let mut defines = vec!["WG_SIZE".to_string()];
    let mut replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    match dtype {
        DType::F32 => {
            defines.push("TYPE_F32".to_string());
            replacements.push(("DataType".to_string(), "f32".to_string()));
        }
        DType::F16 => {
            defines.push("TYPE_F16".to_string());
            replacements.push(("DataType".to_string(), "f16".to_string()));
        }
    }
    Some(preprocess(&source, &defines, &replacements, dtype))
}

pub fn rand_normal_shader(dtype: DType, workgroup_size: u32) -> Option<String> {
    let source = get("rand_normal.wgsl")?.source().replace(
        "#include \"rand_common.wgsl\"",
        get("rand_common.wgsl")?.source(),
    );
    let mut defines = vec!["WG_SIZE".to_string()];
    let mut replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    match dtype {
        DType::F32 => {
            defines.push("TYPE_F32".to_string());
            replacements.push(("DataType".to_string(), "f32".to_string()));
        }
        DType::F16 => {
            defines.push("TYPE_F16".to_string());
            replacements.push(("DataType".to_string(), "f16".to_string()));
        }
    }
    Some(preprocess(&source, &defines, &replacements, dtype))
}

pub fn argmax_shader(workgroup_size: u32) -> Option<String> {
    let source = get("argmax.wgsl")?.source();
    let defines = vec!["WG_SIZE".to_string()];
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn rope_shader(dtype: DType, workgroup_size: u32) -> Option<String> {
    let source = get("rope.wgsl")?.source();
    let mut defines = vec!["WG_SIZE".to_string()];
    let mut replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    match dtype {
        DType::F32 => {
            defines.push("TYPE_F32".to_string());
            replacements.push(("DataType".to_string(), "f32".to_string()));
        }
        DType::F16 => {
            defines.push("TYPE_F16".to_string());
            replacements.push(("DataType".to_string(), "f16".to_string()));
        }
    }
    Some(preprocess(source, &defines, &replacements, dtype))
}

fn argsort_shader_for_type(workgroup_size: u32, asc: bool, src_type: &str) -> Option<String> {
    let source = get("argsort.wgsl")?.source();
    let mut defines = vec!["WG_SIZE".to_string()];
    if asc {
        defines.push("ORDER == 0".to_string());
    }
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        (
            "ORDER".to_string(),
            if asc {
                "0".to_string()
            } else {
                "1".to_string()
            },
        ),
    ];
    let mut replacements = replacements;
    replacements.push(("SRC_TYPE".to_string(), src_type.to_string()));
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn argsort_shader(workgroup_size: u32, asc: bool) -> Option<String> {
    argsort_shader_for_type(workgroup_size, asc, "f32")
}

pub fn argsort_u32_shader(workgroup_size: u32, asc: bool) -> Option<String> {
    argsort_shader_for_type(workgroup_size, asc, "u32")
}

fn argsort_merge_shader_for_type(workgroup_size: u32, asc: bool, src_type: &str) -> Option<String> {
    let source = get("argsort_merge.wgsl")?.source();
    let mut defines = vec!["WG_SIZE".to_string()];
    if asc {
        defines.push("ORDER == 0".to_string());
    }
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        (
            "ORDER".to_string(),
            if asc {
                "0".to_string()
            } else {
                "1".to_string()
            },
        ),
    ];
    let mut replacements = replacements;
    replacements.push(("SRC_TYPE".to_string(), src_type.to_string()));
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn argsort_merge_shader(workgroup_size: u32, asc: bool) -> Option<String> {
    argsort_merge_shader_for_type(workgroup_size, asc, "f32")
}

pub fn argsort_u32_merge_shader(workgroup_size: u32, asc: bool) -> Option<String> {
    argsort_merge_shader_for_type(workgroup_size, asc, "u32")
}

pub fn cumsum_shader(workgroup_size: u32) -> Option<String> {
    let source = get("cumsum.wgsl")?.source();
    let defines = vec!["WG_SIZE".to_string()];
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn softmax_shader(workgroup_size: u32) -> Option<String> {
    let source = get("soft_max.wgsl")?.source();
    let defines = vec!["WG_SIZE".to_string()];
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn rms_norm_mul_shader(workgroup_size: u32) -> Option<String> {
    let source = get("rms_norm_mul.wgsl")?.source();
    let defines = vec!["WG_SIZE".to_string()];
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn get_rows_f32_shader(workgroup_size: u32) -> Option<String> {
    let source = get("get_rows.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let defines = vec!["F32".to_string()];
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("BLOCK_SIZE".to_string(), "1".to_string()),
        ("SRC_TYPE".to_string(), "f32".to_string()),
        ("DST_TYPE".to_string(), "f32".to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F32))
}

pub fn get_rows_f16_shader(workgroup_size: u32) -> Option<String> {
    let source = get("get_rows.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let defines = vec!["F16".to_string()];
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("BLOCK_SIZE".to_string(), "1".to_string()),
        ("SRC_TYPE".to_string(), "f16".to_string()),
        ("DST_TYPE".to_string(), "f32".to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F16))
}

pub fn get_rows_u32_shader(workgroup_size: u32) -> Option<String> {
    let source = get("get_rows.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let defines = vec!["U32".to_string()];
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("BLOCK_SIZE".to_string(), "1".to_string()),
        ("SRC_TYPE".to_string(), "u32".to_string()),
        ("DST_TYPE".to_string(), "u32".to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F32))
}

pub fn set_rows_f32_shader(workgroup_size: u32) -> Option<String> {
    let source = get("set_rows.wgsl")?.source();
    let defines = vec!["DST_F32".to_string()];
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("SRC_TYPE".to_string(), "f32".to_string()),
        ("DST_INNER_TYPE".to_string(), "f32".to_string()),
        ("DST_TYPE".to_string(), "f32".to_string()),
        ("VEC_SIZE".to_string(), "1".to_string()),
    ];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn set_rows_f16_shader(workgroup_size: u32) -> Option<String> {
    let source = get("set_rows.wgsl")?.source();
    let defines = Vec::new();
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("SRC_TYPE".to_string(), "f16".to_string()),
        ("DST_INNER_TYPE".to_string(), "f16".to_string()),
        ("DST_TYPE".to_string(), "f16".to_string()),
        ("VEC_SIZE".to_string(), "1".to_string()),
    ];
    Some(preprocess(source, &defines, &replacements, DType::F16))
}

pub fn set_rows_u32_shader(workgroup_size: u32) -> Option<String> {
    let source = get("set_rows.wgsl")?.source();
    let defines = Vec::new();
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("SRC_TYPE".to_string(), "u32".to_string()),
        ("DST_INNER_TYPE".to_string(), "u32".to_string()),
        ("DST_TYPE".to_string(), "u32".to_string()),
        ("VEC_SIZE".to_string(), "1".to_string()),
    ];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn set_rows_add_f32_shader(workgroup_size: u32) -> Option<String> {
    let source = get("set_rows.wgsl")?.source();
    let defines = vec!["DST_F32".to_string(), "ADD".to_string()];
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("SRC_TYPE".to_string(), "f32".to_string()),
        ("DST_INNER_TYPE".to_string(), "f32".to_string()),
        ("DST_TYPE".to_string(), "f32".to_string()),
        ("VEC_SIZE".to_string(), "1".to_string()),
    ];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn matmul_f32_shader() -> Option<String> {
    let source = get("mul_mat.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let defines = vec!["FLOAT".to_string()];
    let replacements = vec![
        ("SRC0_TYPE".to_string(), "f32".to_string()),
        ("SRC1_TYPE".to_string(), "f32".to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F32))
}

pub fn matmul_bf16_shader() -> Option<String> {
    let source = get("mul_mat_bf16.wgsl")?.source();
    Some(source.to_string())
}

pub fn matmul_f16_shader() -> Option<String> {
    let source = get("mul_mat.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let defines = vec!["FLOAT".to_string()];
    let replacements = vec![
        ("SRC0_TYPE".to_string(), "f16".to_string()),
        ("SRC1_TYPE".to_string(), "f16".to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F16))
}

pub fn matmul_f64_shader() -> Option<&'static str> {
    get("mul_mat_f64.wgsl").map(|module| module.source())
}

pub fn matmul_fast_tile_shape() -> (u32, u32, u32, u32, u32) {
    (
        MUL_MAT_TILE_M,
        MUL_MAT_TILE_N,
        MUL_MAT_WG_SIZE_M,
        MUL_MAT_WG_SIZE_N,
        MUL_MAT_REG_TILE_K_FLOAT,
    )
}

pub fn matmul_fast_shader(dtype: DType, vectorized: bool) -> Option<String> {
    let source = get("mul_mat_reg_tile.wgsl")?
        .source()
        .replace(
            "#include \"common_decls.tmpl\"",
            get("common_decls.tmpl")?.source(),
        )
        .replace(
            "#include \"mul_mat_decls.tmpl\"",
            get("mul_mat_decls.tmpl")?.source(),
        );
    let (inner_type, need_f16_enable) = match dtype {
        // Dense F32 uses f32 workgroup memory for full precision (FLOAT_ACC_SHMEM).
        DType::F32 => ("f32", false),
        // F16 path keeps the historical f16 shmem staging.
        DType::F16 => ("f16", true),
    };
    let mut defines = vec![
        if vectorized {
            "VEC".to_string()
        } else {
            "SCALAR".to_string()
        },
        "INIT_SRC0_SHMEM_FLOAT".to_string(),
        "INIT_SRC1_SHMEM_FLOAT".to_string(),
        "TILE_M".to_string(),
        "TILE_N".to_string(),
        "WORKGROUP_SIZE_M".to_string(),
        "WORKGROUP_SIZE_N".to_string(),
        "TILE_K".to_string(),
    ];
    if matches!(dtype, DType::F32) {
        defines.push("FLOAT_ACC_SHMEM".to_string());
    }
    let replacements = vec![
        ("SRC0_INNER_TYPE".to_string(), inner_type.to_string()),
        ("SRC1_INNER_TYPE".to_string(), inner_type.to_string()),
        ("TILE_M".to_string(), format!("{MUL_MAT_TILE_M}u")),
        ("TILE_N".to_string(), format!("{MUL_MAT_TILE_N}u")),
        (
            "WORKGROUP_SIZE_M".to_string(),
            format!("{MUL_MAT_WG_SIZE_M}u"),
        ),
        (
            "WORKGROUP_SIZE_N".to_string(),
            format!("{MUL_MAT_WG_SIZE_N}u"),
        ),
        ("TILE_K".to_string(), format!("{MUL_MAT_REG_TILE_K_FLOAT}u")),
    ];
    defines.push("TILE_M".to_string());
    defines.push("TILE_N".to_string());
    defines.push("WORKGROUP_SIZE_M".to_string());
    defines.push("WORKGROUP_SIZE_N".to_string());
    defines.push("TILE_K".to_string());
    // preprocess's dtype controls whether `enable f16;` is injected.
    let pre_dtype = if need_f16_enable {
        DType::F16
    } else {
        DType::F32
    };
    Some(preprocess(&source, &defines, &replacements, pre_dtype))
}

pub fn matvec_outputs_per_wg() -> u32 {
    QUANT_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG
}

pub fn matvec_workgroup_size() -> u32 {
    QUANT_MUL_MAT_VEC_WG_SIZE
}

pub fn matvec_shader(dtype: DType, vectorized: bool, use_subgroups: bool) -> Option<String> {
    let source = get("mul_mat_vec.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let inner_type = match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
    };
    let mut defines = vec![
        if vectorized {
            "VEC".to_string()
        } else {
            "SCALAR".to_string()
        },
        if use_subgroups {
            "USE_SUBGROUP_REDUCTION".to_string()
        } else {
            "USE_WORKGROUP_REDUCTION".to_string()
        },
        "MUL_ACC_FLOAT".to_string(),
        "WG_SIZE".to_string(),
        "OUTPUTS_PER_WG".to_string(),
    ];
    let replacements = vec![
        ("WG_SIZE".to_string(), QUANT_MUL_MAT_VEC_WG_SIZE.to_string()),
        (
            "OUTPUTS_PER_WG".to_string(),
            QUANT_MUL_MAT_VEC_FLOAT_OUTPUTS_PER_WG.to_string(),
        ),
        ("SRC0_INNER_TYPE".to_string(), inner_type.to_string()),
        ("SRC1_INNER_TYPE".to_string(), inner_type.to_string()),
    ];
    if !matches!(dtype, DType::F16) {
        defines.retain(|define| define != "USE_SUBGROUP_REDUCTION");
    }
    Some(preprocess(&source, &defines, &replacements, dtype))
}

fn quantized_shader_config(dtype: QuantizedDType) -> (Vec<String>, &'static str, &'static str) {
    match dtype {
        QuantizedDType::Q4_0 => (
            vec![
                "Q4_0".to_string(),
                "BYTE_HELPERS".to_string(),
                "U32_DEQUANT_HELPERS".to_string(),
            ],
            "u32",
            "32",
        ),
        QuantizedDType::Q4_1 => (
            vec![
                "Q4_1".to_string(),
                "Q4_1_T".to_string(),
                "BYTE_HELPERS".to_string(),
            ],
            "q4_1",
            "32",
        ),
        QuantizedDType::Q5_0 => (
            vec![
                "Q5_0".to_string(),
                "BYTE_HELPERS".to_string(),
                "U32_DEQUANT_HELPERS".to_string(),
            ],
            "u32",
            "32",
        ),
        QuantizedDType::Q5_1 => (
            vec![
                "Q5_1".to_string(),
                "Q5_1_T".to_string(),
                "BYTE_HELPERS".to_string(),
            ],
            "q5_1",
            "32",
        ),
        QuantizedDType::Q8_0 => (
            vec![
                "Q8_0".to_string(),
                "BYTE_HELPERS".to_string(),
                "U32_DEQUANT_HELPERS".to_string(),
            ],
            "u32",
            "32",
        ),
        QuantizedDType::Q8_1 => (
            vec![
                "Q8_1".to_string(),
                "Q8_1_T".to_string(),
                "BYTE_HELPERS".to_string(),
            ],
            "q8_1",
            "32",
        ),
        QuantizedDType::Q2_K => (
            vec![
                "Q2_K".to_string(),
                "Q2_K_T".to_string(),
                "BYTE_HELPERS".to_string(),
            ],
            "q2_K",
            "256",
        ),
        QuantizedDType::Q3_K => (
            vec![
                "Q3_K".to_string(),
                "BYTE_HELPERS".to_string(),
                "U32_DEQUANT_HELPERS".to_string(),
            ],
            "u32",
            "256",
        ),
        QuantizedDType::Q4_K => (
            vec![
                "Q4_K".to_string(),
                "Q4_K_T".to_string(),
                "Q4_K_SCALE_MIN".to_string(),
                "BYTE_HELPERS".to_string(),
            ],
            "q4_K",
            "256",
        ),
        QuantizedDType::Q5_K => (
            vec![
                "Q5_K".to_string(),
                "Q5_K_T".to_string(),
                "Q5_K_SCALE_MIN".to_string(),
                "BYTE_HELPERS".to_string(),
            ],
            "q5_K",
            "256",
        ),
        QuantizedDType::Q6_K => (
            vec![
                "Q6_K".to_string(),
                "BYTE_HELPERS".to_string(),
                "U32_DEQUANT_HELPERS".to_string(),
            ],
            "u32",
            "256",
        ),
        QuantizedDType::Q8_K => (
            vec![
                "Q8_K".to_string(),
                "BYTE_HELPERS".to_string(),
                "U32_DEQUANT_HELPERS".to_string(),
            ],
            "u32",
            "256",
        ),
    }
}

fn quantized_mul_acc_define(dtype: QuantizedDType) -> &'static str {
    match dtype {
        QuantizedDType::Q4_0 => "MUL_ACC_Q4_0",
        QuantizedDType::Q4_1 => "MUL_ACC_Q4_1",
        QuantizedDType::Q5_0 => "MUL_ACC_Q5_0",
        QuantizedDType::Q5_1 => "MUL_ACC_Q5_1",
        QuantizedDType::Q8_0 => "MUL_ACC_Q8_0",
        QuantizedDType::Q8_1 => "MUL_ACC_Q8_1",
        QuantizedDType::Q2_K => "MUL_ACC_Q2_K",
        QuantizedDType::Q3_K => "MUL_ACC_Q3_K",
        QuantizedDType::Q4_K => "MUL_ACC_Q4_K",
        QuantizedDType::Q5_K => "MUL_ACC_Q5_K",
        QuantizedDType::Q6_K => "MUL_ACC_Q6_K",
        QuantizedDType::Q8_K => "MUL_ACC_Q8_K",
    }
}

pub fn quantized_matvec_outputs_per_wg(dtype: QuantizedDType) -> u32 {
    match dtype {
        QuantizedDType::Q2_K
        | QuantizedDType::Q3_K
        | QuantizedDType::Q4_K
        | QuantizedDType::Q5_K
        | QuantizedDType::Q6_K
        | QuantizedDType::Q8_K => QUANT_MUL_MAT_VEC_K_Q_OUTPUTS_PER_WG,
        _ => QUANT_MUL_MAT_VEC_LEGACY_Q_OUTPUTS_PER_WG,
    }
}

pub fn quantized_matvec_workgroup_size() -> u32 {
    QUANT_MUL_MAT_VEC_WG_SIZE
}

pub fn quantized_matmul_fast_tile_shape() -> (u32, u32, u32, u32, u32) {
    (
        QUANT_MUL_MAT_TILE_M,
        QUANT_MUL_MAT_TILE_N,
        QUANT_MUL_MAT_WG_SIZE_M,
        QUANT_MUL_MAT_WG_SIZE_N,
        QUANT_MUL_MAT_REG_TILE_K_QUANT,
    )
}

pub fn quantized_matmul_shader(dtype: QuantizedDType, rhs_dtype: DType) -> Option<String> {
    let source = get("mul_mat.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let (mut defines, src0_type, _) = quantized_shader_config(dtype);
    defines.push("DECLARE_BYTE_LOADERS_SRC0".to_string());
    let src1_type = match rhs_dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
    };
    let replacements = vec![
        ("SRC0_TYPE".to_string(), src0_type.to_string()),
        ("SRC1_TYPE".to_string(), src1_type.to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F16))
}

pub fn quantized_matvec_shader(dtype: QuantizedDType, rhs_dtype: DType) -> Option<String> {
    let source = get("mul_mat_vec.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let outputs_per_wg = quantized_matvec_outputs_per_wg(dtype);
    let mut defines = vec![
        "SCALAR".to_string(),
        "USE_WORKGROUP_REDUCTION".to_string(),
        "BYTE_HELPERS".to_string(),
        "U32_DEQUANT_HELPERS".to_string(),
        "DECLARE_BYTE_LOADERS_SRC0".to_string(),
        quantized_mul_acc_define(dtype).to_string(),
    ];
    let src1_type = match rhs_dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
    };
    let replacements = vec![
        ("WG_SIZE".to_string(), QUANT_MUL_MAT_VEC_WG_SIZE.to_string()),
        ("OUTPUTS_PER_WG".to_string(), outputs_per_wg.to_string()),
        ("SRC0_TYPE".to_string(), "u32".to_string()),
        ("SRC1_TYPE".to_string(), src1_type.to_string()),
        ("SRC0_INNER_TYPE".to_string(), "u32".to_string()),
        ("SRC1_INNER_TYPE".to_string(), src1_type.to_string()),
    ];
    defines.push("WG_SIZE".to_string());
    defines.push("OUTPUTS_PER_WG".to_string());
    Some(preprocess(&source, &defines, &replacements, DType::F16))
}

pub fn quantized_matmul_fast_shader(dtype: QuantizedDType, rhs_dtype: DType) -> Option<String> {
    let source = get("mul_mat_reg_tile.wgsl")?
        .source()
        .replace(
            "#include \"common_decls.tmpl\"",
            get("common_decls.tmpl")?.source(),
        )
        .replace(
            "#include \"mul_mat_decls.tmpl\"",
            get("mul_mat_decls.tmpl")?.source(),
        );
    let mut defines = vec![
        "SCALAR".to_string(),
        "BYTE_HELPERS".to_string(),
        "U32_DEQUANT_HELPERS".to_string(),
        "DECLARE_BYTE_LOADERS_SRC0".to_string(),
        quantized_mul_acc_define(dtype).to_string(),
        quantized_mul_mat_id_init_define(dtype).to_string(),
        "INIT_SRC1_SHMEM_FLOAT".to_string(),
    ];
    let src1_type = match rhs_dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
    };
    let replacements = vec![
        ("SRC0_TYPE".to_string(), "u32".to_string()),
        ("SRC1_TYPE".to_string(), src1_type.to_string()),
        ("DST_TYPE".to_string(), "f32".to_string()),
        ("SRC0_INNER_TYPE".to_string(), "u32".to_string()),
        ("SRC1_INNER_TYPE".to_string(), src1_type.to_string()),
        ("TILE_M".to_string(), format!("{QUANT_MUL_MAT_TILE_M}u")),
        ("TILE_N".to_string(), format!("{QUANT_MUL_MAT_TILE_N}u")),
        (
            "WORKGROUP_SIZE_M".to_string(),
            format!("{QUANT_MUL_MAT_WG_SIZE_M}u"),
        ),
        (
            "WORKGROUP_SIZE_N".to_string(),
            format!("{QUANT_MUL_MAT_WG_SIZE_N}u"),
        ),
        (
            "TILE_K".to_string(),
            format!("{QUANT_MUL_MAT_REG_TILE_K_QUANT}u"),
        ),
    ];
    defines.push("TILE_M".to_string());
    defines.push("TILE_N".to_string());
    defines.push("WORKGROUP_SIZE_M".to_string());
    defines.push("WORKGROUP_SIZE_N".to_string());
    defines.push("TILE_K".to_string());
    Some(preprocess(&source, &defines, &replacements, DType::F16))
}

pub fn quantized_get_rows_f32_shader(dtype: QuantizedDType, workgroup_size: u32) -> Option<String> {
    let source = get("get_rows.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let (mut defines, src_type, block_size) = quantized_shader_config(dtype);
    defines.push("DECLARE_BYTE_LOADERS_SRC".to_string());
    let replacements = vec![
        ("WG_SIZE".to_string(), workgroup_size.to_string()),
        ("BLOCK_SIZE".to_string(), block_size.to_string()),
        ("SRC_TYPE".to_string(), src_type.to_string()),
        ("DST_TYPE".to_string(), "f32".to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F16))
}

fn quantized_mul_mat_id_init_define(dtype: QuantizedDType) -> &'static str {
    match dtype {
        QuantizedDType::Q4_0 => "INIT_SRC0_SHMEM_Q4_0",
        QuantizedDType::Q4_1 => "INIT_SRC0_SHMEM_Q4_1",
        QuantizedDType::Q5_0 => "INIT_SRC0_SHMEM_Q5_0",
        QuantizedDType::Q5_1 => "INIT_SRC0_SHMEM_Q5_1",
        QuantizedDType::Q8_0 => "INIT_SRC0_SHMEM_Q8_0",
        QuantizedDType::Q8_1 => "INIT_SRC0_SHMEM_Q8_1",
        QuantizedDType::Q2_K => "INIT_SRC0_SHMEM_Q2_K",
        QuantizedDType::Q3_K => "INIT_SRC0_SHMEM_Q3_K",
        QuantizedDType::Q4_K => "INIT_SRC0_SHMEM_Q4_K",
        QuantizedDType::Q5_K => "INIT_SRC0_SHMEM_Q5_K",
        QuantizedDType::Q6_K => "INIT_SRC0_SHMEM_Q6_K",
        QuantizedDType::Q8_K => "INIT_SRC0_SHMEM_Q8_K",
    }
}

pub fn quantized_mul_mat_id_gather_shader(workgroup_size: u32) -> Option<String> {
    let source = get("mul_mat_id_gather.wgsl")?.source();
    let defines = Vec::new();
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(source, &defines, &replacements, DType::F32))
}

pub fn quantized_mul_mat_id_shader(dtype: QuantizedDType, rhs_dtype: DType) -> Option<String> {
    let source = get("mul_mat_id.wgsl")?
        .source()
        .replace(
            "#include \"common_decls.tmpl\"",
            get("common_decls.tmpl")?.source(),
        )
        .replace(
            "#include \"mul_mat_decls.tmpl\"",
            get("mul_mat_decls.tmpl")?.source(),
        );
    let defines = vec![
        "MUL_MAT_ID".to_string(),
        "SCALAR".to_string(),
        "BYTE_HELPERS".to_string(),
        "DECLARE_BYTE_LOADERS_SRC0".to_string(),
        "INIT_SRC1_SHMEM_FLOAT".to_string(),
        "U32_DEQUANT_HELPERS".to_string(),
        quantized_mul_mat_id_init_define(dtype).to_string(),
    ];
    let src1_type = match rhs_dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
    };
    let replacements = vec![
        ("SRC0_TYPE".to_string(), "u32".to_string()),
        ("SRC1_TYPE".to_string(), src1_type.to_string()),
        ("TILE_M".to_string(), "4u".to_string()),
        ("TILE_N".to_string(), "4u".to_string()),
        ("TILE_K".to_string(), "32u".to_string()),
        ("WORKGROUP_SIZE_M".to_string(), "8u".to_string()),
        ("WORKGROUP_SIZE_N".to_string(), "8u".to_string()),
    ];
    Some(preprocess(&source, &defines, &replacements, DType::F16))
}

pub fn conv2d_f32_shader(workgroup_size: u32) -> Option<String> {
    let source = get("conv2d.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let defines = vec![
        "WEIGHT_F32".to_string(),
        "INPUT_F32".to_string(),
        "OUTPUT_F32".to_string(),
    ];
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(&source, &defines, &replacements, DType::F32))
}

pub fn im2col_f32_shader(workgroup_size: u32) -> Option<String> {
    let source = get("im2col.wgsl")?.source().replace(
        "#include \"common_decls.tmpl\"",
        get("common_decls.tmpl")?.source(),
    );
    let defines = vec!["INPUT_F32".to_string(), "OUTPUT_F32".to_string()];
    let replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    Some(preprocess(&source, &defines, &replacements, DType::F32))
}

pub fn fill_inplace_shader(dtype: DType, workgroup_size: u32) -> String {
    let mut defines = vec![
        "WG_SIZE".to_string(),
        "INPLACE".to_string(),
        UnaryOp::Fill.define().to_string(),
    ];
    let mut replacements = vec![("WG_SIZE".to_string(), workgroup_size.to_string())];
    match dtype {
        DType::F32 => {
            defines.push("TYPE_F32".to_string());
            replacements.push(("TYPE".to_string(), "f32".to_string()));
        }
        DType::F16 => {
            defines.push("TYPE_F16".to_string());
            replacements.push(("TYPE".to_string(), "f16".to_string()));
        }
    }
    preprocess(UNARY_WGSL.source(), &defines, &replacements, dtype)
}

#[derive(Clone, Copy)]
struct Cond {
    parent_active: bool,
    this_active: bool,
    branch_taken: bool,
}

fn preprocess(
    source: &str,
    defines: &[String],
    replacements: &[(String, String)],
    dtype: DType,
) -> String {
    let mut out = String::new();
    let mut stack: Vec<Cond> = Vec::new();
    let mut local_replacements = replacements.to_vec();

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("#ifdef ") {
            let active = is_active(&stack);
            let cond = defines.iter().any(|d| d == name.trim());
            stack.push(Cond {
                parent_active: active,
                this_active: active && cond,
                branch_taken: active && cond,
            });
            continue;
        }
        if let Some(name) = trimmed.strip_prefix("#ifndef ") {
            let active = is_active(&stack);
            let cond = !defines.iter().any(|d| d == name.trim());
            stack.push(Cond {
                parent_active: active,
                this_active: active && cond,
                branch_taken: active && cond,
            });
            continue;
        }
        if let Some(expr) = trimmed.strip_prefix("#if ") {
            let active = is_active(&stack);
            let cond = eval_expr(expr, defines);
            stack.push(Cond {
                parent_active: active,
                this_active: active && cond,
                branch_taken: active && cond,
            });
            continue;
        }
        if let Some(expr) = trimmed.strip_prefix("#elif ") {
            if let Some(top) = stack.last_mut() {
                let cond = top.parent_active && !top.branch_taken && eval_expr(expr, defines);
                top.this_active = cond;
                top.branch_taken |= cond;
            }
            continue;
        }
        if trimmed == "#else" || trimmed.starts_with("#else ") {
            if let Some(top) = stack.last_mut() {
                let cond = top.parent_active && !top.branch_taken;
                top.this_active = cond;
                top.branch_taken |= cond;
            }
            continue;
        }
        if trimmed == "#endif" || trimmed.starts_with("#endif ") {
            stack.pop();
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("#define ") {
            if is_active(&stack) {
                let mut parts = rest.splitn(2, char::is_whitespace);
                if let (Some(name), Some(value)) = (parts.next(), parts.next()) {
                    local_replacements.push((name.to_string(), value.trim().to_string()));
                }
            }
            continue;
        }
        if trimmed.starts_with("#") {
            continue;
        }
        if !is_active(&stack) {
            continue;
        }
        if trimmed == "enable f16;" && matches!(dtype, DType::F32) {
            continue;
        }
        let mut expanded = line.to_string();
        for _ in 0..8 {
            let mut next = expanded.clone();
            for (name, value) in &local_replacements {
                next = replace_token(&next, name, value);
            }
            if next == expanded {
                break;
            }
            expanded = next;
        }
        out.push_str(&expanded);
        out.push('\n');
    }
    out
}

fn is_active(stack: &[Cond]) -> bool {
    stack.last().map(|cond| cond.this_active).unwrap_or(true)
}

fn eval_expr(expr: &str, defines: &[String]) -> bool {
    expr.split("||").any(|part| {
        let part = part.trim();
        if let Some(name) = part
            .strip_prefix("defined(")
            .and_then(|s| s.strip_suffix(')'))
        {
            defines.iter().any(|d| d == name)
        } else {
            defines.iter().any(|d| d == part)
        }
    })
}

fn replace_token(line: &str, name: &str, value: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut token = String::new();
    for ch in line.chars() {
        if ch == '_' || ch.is_ascii_alphanumeric() {
            token.push(ch);
        } else {
            if token == name {
                out.push_str(value);
            } else {
                out.push_str(&token);
            }
            token.clear();
            out.push(ch);
        }
    }
    if token == name {
        out.push_str(value);
    } else {
        out.push_str(&token);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{shader, BinaryOp, DType, ShaderOp, UnaryOp};

    #[test]
    fn preprocesses_rand_uniform_wgsl() {
        let source = super::rand_uniform_shader(DType::F32, 256).expect("rand_uniform f32");
        assert!(source.contains("splitmix64"));
        assert!(source.contains("@compute @workgroup_size(256)"));
    }

    #[test]
    fn preprocesses_rand_normal_wgsl() {
        let source = super::rand_normal_shader(DType::F32, 256).expect("rand_normal f32");
        assert!(source.contains("TWO_PI"));
        assert!(source.contains("log("));
    }

    #[test]
    fn preprocesses_unary_wgsl() {
        let source = shader(ShaderOp::Unary(UnaryOp::Relu), DType::F32, 256);
        assert!(source.contains("@compute @workgroup_size(256)"));
        assert!(source.contains("let res = select"));
        assert!(!source.contains("#ifdef"));
        assert!(!source.contains("#define"));
        assert!(!source.contains("TYPE("));
    }

    #[test]
    fn preprocesses_binary_wgsl() {
        let source = shader(ShaderOp::Binary(BinaryOp::Add), DType::F32, 256);
        assert!(source.contains("@compute @workgroup_size(256)"));
        assert!(source.contains("return a + b;"));
        assert!(!source.contains("#ifdef"));
        assert!(!source.contains("#define"));
        assert!(!source.contains("DataType"));
    }
}
