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
pub enum UnaryOp {
    Abs,
    Ceil,
    Cos,
    Exp,
    Floor,
    Gelu,
    GeluErf,
    GeluQuick,
    Hardsigmoid,
    Hardswish,
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
            Self::Cos => "COS",
            Self::Exp => "EXP",
            Self::Floor => "FLOOR",
            Self::Gelu => "GELU",
            Self::GeluErf => "GELU_ERF",
            Self::GeluQuick => "GELU_QUICK",
            Self::Hardsigmoid => "HARDSIGMOID",
            Self::Hardswish => "HARDSWISH",
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
        if trimmed == "#else" {
            if let Some(top) = stack.last_mut() {
                let cond = top.parent_active && !top.branch_taken;
                top.this_active = cond;
                top.branch_taken |= cond;
            }
            continue;
        }
        if trimmed == "#endif" {
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
        for (name, value) in &local_replacements {
            expanded = replace_token(&expanded, name, value);
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
