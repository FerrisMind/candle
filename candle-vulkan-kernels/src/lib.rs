mod comp {
    use super::Module;

    include!(concat!(env!("OUT_DIR"), "/comp.rs"));
}

mod spv {
    use super::SpirvModule;

    include!(concat!(env!("OUT_DIR"), "/spv.rs"));
}

#[derive(Debug, Clone, Copy)]
pub struct Module {
    name: &'static str,
    source: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct SpirvModule {
    name: &'static str,
    words: &'static [u32],
}

impl Module {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn source(&self) -> &'static str {
        self.source
    }
}

impl SpirvModule {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn words(&self) -> &'static [u32] {
        self.words
    }
}

pub use comp::*;

pub fn get(name: &str) -> Option<Module> {
    ALL_MODULES
        .iter()
        .copied()
        .find(|module| module.name == name)
}

pub fn spirv(name: &str) -> Option<&'static [u32]> {
    spv::ALL_SPIRV_MODULES
        .iter()
        .find(|module| module.name == name)
        .map(|module| module.words)
}

pub fn spirv_modules() -> &'static [SpirvModule] {
    spv::ALL_SPIRV_MODULES
}

#[cfg(test)]
mod tests {
    use super::spirv;

    #[test]
    fn exposes_unary_comp_spirv() {
        let words = spirv("relu_f32").unwrap();
        assert!(words.len() > 16);
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn exposes_binary_comp_spirv() {
        let words = spirv("add_f32_f32_f32").unwrap();
        assert!(words.len() > 16);
        assert_eq!(words[0], 0x0723_0203);
    }
}
