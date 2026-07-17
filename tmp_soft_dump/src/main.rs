fn main() {
    let s = candle_wgpu_kernels::softmax_shader(256).unwrap();
    for (i, line) in s.lines().enumerate() {
        let l = line;
        if i < 15 || l.contains("workgroup") || l.contains("@group") || l.contains("var<storage")
            || l.contains("fn main") || l.contains("ne0") || l.contains("scratch") || l.contains("WG_SIZE")
            || l.contains("array<f32") {
            println!("{:4}|{}", i+1, l);
        }
    }
    println!("LEN={}", s.len());
}
