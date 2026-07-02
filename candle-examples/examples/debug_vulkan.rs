use candle::{DType, Device};
use candle_nn::VarBuilder;

fn main() -> anyhow::Result<()> {
    let device = Device::new_vulkan(0)?;
    let cpu = Device::Cpu;
    let path = std::path::Path::new("/home/mod479711/Downloads/models/Qwen3-0.6B/model.safetensors");

    println!("=== Test 1: mmap -> CPU BF16 -> Vulkan BF16 -> to_dtype F32 ===");
    {
        let vb_cpu = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::BF16, &cpu)? };
        let w_cpu = vb_cpu.get_unchecked("model.embed_tokens.weight")?;
        let w_v = w_cpu.to_device(&device)?;
        let w_f32 = w_v.to_dtype(DType::F32)?;
        let flat = w_f32.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let zeros = flat.iter().filter(|&&x| x == 0.0).count();
        println!("  cpu_bf16->vulkan_bf16->f32: total={} zeros={}", flat.len(), zeros);
    }

    println!("\n=== Test 2: mmap -> Vulkan BF16 directly -> to_dtype F32 ===");
    {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::BF16, &device)? };
        let w = vb.get_unchecked("model.embed_tokens.weight")?;
        let w_f32 = w.to_dtype(DType::F32)?;
        let flat = w_f32.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let zeros = flat.iter().filter(|&&x| x == 0.0).count();
        println!("  mmap->vulkan_bf16->f32: total={} zeros={}", flat.len(), zeros);
    }

    println!("\n=== Test 3: mmap -> Vulkan as F32 ===");
    {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? };
        let w = vb.get((151936, 1024), "model.embed_tokens.weight")?;
        let flat = w.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let zeros = flat.iter().filter(|&&x| x == 0.0).count();
        let first_zero = flat.iter().position(|&x| x == 0.0);
        println!("  mmap->vulkan_f32: total={} zeros={} first_zero={:?}", flat.len(), zeros, first_zero);
    }

    println!("\n=== Test 4: mmap -> CPU -> Vulkan ===");
    {
        let vb_cpu = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &cpu)? };
        let w_cpu = vb_cpu.get((151936, 1024), "model.embed_tokens.weight")?;
        let w_v = w_cpu.to_device(&device)?;
        let flat = w_v.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let zeros = flat.iter().filter(|&&x| x == 0.0).count();
        println!("  mmap->cpu->vulkan: total={} zeros={}", flat.len(), zeros);
    }

    println!("\nAll tests passed if zeros=0 for all.");
    Ok(())
}
