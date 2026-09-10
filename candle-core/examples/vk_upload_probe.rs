// Break down the H2D upload cost with a chrome trace of the vulkan.* spans.
use candle_core::{Device, Tensor};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

fn main() -> anyhow::Result<()> {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().file("vk_upload_trace.json").build();
    let _subscriber = tracing_subscriber::registry().with(chrome_layer).set_default();
    let vk = Device::new_vulkan(0)?;
    let data: Vec<f32> = (0..512 * 1024).map(|i| i as f32 / 1024.0).collect();

    for _ in 0..10 { let _ = Tensor::from_slice(&data, (512, 1024), &vk)?; }
    for i in 0..50 {
        let _span = tracing::info_span!("probe_iter", i).entered();
        let _ = Tensor::from_slice(&data, (512, 1024), &vk)?;
    }

    // Allocation-only: no host bytes are moved, isolates device buffer
    // creation + the deferred free cycle from the copy path.
    for _ in 0..10 { let _ = Tensor::zeros((512, 1024), candle_core::DType::F32, &vk)?; }
    for i in 0..50 {
        let _span = tracing::info_span!("probe_alloc", i).entered();
        let _ = Tensor::zeros((512, 1024), candle_core::DType::F32, &vk)?;
    }

    // CPU-side tensor creation only, no device involvement.
    for i in 0..50 {
        let _span = tracing::info_span!("probe_cpu_tensor", i).entered();
        let _ = Tensor::from_slice(&data, (512, 1024), &candle_core::Device::Cpu)?;
    }

    // Raw Vec clone reference: isolates candle CPU-path overhead from the
    // machine's actual 2MB memcpy speed.
    for i in 0..50 {
        let _span = tracing::info_span!("probe_raw_clone", i).entered();
        let _ = data.clone();
    }

    // Pre-built CPU tensor: to_device only (alloc + upload + record).
    for _ in 0..10 {
        let t = Tensor::from_slice(&data, (512, 1024), &candle_core::Device::Cpu)?;
        let _ = t.to_device(&vk)?;
    }
    for i in 0..50 {
        let _span = tracing::info_span!("probe_to_device", i).entered();
        let t = Tensor::from_slice(&data, (512, 1024), &candle_core::Device::Cpu)?;
        let _ = t.to_device(&vk)?;
    }

    // Full round trip: upload + GPU copy completion + readback. Upper bound
    // on the true cost of one upload when the consumer needs the data.
    for i in 0..50 {
        let _span = tracing::info_span!("probe_roundtrip", i).entered();
        let t = Tensor::from_slice(&data, (512, 1024), &vk)?;
        let _ = t.to_dtype(candle_core::DType::F32)?.to_device(&candle_core::Device::Cpu)?;
    }

    let _ = Tensor::from_slice(&[0f32], (1,), &vk)?;
    drop(guard);
    println!("trace written");
    Ok(())
}
