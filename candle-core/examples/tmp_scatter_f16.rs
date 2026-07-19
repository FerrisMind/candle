// quick f16 scatter_add parity
use candle::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    let dev = Device::new_vulkan(0)?;
    // dim=0 scatter like deepstack: (rows, cols)
    let dst = Tensor::zeros((4, 4), DType::F16, &dev)?;
    let ids = Tensor::from_slice(
        &[1u32, 1u32, 3u32, 0u32, 2u32, 2u32, 1u32, 3u32],
        (2, 4),
        &dev,
    )?;
    // wait - for scatter_add dim 0, ids and src same shape as the "source rows"
    // simpler: last-dim scatter like the f32 smoke
    let dst = Tensor::zeros((1, 8), DType::F16, &dev)?;
    let ids = Tensor::from_slice(&[1u32, 1, 3, 6], (1, 4), &dev)?;
    let src = Tensor::from_slice(&[1.0f32, 2.5, 4.0, 0.5], (1, 4), &dev)?.to_dtype(DType::F16)?;
    let out = dst.scatter_add(&ids, &src, 1)?;
    dev.synchronize()?;
    let v = out.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    println!("vk f16 scatter last-dim: {:?}", v);
    // expected: [0, 3.5, 0, 4, 0, 0, 0.5, 0]

    // dim=0 like deepstack
    let base = Tensor::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        (3, 4),
        &dev,
    )?
    .to_dtype(DType::F16)?;
    // add to rows 1 and 0
    let row_ids = Tensor::from_slice(&[1u32, 1, 1, 1, 0u32, 0, 0, 0], (2, 4), &dev)?;
    let add = Tensor::from_slice(
        &[10.0f32, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0],
        (2, 4),
        &dev,
    )?
    .to_dtype(DType::F16)?;
    let out2 = base.scatter_add(&row_ids, &add, 0)?;
    dev.synchronize()?;
    println!(
        "vk f16 scatter dim0: {:?}",
        out2.to_dtype(DType::F32)?.to_vec2::<f32>()?
    );
    // expected rows: [2,3,4,5], [15,16,17,18], [9,10,11,12]
    Ok(())
}
