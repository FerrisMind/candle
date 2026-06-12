fn main() -> candle_core::Result<()> {
    use candle_core::{Device, Tensor};
    let device = Device::new_vulkan(0)?;
    let a = Tensor::from_slice(
        &[16_777_217u32, 100, 7, 4_000_000_000, 13, 6],
        (2, 3),
        &device,
    )?;
    let b = Tensor::from_slice(&[255u32, 7, 3, 1, 13, 5], (2, 3), &device)?;
    println!("u32 add: {:?}", (&a + &b)?.to_vec2::<u32>()?);
    println!("u32 sub: {:?}", (&a - &b)?.to_vec2::<u32>()?);
    println!("u32 mul: {:?}", (&a * &b)?.to_vec2::<u32>()?);
    println!("u32 div: {:?}", (&a / &b)?.to_vec2::<u32>()?);
    println!("u32 max: {:?}", a.maximum(&b)?.to_vec2::<u32>()?);
    println!("u32 min: {:?}", a.minimum(&b)?.to_vec2::<u32>()?);
    let a8 = Tensor::from_slice(&[20u8, 15, 50, 12, 13, 5], (2, 3), &device)?;
    let b8 = Tensor::from_slice(&[10u8, 7, 4, 1, 13, 5], (2, 3), &device)?;
    println!("u8 add: {:?}", (&a8 + &b8)?.to_vec2::<u8>()?);
    println!("u8 mul: {:?}", (&a8 * &b8)?.to_vec2::<u8>()?);
    println!("u8 max: {:?}", a8.maximum(&b8)?.to_vec2::<u8>()?);
    let a64 = Tensor::from_slice(
        &[-3_000_000_000i64, 3_000_000_000, -7, 42, -1, 6_700_417],
        (2, 3),
        &device,
    )?;
    let b64 = Tensor::from_slice(
        &[3_000_000_000i64, 3_000_000_000, 100, -42, -1, 641],
        (2, 3),
        &device,
    )?;
    println!("i64 add: {:?}", (&a64 + &b64)?.to_vec2::<i64>()?);
    println!("i64 sub: {:?}", (&a64 - &b64)?.to_vec2::<i64>()?);
    println!("i64 mul: {:?}", (&a64 * &b64)?.to_vec2::<i64>()?);
    println!("i64 max: {:?}", a64.maximum(&b64)?.to_vec2::<i64>()?);
    Ok(())
}
