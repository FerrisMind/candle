use candle::{Device, Result, Tensor, D};
use candle_nn::ops::softmax;

#[test]
fn soft_vk() -> Result<()> {
    let vk = Device::new_vulkan(0)?;
    let t = Tensor::from_vec(vec![2.5f32], 1, &vk)?;
    let s = softmax(&t, 0)?;
    vk.synchronize()?;
    println!("vk out {:?}", s.to_vec1::<f32>()?);
    let t2 = Tensor::from_vec(vec![1.0f32,2.0,3.0,4.0], (1,1,1,4), &vk)?;
    let s2 = softmax(&t2, D::Minus1)?;
    vk.synchronize()?;
    println!("vk 4 {:?}", s2.flatten_all()?.to_vec1::<f32>()?);
    Ok(())
}
