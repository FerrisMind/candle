use candle::{Device, Result, Tensor, D};
use candle_nn::ops::softmax;

#[test]
fn soft1() -> Result<()> {
    let wg = Device::new_wgpu(0)?;
    // print params-like: create [2.5] and softmax
    let t = Tensor::from_vec(vec![2.5f32], 1, &wg)?;
    println!("in {:?}", t.to_vec1::<f32>()?);
    let s = softmax(&t, 0)?;
    wg.synchronize()?;
    println!("out {:?}", s.to_vec1::<f32>()?);
    // also try contiguous after ops chain like model
    let t2 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1,6,1,1), &wg)?;
    let s2 = softmax(&t2, D::Minus1)?;
    wg.synchronize()?;
    println!("attn1 {:?}", s2.flatten_all()?.to_vec1::<f32>()?);
    // force flush between each?
    let t3 = Tensor::from_vec(vec![4.38f32], (1,1,1,1), &wg)?;
    let s3 = softmax(&t3, D::Minus1)?;
    wg.synchronize()?;
    println!("1x1x1x1 {:?}", s3.to_vec1::<f32>()?);
    Ok(())
}
