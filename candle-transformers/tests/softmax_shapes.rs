use candle::{Device, Result, Tensor, D};
use candle_nn::ops::softmax;

#[test]
fn softmax_shapes() -> Result<()> {
    let cpu = Device::Cpu;
    let wg = Device::new_wgpu(0)?;
    let cases: &[(&[usize], &str)] = &[
        (&[1], "1d1"),
        (&[4], "1d4"),
        (&[1,1], "2d1x1"),
        (&[1,4], "2d1x4"),
        (&[1,6,1,1], "attn1"),
        (&[1,6,4,4], "attn4"),
        (&[1,6,1,8], "attn1x8"),
    ];
    for (shape, name) in cases {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let t_c = Tensor::from_vec(data.clone(), *shape, &cpu)?;
        let t_g = Tensor::from_vec(data, *shape, &wg)?;
        let s_c = softmax(&t_c, D::Minus1)?;
        let s_g = softmax(&t_g, D::Minus1)?;
        wg.synchronize()?;
        let vc = s_c.flatten_all()?.to_vec1::<f32>()?;
        let vg = s_g.flatten_all()?.to_vec1::<f32>()?;
        let md = vc.iter().zip(vg.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32, f32::max);
        let sum_c: f32 = vc.iter().sum();
        let sum_g: f32 = vg.iter().sum();
        println!("{name:?} shape={shape:?} maxdiff={md:.3e} sum_cpu={sum_c:.4} sum_gpu={sum_g:.4} first_cpu={} first_gpu={}", vc[0], vg[0]);
    }
    Ok(())
}
