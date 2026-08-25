//! Worker-F2 diagnostic: replicate the real yolo-v8 (small) graph as a candle-nn
//! test fixture (model struct copied from candle-examples/examples/yolo-v8/model.rs,
//! tracing/report/NMS stripped), load yolov8s.safetensors, and diff the real-model
//! intermediates on CPU vs wgpu stage-by-stage. Also checks wgpu run-to-run
//! determinism. This is the "crime-scene" bisect F1 could not reach because the
//! real model needs VarBuilder (candle-nn), unreachable from a candle-core test.
//!
//! Run:
//!   cargo test -p candle-nn --features wgpu --test wgpu_yolo_dump_bisect -- --test-threads=1
//!
//! Env overrides:
//!   CANDLE_YOLO_MODEL  path to yolov8s.safetensors (default G:\models\candle-yolo-v8\yolov8s.safetensors)
//!   CANDLE_WGPU_POOL_FLUSH_PENDING  forwarded to the wgpu backend (used by the A/B experiment)
#![cfg(feature = "wgpu")]

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{batch_norm, conv2d, conv2d_no_bias, Conv2d, Conv2dConfig, Module, VarBuilder};

fn relerr(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (((x - y).abs()) as f64) / ((x.hypot(*y)) as f64).max(1e-6))
        .fold(0.0f64, f64::max)
}

fn maxabs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn downcast(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn lcg(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 13) % 4093) as f32 / 61.0 - 32.0
        })
        .collect()
}

// ---- yolo-v8 model fixture (copied from candle-examples) ----

struct Upsample {
    scale_factor: usize,
}
impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(self.scale_factor * h, self.scale_factor * w)
    }
}

struct ConvBlock {
    conv: Conv2d,
}
impl ConvBlock {
    fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize, stride: usize, padding: Option<usize>) -> Result<Self> {
        let padding = padding.unwrap_or(k / 2);
        let cfg = Conv2dConfig { padding, stride, groups: 1, dilation: 1, cudnn_fwd_algo: None };
        let bn = batch_norm(c2, 1e-3, vb.pp("bn"))?;
        let conv = conv2d_no_bias(c1, c2, k, cfg, vb.pp("conv"))?.absorb_bn(&bn)?;
        Ok(Self { conv })
    }
}
impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(&self.conv.forward(xs)?)
    }
}

struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    residual: bool,
}
impl Bottleneck {
    fn load(vb: VarBuilder, c1: usize, c2: usize, shortcut: bool) -> Result<Self> {
        let c_ = c2;
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c1, c_, 3, 1, None)?,
            cv2: ConvBlock::load(vb.pp("cv2"), c_, c2, 3, 1, None)?,
            residual: c1 == c2 && shortcut,
        })
    }
}
impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.cv2.forward(&self.cv1.forward(xs)?)?;
        if self.residual {
            xs + ys
        } else {
            Ok(ys)
        }
    }
}

struct C2f {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottleneck: Vec<Bottleneck>,
}
impl C2f {
    fn load(vb: VarBuilder, c1: usize, c2: usize, n: usize, shortcut: bool) -> Result<Self> {
        let c = (c2 as f64 * 0.5) as usize;
        let mut bottleneck = Vec::with_capacity(n);
        for idx in 0..n {
            bottleneck.push(Bottleneck::load(vb.pp(format!("bottleneck.{idx}")), c, c, shortcut)?);
        }
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c1, 2 * c, 1, 1, None)?,
            cv2: ConvBlock::load(vb.pp("cv2"), (2 + n) * c, c2, 1, 1, None)?,
            bottleneck,
        })
    }
}
impl Module for C2f {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.cv1.forward(xs)?;
        let mut ys = ys.chunk(2, 1)?;
        for m in self.bottleneck.iter() {
            ys.push(m.forward(ys.last().unwrap())?);
        }
        let zs = Tensor::cat(ys.as_slice(), 1)?;
        self.cv2.forward(&zs)
    }
}

struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
}
impl Sppf {
    fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize) -> Result<Self> {
        let c_ = c1 / 2;
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c1, c_, 1, 1, None)?,
            cv2: ConvBlock::load(vb.pp("cv2"), c_ * 4, c2, 1, 1, None)?,
            k,
        })
    }
}
impl Module for Sppf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.cv1.forward(xs)?;
        let xs2 = xs.pad_with_zeros(2, self.k / 2, self.k / 2)?.pad_with_zeros(3, self.k / 2, self.k / 2)?.max_pool2d_with_stride(self.k, 1)?;
        let xs3 = xs2.pad_with_zeros(2, self.k / 2, self.k / 2)?.pad_with_zeros(3, self.k / 2, self.k / 2)?.max_pool2d_with_stride(self.k, 1)?;
        let xs4 = xs3.pad_with_zeros(2, self.k / 2, self.k / 2)?.pad_with_zeros(3, self.k / 2, self.k / 2)?.max_pool2d_with_stride(self.k, 1)?;
        self.cv2.forward(&Tensor::cat(&[&xs, &xs2, &xs3, &xs4], 1)?)
    }
}

impl Sppf {
    // Returns (cv1_out, pad2, xs2, xs3, xs4, cat, cv2_out). pad2 is the
    // double-padded tensor right before the max-pool.
    #[allow(clippy::type_complexity)]
    fn forward_trace(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let x0 = self.cv1.forward(xs)?;
        let pad2 = x0.pad_with_zeros(2, self.k / 2, self.k / 2)?.pad_with_zeros(3, self.k / 2, self.k / 2)?;
        let xs2 = pad2.max_pool2d_with_stride(self.k, 1)?;
        let xs3 = xs2.pad_with_zeros(2, self.k / 2, self.k / 2)?.pad_with_zeros(3, self.k / 2, self.k / 2)?.max_pool2d_with_stride(self.k, 1)?;
        let xs4 = xs3.pad_with_zeros(2, self.k / 2, self.k / 2)?.pad_with_zeros(3, self.k / 2, self.k / 2)?.max_pool2d_with_stride(self.k, 1)?;
        let cat = Tensor::cat(&[&x0, &xs2, &xs3, &xs4], 1)?;
        let out = self.cv2.forward(&cat)?;
        Ok((x0, pad2, xs2, xs3, xs4, cat, out))
    }
}

struct Dfl {
    conv: Conv2d,
    num_classes: usize,
}
impl Dfl {
    fn load(vb: VarBuilder, num_classes: usize) -> Result<Self> {
        let conv = conv2d_no_bias(num_classes, 1, 1, Default::default(), vb.pp("conv"))?;
        Ok(Self { conv, num_classes })
    }
}
impl Module for Dfl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, _channels, anchors) = xs.dims3()?;
        let xs = xs.reshape((b_sz, 4, self.num_classes, anchors))?.transpose(2, 1)?;
        let xs = candle_nn::ops::softmax(&xs, 1)?;
        self.conv.forward(&xs)?.reshape((b_sz, 4, anchors))
    }
}

struct DarkNet {
    b1_0: ConvBlock,
    b1_1: ConvBlock,
    b2_0: C2f,
    b2_1: ConvBlock,
    b2_2: C2f,
    b3_0: ConvBlock,
    b3_1: C2f,
    b4_0: ConvBlock,
    b4_1: C2f,
    b5: Sppf,
}
impl DarkNet {
    fn load(vb: VarBuilder, w: f64, d: f64, r: f64) -> Result<Self> {
        Ok(Self {
            b1_0: ConvBlock::load(vb.pp("b1.0"), 3, (64. * w) as usize, 3, 2, Some(1))?,
            b1_1: ConvBlock::load(vb.pp("b1.1"), (64. * w) as usize, (128. * w) as usize, 3, 2, Some(1))?,
            b2_0: C2f::load(vb.pp("b2.0"), (128. * w) as usize, (128. * w) as usize, (3. * d).round() as usize, true)?,
            b2_1: ConvBlock::load(vb.pp("b2.1"), (128. * w) as usize, (256. * w) as usize, 3, 2, Some(1))?,
            b2_2: C2f::load(vb.pp("b2.2"), (256. * w) as usize, (256. * w) as usize, (6. * d).round() as usize, true)?,
            b3_0: ConvBlock::load(vb.pp("b3.0"), (256. * w) as usize, (512. * w) as usize, 3, 2, Some(1))?,
            b3_1: C2f::load(vb.pp("b3.1"), (512. * w) as usize, (512. * w) as usize, (6. * d).round() as usize, true)?,
            b4_0: ConvBlock::load(vb.pp("b4.0"), (512. * w) as usize, (512. * w * r) as usize, 3, 2, Some(1))?,
            b4_1: C2f::load(vb.pp("b4.1"), (512. * w * r) as usize, (512. * w * r) as usize, (3. * d).round() as usize, true)?,
            b5: Sppf::load(vb.pp("b5.0"), (512. * w * r) as usize, (512. * w * r) as usize, 5)?,
        })
    }
}

struct YoloV8Neck {
    up: Upsample,
    n1: C2f,
    n2: C2f,
    n3: ConvBlock,
    n4: C2f,
    n5: ConvBlock,
    n6: C2f,
}
impl YoloV8Neck {
    fn load(vb: VarBuilder, w: f64, d: f64, r: f64) -> Result<Self> {
        let n = (3. * d).round() as usize;
        Ok(Self {
            up: Upsample { scale_factor: 2 },
            n1: C2f::load(vb.pp("n1"), (512. * w * (1. + r)) as usize, (512. * w) as usize, n, false)?,
            n2: C2f::load(vb.pp("n2"), (768. * w) as usize, (256. * w) as usize, n, false)?,
            n3: ConvBlock::load(vb.pp("n3"), (256. * w) as usize, (256. * w) as usize, 3, 2, Some(1))?,
            n4: C2f::load(vb.pp("n4"), (768. * w) as usize, (512. * w) as usize, n, false)?,
            n5: ConvBlock::load(vb.pp("n5"), (512. * w) as usize, (512. * w) as usize, 3, 2, Some(1))?,
            n6: C2f::load(vb.pp("n6"), (512. * w * (1. + r)) as usize, (512. * w * r) as usize, n, false)?,
        })
    }
}

struct DetectionHead {
    dfl: Dfl,
    cv2: [(ConvBlock, ConvBlock, Conv2d); 3],
    cv3: [(ConvBlock, ConvBlock, Conv2d); 3],
    ch: usize,
    no: usize,
}
impl DetectionHead {
    fn load(vb: VarBuilder, nc: usize, filters: (usize, usize, usize)) -> Result<Self> {
        let ch = 16;
        let c1 = usize::max(filters.0, nc);
        let c2 = usize::max(filters.0 / 4, ch * 4);
        Ok(Self {
            dfl: Dfl::load(vb.pp("dfl"), ch)?,
            cv2: [
                Self::load_cv2(vb.pp("cv2.0"), c2, ch, filters.0)?,
                Self::load_cv2(vb.pp("cv2.1"), c2, ch, filters.1)?,
                Self::load_cv2(vb.pp("cv2.2"), c2, ch, filters.2)?,
            ],
            cv3: [
                Self::load_cv3(vb.pp("cv3.0"), c1, nc, filters.0)?,
                Self::load_cv3(vb.pp("cv3.1"), c1, nc, filters.1)?,
                Self::load_cv3(vb.pp("cv3.2"), c1, nc, filters.2)?,
            ],
            ch,
            no: nc + ch * 4,
        })
    }
    fn load_cv3(vb: VarBuilder, c1: usize, nc: usize, filter: usize) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        Ok((
            ConvBlock::load(vb.pp("0"), filter, c1, 3, 1, None)?,
            ConvBlock::load(vb.pp("1"), c1, c1, 3, 1, None)?,
            conv2d(c1, nc, 1, Default::default(), vb.pp("2"))?,
        ))
    }
    fn load_cv2(vb: VarBuilder, c2: usize, ch: usize, filter: usize) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        Ok((
            ConvBlock::load(vb.pp("0"), filter, c2, 3, 1, None)?,
            ConvBlock::load(vb.pp("1"), c2, c2, 3, 1, None)?,
            conv2d(c2, 4 * ch, 1, Default::default(), vb.pp("2"))?,
        ))
    }
}

fn make_anchors(xs0: &Tensor, xs1: &Tensor, xs2: &Tensor, (s0, s1, s2): (usize, usize, usize), grid_cell_offset: f64) -> Result<(Tensor, Tensor)> {
    let dev = xs0.device();
    let mut anchor_points = vec![];
    let mut stride_tensor = vec![];
    for (xs, stride) in [(xs0, s0), (xs1, s1), (xs2, s2)] {
        let (_, _, h, w) = xs.dims4()?;
        let sx = (Tensor::arange(0, w as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sy = (Tensor::arange(0, h as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sx = sx.reshape((1, sx.elem_count()))?.repeat((h, 1))?.flatten_all()?;
        let sy = sy.reshape((sy.elem_count(), 1))?.repeat((1, w))?.flatten_all()?;
        anchor_points.push(Tensor::stack(&[&sx, &sy], D::Minus1)?);
        stride_tensor.push((Tensor::ones(h * w, DType::F32, dev)? * stride as f64)?);
    }
    let anchor_points = Tensor::cat(anchor_points.as_slice(), 0)?;
    let stride_tensor = Tensor::cat(stride_tensor.as_slice(), 0)?.unsqueeze(1)?;
    Ok((anchor_points, stride_tensor))
}

fn dist2bbox(distance: &Tensor, anchor_points: &Tensor) -> Result<Tensor> {
    let chunks = distance.chunk(2, 1)?;
    let lt = &chunks[0];
    let rb = &chunks[1];
    let x1y1 = anchor_points.sub(lt)?;
    let x2y2 = anchor_points.add(rb)?;
    let c_xy = ((&x1y1 + &x2y2)? * 0.5)?;
    let wh = (&x2y2 - &x1y1)?;
    Tensor::cat(&[c_xy, wh], 1)
}

struct YoloV8 {
    net: DarkNet,
    fpn: YoloV8Neck,
    head: DetectionHead,
}
impl YoloV8 {
    fn load(vb: VarBuilder, m: &MultiplesRef, num_classes: usize) -> Result<Self> {
        Ok(Self {
            net: DarkNet::load(vb.pp("net"), m.w, m.d, m.r)?,
            fpn: YoloV8Neck::load(vb.pp("fpn"), m.w, m.d, m.r)?,
            head: DetectionHead::load(vb.pp("head"), num_classes, (m.f1(), m.f2(), m.f3()))?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (xs1, xs2, xs3) = self.net.forward_stage(xs)?;
        let (xs1, xs2, xs3) = self.fpn.forward_stage(&xs1, &xs2, &xs3)?;
        Ok(self.head.forward_stage(&xs1, &xs2, &xs3)?)
    }
}

// Forward with stage readback for localization. Each `out` array is filled with
// the flattened tensor of that stage.
struct Stages {
    net1: Tensor, // darknet x2 (p3), *before* fpn
    net2: Tensor, // darknet x3 (p4)
    net3: Tensor, // darknet x5 (p5)
    head1: Tensor,
    head2: Tensor,
    head3: Tensor,
    pred: Tensor,
}

impl DarkNet {
    fn forward_stage(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x1 = self.b1_1.forward(&self.b1_0.forward(xs)?)?;
        let x2 = self.b2_2.forward(&self.b2_1.forward(&self.b2_0.forward(&x1)?)?)?;
        let x3 = self.b3_1.forward(&self.b3_0.forward(&x2)?)?;
        let x4 = self.b4_1.forward(&self.b4_0.forward(&x3)?)?;
        let x5 = self.b5.forward(&x4)?;
        Ok((x2, x3, x5))
    }
    /// Fine trace: (b4_0_out, b4_1_out, sppf_cv1, sppf_pad2, sppf_xs2, sppf_xs3, sppf_xs4, sppf_cat, sppf_out).
    #[allow(clippy::type_complexity)]
    fn forward_fine(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let x1 = self.b1_1.forward(&self.b1_0.forward(xs)?)?;
        let x2 = self.b2_2.forward(&self.b2_1.forward(&self.b2_0.forward(&x1)?)?)?;
        let x3 = self.b3_1.forward(&self.b3_0.forward(&x2)?)?;
        let b4_0_out = self.b4_0.forward(&x3)?;
        let b4_1_out = self.b4_1.forward(&b4_0_out)?;
        let (cv1, pad2, xs2, xs3, xs4, cat, out) = self.b5.forward_trace(&b4_1_out)?;
        Ok((b4_0_out, b4_1_out, cv1, pad2, xs2, xs3, xs4, cat, out))
    }
}

impl YoloV8Neck {
    fn forward_stage(&self, p3: &Tensor, p4: &Tensor, p5: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.n1.forward(&Tensor::cat(&[&self.up.forward(p5)?, p4], 1)?)?;
        let head_1 = self.n2.forward(&Tensor::cat(&[&self.up.forward(&x)?, p3], 1)?)?;
        let head_2 = self.n4.forward(&Tensor::cat(&[&self.n3.forward(&head_1)?, &x], 1)?)?;
        let head_3 = self.n6.forward(&Tensor::cat(&[&self.n5.forward(&head_2)?, p5], 1)?)?;
        Ok((head_1, head_2, head_3))
    }
}

impl DetectionHead {
    fn forward_stage(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<Tensor> {
        let forward_cv = |xs: &Tensor, i: usize| {
            let xs_2 = self.cv2[i].0.forward(xs)?;
            let xs_2 = self.cv2[i].1.forward(&xs_2)?;
            let xs_2 = self.cv2[i].2.forward(&xs_2)?;
            let xs_3 = self.cv3[i].0.forward(xs)?;
            let xs_3 = self.cv3[i].1.forward(&xs_3)?;
            let xs_3 = self.cv3[i].2.forward(&xs_3)?;
            Tensor::cat(&[&xs_2, &xs_3], 1)
        };
        let xs0 = forward_cv(xs0, 0)?;
        let xs1 = forward_cv(xs1, 1)?;
        let xs2 = forward_cv(xs2, 2)?;

        let (anchors, strides) = make_anchors(&xs0, &xs1, &xs2, (8, 16, 32), 0.5)?;
        let anchors = anchors.transpose(0, 1)?.unsqueeze(0)?;
        let strides = strides.transpose(0, 1)?;

        let reshape = |xs: &Tensor| {
            let d = xs.dim(0)?;
            let el = xs.elem_count();
            xs.reshape((d, self.no, el / (d * self.no)))
        };
        let ys0 = reshape(&xs0)?;
        let ys1 = reshape(&xs1)?;
        let ys2 = reshape(&xs2)?;

        let x_cat = Tensor::cat(&[ys0, ys1, ys2], 2)?;
        let box_ = x_cat.i((.., ..self.ch * 4))?;
        let cls = x_cat.i((.., self.ch * 4..))?;

        let dbox = dist2bbox(&self.dfl.forward(&box_)?, &anchors)?;
        let dbox = dbox.broadcast_mul(&strides)?;
        Tensor::cat(&[dbox, candle_nn::ops::sigmoid(&cls)?], 1)
    }
}

impl YoloV8 {
    // Returns (net, fpn, pred) stage tensors, forcing a readback at each stage on
    // the GPU side. NOTE: the readback implies a synchronize, so this localizes a
    // corruption that ALREADY happened; a pure in-flight reuse race that a drain
    // would have prevented is NOT captured here (that is handled by `forward`).
    fn forward_trace(&self, xs: &Tensor) -> Result<Stages> {
        let (x2, x3, x5) = self.net.forward_stage(xs)?;
        let (h1, h2, h3) = self.fpn.forward_stage(&x2, &x3, &x5)?;
        let pred = self.head.forward_stage(&h1, &h2, &h3)?;
        Ok(Stages { net1: x2, net2: x3, net3: x5, head1: h1, head2: h2, head3: h3, pred })
    }
}

struct MultiplesRef {
    w: f64,
    d: f64,
    r: f64,
}
impl MultiplesRef {
    fn s() -> Self {
        Self { w: 0.50, d: 0.33, r: 2.0 }
    }
    fn f1(&self) -> usize {
        (256. * self.w) as usize
    }
    fn f2(&self) -> usize {
        (512. * self.w) as usize
    }
    fn f3(&self) -> usize {
        (512. * self.w * self.r) as usize
    }
}

fn model_path() -> String {
    std::env::var("CANDLE_YOLO_MODEL").unwrap_or_else(|_| r"G:\models\candle-yolo-v8\yolov8s.safetensors".to_string())
}

// Preprocessed real input (same form the candle-examples yolo main.rs builds):
// resized to 640x{416}, NHWC->NCHW, /255, f32, contiguous [1,3,416,640]. Written
// with python+PIL (BICUBIC). Read as raw f32 (IEEE little-endian).
fn read_real_input() -> Result<(Vec<f32>, Vec<usize>)> {
    let path = std::env::var("CANDLE_YOLO_IMG_BIN").unwrap_or_else(|_| "/tmp/t1/bike_raw_nchw_f32.bin".to_string());
    let bytes = std::fs::read(&path).map_err(candle::Error::wrap)?;
    let n = (bytes.len() / 4) as usize;
    let mut v = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        v.push(f32::from_bits(bits));
    }
    Ok((v, vec![1, 3, 416, 640]))
}

#[test]
fn yolo_real_model_cpu_vs_wgpu_bisect() -> Result<()> {
    let cpu = Device::Cpu;
    // wgpu best-effort.
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    let mp = model_path();
    if std::path::Path::new(&mp).exists() {
        // Full real-model bisect (needs the safetensors + a preprocessed input).
        run_real_model_bisect(&cpu, &wgpu)
    } else {
        eprintln!("[bisect] model file not found at {mp}; skipping real-model bisect");
        Ok(())
    }
}

// The actual full-model comparison (separated so the regression gate can skip
// when the large safetensors file is not checked out).
fn run_real_model_bisect(cpu: &Device, wgpu: &Device) -> Result<()> {
    // Build two model copies (cpu + wgpu) from the same safetensors file.
    let cpu_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path()], DType::F32, cpu) }?;
    let wgpu_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path()], DType::F32, wgpu) }?;
    let cpu_model = YoloV8::load(cpu_vb, &MultiplesRef::s(), 80)?;
    let wgpu_model = YoloV8::load(wgpu_vb, &MultiplesRef::s(), 80)?;

    // Deterministic input (same on both devices) = the real preprocessed image.
    let (data, shape) = read_real_input()?;
    let input_cpu = Tensor::from_vec(data.clone(), shape.as_slice(), cpu)?.to_dtype(DType::F32)?;
    let input_wgpu = Tensor::from_vec(data, shape.as_slice(), wgpu)?.to_dtype(DType::F32)?;
    eprintln!("[bisect] input shape={shape:?} n={}", shape[0] * shape[1] * shape[2] * shape[3]);

    eprintln!("[bisect] model loaded; running full forward (no stage readback)");

    // FIRST wgpu computation (fresh process, empty buffer pool) — read back once.
    // This is the cleanest signal: is a brand-new forward correct?
    let pred_wgpu = downcast(&wgpu_model.forward(&input_wgpu)?)?;
    let pred_cpu = downcast(&cpu_model.forward(&input_cpu)?)?;
    let r0 = relerr(&pred_cpu, &pred_wgpu);
    let ma0 = maxabs(&pred_cpu, &pred_wgpu);
    eprintln!("[bisect] FRESH(1st) wgpu pred cpu-vs-wgpu relerr={r0:.6} maxabs={ma0:.6}");

    // Determinism of wgpu full forward across REUSING runs.
    for run in 1..4 {
        let p = wgpu_model.forward(&input_wgpu)?;
        let pv = downcast(&p)?;
        let d = relerr(&pred_wgpu, &pv);
        eprintln!("[bisect] wgpu full-forward reuse-run{run}: relerr vs 1st = {d:.6}");
    }
    // pred is [1, 84, 8400] -> channel c is slices [c*8400, (c+1)*8400).
    let anchors = pred_cpu.len() / 84;
    eprintln!("[bisect] pred shape anchors={anchors}");
    let r = relerr(&pred_cpu, &pred_wgpu);
    let ma = maxabs(&pred_cpu, &pred_wgpu);
    eprintln!("[bisect] FULL pred cpu-vs-wgpu relerr={r:.6} maxabs={ma:.6}");

    // Per-channel breakdown: which output channel diverges.
    let mut ch_diff: Vec<f32> = Vec::with_capacity(84);
    for c in 0..84 {
        let a = &pred_cpu[c * anchors..(c + 1) * anchors];
        let b = &pred_wgpu[c * anchors..(c + 1) * anchors];
        ch_diff.push(maxabs(a, b));
    }
    for c in 0..84 {
        if ch_diff[c] > 1e-3 {
            eprintln!("[bisect] pred channel {c:2} maxabs={:.6}", ch_diff[c]);
        }
    }
    // Class channels are indices 4..84 (predictions after dbox). motorbike (class
    // 3) is channel 7. Report its cpu/vgpu means.
    for (name, c) in [("bbox0", 0), ("bbox1", 1), ("bbox2", 2), ("bbox3", 3), ("cls_3(motorbike)", 7), ("cls_0(person)", 4), ("cls_1(bicycle)", 5), ("cls_2(car)", 6)] {
        let a = &pred_cpu[c * anchors..(c + 1) * anchors];
        let b = &pred_wgpu[c * anchors..(c + 1) * anchors];
        let ma: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let amean: f32 = a.iter().sum::<f32>() / anchors as f32;
        let bmean: f32 = b.iter().sum::<f32>() / anchors as f32;
        eprintln!("[bisect] ch {name:18} cpu_mean={amean:.4} wgpu_mean={bmean:.4} maxabs={ma:.4}");
    }

    // Stage-level trace (forces a readback per stage; localizes which stage's
    // output is already wrong even with drains between stages).
    eprintln!("[bisect] running traced forward (stage readback)");
    let s_cpu = cpu_model.forward_trace(&input_cpu)?;
    let s_wgpu = wgpu_model.forward_trace(&input_wgpu)?;
    let stage_labels = ["net.x2", "net.x3", "net.x5", "head1", "head2", "head3", "pred"];
    let cpu_stages = [
        downcast(&s_cpu.net1)?,
        downcast(&s_cpu.net2)?,
        downcast(&s_cpu.net3)?,
        downcast(&s_cpu.head1)?,
        downcast(&s_cpu.head2)?,
        downcast(&s_cpu.head3)?,
        downcast(&s_cpu.pred)?,
    ];
    let wgpu_stages = [
        downcast(&s_wgpu.net1)?,
        downcast(&s_wgpu.net2)?,
        downcast(&s_wgpu.net3)?,
        downcast(&s_wgpu.head1)?,
        downcast(&s_wgpu.head2)?,
        downcast(&s_wgpu.head3)?,
        downcast(&s_wgpu.pred)?,
    ];
    for i in 0..stage_labels.len() {
        let r = relerr(&cpu_stages[i], &wgpu_stages[i]);
        let ma = maxabs(&cpu_stages[i], &wgpu_stages[i]);
        eprintln!("[bisect] stage {:8} cpu-vs-wgpu relerr={r:.6} maxabs={ma:.6} (n={})", stage_labels[i], cpu_stages[i].len());
    }

    // Fine per-op trace inside DarkNet b4/b5 (SPPF) region.
    eprintln!("[bisect] running fine b4/b5 trace");
    let f_cpu = cpu_model.net.forward_fine(&input_cpu)?;
    let f_wgpu = wgpu_model.net.forward_fine(&input_wgpu)?;
    let fine_labels = ["b4_0_out", "b4_1_out", "sppf_cv1", "sppf_pad2", "sppf_xs2", "sppf_xs3", "sppf_xs4", "sppf_cat", "sppf_out"];
    let fine_cpu = [
        downcast(&f_cpu.0)?,
        downcast(&f_cpu.1)?,
        downcast(&f_cpu.2)?,
        downcast(&f_cpu.3)?,
        downcast(&f_cpu.4)?,
        downcast(&f_cpu.5)?,
        downcast(&f_cpu.6)?,
        downcast(&f_cpu.7)?,
        downcast(&f_cpu.8)?,
    ];
    let fine_wgpu = [
        downcast(&f_wgpu.0)?,
        downcast(&f_wgpu.1)?,
        downcast(&f_wgpu.2)?,
        downcast(&f_wgpu.3)?,
        downcast(&f_wgpu.4)?,
        downcast(&f_wgpu.5)?,
        downcast(&f_wgpu.6)?,
        downcast(&f_wgpu.7)?,
        downcast(&f_wgpu.8)?,
    ];
    for i in 0..fine_labels.len() {
        let r = relerr(&fine_cpu[i], &fine_wgpu[i]);
        let ma = maxabs(&fine_cpu[i], &fine_wgpu[i]);
        eprintln!("[bisect] fine {:10} cpu-vs-wgpu relerr={r:.6} maxabs={ma:.6} (n={})", fine_labels[i], fine_cpu[i].len());
    }

    // Assertion: real-model intermediate parity. This FAILS on the buggy backend
    // (the crime scene) and PASSES after the fix. Use a loose f32-reduction
    // tolerance; the motorbike-zeroed symptom is a gross error far above it.
    assert!(r < 5e-3, "yolo real-model pred cpu-vs-wgpu relerr={r:.5} >= 5e-3 (bug present)");
    Ok(())
}

// Isolation: SD UNet Downsamp2D uses avg_pool2d(2). Verify the native F32
// avg-pool shader is exact (kernel=2 stride=2), to rule it in/out for the SD
// green-channel defect.
#[test]
fn avgpool_kernel2_isolated() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    for (name, shape) in [
        ("unet_deep [1,1280,8,8]", vec![1usize, 1280, 8, 8]),
        ("unet_mid [1,320,16,16]", vec![1usize, 320, 16, 16]),
    ] {
        let n = shape.iter().product::<usize>();
        let data = lcg(555, n);
        let c = Tensor::from_vec(data.clone(), shape.as_slice(), &cpu)?.to_dtype(DType::F32)?;
        let g = Tensor::from_vec(data, shape.as_slice(), &wgpu)?.to_dtype(DType::F32)?;
        let co = c.avg_pool2d(2)?;
        let go = g.avg_pool2d(2)?;
        let cv = downcast(&co)?;
        let gv = downcast(&go)?;
        let r = relerr(&cv, &gv);
        let ma = maxabs(&cv, &gv);
        eprintln!("[avg2] {name:20} cpu-vs-wgpu relerr={r:.5} maxabs={ma:.5}");
        assert!(ma < 1e-3 && r < 1e-3, "[{name}] avg_pool2d(2) wgpu-vs-cpu relerr={r:.5} maxabs={ma:.5}");
    }
    Ok(())
}

// Isolation: VAE-decode final-stage conv (512->3, k3 pad1). Tests whether the
// green output channel (channel 1) is zeroed on wgpu — the suspected SD defect.
#[test]
fn vae_conv_out_green() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    let nw = 3 * 512 * 3 * 3;
    let wdata = lcg(7, nw);
    let cfg = candle_nn::Conv2dConfig { padding: 1, ..Default::default() };
    let cpu_conv = candle_nn::Conv2d::new(
        Tensor::from_vec(wdata.clone(), [3, 512, 3, 3].as_slice(), &cpu)?.to_dtype(DType::F32)?,
        None,
        cfg,
    );
    let wgpu_conv = candle_nn::Conv2d::new(
        Tensor::from_vec(wdata, [3, 512, 3, 3].as_slice(), &wgpu)?.to_dtype(DType::F32)?,
        None,
        cfg,
    );
    let data = lcg(9, 512 * 32 * 32);
    let in_cpu = Tensor::from_vec(data.clone(), [1, 512, 32, 32].as_slice(), &cpu)?.to_dtype(DType::F32)?;
    let in_wgpu = Tensor::from_vec(data, [1, 512, 32, 32].as_slice(), &wgpu)?.to_dtype(DType::F32)?;
    let out_cpu = cpu_conv.forward(&in_cpu)?;
    let out_wgpu = wgpu_conv.forward(&in_wgpu)?;
    let c = downcast(&out_cpu)?;
    let g = downcast(&out_wgpu)?;
    let ch = 32 * 32;
    // channel 0 = R, 1 = G, 2 = B (NCHW -> per-channel slice).
    for (name, idx) in [("R", 0), ("G", 1), ("B", 2)] {
        let c_slice = &c[idx * ch..(idx + 1) * ch];
        let g_slice = &g[idx * ch..(idx + 1) * ch];
        let m: f32 = c_slice.iter().zip(g_slice.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let cmean: f32 = c_slice.iter().sum::<f32>() / ch as f32;
        let gmean: f32 = g_slice.iter().sum::<f32>() / ch as f32;
        eprintln!("[vae] ch {name} cpu_mean={cmean:.4} wgpu_mean={gmean:.4} maxabs={m:.4}");
    }
    Ok(())
}

// Isolation: does F32 max_pool2d_with_stride alone corrupt on wgpu vs cpu, and
// does the corruption depend on shape (bc vs w) — the suspected im2col gate.
#[test]
fn maxpool_isolated_shapes() -> Result<()> {
    let cpu = Device::Cpu;
    let Some(wgpu) = Device::new_wgpu(0).ok() else {
        eprintln!("wgpu unavailable; skipping");
        return Ok(());
    };
    for (name, shape) in [
        ("yolo_sppf [1,256,20,26]", vec![1usize, 256, 20, 26]),
        ("yolo_sppf [1,256,26,20]", vec![1usize, 256, 26, 20]),
        ("yolo_sppf [1,256,40,52]", vec![1usize, 256, 40, 52]),
        ("resnet_pool [1,64,112,112]", vec![1usize, 64, 112, 112]),
    ] {
        let n = shape.iter().product::<usize>();
        let data = lcg(123, n);
        // pad 2 each side (k/2=2 for k=5) then maxpool 5x5 stride 1 (SPPF style).
        let cpu_t = Tensor::from_vec(data.clone(), shape.as_slice(), &cpu)?.to_dtype(DType::F32)?;
        let gpu_t = Tensor::from_vec(data, shape.as_slice(), &wgpu)?.to_dtype(DType::F32)?;
        let cpu_out = cpu_t
            .pad_with_zeros(2, 2, 2)?
            .pad_with_zeros(3, 2, 2)?
            .max_pool2d_with_stride(5, 1)?;
        let gpu_out = gpu_t
            .pad_with_zeros(2, 2, 2)?
            .pad_with_zeros(3, 2, 2)?
            .max_pool2d_with_stride(5, 1)?;
        let c = downcast(&cpu_out)?;
        let g = downcast(&gpu_out)?;
        let r = relerr(&c, &g);
        let ma = maxabs(&c, &g);
        eprintln!("[maxpool] {name:24} cpu-vs-wgpu relerr={r:.5} maxabs={ma:.5}");
        // The im2col F32 pool path returned maxabs ~35 (the input's global max)
        // for every shape where b*c > width. The native pool path must be exact
        // for every shape, including resnet (b*c <= width).
        assert!(ma < 1e-3 && r < 1e-3, "[{name}] max_pool2d wgpu-vs-cpu relerr={r:.5} maxabs={ma:.5} (pool corruption)");
    }
    Ok(())
}
