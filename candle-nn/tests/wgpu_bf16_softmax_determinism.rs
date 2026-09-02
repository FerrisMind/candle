// Regression test for the wgpu BF16 softmax cross-process nondeterminism.
//
// The wgpu BF16 softmax kernel packs two BF16 elements per u32 word and runs the
// softmax IN-PLACE, one workgroup per row. For an ODD row length (e.g. the
// prefill sequence length Lk=13 of Qwen3-16B-A3B) the LAST element of a row
// shares its u32 word with the FIRST element of the NEXT row, and the two rows
// are independent, concurrently-executing workgroups. The old store performed a
// plain read-modify-write (`src[wi] = word`) on that shared word (starting from
// a zeroed word), so it either clobbered the neighbor row's element with 0 or
// raced with it -> the softmax output was nondeterministic at the odd last
// column. This is why temp-0 Qwen3-16B-A3B on wgpu produced different output
// across processes (16B shapes hit odd Lk, while qwen3-0.6b uses an even row
// width and is bit-identical).
//
// This test locks the fix (atomic CAS store that preserves the sibling half):
// the BF16 softmax on a [1,32,1,L] row with ODD L (and, for good measure, a
// range of even/odd L) must be bit-identical for two identical inputs.
use candle::{DType, Device, Tensor};
use candle_nn::ops;

fn softmax_deterministic(dev: &Device, rows: usize, l: usize) -> bool {
    // Deterministic input: build on CPU, then copy to wgpu (two separate tensors
    // with identical content, exercising the pool/allocator rather than sharing).
    let cpu = Tensor::randn(0f32, 1.0, (1usize, rows, 1usize, l), &Device::Cpu).unwrap();
    let a = cpu.to_dtype(DType::BF16).unwrap().to_device(dev).unwrap();
    let b = cpu.to_dtype(DType::BF16).unwrap().to_device(dev).unwrap();
    let s1 = ops::softmax_last_dim(&a).unwrap();
    let s2 = ops::softmax_last_dim(&b).unwrap();
    let d1 = s1.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let d2 = s2.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    if d1.len() != d2.len() {
        return false;
    }
    d1.iter().zip(d2.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn wgpu_bf16_softmax_deterministic() {
    let Ok(dev) = Device::new_wgpu(0) else {
        eprintln!("skipping: no wgpu device");
        return;
    };
    // Odd length triggers the shared-word race; even length is the control that
    // must remain deterministic too.
    for l in [13usize, 15, 17, 5, 7, 33] {
        assert!(
            softmax_deterministic(&dev, 32, l),
            "wgpu BF16 softmax nondeterministic at odd ne0={l}"
        );
    }
    for l in [14usize, 16, 12, 8, 32, 64, 128] {
        assert!(
            softmax_deterministic(&dev, 32, l),
            "wgpu BF16 softmax nondeterministic at even ne0={l}"
        );
    }
}
