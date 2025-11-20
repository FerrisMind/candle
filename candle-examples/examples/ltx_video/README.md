# LTX-Video Example Application

This example demonstrates how to use the LTX-Video model for text-to-video and image-to-video generation with Candle.

## Overview

LTX-Video is a DiT-based (Diffusion Transformer) video generation model capable of:
- **Text-to-video generation**: Create videos from natural language descriptions
- **Image-to-video generation**: Generate videos conditioned on input images
- **Flexible configurations**: Support for various resolutions and frame counts

## Features

- **Multi-backend support**: Run on CPU, CUDA, or Metal
- **Flexible precision**: Support for FP32, FP16, and BF16 precision
- **Conditioning support**: Image-to-video mode with adjustable conditioning strength
- **Classifier-free guidance**: Control video quality and prompt adherence
- **Batch processing**: Generate videos with optimized memory usage
- **Frame export**: Save individual frames as PNG images
- **Verbose logging**: Detailed progress and timing information

## Installation

### Prerequisites

- Rust 1.70+ with Cargo
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Model weights in safetensors format

### Building

```bash
# Build with CPU support only
cargo build --example ltx_video --release

# Build with CUDA support
cargo build --example ltx_video --release --features cuda

# Build with cuDNN for faster CUDA operations
cargo build --example ltx_video --release --features cudnn

# Build with Metal support (macOS)
cargo build --example ltx_video --release --features metal
```

## Model Weights

### Automatic Download (Recommended)

Model weights are **automatically downloaded** from Hugging Face Hub on first run. Simply run the example without any weight paths:

```bash
cargo run --example ltx_video --release -- \
  --prompt "A serene beach landscape with gentle waves"
```

The example will:
1. Automatically download all required weights from the official repository
2. Cache them locally for faster subsequent runs
3. Use default parameters for video generation

### Manual Download

If you prefer to download weights manually or use a custom repository:

```bash
# Create a directory for weights
mkdir -p ./weights

# Download from HuggingFace (or use huggingface-cli):
# https://huggingface.co/NousResearch/LTX-Video

# Place files in ./weights/:
# - weights/transformer.safetensors
# - weights/vae.safetensors
# - weights/text_encoder.safetensors
# - weights/tokenizer.json
```

Then use the manual paths:

```bash
cargo run --example ltx_video --release -- \
  --prompt "Your prompt here" \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json
```

### Custom Repository

To use a different model repository:

```bash
cargo run --example ltx_video --release -- \
  --prompt "Your prompt here" \
  --model-repo "username/custom-ltx-model"
```

## Usage

### Quick Start (Automatic Download)

**No configuration needed! Just run:**

```bash
cargo run --example ltx_video --release -- \
  --prompt "A serene beach landscape with gentle waves"
```

This will:
- Automatically download all model weights (~20GB)
- Cache them for future runs
- Generate a video with default settings (512x768, 25 frames)
- Save to `output_video.mp4`

### Basic Text-to-Video Generation

```bash
# With auto-download
cargo run --example ltx_video --release -- \
  --prompt "A magical forest with glowing particles" \
  --num-frames 25 \
  --guidance-scale 7.5

# With manual weights (skip download)
cargo run --example ltx_video --release -- \
  --prompt "A serene beach landscape with gentle waves" \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json \
  --output-path output.mp4
```

### Image-to-Video Generation

```bash
# With auto-download
cargo run --example ltx_video --release -- \
  --prompt "The beach waves get more dramatic" \
  --conditioning-image ./beach_image.png \
  --conditioning-frame-index 0 \
  --conditioning-strength 0.8

# With manual weights
cargo run --example ltx_video --release -- \
  --prompt "The beach waves get more dramatic" \
  --conditioning-image ./beach_image.png \
  --conditioning-frame-index 0 \
  --conditioning-strength 0.8 \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json \
  --output-path beach_video.mp4
```

### Advanced Configuration

```bash
cargo run --example ltx_video --release -- \
  --prompt "A magical forest with floating particles" \
  --negative-prompt "blurry, low quality" \
  --height 512 \
  --width 768 \
  --num-frames 25 \
  --num-inference-steps 50 \
  --guidance-scale 7.5 \
  --seed 42 \
  --fps 25 \
  --use-fp16 \
  --save-frames \
  --frames-dir ./output_frames \
  --verbose \
  --output-path output.mp4
  # Note: weights auto-download if not specified
```

### Using CUDA Acceleration

```bash
cargo run --example ltx_video --release --features cuda -- \
  --prompt "Your prompt here" \
  --verbose
  # Weights auto-download
```

## Command-Line Options

### Basic Parameters

- `--prompt TEXT`: Text prompt for video generation (required for text-to-video)
- `--negative-prompt TEXT`: Negative prompt to avoid unwanted features
- `--output-path FILE`: Path to save the generated video (default: `output_video.mp4`)

### Video Configuration

- `--height PIXELS`: Video height, must be divisible by 32 (default: 512)
- `--width PIXELS`: Video width, must be divisible by 32 (default: 768)
- `--num-frames N`: Number of frames to generate, must follow 8n+1 pattern (default: 25)
  - Valid values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, ...
- `--fps FPS`: Frames per second for output video (default: 25)

### Generation Parameters

- `--num-inference-steps N`: Denoising steps (default: 50)
  - For distilled models: 7-8 steps recommended
  - For full models: 50+ steps for better quality
- `--guidance-scale SCALE`: Classifier-free guidance scale (default: 7.5)
  - 1.0 = no guidance
  - Higher values = stronger prompt adherence, less diversity
  - Range: 1.0 - 20.0
- `--seed N`: Random seed for reproducibility (optional)

### Image-to-Video Parameters

- `--conditioning-image FILE`: Path to conditioning image for image-to-video mode
- `--conditioning-frame-index N`: Frame index where conditioning is applied (default: 0)
- `--conditioning-strength STRENGTH`: Conditioning strength 0.0-1.0 (default: 1.0)
  - 0.0 = no conditioning influence
  - 1.0 = full replacement with conditioning

### Model Configuration

- `--model-variant NAME`: Model variant to use (default: `distilled`)
  - `distilled`: 2B parameter distilled model (faster, lower quality)
  - `base`: 13B parameter base model (slower, higher quality)
- `--model-repo REPO`: Hugging Face model repository ID (default: `NousResearch/LTX-Video`)
- `--weights FILE`: Path to transformer weights (auto-downloads if not provided)
- `--vae-weights FILE`: Path to VAE weights (auto-downloads if not provided)
- `--text-encoder-weights FILE`: Path to text encoder weights (auto-downloads if not provided)
- `--tokenizer-path FILE`: Path to T5 tokenizer file (auto-downloads if not provided)
- `--cache-dir DIR`: Custom cache directory for downloaded models

### Hardware Configuration

- `--cpu`: Force CPU computation (disables GPU acceleration)
- `--use-fp16`: Use FP16 precision (requires GPU with FP16 support, ~50% memory vs FP32)
- `--use-bf16`: Use BF16 precision (requires GPU with BF16 support, ~50% memory vs FP32)
- `--use-fp8`: Use FP8 precision (requires GPU with FP8 support, ~25% memory vs FP32, fastest)
- `--backend NAME`: Backend selection (default: `auto`)
  - `auto`: Auto-detect available backend
  - `cpu`: Force CPU computation
  - `cuda`: Force CUDA backend
  - `metal`: Force Metal backend (macOS only)

### Output Options

- `--save-frames`: Save individual frames as PNG images
- `--frames-dir DIR`: Directory for saved frames (default: `frames`)

### Debugging

- `--verbose`: Enable verbose output with detailed timing and memory information

## Examples

### Quick Start (Text-to-Video)

```bash
cargo run --example ltx_video --release -- \
  --prompt "A beautiful sunset over the ocean" \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json
```

### High-Quality Generation

```bash
cargo run --example ltx_video --release -- \
  --prompt "A cinematic shot of mountains at sunrise" \
  --num-frames 121 \
  --height 768 \
  --width 1024 \
  --num-inference-steps 100 \
  --guidance-scale 10.0 \
  --seed 12345 \
  --fps 30 \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json \
  --output-path high_quality.mp4
```

### Fast Generation (Distilled Model)

```bash
cargo run --example ltx_video --release -- \
  --prompt "A quick video" \
  --model-variant distilled \
  --num-frames 9 \
  --num-inference-steps 7 \
  --guidance-scale 1.0 \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json
```

### Memory-Efficient Generation (FP8 Precision)

```bash
cargo run --example ltx_video --release -- \
  --prompt "A beautiful landscape" \
  --use-fp8 \
  --height 512 \
  --width 768 \
  --num-frames 25 \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json
```

### Image-to-Video with Strong Conditioning

```bash
cargo run --example ltx_video --release -- \
  --prompt "The scene becomes more dramatic and colorful" \
  --conditioning-image ./input_image.png \
  --conditioning-frame-index 0 \
  --conditioning-strength 0.95 \
  --num-frames 25 \
  --guidance-scale 5.0 \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json
```

### Reproducible Generation

```bash
cargo run --example ltx_video --release -- \
  --prompt "Consistent results with seed" \
  --seed 42 \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json
```

## Troubleshooting

### Out of Memory (OOM) Error

**Problem**: CUDA/GPU runs out of memory

**Solutions**:
1. Reduce `--height` and `--width`
2. Reduce `--num-frames`
3. Reduce `--num-inference-steps`
4. Use `--use-fp16` for lower memory consumption
5. Use `--cpu` to fall back to CPU (slower but more memory efficient)

```bash
# Example: Lower memory configuration
cargo run --example ltx_video --release -- \
  --prompt "Lower resolution video" \
  --height 256 \
  --width 512 \
  --num-frames 9 \
  --use-fp16 \
  --weights ./weights/transformer.safetensors \
  --vae-weights ./weights/vae.safetensors \
  --text-encoder-weights ./weights/text_encoder.safetensors \
  --tokenizer-path ./weights/tokenizer.json
```

### Invalid Frame Count

**Problem**: `Frame count must follow pattern 8n+1`

**Reason**: LTX-Video requires frames to follow the pattern 8n+1

**Solutions**: Use valid frame counts: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121

```bash
# Correct:
--num-frames 25  # 8*3 + 1

# Incorrect:
--num-frames 24  # Would cause error
```

### Invalid Dimensions

**Problem**: `Width and height must be divisible by 32`

**Solutions**: Ensure both width and height are multiples of 32

```bash
# Correct:
--height 512 --width 768   # Both divisible by 32

# Incorrect:
--height 500 --width 700   # Not divisible by 32
```

### Slow Generation

**Problem**: Generation is very slow on CPU

**Solutions**:
1. Use GPU acceleration with `--features cuda` or `--features metal`
2. Reduce `--num-inference-steps`
3. Reduce video dimensions (`--height` and `--width`)
4. Use FP16 precision with `--use-fp16`
5. Use distilled model variant

### Missing Model Weights

**Problem**: `Weights file not found`

**Solutions**:
1. Check file paths are correct
2. Download weights from HuggingFace
3. Ensure path is absolute or relative to current directory

```bash
# Check weights exist
ls -la ./weights/

# Correct usage:
--weights ./weights/transformer.safetensors \
--vae-weights ./weights/vae.safetensors \
--text-encoder-weights ./weights/text_encoder.safetensors
```

## Performance Tips

### For Speed

- Use distilled model: `--model-variant distilled`
- Reduce inference steps: `--num-inference-steps 7`
- Use lower resolution: `--height 256 --width 512`
- Use fewer frames: `--num-frames 9`
- Enable GPU: Build with `--features cuda`
- Use FP8 precision: `--use-fp8` (fastest, lowest memory)

### For Quality

- Use base model: `--model-variant base`
- Increase inference steps: `--num-inference-steps 100`
- Use higher resolution: `--height 768 --width 1024`
- Use more frames: `--num-frames 121`
- Increase guidance scale: `--guidance-scale 10.0`
- Use seed for consistency: `--seed 42`

### Memory Optimization

- Use `--use-fp8` (75% memory reduction vs FP32, fastest)
- Use `--use-fp16` or `--use-bf16` (50% memory reduction vs FP32)
- Reduce `--num-frames`
- Reduce `--height` and `--width`
- Reduce `--num-inference-steps`
- Use `--cpu` if GPU memory is insufficient

## Output

The example generates:

1. **Video file** (MP4 H.264 encoded)
   - Located at path specified by `--output-path`
   - Default: `output_video.mp4`
   - Frame rate: Specified by `--fps` (default: 25 FPS)

2. **Individual frames** (Optional, with `--save-frames`)
   - PNG images in directory specified by `--frames-dir`
   - Named as `frame_0000.png`, `frame_0001.png`, etc.
   - Useful for debugging or post-processing

## Architecture

The LTX-Video pipeline consists of:

1. **Text Encoder** (T5-XXL): Encodes text prompts to embeddings
2. **Transformer3D**: Core model processing spatiotemporal latents
3. **VAE (Video Autoencoder)**: Encodes/decodes between pixel and latent space
4. **Scheduler (RectifiedFlow)**: Manages the denoising process
5. **Patchifier**: Converts between video and patch representations

For detailed architecture information, see the design document in the specifications.

## References

- [LTX-Video Paper](https://arxiv.org/abs/2311.04204)
- [Candle Repository](https://github.com/huggingface/candle)
- [Lightricks LTX-Video](https://github.com/Lightricks/LTX-Video)
