# candle-quantized-qwen3

[Qwen3]((https://qwenlm.github.io/blog/qwen3/)) is an upgraded version of Qwen2.5, released by Alibaba Cloud.

## Running the example

```bash
cargo run --example quantized-qwen3 --release -- --prompt "Write a function to count prime numbers up to N."
```


0.6b is used by default, 1.7b, 4b, 8b, 14b, and 32b models are available via `--which` argument.

```bash
cargo run --example quantized-qwen3 --release -- --which 4b   --prompt "A train is travelling at 120mph, how far does it travel in 3 minutes 30 seconds?"
```

## Standard Vulkan profiling

Use the Vulkan SDK layers instead of vendor-specific profilers:

```powershell
pwsh .\candle-examples\examples\quantized-qwen3\profile-vulkan-sdk.ps1 `
  -Model G:\BURN-ML\Qwen3-0.6B\Qwen3-0.6B-Q4_K_M.gguf `
  -Tokenizer G:\BURN-ML\Qwen3-0.6B\tokenizer.json `
  -Prompt "Write a Rust function to calculate factorial of a number." `
  -SampleLen 128 `
  -OutputDir G:\BURN-ML\artifacts\vulkan-sdk
```

The script builds the release binary, runs `VK_LAYER_LUNARG_api_dump`, runs `VK_LAYER_LUNARG_gfxreconstruct`, and writes a small summary with Vulkan API call counts next to the capture artifacts.
