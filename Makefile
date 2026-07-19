.PHONY: clean-ptx clean test backend-parity-audit backend-smoke-vulkan backend-smoke-wgpu

clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > candle-kernels/src/lib.rs
	touch candle-kernels/build.rs
	touch candle-examples/build.rs
	touch candle-flash-attn/build.rs

clean:
	cargo clean

test:
	cargo test

# Static CUDA↔Vulkan↔WebGPU surface + curated three-profile manifest gate.
backend-parity-audit:
	python scripts/backend_parity_audit.py

backend-smoke-vulkan:
	cargo test -p candle-core --features vulkan --test backend_smoke_tests

backend-smoke-wgpu:
	cargo test -p candle-core --features wgpu --test backend_smoke_tests

all: test
