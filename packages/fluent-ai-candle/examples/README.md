# Fluent AI Candle Examples

This directory contains comprehensive examples demonstrating how to use the fluent-ai-candle framework with various models and configurations.

## Available Examples

### Kimi K2 Model Example

**File:** `kimi_k2.rs`

A complete example showing how to use the Kimi K2 model (moonshotai/Kimi-K2-Instruct) with the fluent-ai-candle framework.

**Features demonstrated:**
- Model loading from Hugging Face with streaming progress
- Tokenizer initialization with chat templates
- Inference with Mixture-of-Experts (MoE) architecture
- Zero-allocation streaming generation
- KV cache management
- Error handling and performance statistics

**Usage:**
```bash
# Run with default settings (CPU, FP16)
cargo run --example kimi_k2

# Run with CUDA acceleration
cargo run --example kimi_k2 -- --device cuda

# Run with FP8 quantization
cargo run --example kimi_k2 -- --quantization fp8

# Run with both CUDA and FP8
cargo run --example kimi_k2 -- --device cuda --quantization fp8
```

**Command line options:**
- `--device <DEVICE>`: Device to use (cpu, cuda, metal) - default: cpu
- `--quantization <QUANT>`: Quantization format (fp16, fp8) - default: fp16

**Model specifications:**
- **Parameters:** 1 trillion total, 32 billion activated
- **Architecture:** DeepSeek V3 with Mixture-of-Experts
- **Context length:** ~131k tokens with YARN RoPE scaling
- **Experts:** 384 routed experts, 8 active per token
- **Quantization:** FP16 and FP8 support

## Running Examples

All examples are designed to be run as standalone binaries using Cargo's example system:

```bash
cargo run --example <example_name>
```

## Requirements

- **Rust:** 1.70+ with Cargo
- **Dependencies:** All required dependencies are automatically managed by Cargo
- **Hardware:** 
  - CPU: Any modern x86_64 or ARM64 processor
  - GPU (optional): CUDA-compatible GPU for CUDA acceleration
  - Memory: At least 16GB RAM recommended for large models

## Architecture

The examples demonstrate the fluent-ai-candle framework's key architectural patterns:

- **Zero-allocation streaming:** All operations use `AsyncStream<T>` for memory efficiency
- **Lock-free concurrency:** Atomic operations and lock-free data structures
- **Modular design:** Clean separation between models, tokenizers, and cache management
- **Error handling:** Comprehensive error types and graceful degradation
- **Performance optimization:** Optimized tensor operations and memory management

## Adding New Examples

When adding new examples:

1. Create a new `.rs` file in this directory
2. Follow the existing pattern with proper documentation
3. Include command-line argument parsing for flexibility
4. Demonstrate key framework features
5. Add comprehensive error handling
6. Update this README with the new example

## Support

For questions or issues with the examples:
- Check the main project documentation
- Review the source code in `/src/model/fluent/`
- Examine the test cases in `/tests/`
