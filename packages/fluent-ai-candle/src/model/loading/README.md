# Model Loading System

This module provides a robust, production-ready system for loading machine learning models with support for progressive loading, quantization, and error recovery.

## Features

- **Progressive Loading**: Load models in stages with detailed progress tracking
- **Memory Efficiency**: Memory-mapped file access for large models
- **Quantization**: Built-in support for various quantization schemes
- **Error Recovery**: Configurable recovery strategies for handling failures
- **Device-Aware**: Automatic device detection and tensor placement
- **Validation**: Comprehensive model validation and compatibility checking

## Architecture

The model loading system is built around several key components:

1. **ModelLoader**: The main entry point for loading models
2. **VarBuilderFactory**: Handles tensor loading and memory management
3. **ProgressTracker**: Tracks loading progress across multiple stages
4. **RecoveryContext**: Manages error recovery strategies
5. **Quantization**: Handles model quantization during loading

## Usage

### Basic Usage

```rust
use candle_core::Device;
use fluent_ai_candle::model::loading::{
    ModelLoader, ModelLoaderConfig, ProgressCallback, LoadingStage
};

// Create a loader with default configuration
let config = ModelLoaderConfig {
    device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
    ..Default::default()
};

let mut loader = ModelLoader::new(config)
    .with_progress_callback(|progress, message| {
        println!("[{:.1}%] {}", progress * 100.0, message);
    });

// Load a model from a file
let (metadata, var_builder) = loader.load_from_file("model.safetensors")?;

// Use the loaded model
// ...
```

### Progress Tracking

The `ProgressTracker` provides detailed progress information:

```rust
let progress = loader.progress_tracker();
println!("Current stage: {:?}", progress.current_stage());
println!("Progress: {}%", progress.current_progress());
```

### Error Recovery

Configure recovery strategies for handling failures:

```rust
use fluent_ai_candle::model::loading::{RecoveryStrategy, RecoveryAction};

let config = ModelLoaderConfig {
    recovery_strategy: RecoveryStrategy::Retry(3), // Retry up to 3 times
    ..Default::default()
};

// Or handle recovery manually
match loader.load_from_file("model.safetensors") {
    Ok((metadata, var_builder)) => {
        // Success
    },
    Err(e) => {
        match loader.recovery_context().should_recover(&e) {
            true => {
                // Handle recovery
                let action = loader.recovery_context_mut().recovery_action(e);
                // ...
            },
            false => return Err(e),
        }
    }
}
```

### Quantization

Quantize models during loading:

```rust
use fluent_ai_candle::model::loading::{QuantizationType, QuantizationConfig};

let config = ModelLoaderConfig {
    quant_config: Some(QuantizationConfig {
        quant_type: QuantizationType::Int8,
        in_place: true,
        keep_original: false,
        ..Default::default()
    }),
    ..Default::default()
};
```

## Error Handling

All errors implement `std::error::Error` and provide detailed error messages. The module uses the `CandleError` type from the parent crate.

## Performance Considerations

- Enable memory mapping (`use_mmap: true`) for large models to reduce memory usage
- Use quantization to reduce model size and improve inference speed
- Consider using `keep_original: false` to save memory when quantization is enabled
- Use the progress callback to provide user feedback during long-running operations

## Testing

Run the tests with:

```bash
cargo test --package fluent-ai-candle --lib -- model::loading --nocapture
```

## License

This module is part of the `fluent-ai-candle` crate and is licensed under the same terms.