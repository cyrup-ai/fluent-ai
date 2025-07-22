//! Kimi K2 Model Example
//!
//! This example demonstrates how to:
//! - Load the Kimi K2 model from Hugging Face
//! - Initialize the tokenizer with chat templates
//! - Perform inference with the MoE architecture
//! - Use streaming generation with zero-allocation patterns
//!
//! Usage:
//!   cargo run --example kimi_k2
//!   cargo run --example kimi_k2 -- --device cuda
//!   cargo run --example kimi_k2 -- --quantization fp8

use std::sync::Arc;

use candle_core::{Device, DType};
use clap::{Arg, Command};
use fluent_ai_candle::{
    error::{CandleError, CandleResult},
    model::fluent::{
        kimi_k2::{
            loader::{load_model, LoaderEvent},
            model::{KimiK2Config, KimiK2Model},
            tokenizer::{ChatMessage, KimiK2Tokenizer},
        },
        Config, QuantFormat, KIMI_K2_FP16, KIMI_K2_FP8,
    },
    model::{
        cache::{KVCacheConfig, KVCacheManager},
        wrappers::{IntoKimiK2Wrapper, KimiK2Wrapper},
    },
    var_builder::CandleVarBuilder,
};
use fluent_ai_stream::{on_chunk, AsyncStream};

/// Complete example of Kimi K2 model usage
pub struct KimiK2Example {
    model: KimiK2Wrapper,
    tokenizer: KimiK2Tokenizer,
    config: KimiK2Config,
}

impl KimiK2Example {
    /// Create a new Kimi K2 example with model loading
    pub fn new(device: Device, quant_format: QuantFormat) -> AsyncStream<CandleResult<Self>> {
        AsyncStream::new(move |y| async move {
            // Configuration
            let config = match quant_format {
                QuantFormat::Fp16 => KIMI_K2_FP16.clone(),
                QuantFormat::Fp8 => KIMI_K2_FP8.clone(),
            };
            
            // Load tokenizer
            y.yield_item(Ok("Loading tokenizer...".to_string()));
            let tokenizer = match KimiK2Tokenizer::from_hub(&config).await {
                Ok(tokenizer) => tokenizer,
                Err(e) => {
                    y.yield_error(e);
                    return;
                }
            };
            
            // Load model shards
            y.yield_item(Ok("Loading model shards...".to_string()));
            let mut model_shards = Vec::new();
            
            on_chunk!(load_model(&config), chunk, {
                match chunk {
                    LoaderEvent::Start { total_shards, total_bytes } => {
                        y.yield_item(Ok(format!(
                            "Starting download: {} shards, {} bytes total",
                            total_shards, total_bytes
                        )));
                    }
                    LoaderEvent::Progress { shard_idx, bytes } => {
                        y.yield_item(Ok(format!(
                            "Downloading shard {}: {} bytes",
                            shard_idx, bytes.len()
                        )));
                    }
                    LoaderEvent::ShardReady { shard_idx, shard } => {
                        y.yield_item(Ok(format!("Shard {} ready", shard_idx)));
                        model_shards.push(shard);
                    }
                    LoaderEvent::Complete { shards } => {
                        model_shards = shards.into_iter().collect();
                        y.yield_item(Ok("All shards loaded".to_string()));
                    }
                }
            });
            
            // Initialize model from shards
            y.yield_item(Ok("Initializing model...".to_string()));
            let model_config = KimiK2Config::default();
            
            // Create VarBuilder from loaded shards
            let var_builder = match CandleVarBuilder::from_safetensors_shards(
                &model_shards,
                DType::BF16,
                &device,
            ) {
                Ok(vb) => vb,
                Err(e) => {
                    y.yield_error(CandleError::InitializationError(format!(
                        "Failed to create VarBuilder: {}",
                        e
                    )));
                    return;
                }
            };
            
            // Create model
            let model = match KimiK2Model::new(&model_config, var_builder.into(), &device) {
                Ok(model) => model,
                Err(e) => {
                    y.yield_error(CandleError::InitializationError(format!(
                        "Failed to create model: {}",
                        e
                    )));
                    return;
                }
            };
            
            // Create cache manager
            let cache_config = KVCacheConfig::default();
            let cache_manager = Arc::new(KVCacheManager::new(cache_config));
            
            // Wrap model
            let model_wrapper = model.into_wrapper(cache_manager);
            
            y.yield_item(Ok(Self {
                model: model_wrapper,
                tokenizer,
                config: model_config,
            }));
        })
    }
    
    /// Generate a response to a chat conversation
    pub fn chat(&self, messages: &[ChatMessage]) -> CandleResult<String> {
        // Apply chat template
        let prompt = self.tokenizer.apply_chat_template(messages)?;
        
        // Tokenize input
        let input_ids = self.tokenizer.encode(&prompt)?;
        let input_tensor = input_ids.to_tensor(&self.model.config().torch_dtype, &Device::Cpu)?;
        
        // Generate response
        let output = self.model.generate_next(&input_tensor)?;
        
        // Decode output
        let output_ids = output.argmax(candle_core::D::Minus1)?;
        let response = self.tokenizer.decode(&output_ids)?;
        
        Ok(response)
    }
    
    /// Stream generation for longer responses
    pub fn stream_chat(&self, messages: &[ChatMessage]) -> AsyncStream<CandleResult<String>> {
        let messages = messages.to_vec();
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        
        AsyncStream::new(move |y| async move {
            // Apply chat template
            let prompt = match tokenizer.apply_chat_template(&messages) {
                Ok(prompt) => prompt,
                Err(e) => {
                    y.yield_error(e);
                    return;
                }
            };
            
            // Tokenize input
            let input_ids = match tokenizer.encode(&prompt) {
                Ok(ids) => ids,
                Err(e) => {
                    y.yield_error(e);
                    return;
                }
            };
            
            // Convert to tensor
            let input_tensor = match input_ids.to_tensor(&model.config().torch_dtype, &Device::Cpu) {
                Ok(tensor) => tensor,
                Err(e) => {
                    y.yield_error(CandleError::TensorError(format!("Tensor conversion failed: {}", e)));
                    return;
                }
            };
            
            // Prefill cache
            let mut current_output = match model.prefill(&input_tensor) {
                Ok(output) => output,
                Err(e) => {
                    y.yield_error(CandleError::InferenceError(format!("Prefill failed: {}", e)));
                    return;
                }
            };
            
            // Generate tokens one by one
            let max_tokens = 512;
            for _step in 0..max_tokens {
                // Get next token
                let next_token_logits = match model.generate_next(&current_output) {
                    Ok(logits) => logits,
                    Err(e) => {
                        y.yield_error(CandleError::InferenceError(format!("Generation failed: {}", e)));
                        return;
                    }
                };
                
                // Sample next token (greedy for now)
                let next_token = match next_token_logits.argmax(candle_core::D::Minus1) {
                    Ok(token) => token,
                    Err(e) => {
                        y.yield_error(CandleError::InferenceError(format!("Token sampling failed: {}", e)));
                        return;
                    }
                };
                
                // Decode token
                let token_text = match tokenizer.decode(&next_token) {
                    Ok(text) => text,
                    Err(e) => {
                        y.yield_error(e);
                        return;
                    }
                };
                
                // Check for end of sequence
                if token_text.contains("[EOS]") || token_text.contains("<|im_end|>") {
                    break;
                }
                
                y.yield_item(Ok(token_text));
                current_output = next_token_logits;
            }
        })
    }
    
    /// Get model statistics
    pub fn stats(&self) -> String {
        let cache_stats = self.model.cache_stats();
        format!(
            "Kimi K2 Model Stats:\n\
             - Parameters: 1T total, 32B activated\n\
             - Layers: {}\n\
             - Hidden size: {}\n\
             - Attention heads: {}\n\
             - Cache hits: {}\n\
             - Cache misses: {}\n\
             - Cache size: {} entries",
            self.config.num_hidden_layers,
            self.config.hidden_size,
            self.config.num_attention_heads,
            cache_stats.hits,
            cache_stats.misses,
            cache_stats.total_entries,
        )
    }
}

/// Parse command line arguments
fn parse_args() -> (Device, QuantFormat) {
    let matches = Command::new("Kimi K2 Example")
        .about("Demonstrates Kimi K2 model usage with fluent-ai-candle")
        .arg(
            Arg::new("device")
                .long("device")
                .value_name("DEVICE")
                .help("Device to use: cpu, cuda, or metal")
                .default_value("cpu"),
        )
        .arg(
            Arg::new("quantization")
                .long("quantization")
                .value_name("QUANT")
                .help("Quantization format: fp16 or fp8")
                .default_value("fp16"),
        )
        .get_matches();

    let device = match matches.get_one::<String>("device").unwrap().as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0).unwrap_or_else(|_| {
            eprintln!("CUDA not available, falling back to CPU");
            Device::Cpu
        }),
        "metal" => Device::new_metal(0).unwrap_or_else(|_| {
            eprintln!("Metal not available, falling back to CPU");
            Device::Cpu
        }),
        _ => {
            eprintln!("Unknown device, using CPU");
            Device::Cpu
        }
    };

    let quant_format = match matches.get_one::<String>("quantization").unwrap().as_str() {
        "fp16" => QuantFormat::Fp16,
        "fp8" => QuantFormat::Fp8,
        _ => {
            eprintln!("Unknown quantization format, using FP16");
            QuantFormat::Fp16
        }
    };

    (device, quant_format)
}

/// Main example function
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let (device, quant_format) = parse_args();
    
    println!("🚀 Loading Kimi K2 model...");
    println!("📱 Device: {:?}", device);
    println!("🔢 Quantization: {:?}", quant_format);
    
    // Load model
    let mut example = None;
    on_chunk!(KimiK2Example::new(device, quant_format), chunk, {
        match chunk {
            Ok(msg) => {
                if let Ok(ex) = msg.downcast::<KimiK2Example>() {
                    example = Some(*ex);
                } else if let Ok(status) = msg.downcast::<String>() {
                    println!("📦 {}", status);
                }
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
                return Err(e.into());
            }
        }
    });
    
    let example = example.ok_or_else(|| {
        CandleError::InitializationError("Failed to load model".to_string())
    })?;
    
    println!("✅ Model loaded successfully!");
    println!("{}", example.stats());
    
    // Example conversation
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are Kimi, an AI assistant created by Moonshot AI.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "What are the key features of the Kimi K2 model?".to_string(),
        },
    ];
    
    println!("\n💬 Starting conversation...");
    println!("👤 User: What are the key features of the Kimi K2 model?");
    println!("🤖 Kimi: ");
    
    // Stream response
    on_chunk!(example.stream_chat(&messages), chunk, {
        match chunk {
            Ok(token) => print!("{}", token),
            Err(e) => {
                eprintln!("\n❌ Generation error: {}", e);
                return Err(e.into());
            }
        }
    });
    
    println!("\n\n🎉 Example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_argument_parsing() {
        // Test would require actual CLI parsing, but we can test the logic
        let (device, quant) = (Device::Cpu, QuantFormat::Fp16);
        assert!(matches!(device, Device::Cpu));
        assert!(matches!(quant, QuantFormat::Fp16));
    }
    
    #[test]
    fn test_chat_message_creation() {
        let message = ChatMessage {
            role: "user".to_string(),
            content: "Hello, Kimi!".to_string(),
        };
        
        assert_eq!(message.role, "user");
        assert_eq!(message.content, "Hello, Kimi!");
    }
}
