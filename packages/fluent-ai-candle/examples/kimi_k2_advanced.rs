//! Advanced Kimi K2 integration example demonstrating all framework features
//!
//! This example showcases the complete integration between Kimi K2 and the
//! fluent-ai-candle framework, including:
//! - SIMD optimizations for high-performance inference
//! - Constrained generation with JSON schema support
//! - KV cache optimization for long conversations
//! - Real-time streaming with AsyncStream architecture
//! - MoE (Mixture of Experts) optimizations
//! - Advanced sampling strategies (nucleus, typical, mirostat)

use std::time::Instant;
use candle_core::Device;
use fluent_ai_async::StreamExt;
use fluent_ai_candle::{
    CandleCompletionRequest, CandleMessage, CandleMessageRole,
    model::fluent::kimi_k2::{
        integration::{KimiK2GeneratorBuilder, KimiK2Features},
        KimiK2Config, QuantFormat,
    },
    sampling::{Sampling, SamplingConfig},
    streaming::StreamingConfig,
    kv_cache::KVCacheConfig,
    generator::types::GenerationConfig,
    types::ZeroOneOrMany,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Kimi K2 Integration Example");
    println!("========================================");

    // 1. Configure Kimi K2 with all advanced features
    let kimi_k2_generator = KimiK2GeneratorBuilder::new()
        .with_device(Device::auto_detect_best()?)
        .with_model_config(KimiK2Config::production_config())
        .with_generation_config(GenerationConfig {
            temperature: 0.7,
            max_tokens: 2048,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: Some(42),
            stop_sequences: vec!["</chat>".to_string()],
        })
        .with_sampling(Sampling::sophisticated_conversation())
        .with_streaming(StreamingConfig::real_time())
        .with_kv_cache(Some(KVCacheConfig::kimi_k2_optimized()))
        .with_simd(true)
        .with_constraints(true)
        .build()
        .await?;

    println!("✅ Kimi K2 Generator initialized with {} parameters", 
             format_number(kimi_k2_generator.stats().total_parameters));

    // 2. Demonstrate basic text completion
    println!("\n📝 Example 1: Basic Text Completion");
    println!("----------------------------------");
    
    let basic_request = CandleCompletionRequest {
        system_prompt: "You are Kimi, a helpful AI assistant created by Moonshot AI.".to_string(),
        chat_history: ZeroOneOrMany::One(CandleMessage {
            role: CandleMessageRole::User,
            content: "Explain quantum computing in simple terms.".to_string(),
            name: None,
        }),
        documents: ZeroOneOrMany::None,
        temperature: 0.7,
        max_tokens: Some(std::num::NonZeroU32::new(512).unwrap()),
    };

    let start_time = Instant::now();
    let mut completion_stream = kimi_k2_generator.complete(&basic_request);
    
    while let Some(response) = completion_stream.next().await {
        let duration = start_time.elapsed();
        println!("⚡ Generated in {:.2}ms: {}", 
                duration.as_millis(), 
                response.text.chars().take(100).collect::<String>());
        break; // Just show first response for brevity
    }

    // 3. Demonstrate streaming completion with real-time output
    println!("\n🌊 Example 2: Real-time Streaming");
    println!("--------------------------------");

    let streaming_request = CandleCompletionRequest {
        system_prompt: "You are a creative writer.".to_string(),
        chat_history: ZeroOneOrMany::One(CandleMessage {
            role: CandleMessageRole::User,
            content: "Write a short story about a robot discovering emotions.".to_string(),
            name: None,
        }),
        documents: ZeroOneOrMany::None,
        temperature: 0.9,
        max_tokens: Some(std::num::NonZeroU32::new(1024).unwrap()),
    };

    let mut streaming_response = kimi_k2_generator.complete_stream(&streaming_request);
    let mut token_count = 0;
    let stream_start = Instant::now();

    println!("📖 Story generation (streaming):");
    while let Some(chunk) = streaming_response.next().await {
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                print!("{}", content);
                token_count += 1;
                
                // Show real-time statistics every 50 tokens
                if token_count % 50 == 0 {
                    let elapsed = stream_start.elapsed();
                    let tokens_per_sec = token_count as f64 / elapsed.as_secs_f64();
                    println!("\n💨 Tokens/sec: {:.1}", tokens_per_sec);
                }
            }
            
            if choice.finish_reason.is_some() {
                break;
            }
        }
    }
    println!("\n");

    // 4. Demonstrate constrained generation (JSON output)
    println!("\n🔒 Example 3: Constrained JSON Generation");
    println!("------------------------------------------");

    let json_request = CandleCompletionRequest {
        system_prompt: r#"You are a data extraction AI. Always respond with valid JSON matching this schema:
{
  "name": "string",
  "age": "number", 
  "skills": ["array", "of", "strings"],
  "location": "string"
}"#.to_string(),
        chat_history: ZeroOneOrMany::One(CandleMessage {
            role: CandleMessageRole::User,
            content: "Extract information about: John Smith, 35 years old, software engineer and data scientist from San Francisco".to_string(),
            name: None,
        }),
        documents: ZeroOneOrMany::None,
        temperature: 0.3, // Lower temperature for structured output
        max_tokens: Some(std::num::NonZeroU32::new(256).unwrap()),
    };

    let mut json_stream = kimi_k2_generator.complete(&json_request);
    if let Some(json_response) = json_stream.next().await {
        println!("📊 Structured JSON output:");
        println!("{}", json_response.text);
        
        // Verify it's valid JSON
        match serde_json::from_str::<serde_json::Value>(&json_response.text) {
            Ok(_) => println!("✅ Valid JSON generated"),
            Err(e) => println!("⚠️  JSON validation failed: {}", e),
        }
    }

    // 5. Demonstrate long conversation with KV cache optimization
    println!("\n💬 Example 4: Long Conversation with KV Cache");
    println!("---------------------------------------------");

    let mut conversation_history = vec![
        CandleMessage {
            role: CandleMessageRole::System,
            content: "You are participating in a technical discussion about AI systems.".to_string(),
            name: None,
        },
        CandleMessage {
            role: CandleMessageRole::User,
            content: "What are the key challenges in scaling transformer models?".to_string(),
            name: None,
        },
    ];

    // Simulate a multi-turn conversation
    for turn in 1..=3 {
        let long_request = CandleCompletionRequest {
            system_prompt: String::new(),
            chat_history: ZeroOneOrMany::Many(conversation_history.clone()),
            documents: ZeroOneOrMany::None,
            temperature: 0.8,
            max_tokens: Some(std::num::NonZeroU32::new(300).unwrap()),
        };

        let turn_start = Instant::now();
        let mut turn_stream = kimi_k2_generator.complete(&long_request);
        
        if let Some(response) = turn_stream.next().await {
            let turn_duration = turn_start.elapsed();
            println!("🔄 Turn {}: Generated in {:.2}ms (KV cache: {})", 
                    turn, 
                    turn_duration.as_millis(),
                    if turn > 1 { "ACTIVE" } else { "WARMING" });
            
            // Add AI response to conversation history
            conversation_history.push(CandleMessage {
                role: CandleMessageRole::Assistant,
                content: response.text.to_string(),
                name: None,
            });
            
            // Add follow-up user message
            let follow_ups = [
                "Can you elaborate on the memory requirements?",
                "How do MoE models help with this?",
                "What are the latest developments in this area?",
            ];
            
            if let Some(follow_up) = follow_ups.get(turn - 1) {
                conversation_history.push(CandleMessage {
                    role: CandleMessageRole::User,
                    content: follow_up.to_string(),
                    name: None,
                });
            }
        }
    }

    // 6. Display final statistics and performance metrics
    println!("\n📊 Final Performance Statistics");
    println!("==============================");
    
    let stats = kimi_k2_generator.stats();
    println!("🧠 Model: {} parameters", format_number(stats.total_parameters));
    println!("🔧 Experts: {}/{} active per token", 
             stats.active_experts_per_token, stats.total_experts);
    println!("📏 Context: {} tokens", format_number(stats.context_length as u64));
    println!("🎯 Hidden size: {}", stats.hidden_size);
    println!("🏗️  Layers: {}", stats.num_layers);
    println!("📈 Log probability: {:.4}", stats.cumulative_log_prob);
    
    let features = kimi_k2_generator.features();
    println!("⚡ SIMD enabled: {}", features.simd_enabled);
    println!("🔒 Constraints enabled: {}", features.constraints_enabled);

    // 7. Memory and performance analysis
    println!("\n🔍 Performance Analysis");
    println!("======================");
    
    let generator = kimi_k2_generator.generator();
    let config = generator.config();
    
    println!("🌡️  Temperature: {}", config.temperature);
    println!("🎰 Max tokens: {}", config.max_tokens);
    println!("🔢 Top-k: {}", config.top_k);
    println!("🎪 Top-p: {}", config.top_p);
    println!("🔁 Repetition penalty: {}", config.repetition_penalty);
    
    if let Some(seed) = config.seed {
        println!("🌱 Seed: {}", seed);
    }

    println!("\n✨ Kimi K2 Advanced Integration Complete!");
    println!("This example demonstrated:");
    println!("  • Complete model initialization with all features");
    println!("  • Basic and streaming text generation");
    println!("  • Constrained JSON output generation");
    println!("  • Multi-turn conversation with KV cache optimization");
    println!("  • SIMD and MoE performance optimizations");
    println!("  • Real-time performance monitoring");

    Ok(())
}

/// Format large numbers with appropriate suffixes
fn format_number(num: u64) -> String {
    if num >= 1_000_000_000_000 {
        format!("{:.1}T", num as f64 / 1_000_000_000_000.0)
    } else if num >= 1_000_000_000 {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}

/// Extension trait for KimiK2Config to add production configurations
trait ProductionConfig {
    fn production_config() -> KimiK2Config;
}

impl ProductionConfig for KimiK2Config {
    fn production_config() -> KimiK2Config {
        KimiK2Config::new("moonshotai/Kimi-K2-Instruct", QuantFormat::Fp8)
    }
}

/// Extension trait for Sampling to add sophisticated configurations
trait SophisticatedSampling {
    fn sophisticated_conversation() -> Sampling;
}

impl SophisticatedSampling for Sampling {
    fn sophisticated_conversation() -> Sampling {
        SamplingConfig::default()
            .with_temperature(0.8)
            .with_top_k(40)
            .with_top_p(0.9)
            .with_typical_p(0.95)
            .with_repetition_penalty(1.05)
            .with_frequency_penalty(0.1)
            .with_presence_penalty(0.1)
            .build_sampling()
    }
}

/// Extension trait for StreamingConfig to add real-time configurations
trait RealTimeStreaming {
    fn real_time() -> StreamingConfig;
}

impl RealTimeStreaming for StreamingConfig {
    fn real_time() -> StreamingConfig {
        StreamingConfig {
            buffer_size: 1, // Immediate token emission
            chunk_size: 1,  // Single token chunks
            enable_flow_control: true,
            max_latency_ms: 50,
            enable_compression: false, // Disable for lowest latency
        }
    }
}

/// Extension trait for Device to add auto-detection
trait AutoDetect {
    fn auto_detect_best() -> Result<Device, Box<dyn std::error::Error>>;
}

impl AutoDetect for Device {
    fn auto_detect_best() -> Result<Device, Box<dyn std::error::Error>> {
        // Try CUDA first, then Metal, fallback to CPU
        if candle_core::cuda_is_available() {
            Ok(Device::new_cuda(0)?)
        } else if candle_core::metal_is_available() {
            Ok(Device::new_metal(0)?)
        } else {
            Ok(Device::Cpu)
        }
    }
}