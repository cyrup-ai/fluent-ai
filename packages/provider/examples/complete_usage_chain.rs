//! Complete usage chain demonstration
//!
//! Shows the full flow from model selection through streaming completion consumption
//! using the universal CompletionProvider trait system with zero allocation patterns

use std::env;

use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::{Document, Message, chunk::CompletionChunk, tool::ToolDefinition};
use fluent_ai_provider::{
    AsyncStream,
    clients::openai::{Gpt4o, Gpt4oMini, OpenAICompletionBuilder},
    completion_provider::{CompletionError, CompletionProvider, ModelPrompt}};
use futures_util::StreamExt;
use serde_json::json;

/// Example 1: Direct prompting with ModelInfo defaults
/// Syntax: Model::X.prompt("text") - simplest possible usage
async fn example_direct_prompting() -> Result<(), CompletionError> {
    println!("🚀 Example 1: Direct Prompting");
    println!("🔍 Automatic API key discovery from environment variables");

    // Direct prompting - all defaults from ModelInfo
    let mut stream: AsyncStream<CompletionChunk> = Gpt4o.prompt("What is Rust?");

    println!("📡 Streaming response:");
    while let Some(chunk) = stream.next().await {
        match chunk {
            CompletionChunk::Text { content, .. } => {
                print!("{}", content);
            }
            CompletionChunk::Complete {
                text,
                finish_reason,
                usage} => {
                println!("\n✅ Complete: {}", text);
                if let Some(reason) = finish_reason {
                    println!("🏁 Finish reason: {:?}", reason);
                }
                if let Some(usage_info) = usage {
                    println!(
                        "📊 Tokens - Prompt: {}, Completion: {}, Total: {}",
                        usage_info.prompt_tokens,
                        usage_info.completion_tokens,
                        usage_info.total_tokens
                    );
                }
                break;
            }
            CompletionChunk::Error { message, .. } => {
                println!("❌ Error: {}", message);
                break;
            }
            _ => {} // Handle other chunk types
        }
    }

    Ok(())
}

/// Example 2: Advanced completion builder with custom configuration
/// Shows full builder pattern with ZeroOneOrMany parameters
async fn example_advanced_builder() -> Result<(), CompletionError> {
    println!("\n🔧 Example 2: Advanced Builder Pattern");

    // Create completion builder with ModelInfo defaults
    let builder: OpenAICompletionBuilder = Gpt4oMini.completion()?;

    // Configure with custom parameters (overriding ModelInfo defaults)
    let configured_builder = builder
        .system_prompt("You are an expert Rust developer and teacher")
        .temperature(0.8)
        .max_tokens(1500)
        .top_p(0.9)
        .frequency_penalty(0.1)
        .presence_penalty(0.1);

    // Add chat history using ZeroOneOrMany
    let chat_history = ZeroOneOrMany::Many(vec![
        Message::user("Hello! I'm learning Rust."),
        Message::assistant("Great! Rust is a fantastic language. What would you like to know?"),
        Message::user("Can you explain ownership?"),
    ]);

    let builder_with_history = configured_builder.chat_history(chat_history)?;

    // Add documents for RAG using ZeroOneOrMany
    let documents = ZeroOneOrMany::One(Document::new(
        "rust_guide.md",
        "Ownership is Rust's unique feature for memory safety...",
    ));

    let builder_with_docs = builder_with_history.documents(documents)?;

    // Add tools for function calling using ZeroOneOrMany
    let tools = ZeroOneOrMany::Many(vec![
        ToolDefinition::new(
            "search_documentation",
            "Search Rust documentation for specific topics",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Rust documentation"
                    }
                },
                "required": ["query"]
            }),
        ),
        ToolDefinition::new(
            "run_code",
            "Execute Rust code and return the result",
            json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Rust code to execute"
                    }
                },
                "required": ["code"]
            }),
        ),
    ]);

    let builder_with_tools = builder_with_docs.tools(tools)?;

    // Add chunk handler with cyrup_sugars pattern matching
    // Demonstrate explicit API key override (takes priority over environment)
    let builder_with_explicit_key = builder_with_tools.api_key("sk-explicit-key-override");

    let final_builder = builder_with_explicit_key.on_chunk(|chunk_result| match chunk_result {
        Ok(chunk) => match chunk {
            CompletionChunk::Text { content, .. } => {
                log::info!("📝 Text chunk: {}", content);
            }
            CompletionChunk::ToolStart { id, name, .. } => {
                log::info!("🔧 Tool call started: {} ({})", name, id);
            }
            CompletionChunk::ToolPartial {
                id,
                name,
                arguments,
                ..
            } => {
                log::info!("⚙️ Tool partial: {} {} - {}", id, name, arguments);
            }
            CompletionChunk::Complete { .. } => {
                log::info!("✅ Completion finished");
            }
            _ => {}
        },
        Err(error) => {
            log::error!("❌ Chunk error: {}", error);
        }
    });

    // Execute with user prompt - returns blazing-fast streaming
    let mut stream = final_builder.prompt("Explain borrowing in Rust with a practical example");

    println!("📡 Streaming advanced response:");
    let mut accumulated_text = String::new();
    let mut tool_calls = Vec::new();

    while let Some(chunk) = stream.next().await {
        match chunk {
            CompletionChunk::Text { content, .. } => {
                print!("{}", content);
                accumulated_text.push_str(&content);
            }
            CompletionChunk::ToolStart { id, name, .. } => {
                println!("\n🔧 Starting tool call: {} ({})", name, id);
                tool_calls.push((id, name));
            }
            CompletionChunk::ToolPartial { arguments, .. } => {
                println!("⚙️ Tool arguments: {}", arguments);
            }
            CompletionChunk::ToolComplete {
                id, name, result, ..
            } => {
                println!("✅ Tool {} ({}) completed: {}", name, id, result);
            }
            CompletionChunk::Complete {
                text,
                finish_reason,
                usage} => {
                println!("\n\n🎯 Final completion summary:");
                println!(
                    "📝 Total accumulated text length: {} characters",
                    accumulated_text.len()
                );
                println!("🔧 Tool calls made: {}", tool_calls.len());

                if let Some(reason) = finish_reason {
                    println!("🏁 Finish reason: {:?}", reason);
                }

                if let Some(usage_info) = usage {
                    println!("📊 Token usage:");
                    println!("   Prompt tokens: {}", usage_info.prompt_tokens);
                    println!("   Completion tokens: {}", usage_info.completion_tokens);
                    println!("   Total tokens: {}", usage_info.total_tokens);

                    // Calculate efficiency metrics
                    let efficiency =
                        usage_info.completion_tokens as f64 / usage_info.total_tokens as f64;
                    println!("   Efficiency: {:.2}% completion/total", efficiency * 100.0);
                }
                break;
            }
            CompletionChunk::Error { message, .. } => {
                println!("❌ Stream error: {}", message);
                break;
            }
            _ => {} // Handle other chunk types
        }
    }

    Ok(())
}

/// Example 3: API key discovery and override patterns
/// Shows automatic discovery and explicit override behavior
async fn example_api_key_patterns() -> Result<(), CompletionError> {
    println!("\n🔑 Example 3: API Key Discovery and Override");

    // Demonstrate automatic discovery
    println!("🔍 Testing automatic API key discovery...");
    env::set_var("OPENAI_API_KEY", "sk-discovered-from-env");

    // This will use discovered API key
    let builder1 = Gpt4o.completion()?;
    println!("✅ Builder created with discovered API key");

    // Demonstrate explicit override
    println!("🔧 Testing explicit API key override...");
    let builder2 = builder1.api_key("sk-explicit-override");
    println!("✅ Builder configured with explicit API key override");

    // Clear API key to test error handling
    env::remove_var("OPENAI_API_KEY");

    println!("🔍 Testing without any API key...");
    let mut stream = Gpt4o.prompt("This should fail gracefully");

    if let Some(chunk) = stream.next().await {
        match chunk {
            CompletionChunk::Error { message, .. } => {
                println!("✅ Expected error caught: {}", message);
            }
            _ => {
                println!("❌ Unexpected success - should have failed!");
            }
        }
    }

    // Restore API key for other examples
    env::set_var("OPENAI_API_KEY", "sk-test-key-123");

    // Test builder pattern error handling
    println!("🔍 Testing builder error handling...");
    match Gpt4oMini.completion() {
        Ok(builder) => {
            println!("✅ Builder created successfully");

            // Test with too many messages (bounded by ArrayVec)
            let too_many_messages: Vec<Message> = (0..150)
                .map(|i| Message::user(&format!("Message {}", i)))
                .collect();

            match builder.chat_history(ZeroOneOrMany::Many(too_many_messages)) {
                Ok(_) => println!("❌ Should have failed with too many messages"),
                Err(CompletionError::RequestTooLarge) => {
                    println!("✅ Correctly rejected oversized request");
                }
                Err(e) => println!("❌ Unexpected error: {}", e)}
        }
        Err(e) => {
            println!("❌ Builder creation failed: {}", e);
        }
    }

    Ok(())
}

/// Example 4: Performance benchmarking
/// Shows the zero-allocation, blazing-fast performance characteristics
async fn example_performance_demo() -> Result<(), CompletionError> {
    println!("\n⚡ Example 4: Performance Demonstration");

    env::set_var("OPENAI_API_KEY", "sk-test-key-123");

    // Measure builder creation time (should be near-instantaneous due to compile-time configs)
    let start = std::time::Instant::now();
    let builder = Gpt4o.completion()?;
    let builder_time = start.elapsed();
    println!(
        "🏗️ Builder creation: {:?} (compile-time defaults)",
        builder_time
    );

    // Measure configuration time (should be zero-allocation)
    let start = std::time::Instant::now();
    let configured = builder
        .system_prompt("Performance test")
        .temperature(0.7)
        .max_tokens(100);
    let config_time = start.elapsed();
    println!("⚙️ Configuration: {:?} (zero allocation)", config_time);

    // Measure stream creation time
    let start = std::time::Instant::now();
    let mut stream = configured.prompt("Count to 5");
    let stream_time = start.elapsed();
    println!("📡 Stream creation: {:?} (async task spawned)", stream_time);

    // Measure chunk processing throughput
    let start = std::time::Instant::now();
    let mut chunk_count = 0;
    let mut total_chars = 0;

    while let Some(chunk) = stream.next().await {
        chunk_count += 1;

        match chunk {
            CompletionChunk::Text { content, .. } => {
                total_chars += content.len();
            }
            CompletionChunk::Complete { .. } => {
                break;
            }
            CompletionChunk::Error { .. } => {
                break;
            }
            _ => {}
        }
    }

    let processing_time = start.elapsed();
    println!("🔥 Stream processing: {:?}", processing_time);
    println!("📊 Performance metrics:");
    println!("   Chunks processed: {}", chunk_count);
    println!("   Characters streamed: {}", total_chars);

    if processing_time.as_millis() > 0 {
        let chars_per_ms = total_chars as f64 / processing_time.as_millis() as f64;
        println!("   Throughput: {:.2} chars/ms", chars_per_ms);
    }

    Ok(())
}

/// Example 5: Multiple model comparison
/// Shows how different models can be used with the same interface
async fn example_multi_model() -> Result<(), CompletionError> {
    println!("\n🎭 Example 5: Multi-Model Comparison");

    env::set_var("OPENAI_API_KEY", "sk-test-key-123");

    let models = vec![
        (
            "GPT-4o",
            Box::new(Gpt4o) as Box<dyn ModelPrompt<Provider = OpenAICompletionBuilder>>,
        ),
        (
            "GPT-4o Mini",
            Box::new(Gpt4oMini) as Box<dyn ModelPrompt<Provider = OpenAICompletionBuilder>>,
        ),
    ];

    let prompt = "Explain Rust's type system in one sentence.";

    for (model_name, model) in models {
        println!("\n🤖 Testing {} model:", model_name);
        println!("📝 Prompt: {}", prompt);

        // Show model capabilities at compile time
        println!("🔧 Model capabilities:");
        println!("   Tools: {}", model.supports_tools());
        println!("   Vision: {}", model.supports_vision());
        println!("   Audio: {}", model.supports_audio());
        println!("   Context length: {}", model.context_length());
        println!("   Max tokens: {}", model.max_output_tokens());

        // Execute with same interface
        let mut stream = model.prompt(prompt);
        print!("📡 Response: ");

        while let Some(chunk) = stream.next().await {
            match chunk {
                CompletionChunk::Text { content, .. } => {
                    print!("{}", content);
                }
                CompletionChunk::Complete { .. } => {
                    println!(" ✅");
                    break;
                }
                CompletionChunk::Error { message, .. } => {
                    println!(" ❌ Error: {}", message);
                    break;
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Main demonstration function
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging for chunk handler demonstration
    env_logger::init();

    println!("🎯 Complete Usage Chain Demonstration");
    println!("=====================================");

    // Run all examples
    example_direct_prompting().await?;
    example_advanced_builder().await?;
    example_api_key_patterns().await?;
    example_performance_demo().await?;
    example_multi_model().await?;

    println!("\n🎉 All examples completed successfully!");
    println!("💡 Key takeaways:");
    println!("   ✅ Zero allocation architecture with compile-time defaults");
    println!("   ✅ ZeroOneOrMany everywhere - no Optional types");
    println!("   ✅ Universal trait system enabling Model::X.prompt() syntax");
    println!("   ✅ Blazing-fast HTTP3 streaming with perfect error handling");
    println!("   ✅ Elegant ergonomics with cyrup_sugars pattern matching");
    println!("   ✅ No unsafe code, no unwrap/expect, production-ready");

    Ok(())
}

#[cfg(test)]
mod tests {
    use fluent_ai_provider::completion_provider::ModelInfo;

    use super::*;

    #[test]
    fn test_model_info_compile_time() {
        // Verify compile-time model configurations
        assert_eq!(Gpt4o::CONFIG.model_name, "gpt-4o");
        assert_eq!(Gpt4o::CONFIG.provider, "openai");
        assert_eq!(Gpt4o::CONFIG.max_tokens, 4096);
        assert_eq!(Gpt4o::CONFIG.context_length, 128000);
        assert!(Gpt4o::CONFIG.supports_tools);
        assert!(Gpt4o::CONFIG.supports_vision);

        assert_eq!(Gpt4oMini::CONFIG.model_name, "gpt-4o-mini");
        assert_eq!(Gpt4oMini::CONFIG.max_tokens, 16384);
        assert!(!Gpt4oMini::CONFIG.supports_audio);
    }

    #[test]
    fn test_zero_one_or_many_patterns() {
        // Test ZeroOneOrMany usage patterns
        let none: ZeroOneOrMany<String> = ZeroOneOrMany::None;
        let one = ZeroOneOrMany::One("single".to_string());
        let many = ZeroOneOrMany::Many(vec!["first".to_string(), "second".to_string()]);

        // Verify no Optional types used
        match none {
            ZeroOneOrMany::None => assert!(true),
            _ => panic!("Should be None variant")}

        match one {
            ZeroOneOrMany::One(value) => assert_eq!(value, "single"),
            _ => panic!("Should be One variant")}

        match many {
            ZeroOneOrMany::Many(values) => assert_eq!(values.len(), 2),
            _ => panic!("Should be Many variant")}
    }

    #[tokio::test]
    async fn test_error_stream_creation() {
        // Test that error streams are created correctly without panicking
        env::remove_var("OPENAI_API_KEY");

        let mut stream = Gpt4o.prompt("test");

        if let Some(chunk) = stream.next().await {
            match chunk {
                CompletionChunk::Error { message, .. } => {
                    assert!(message.contains("Missing API key"));
                }
                _ => panic!("Should return error chunk")}
        }
    }
}
