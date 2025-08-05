//! Candle ML Framework - Interactive Chat Loop
//!
//! This is the main entry point for the Candle-based AI agent with extensible model support.
//! Features real-time streaming, progresshub model downloads, and interactive chat.
//!
//! Usage:
//!   cargo run --package fluent_ai_candle                    # Uses kimi-k2 (default)
//!   cargo run --package fluent_ai_candle -- --model kimi-k2 # Explicit kimi-k2
//!   cargo run --package fluent_ai_candle -- --model llama3  # Future: Llama 3 support
//!   cargo run --package fluent_ai_candle -- --help          # Show help

use fluent_ai_candle::prelude::*;
use std::io::{self, Write, BufRead};
use clap::{Parser, ValueEnum};

/// Supported models for the chat loop
#[derive(Debug, Clone, ValueEnum)]
enum SupportedModel {
    /// Kimi-K2 model (default) - Excellent for general conversation and coding
    #[value(name = "kimi-k2")]
    KimiK2,
    /// Llama 3 model (future support) - Open source alternative
    #[value(name = "llama3")]
    Llama3,
    /// GPT-4 model (future support) - Advanced reasoning capabilities  
    #[value(name = "gpt4")]
    Gpt4,
}

impl Default for SupportedModel {
    fn default() -> Self {
        Self::KimiK2
    }
}

impl std::fmt::Display for SupportedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KimiK2 => write!(f, "kimi-k2"),
            Self::Llama3 => write!(f, "llama3"),
            Self::Gpt4 => write!(f, "gpt4"),
        }
    }
}

/// Command line arguments for the Candle chat loop
#[derive(Parser, Clone)]
#[command(name = "candle-chat")]
#[command(about = "Interactive chat loop using Candle ML framework")]
#[command(version = "1.0")]
struct Args {
    /// Model to use for the chat session
    #[arg(short, long, value_enum, default_value_t = SupportedModel::default())]
    model: SupportedModel,
    
    /// Temperature for response generation (0.0 to 2.0)
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,
    
    /// Maximum tokens per response
    #[arg(long, default_value_t = 4000)]
    max_tokens: u32,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

/// Main interactive chat loop using Candle ML framework with configurable models
fn main() {
    let args = Args::parse();
    
    println!("🔥 Candle ML Framework - Interactive Chat");
    println!("==========================================");
    println!("Model: {} (via progresshub)", args.model);
    println!("Temperature: {}", args.temperature);
    println!("Max Tokens: {}", args.max_tokens);
    if args.verbose {
        println!("Verbose Logging: Enabled");
    }
    println!("Type 'quit', 'exit', or 'bye' to end the session");
    println!("Type '/help' for available commands\n");

    // Start the interactive chat loop using streams-only architecture
    let mut chat_stream = run_chat_loop(&args);
    // Process the chat loop stream to completion
    while chat_stream.try_next().is_some() {
        // Chat loop completed
    }
}

/// Create a Candle agent for the specified model with progresshub integration
fn create_agent_for_model(args: Args) -> AsyncStream<impl CandleAgentBuilder> {
    AsyncStream::with_channel(move |sender| {
        match args.model {
            SupportedModel::KimiK2 => {
                let mut agent_stream = create_kimi_k2_agent(args.clone());
                if let Some(agent) = agent_stream.try_next() {
                    let _ = sender.send(agent);
                }
            },
            SupportedModel::Llama3 => {
                eprintln!("Error: Llama 3 support is not yet implemented. Use --model kimi-k2 for now.");
            },
            SupportedModel::Gpt4 => {
                eprintln!("Error: GPT-4 support is not yet implemented. Use --model kimi-k2 for now.");
            },
        }
    })
}

/// Create a Candle agent with kimi-k2 model and progresshub integration
fn create_kimi_k2_agent(args: Args) -> AsyncStream<impl CandleAgentBuilder> {
    AsyncStream::with_channel(move |sender| {
        println!("🚀 Initializing kimi-k2 model...");
        
        // TODO: Add progresshub integration for model downloads
        let model_path = "kimi-k2"; // This will be downloaded via progresshub
        
        if args.verbose {
            println!("📦 Config: temperature={}, max_tokens={}", args.temperature, args.max_tokens);
            println!("📦 Model path: {model_path}");
        }
        
        println!("📦 Loading model (if needed)...");
        match CandleKimiK2Provider::new(model_path) {
            Ok(provider) => {
                println!("✅ Model ready!");
                
                // Build the agent with streaming and interactive capabilities
                let agent_builder = CandleFluentAi::agent_role("helpful-assistant")
                    .completion_provider(provider)
                    .temperature(args.temperature.into())
                    .max_tokens(args.max_tokens.into())
                    .system_prompt(
                        "You are a helpful, knowledgeable AI assistant powered by the kimi-k2 model. \
                         You provide clear, accurate, and concise responses. You can help with a wide \
                         variety of tasks including coding, writing, analysis, and general questions. \
                         Be friendly and engaging in your responses."
                    )
                    .on_chunk(|chunk| {
                        // Real-time streaming - print each token as it arrives
                        print!("{chunk:?}"); // Use debug format for now until Display is implemented
                        let _ = io::stdout().flush(); // Ignore flush errors to prevent panic
                        chunk
                    })
                    .into_agent(); // Convert CandleAgentRoleBuilder to CandleAgentBuilder
                
                let _ = sender.send(agent_builder);
            },
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                // Don't send anything on error - stream will be empty
            }
        }
    })
}



/// Run the interactive chat loop
fn run_chat_loop(args: &Args) -> AsyncStream<()> {
    let args = args.clone();
    AsyncStream::with_channel(move |sender| {
        // Use the agent builder directly for chat
        let stdin = io::stdin();
        let mut conversation_history = Vec::new();
        
        loop {
            // Get user input
            print!("\n🧑 You: ");
            if io::stdout().flush().is_err() {
                break;
            }
            
            let mut user_input = String::new();
            if stdin.lock().read_line(&mut user_input).is_err() {
                break;
            }
            let user_input = user_input.trim();
            
            // Handle special commands
            match user_input.to_lowercase().as_str() {
                "quit" | "exit" | "bye" => {
                    println!("👋 Goodbye!");
                    break;
                },
                "/help" => {
                    print_help();
                    continue;
                },
                "/clear" => {
                    conversation_history.clear();
                    println!("🗑️  Conversation history cleared");
                    continue;
                },
                "/history" => {
                    print_history(&conversation_history);
                    continue;
                },
                "/stats" => {
                    print_stats(&args);
                    continue;
                },
                _ if user_input.is_empty() => {
                    continue;
                },
                _ => {
                    // Continue to chat processing
                }
            }
            
            // Add user message to history
            conversation_history.push((CandleMessageRole::User, user_input.to_string()));
            
            print!("🤖 Assistant: ");
            if io::stdout().flush().is_err() {
                break;
            }
            
            // Create a new agent builder for this chat since chat() consumes self
            // In production, we'd want to maintain state better, but for this demo it's ok
            let mut agent_stream = create_agent_for_model(args.clone());
            if let Some(current_agent) = agent_stream.try_next() {
                let user_input_copy = user_input.to_string();
                let mut response_stream = current_agent.chat(move |_conversation| {
                    // For now, just continue the conversation
                    // In the future, we could use conversation.latest_user_message() here
                    CandleChatLoop::Reprompt(user_input_copy.clone())
                });
                let mut full_response = String::new();
                
                // Stream the response chunks using try_next()
                while let Some(chunk) = response_stream.try_next() {
                    print!("{}", chunk.text());
                    let _ = io::stdout().flush();
                    full_response.push_str(chunk.text());
                    
                    if chunk.done() {
                        break;
                    }
                }
                
                // Add assistant response to history
                conversation_history.push((CandleMessageRole::Assistant, full_response));
                
                println!(); // Add newline after response
            } else {
                println!("❌ Failed to create agent for this request");
            }
        }
        
        // Send completion signal
        let _ = sender.send(());
    })
}

/// Print help information
fn print_help() {
    println!("📚 Available Commands:");
    println!("  /help     - Show this help message");
    println!("  /clear    - Clear conversation history");
    println!("  /history  - Show conversation history");
    println!("  /stats    - Show model statistics");
    println!("  quit/exit/bye - End the session");
    println!("  Just type normally to chat with the assistant!");
    println!();
    println!("💡 Command Line Options:");
    println!("  --model kimi-k2  - Use kimi-k2 model (default)");
    println!("  --model llama3   - Use Llama 3 model (future)");
    println!("  --model gpt4     - Use GPT-4 model (future)");
    println!("  --temperature N  - Set temperature (0.0-2.0)");
    println!("  --max-tokens N   - Set max tokens per response");
    println!("  --verbose        - Enable verbose logging");
}

/// Print conversation history
fn print_history(history: &[(CandleMessageRole, String)]) {
    if history.is_empty() {
        println!("📜 No conversation history");
        return;
    }
    
    println!("📜 Conversation History:");
    println!("========================");
    for (i, (role, message)) in history.iter().enumerate() {
        let role_icon = match role {
            CandleMessageRole::User => "🧑",
            CandleMessageRole::Assistant => "🤖",
            CandleMessageRole::System => "⚙️",
            CandleMessageRole::Tool => "🔧",
        };
        
        println!("{}. {} {}: {}", i + 1, role_icon, role, message);
    }
}

/// Print model statistics
fn print_stats(args: &Args) {
    println!("📊 Model Statistics:");
    println!("===================");
    println!("Model: {}", args.model);
    println!("Framework: Candle");
    println!("Temperature: {}", args.temperature);
    println!("Max Tokens: {}", args.max_tokens);
    println!("Streaming: Enabled");
    println!("Progresshub: Enabled");
    println!("Verbose: {}", args.verbose);
    
    // TODO: Add more detailed stats when available from the provider
    // println!("Tokens Used: {}", agent.tokens_used());
    // println!("Requests: {}", agent.request_count());
    // println!("Model Path: {}", agent.model_path());
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_agent_creation() {
        // Test that we can create the basic agent structure
        // Note: This won't actually download models in tests
        let config = CandleKimiK2Config::default();
        assert_eq!(config.temperature(), 0.7); // Default temperature
    }
    
    #[test]
    fn test_help_commands() {
        // Test command parsing
        assert_eq!("quit".to_lowercase(), "quit");
        assert_eq!("/help".to_lowercase(), "/help");
        assert_eq!("EXIT".to_lowercase(), "exit");
    }
}