use std::io::{self, Write};
use fluent_ai_candle::{
    builders::agent_role::{CandleFluentAi, CandleAgentRoleBuilder, CandleAgentBuilder},
    providers::{kimi_k2::CandleKimiK2Provider, qwen3_coder::CandleQwen3CoderProvider},
    domain::{
        chat::CandleChatLoop,
        completion::{CandleCompletionModel, CandleCompletionParams},
        prompt::CandlePrompt,
    },
};
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The model to use for inference
    #[arg(short, long, default_value = "kimi-k2")]
    model: String,
    
    /// Temperature for sampling (0.0 to 2.0)
    #[arg(short, long, default_value = "0.7")]
    temperature: f64,
    
    /// Maximum tokens to generate
    #[arg(long, default_value = "2000")]
    max_tokens: u64,
    
    /// System prompt to use
    #[arg(long, default_value = "You are a helpful AI assistant using the Candle ML framework for local inference.")]
    system_prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("🤖 Starting Candle Agent Chat Completion with model: {}", args.model);
    println!("📦 Downloading model and initializing provider...");
    
    // Create the provider based on the model argument
    match args.model.as_str() {
        "kimi-k2" => {
            println!("🚀 Initializing Kimi-K2 provider with ProgressHub...");
            let provider = CandleKimiK2Provider::new().await.map_err(|e| format!("Failed to create Kimi-K2 provider: {}", e))?;
            run_chat(provider, args.temperature, args.max_tokens, &args.system_prompt).await
        },
        "qwen-coder" => {
            println!("🚀 Initializing Qwen3-Coder provider with ProgressHub...");
            let provider = CandleQwen3CoderProvider::new().await.map_err(|e| format!("Failed to create Qwen3-Coder provider: {}", e))?;
            run_chat_qwen(provider, args.temperature, args.max_tokens, &args.system_prompt).await
        },
        _ => {
            return Err(format!("Unknown model: {}. Available models: kimi-k2, qwen-coder", args.model).into());
        }
    }
}

async fn run_chat(
    provider: CandleKimiK2Provider, 
    temperature: f64,
    max_tokens: u64,
    system_prompt: &str
) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Provider ready! Starting chat...");
    
    // Use the beautiful fluent API 
    let _stream = CandleFluentAi::agent_role("helpful-assistant")
        .completion_provider(provider)
        .temperature(temperature)
        .max_tokens(max_tokens)
        .system_prompt(system_prompt)
        .on_chunk(|chunk| {
            // Real-time streaming - print each token as it arrives
            print!("{}", chunk);
            io::stdout().flush().unwrap();
            chunk
        })
        .into_agent()
        .chat(|conversation| {
            let user_input = conversation.latest_user_message();
            
            match user_input.to_lowercase().as_str() {
                "quit" | "exit" | "bye" => {
                    println!("\n👋 Goodbye!");
                    CandleChatLoop::Break
                },
                _ => {
                    CandleChatLoop::Reprompt("Hello, can you help me understand how the Candle ML framework works with the Kimi-K2 model?".to_string())
                }
            }
        });
    
    Ok(())
}

async fn run_chat_qwen(
    provider: CandleQwen3CoderProvider, 
    _temperature: f64,
    _max_tokens: u64,
    _system_prompt: &str
) -> Result<(), Box<dyn std::error::Error>> {
    println!("✅ Provider ready! Starting chat...");
    
    // For Qwen3-Coder, we'll implement basic streaming for now since it doesn't have the full CandleCompletionModel trait
    println!("🤖 Qwen3-Coder: Hello! I'm ready to help you with coding tasks.");
    println!("💻 Ask me anything about programming, and I'll generate code using the Qwen3-Coder model.");
    println!("📝 Type 'quit', 'exit', or 'bye' to end the chat.\n");
    
    loop {
        print!("You: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "bye" => {
                println!("\n👋 Goodbye!");
                break;
            },
            _ => {
                print!("🤖 Qwen3-Coder: ");
                io::stdout().flush()?;
                
                // Use the provider's prompt method (new API)
                use std::num::NonZeroU64;
                let candle_prompt = CandlePrompt::new(input);
                let mut params = CandleCompletionParams::default();
                params.temperature = 0.7;
                params.max_tokens = NonZeroU64::new(1000);
                let mut stream = provider.prompt(candle_prompt, &params);
                
                while let Some(chunk) = stream.next() {
                    use fluent_ai_candle::domain::completion::CandleCompletionChunk;
                    match chunk {
                        CandleCompletionChunk::Text(text) => {
                            print!("{}", text);
                            io::stdout().flush()?;
                        },
                        CandleCompletionChunk::Complete { .. } => break,
                        CandleCompletionChunk::Error(err) => {
                            println!("Error: {}", err);
                            break;
                        },
                        _ => {}
                    }
                }
                println!(); // New line after response
            }
        }
    }
    
    Ok(())
}