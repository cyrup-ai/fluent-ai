use std::io::{self, Write};
use fluent_ai_candle::{
    builders::agent_role::{CandleFluentAi, CandleAgentRoleBuilder, CandleAgentBuilder},
    providers::kimi_k2::CandleKimiK2Provider,
    chat::CandleChatLoop,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Starting Candle Agent Chat Completion...");
    
    // Create the provider
    let provider = CandleKimiK2Provider::new("./models/kimi-k2")?;
    
    // Use the beautiful fluent API exactly like ARCHITECTURE.md
    let stream = CandleFluentAi::agent_role("helpful-assistant")
        .completion_provider(provider)
        .temperature(0.7)
        .max_tokens(2000)
        .system_prompt("You are a helpful AI assistant using the Candle ML framework for local inference.")
        .into_agent()
        .chat("Hello, can you help me understand how the Candle ML framework works?")
        .collect();
    
    // Print the response
    for chunk in stream {
        print!("{}", chunk.text());
        io::stdout().flush()?;
    }
    println!();
    
    Ok(())
}