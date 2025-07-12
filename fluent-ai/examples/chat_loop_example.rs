use fluent_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example demonstrating the pure ChatLoop pattern
    // All formatting, streaming, and I/O are handled automatically by the builder
    
    FluentAi::agent("gpt-4o-mini")
        .system_prompt("You are a helpful assistant.")
        .on_chunk(|result| {
            // Handle streaming chunks automatically - no user code needed for formatting
            match result {
                Ok(chunk) => print!("{}", chunk),
                Err(error) => eprintln!("Error: {}", error),
            }
        })
        .chat(|conversation| {
            // Pure chat closure - no manual loop, no I/O, no formatting
            // Just read conversation and return ChatLoop control flow
            
            let last_message = conversation.last_user_message().unwrap_or("");
            
            match last_message.to_lowercase().as_str() {
                "exit" | "quit" | "bye" => ChatLoop::Break,
                "hello" => ChatLoop::Reprompt("Hello! How can I help you today?"),
                "help" => ChatLoop::Reprompt(
                    "I can assist with various tasks. Try asking me questions or type 'exit' to quit."
                ),
                _ => {
                    // Default response - continue conversation
                    ChatLoop::Reprompt(
                        "I understand. Let me help you with that. What would you like to know more about?"
                    )
                }
            }
        })
        .await?;

    Ok(())
}
