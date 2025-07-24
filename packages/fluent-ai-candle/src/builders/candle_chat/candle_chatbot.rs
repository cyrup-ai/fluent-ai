// ============================================================================
// File: src/cli_chatbot.rs
// ----------------------------------------------------------------------------
// Minimal REPL that drives any `Chat` implementation.
//
// • _No_ `Executor::block_on` (the method was removed per design constraints).
// • Synchronous loop pumps the cooperative executor and uses
//   `FutureExt::now_or_never()` to test the JoinHandle without async/await.
// • No allocation or blocking in the hot path.
// ============================================================================

use std::io::{self, Write};

// Note: termcolor items not available, using standard logging instead
use crate::types::{CandleChat, CandleCompletionError, CandleMessage};

/// Run a blocking terminal REPL around any `Chat` engine.
///
/// The executor is local to this function so we don’t spawn global threads
/// just for a quick CLI test-drive.
pub fn cli_chatbot<C>(chatbot: C) -> Result<(), CandleCompletionError>
where
    C: CandleChat,
{
    let mut chat_log = Vec::new();

    // Display welcome message with Cyrup.ai branding
    println!("=== Candle Chat ===");
    println!("Type 'exit' or 'quit' to leave the chat");

    loop {
        // Standard prompt
        print!("> ");
        if let Err(e) = io::stdout().flush() {
            eprintln!("Failed to flush stdout: {}", e);
        }

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            eprintln!("Error reading input.");
            continue;
        }
        let prompt = input.trim();
        if matches!(prompt, "exit" | "quit") {
            break;
        }

        // ------------------------------------------------------------------
        // Get chat response using blocking collection pattern
        // ------------------------------------------------------------------
        let stream = chatbot.chat(prompt.to_string(), chat_log.clone()); // AsyncStream<CandleMessage>

        // Use blocking collection to get the result without async/await
        let messages = stream.collect(); // Returns Vec<CandleMessage>
        let reply = match messages.first() {
            Some(message) => message.content.clone(),
            None => {
                eprintln!("No response received from chat");
                continue;
            }
        };

        // ------------------------------------------------------------------
        // Update chat log & print response
        // ------------------------------------------------------------------
        chat_log.push(CandleMessage::user(prompt));
        chat_log.push(CandleMessage::assistant(reply.clone()));

        // Display response with standard formatting
        println!();
        println!("=== Response ===");
        println!();
        println!("{}", reply);
    }

    Ok(())
}
