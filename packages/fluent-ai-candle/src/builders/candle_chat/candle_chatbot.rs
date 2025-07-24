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
use std::thread::yield_now;

use futures_util::FutureExt; // for `now_or_never`
// Note: termcolor items not available, using standard logging instead

use crate::types::{
    CandleMessage, CandleCompletionError,
    CandleChat,
};
use fluent_ai_async::AsyncTask as Executor; // single-threaded coop executor

/// Run a blocking terminal REPL around any `Chat` engine.
///
/// The executor is local to this function so we don’t spawn global threads
/// just for a quick CLI test-drive.
pub fn cli_chatbot<C>(chatbot: C) -> Result<(), CandleCompletionError>
where
    C: CandleChat,
{
    let mut chat_log = Vec::new();
    let executor = Executor::default();

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
        // Spawn the chat task onto our single-thread executor
        // ------------------------------------------------------------------
        let task = chatbot.chat(prompt, chat_log.clone()); // AsyncTask<Result<_,_>>
        let mut handle = executor.spawn(task); // runtime::JoinHandle<_>

        // Drive the executor cooperatively until the task completes.
        let reply = loop {
            executor.drain(); // runs any ready tasks

            if let Some(res) = handle.now_or_never() {
                break res?; // propagate PromptError
            }

            // Give other threads a chance (keeps CLI responsive on heavy load)
            yield_now();
        };

        // ------------------------------------------------------------------
        // Update chat log & print response
        // ------------------------------------------------------------------
        chat_log.push(CandleMessage::user(prompt));
        chat_log.push(CandleMessage::assistant(&reply));

        // Display response with standard formatting
        println!();
        println!("=== Response ===");
        println!();
        println!("{}", reply);
    }

    Ok(())
}
