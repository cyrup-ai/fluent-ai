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

use crate::{
    completion::{Chat, Message, PromptError},
    runtime::Executor, // single-threaded coop executor
};

/// Run a blocking terminal REPL around any `Chat` engine.
///
/// The executor is local to this function so we don’t spawn global threads
/// just for a quick CLI test-drive.
pub fn cli_chatbot<C>(chatbot: C) -> Result<(), PromptError>
where
    C: Chat,
{
    let mut chat_log = Vec::new();
    let executor = Executor::default();

    println!("Welcome to the chatbot! Type `exit` or `quit` to leave.");

    loop {
        print!("> ");
        io::stdout().flush().expect("stdout flush");

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
        chat_log.push(Message::user(prompt));
        chat_log.push(Message::assistant(&reply));

        println!("\n=== Response ===\n{reply}\n");
    }

    Ok(())
}
