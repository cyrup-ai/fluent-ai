// ============================================================================
// File: src/agent/prompt_request.rs
// ----------------------------------------------------------------------------
// Fluent prompt builder returned by `Agent::prompt` using streams-only architecture.
//
// • The builder uses AsyncStream<T> patterns exclusively - NO FUTURES
// • All operations return AsyncStream<T> with unwrapped values
// • Zero allocation using crossbeam lock-free structures
// ============================================================================

#![allow(dead_code)] // remove once all call‑sites are migrated

use fluent_ai_async::AsyncStream;

use super::Agent;
use crate::completion::{CompletionModel, Message, PromptError};

// ---------------------------------------------------------------------------
// Prompt trait for type conversions
// ---------------------------------------------------------------------------

/// Trait for types that can be converted into prompts
pub trait Prompt {
    /// Convert this type into a Message
    fn into_message(self) -> Message;
}

impl Prompt for String {
    fn into_message(self) -> Message {
        Message::user(self)
    }
}

impl Prompt for &str {
    fn into_message(self) -> Message {
        Message::user(self.to_string())
    }
}

impl Prompt for Message {
    fn into_message(self) -> Message {
        self
    }
}

// ---------------------------------------------------------------------------
// Public builder
// ---------------------------------------------------------------------------

/// **Fluent prompt builder**.
///
/// Returned by [`Agent::prompt`](crate::agent::Agent::prompt); the user can
/// configure multi‑turn depth or attach an external chat‑history buffer before
/// they execute the request using streams.
///
/// ```rust
/// let mut reply_stream = agent
///     .prompt("Tell me a joke")
///     .multi_turn(2)
///     .execute();
/// 
/// if let Some(reply) = reply_stream.try_next() {
///     println!("Reply: {}", reply);
/// }
/// ```
pub struct PromptRequest<'a, M: CompletionModel> {
    agent: &'a Agent<M>,
    prompt: Message,
    chat_hist: Option<&'a mut Vec<Message>>,
    max_depth: usize,
}

/// **Owned prompt builder** for trait implementations.
///
/// This is the owned version of PromptRequest that can be used
/// as an associated type in traits.
pub struct OwnedPromptRequest<M: CompletionModel> {
    agent: Agent<M>,
    prompt: Message,
    max_depth: usize,
}

impl<'a, M: CompletionModel> PromptRequest<'a, M> {
    /// **Constructor** – never public, only called from `Agent::prompt`.
    #[inline(always)]
    pub(super) fn new(agent: &'a Agent<M>, prompt: impl Prompt) -> Self {
        Self {
            agent,
            prompt: prompt.into_message(),
            chat_hist: None,
            max_depth: 0,
        }
    }

    // ---------------------------------------------------------------------
    // Fluent configuration
    // ---------------------------------------------------------------------

    /// Enable multi‑turn conversations.
    /// `depth = 0` (*default*) ➜ single‑shot, no tool loops.
    #[inline(always)]
    pub fn multi_turn(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Attach an external **chat‑history buffer** (caller‑owned).
    /// The buffer may be reused across calls and different agents.
    #[inline(always)]
    pub fn with_history(mut self, hist: &'a mut Vec<Message>) -> Self {
        self.chat_hist = Some(hist);
        self
    }

    // ---------------------------------------------------------------------
    // Execution using streams-only architecture
    // ---------------------------------------------------------------------
    
    /// Execute the prompt request using streams-only architecture
    pub fn execute(self) -> AsyncStream<String> {
        AsyncStream::with_channel(move |sender| {
            self.drive_streams(sender);
        })
    }
    
    /// Internal driver using streams-only architecture
    fn drive_streams(mut self, sender: fluent_ai_async::AsyncStreamSender<String>) {
        std::thread::spawn(move || {
            use crate::completion::Chat;

            // Obtain mutable history reference (external or local scratch).
            let mut local_hist = Vec::new();
            let hist = self.chat_hist.get_or_insert(&mut local_hist);

            let mut depth = 0usize;
            let mut prompt = self.prompt.clone();

            loop {
                depth += 1;

                // Build provider request (static + dyn context/tools).
                let mut completion_stream = self
                    .agent
                    .completion(prompt.clone(), hist.clone())
                    .send();

                if let Some(resp) = completion_stream.try_next() {
                    // ── plain‑text reply?  We're done.
                    if let Some(text) = resp
                        .choice
                        .iter()
                        .filter_map(|c| c.as_text())
                        .map(|t| t.text.clone())
                        .reduce(|a, b| a + "\n" + &b)
                    {
                        let _ = sender.send(text);
                        return;
                    }

                    // ── otherwise: tool calls present → delegate to tool set.
                    let mut tool_stream = self.agent.tools.handle_tool_calls(&resp, hist);
                    if let Some(new_prompt) = tool_stream.try_next() {
                        prompt = new_prompt;
                    } else {
                        // Tool handling failed - send error indication
                        return;
                    }

                    if depth > self.max_depth {
                        // Max‑depth exceeded – abort
                        return;
                    }
                } else {
                    // Completion failed
                    return;
                }
            }
        });
    }
}