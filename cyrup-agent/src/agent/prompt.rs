// ============================================================================
// File: src/agent/prompt_request.rs
// ----------------------------------------------------------------------------
// Fluent *synchronous* prompt builder returned by `Agent::prompt`.
//
// • The builder is 100 % sync; **no `async fn` in the public API**.
// • `agent.prompt("hi").await` works via `IntoFuture` that spawns *one*
//   allocation‑free [`AsyncTask`] behind the scenes (see `crate::runtime`).
// • Absolutely **no `BoxFuture`, no `async_trait`, no heap allocation** on the
//   hot path once the `AsyncTask` is running.
// ============================================================================

#![allow(dead_code)] // remove once all call‑sites are migrated

use std::future::IntoFuture;

use crate::{
    completion::{CompletionModel, Message, PromptError},
    runtime as rt,      // re‑export of the zero‑alloc runtime
    runtime::AsyncTask, // one‑shot task handle
};

use super::Agent;

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
/// they `.await` the request.
///
/// ```rust
/// let reply = agent
///     .prompt("Tell me a joke")
///     .multi_turn(2)
///     .await?;
/// ```
pub struct PromptRequest<'a, M: CompletionModel> {
    agent: &'a Agent<M>,
    prompt: Message,
    chat_hist: Option<&'a mut Vec<Message>>,
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
    /// `depth = 0` (*default*) ➜ single‑shot, no tool loops.
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
    // Internal async driver — **never** exposed in the public surface.
    // ---------------------------------------------------------------------
    async fn drive(mut self) -> Result<String, PromptError> {
        use crate::completion::Chat;

        // Obtain mutable history reference (external or local scratch).
        let mut local_hist = Vec::new();
        let hist = self.chat_hist.get_or_insert(&mut local_hist);

        let mut depth = 0usize;
        let mut prompt = self.prompt.clone();

        loop {
            depth += 1;

            // Build provider request (static + dyn context/tools).
            let resp = self
                .agent
                .completion(prompt.clone(), hist.clone())
                .await? // → CompletionRequestBuilder
                .send() // provider network I/O
                .await?;

            // ── plain‑text reply?  We’re done.
            if let Some(text) = resp
                .choice
                .iter()
                .filter_map(|c| c.as_text())
                .map(|t| t.text.clone())
                .reduce(|a, b| a + "\n" + &b)
            {
                return Ok(text);
            }

            // ── otherwise: tool calls present → delegate to tool set.
            prompt = self.agent.tools.handle_tool_calls(&resp, hist).await?;

            if depth > self.max_depth {
                // Max‑depth exceeded – abort with detailed error.
                return Err(PromptError::MaxDepthError {
                    max_depth: self.max_depth,
                    chat_history: hist.clone(),
                    prompt,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `IntoFuture` glue so that `.await` on the builder just works.
// ---------------------------------------------------------------------------

impl<'a, M: CompletionModel> IntoFuture for PromptRequest<'a, M> {
    type Output = Result<String, PromptError>;
    type IntoFuture = AsyncTask<Self::Output>; // ← zero‑alloc task handle

    #[inline(always)]
    fn into_future(self) -> Self::IntoFuture {
        rt::spawn_async(self.drive())
    }
}
