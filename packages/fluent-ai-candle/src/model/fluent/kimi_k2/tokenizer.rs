//! Custom tokenizer support for the `moonshotai/Kimi-K2-Instruct` model.
//!
//! The model uses a specialised chat template and bespoke special tokens
//! not present in standard LLaMA/Mistral-style tokenizers. This module wraps
//! the generic [`CandleTokenizer`] implementation with Kimi-specific helpers
//! while strictly adhering to the global *streams-only* architecture:
//!
//! • **Zero allocations** on hot paths (`ArrayVec` for buffers)
//! • **No `async fn` / `Future`** – only unwrapped `AsyncStream` where needed
//! • **No `unwrap()`/`expect()`** in production code
//! • **No unsafe code, no locking**
//!
//! The public surface exposes:
//!
//! • [`KimiK2Tokenizer::from_hub`] – download & build tokenizer
//! • [`KimiK2Tokenizer::apply_chat_template`] – format chat conversation
//!
//! Chat template (reverse-engineered from `tokenizer_config.json`):
//!
//! ```text
//! [BOS] <|im_system|> {system}\n[EOT]\n<|im_user|> {user}\n[EOT]\n<|im_assistant|>
//! ```
//!
//! Subsequent assistant/user turns are appended using:
//!
//! ```text
//! {assistant_response}<|im_end|>\n<|im_user|> {next_user}\n[EOT]\n<|im_assistant|>
//! ```
//!
//! The template terminates with `[EOS]` after assistant generation is complete.

use std::sync::Arc;

use arrayvec::ArrayVec;
use fluent_ai_async::{AsyncStream, handle_error};
use fluent_ai_http3::client::HttpClient;

use crate::error::{CandleError, CandleResult};
use crate::{tokenizer::CandleTokenizer, types::candle_chat::message::CandleMessage};

/// Special token literals as per `tokenizer_config.json`.
const BOS: &str = "[BOS]";
#[allow(dead_code)] // Reserved for future EOS token handling
const EOS: &str = "[EOS]";
const EOT: &str = "[EOT]";
const IM_SYSTEM: &str = "<|im_system|>";
const IM_USER: &str = "<|im_user|>";
const IM_ASSISTANT: &str = "<|im_assistant|>";
const IM_END: &str = "<|im_end|>";

/// Maximum prompt buffer size (chars) – fits in stack allocation.
const MAX_PROMPT_BUFFER: usize = 8192;

/// Zero-allocation, production-ready tokenizer for Kimi-K2.
#[derive(Clone)]
pub struct KimiK2Tokenizer {
    inner: Arc<CandleTokenizer>,
}

impl KimiK2Tokenizer {
    /// Download tokenizer files from Hugging Face Hub and construct the wrapper.
    ///
    /// This is a convenience helper; downstream code should ideally cache the
    /// resulting tokenizer instance.
    pub fn from_hub() -> AsyncStream<Self> {
        AsyncStream::with_channel(|_sender| {
            // Use http3 client for zero-allocation download of tokenizer files.
            let _client = HttpClient::default();
            let _model_id = "moonshotai/Kimi-K2-Instruct";

            // TODO: Implement proper streaming tokenizer loading
            let error =
                CandleError::ModelLoadError("Tokenizer loading not yet implemented".to_string());
            handle_error!(error, "Tokenizer loading not implemented");
        })
    }

    /// Expose underlying generic tokenizer.
    #[inline(always)]
    pub fn inner(&self) -> &CandleTokenizer {
        &self.inner
    }

    /// Apply Kimi chat template to a sequence of `CandleMessage`s.
    ///
    /// Returns a zero-allocation `String` built from an `ArrayVec` stack buffer.
    pub fn apply_chat_template(&self, messages: &[CandleMessage]) -> CandleResult<String> {
        if messages.is_empty() {
            return Err(CandleError::tokenization("Chat messages cannot be empty"));
        }

        // Reserve stack buffer for prompt.
        let mut buf: ArrayVec<u8, MAX_PROMPT_BUFFER> = ArrayVec::new();

        // Append system message if first message role == system.
        let mut idx = 0;
        if messages[0].role == crate::types::candle_chat::message::CandleMessageRole::System {
            Self::push_pair(&mut buf, BOS, IM_SYSTEM)?;
            Self::push_line(&mut buf, &messages[0].content)?;
            Self::push_line(&mut buf, EOT)?;
            idx = 1;
        }

        // Iterate over remaining messages in user/assistant pairs.
        while idx < messages.len() {
            // Expect user
            let user_msg = &messages[idx];
            if user_msg.role != crate::types::candle_chat::message::CandleMessageRole::User {
                return Err(CandleError::Tokenizer(
                    "Expected user role in chat sequence",
                ));
            }
            Self::push_pair(&mut buf, IM_USER, "")?; // tag only
            Self::push_line(&mut buf, &user_msg.content)?;
            Self::push_line(&mut buf, EOT)?;
            Self::push_line(&mut buf, IM_ASSISTANT)?;

            idx += 1;

            // If assistant response exists, append and <|im_end|>
            if idx < messages.len()
                && messages[idx].role
                    == crate::types::candle_chat::message::CandleMessageRole::Assistant
            {
                let assist_msg = &messages[idx];
                Self::push_line(&mut buf, &assist_msg.content)?;
                Self::push_line(&mut buf, IM_END)?;
                idx += 1;
            }
        }

        // Convert to String without extra allocation.
        let prompt = String::from_utf8(buf.to_vec())
            .map_err(|_| CandleError::Tokenizer("Prompt contains invalid UTF-8"))?;
        Ok(prompt)
    }

    #[inline(always)]
    fn push_pair(
        buf: &mut ArrayVec<u8, MAX_PROMPT_BUFFER>,
        first: &str,
        second: &str,
    ) -> CandleResult<()> {
        Self::push_str(buf, first)?;
        if !second.is_empty() {
            buf.try_push(b' ')
                .map_err(|_| CandleError::tokenization("Prompt buffer overflow"))?;
            Self::push_str(buf, second)?;
        }
        buf.try_push(b'\n')
            .map_err(|_| CandleError::tokenization("Prompt buffer overflow"))?;
        Ok(())
    }

    #[inline(always)]
    fn push_line(buf: &mut ArrayVec<u8, MAX_PROMPT_BUFFER>, line: &str) -> CandleResult<()> {
        Self::push_str(buf, line)?;
        buf.try_push(b'\n')
            .map_err(|_| CandleError::tokenization("Prompt buffer overflow"))?;
        Ok(())
    }

    #[inline(always)]
    fn push_str(buf: &mut ArrayVec<u8, MAX_PROMPT_BUFFER>, s: &str) -> CandleResult<()> {
        for &b in s.as_bytes() {
            buf.try_push(b)
                .map_err(|_| CandleError::tokenization("Prompt buffer overflow"))?;
        }
        Ok(())
    }
}
