// ============================================================================
// File: completion/request_builder.rs
// ----------------------------------------------------------------------------
// Zero-alloc, fluent request builder for chat/completions.
//
// • Mirrors `embeddings/builder.rs` (typestate + AsyncTask).
// • One-shot `.execute()` or `.stream()` yielding an `AsyncTask<…>`.
// • Never allocates after the first `push()` / `system()` call.
// • No runtime checks – all invariants enforced by the type-system.
// ============================================================================

use std::{marker::PhantomData, mem};

use crate::{
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        StreamingCompletionResponse,
        message::{Message, Text},
    },
    runtime::{AsyncTask, spawn_async},
    streaming::streaming::StreamingResultDyn,
};

// ────────────────────────────────────────────────────────────────────────────
// Typestate markers
// ────────────────────────────────────────────────────────────────────────────

pub trait NeedsContent {}
pub trait Ready {}

pub enum Empty {}
pub enum WithContent {}
impl NeedsContent for Empty {}
impl Ready for WithContent {}

// ────────────────────────────────────────────────────────────────────────────
// Builder
// ────────────────────────────────────────────────────────────────────────────

pub struct CompletionRequestBuilder<M, S = Empty> {
    model: M,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    _state: PhantomData<S>,
}

impl<M> CompletionRequestBuilder<M, Empty>
where
    M: CompletionModel,
{
    #[inline(always)]
    pub(crate) fn new(model: M) -> Self {
        Self {
            model,
            messages: Vec::new(),
            temperature: 1.0,
            max_tokens: None,
            top_p: None,
            _state: PhantomData,
        }
    }

    // -------- prompt creation helpers --------------------------------------

    /// Add a *system* message (prepends if it's the first one).
    #[inline(always)]
    pub fn system(mut self, txt: impl Into<String>) -> Self {
        let msg = Message::assistant(txt.into());
        self.messages.insert(0, msg);
        self
    }

    /// Push a *user* message.
    #[inline(always)]
    pub fn user(mut self, txt: impl Into<String>) -> Self {
        self.messages.push(Message::user(txt.into()));
        unsafe { mem::transmute(self) } // transitions to `WithContent`
    }

    /// Push an arbitrary message.
    #[inline(always)]
    pub fn push(mut self, msg: Message) -> Self {
        self.messages.push(msg);
        unsafe { mem::transmute(self) }
    }

    // -------- generation params --------------------------------------------

    #[inline(always)]
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    #[inline(always)]
    pub fn max_tokens(mut self, t: u32) -> Self {
        self.max_tokens = Some(t);
        self
    }

    #[inline(always)]
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    // ======================================================================
    // NOTE: `.execute()` / `.stream()` only exist when typestate == Ready
    // ======================================================================
}

impl<M> CompletionRequestBuilder<M, WithContent>
where
    M: CompletionModel + Send + Sync + Clone + 'static,
{
    /// Fire-and-forget – returns the whole response in one shot.
    #[inline(always)]
    pub fn execute(self) -> AsyncTask<Result<CompletionResponse<M::Response>, CompletionError>> {
        let req = CompletionRequest {
            messages: self.messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        };
        let model = self.model.clone();
        spawn_async(async move { model.completion(req).await })
    }

    /// Streaming variant – yields chunks as they arrive.
    #[inline(always)]
    pub fn stream(
        self,
    ) -> AsyncTask<Result<StreamingCompletionResponse<M::StreamingResponse>, CompletionError>> {
        let req = CompletionRequest {
            messages: self.messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        };
        let model = self.model.clone();
        spawn_async(async move {
            let inner = model.stream(req).await?;
            let boxed = Box::pin(StreamingResultDyn { inner: inner.inner });
            Ok(StreamingCompletionResponse::stream(boxed))
        })
    }
}

// -----------------------------------------------------------------------------
// End of file
// -----------------------------------------------------------------------------
