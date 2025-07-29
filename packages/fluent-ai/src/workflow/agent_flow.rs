// ============================================================================
// File: src/workflow/agent_flow.rs
// ----------------------------------------------------------------------------
// Provider-agnostic “ops” glued into the workflow DSL.
// Each op is a tiny, allocation-free adapter around a lower-level primitive
// (vector search, prompt, extractor) so that complex chains can be composed
// ergonomically while still compiling down to zero overhead.
// ============================================================================

use std::future::IntoFuture;

use fluent_ai_async::AsyncStream;

use crate::{
    completion::message::Message,
    completion::{self, CompletionModel, Prompt, PromptError},
    extractor::{ExtractionError, Extractor},
    vector_store::{self, VectorStoreError, VectorStoreIndex, VectorStoreIndexDyn},
    workflow::Op, // updated path
};

// ============================================================================
// 1. Semantic-search lookup  --------------------------------------------------
// ============================================================================

/// Perform *n*-nearest-neighbour search against any [`VectorStoreIndex`].
#[derive(Clone)]
pub struct Lookup<I, In, T>
where
    I: VectorStoreIndexDyn + Send + Sync,
{
    index: I,
    n: usize,
    _in: std::marker::PhantomData<In>,
    _t: std::marker::PhantomData<T>}

impl<I, In, T> Lookup<I, In, T>
where
    I: VectorStoreIndexDyn + Send + Sync,
{
    #[inline(always)]
    pub(crate) fn new(index: I, n: usize) -> Self {
        Self {
            index,
            n,
            _in: std::marker::PhantomData,
            _t: std::marker::PhantomData}
    }
}

impl<I, In, T> Op for Lookup<I, In, T>
where
    I: VectorStoreIndexDyn + Send + Sync,
    In: Into<String> + Send + Sync,
    T: for<'de> serde::Deserialize<'de> + Send + Sync,
{
    type Input = In;
    type Output = Result<Vec<(f64, String, T)>, VectorStoreError>;

    #[inline(always)]
    fn call(&self, input: Self::Input) -> AsyncStream<Self::Output> {
        let query = input.into();
        let n = self.n;
        let index = self.index.clone();

        let (tx, stream) = AsyncStream::channel();
        tokio::spawn(async move {
            let result = index.top_n::<T>(&query, n).await;
            let _ = tx.send(result);
        });
        stream
    }
}

#[inline(always)]
pub fn lookup<I, In, T>(index: I, n: usize) -> Lookup<I, In, T>
where
    I: VectorStoreIndexDyn + Send + Sync,
    In: Into<String> + Send + Sync,
    T: for<'de> serde::Deserialize<'de> + Send + Sync,
{
    Lookup::new(index, n)
}

// ============================================================================
// 2. Prompt wrapper  ----------------------------------------------------------
// ============================================================================

/// Thin adapter converting any `Into<String>` into a user [`Message`] and
/// forwarding to [`Prompt::prompt`].
#[derive(Clone)]
pub struct PromptOp<P, In>
where
    P: Prompt + Send + Sync,
{
    prompt: P,
    _in: std::marker::PhantomData<In>}

impl<P, In> PromptOp<P, In>
where
    P: Prompt + Send + Sync,
{
    #[inline(always)]
    pub(crate) fn new(prompt: P) -> Self {
        Self {
            prompt,
            _in: std::marker::PhantomData}
    }
}

impl<P, In> Op for PromptOp<P, In>
where
    P: Prompt + Send + Sync,
    In: Into<String> + Send + Sync,
{
    type Input = In;
    type Output = Result<String, PromptError>;

    #[inline(always)]
    fn call(&self, input: Self::Input) -> AsyncStream<Self::Output> {
        let msg = Message::user(input.into());
        let (tx, stream) = AsyncStream::channel();
        tokio::spawn(async move {
            let result = self.prompt.prompt(msg).into_future().await;
            let _ = tx.send(result);
        });
        stream
    }
}

#[inline(always)]
pub fn prompt<P, In>(model: P) -> PromptOp<P, In>
where
    P: Prompt + Send + Sync,
    In: Into<String> + Send + Sync,
{
    PromptOp::new(model)
}

// ============================================================================
// 3. Structured-data extraction  ---------------------------------------------
// ============================================================================

/// Op invoking an [`Extractor`] and returning strongly-typed output.
#[derive(Clone)]
pub struct ExtractOp<M, In, Out>
where
    M: CompletionModel,
    Out: schemars::JsonSchema + for<'de> serde::Deserialize<'de> + Send + Sync,
{
    extractor: Extractor<M, Out>,
    _in: std::marker::PhantomData<In>}

impl<M, In, Out> ExtractOp<M, In, Out>
where
    M: CompletionModel,
    Out: schemars::JsonSchema + for<'de> serde::Deserialize<'de> + Send + Sync,
{
    #[inline(always)]
    pub(crate) fn new(extractor: Extractor<M, Out>) -> Self {
        Self {
            extractor,
            _in: std::marker::PhantomData}
    }
}

impl<M, In, Out> Op for ExtractOp<M, In, Out>
where
    M: CompletionModel,
    Out: schemars::JsonSchema + for<'de> serde::Deserialize<'de> + Send + Sync,
    In: Into<Message> + Send + Sync,
{
    type Input = In;
    type Output = Result<Out, ExtractionError>;

    #[inline(always)]
    fn call(&self, input: Self::Input) -> AsyncStream<Self::Output> {
        let message = input.into();
        let extractor = self.extractor.clone();

        let (tx, stream) = AsyncStream::channel();
        tokio::spawn(async move {
            let result = extractor.extract(message).await;
            let _ = tx.send(result);
        });
        stream
    }
}

#[inline(always)]
pub fn extract<M, In, Out>(extractor: Extractor<M, Out>) -> ExtractOp<M, In, Out>
where
    M: CompletionModel,
    Out: schemars::JsonSchema + for<'de> serde::Deserialize<'de> + Send + Sync,
    In: Into<Message> + Send + Sync,
{
    ExtractOp::new(extractor)
}

// ============================================================================
// End of file
// ============================================================================
