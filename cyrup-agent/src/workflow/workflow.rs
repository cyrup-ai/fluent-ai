// ============================================================================
// File: src/workflow/workflow.rs
// ----------------------------------------------------------------------------
// Core building-blocks for the pipeline DSL.
//
//  • `Op`      – zero-cost, async-aware transformation node
//  • `Sequential` combinator for fluent chaining
//  • Helpers: `map`, `then`, `lookup`, `prompt`, `passthrough`
//
// Every combinator returns a concrete type (no boxing) so the compiler can
// fully optimise the call-graph.  All hot-path methods are `#[inline]`.
// ============================================================================

use std::future::Future;

#[allow(unused_imports)] // used in downstream macro expansion
use futures::join;
use futures::stream;

use crate::{completion, vector_store};

// ================================================================
// 0. Op trait – the universal node
// ================================================================
pub trait Op: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;

    fn call(&self, input: Self::Input) -> impl Future<Output = Self::Output> + Send;

    /// Execute this op over an iterator of inputs with at most `n` concurrent
    /// in-flight tasks.
    fn batch_call<I>(&self, n: usize, input: I) -> impl Future<Output = Vec<Self::Output>> + Send
    where
        I: IntoIterator<Item = Self::Input> + Send,
        I::IntoIter: Send,
        Self: Sized,
    {
        use futures::stream::StreamExt;

        async move {
            stream::iter(input)
                .map(|v| self.call(v))
                .buffered(n)
                .collect()
                .await
        }
    }

    /* ---------------------------------------------------------------------
     * Fluent combinators
     * ------------------------------------------------------------------ */
    #[inline]
    fn map<F, Out2>(self, f: F) -> Sequential<Self, Map<F, Self::Output>>
    where
        Self: Sized,
        F: Fn(Self::Output) -> Out2 + Send + Sync,
        Out2: Send + Sync,
    {
        Sequential::new(self, Map::new(f))
    }

    #[inline]
    fn then<F, Fut>(self, f: F) -> Sequential<Self, Then<F, Fut::Output>>
    where
        Self: Sized,
        F: Fn(Self::Output) -> Fut + Send + Sync,
        Fut: Future + Send,
        Fut::Output: Send + Sync,
    {
        Sequential::new(self, Then::new(f))
    }

    #[inline]
    fn chain<O>(self, op: O) -> Sequential<Self, O>
    where
        Self: Sized,
        O: Op<Input = Self::Output>,
    {
        Sequential::new(self, op)
    }

    #[inline]
    fn lookup<Ix, Doc>(
        self,
        index: Ix,
        n: usize,
    ) -> Sequential<Self, crate::workflow::agent_ops::Lookup<Ix, Self::Output, Doc>>
    where
        Self: Sized,
        Ix: vector_store::VectorStoreIndex,
        Doc: for<'a> serde::Deserialize<'a> + Send + Sync,
        Self::Output: Into<String>,
    {
        Sequential::new(self, crate::workflow::agent_ops::Lookup::new(index, n))
    }

    #[inline]
    fn prompt<P>(
        self,
        agent: P,
    ) -> Sequential<Self, crate::workflow::agent_ops::Prompt<P, Self::Output>>
    where
        Self: Sized,
        P: completion::Prompt,
        Self::Output: Into<String>,
    {
        Sequential::new(self, crate::workflow::agent_ops::Prompt::new(agent))
    }
}

// Blanket impl so `&Op` can be used interchangeably.
impl<T: Op> Op for &T {
    type Input = T::Input;
    type Output = T::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (**self).call(input).await
    }
}

// ================================================================
// 1. Sequential – glue two ops together
// ================================================================
pub struct Sequential<A, B> {
    a: A,
    b: B,
}

impl<A, B> Sequential<A, B> {
    #[inline(always)]
    pub(crate) fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Op for Sequential<A, B>
where
    A: Op,
    B: Op<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        let mid = self.a.call(input).await;
        self.b.call(mid).await
    }
}

// ================================================================
// 2. Primitive ops
// ================================================================
pub struct Map<F, In> {
    f: F,
    _pd: std::marker::PhantomData<In>,
}

impl<F, In> Map<F, In> {
    #[inline(always)]
    fn new(f: F) -> Self {
        Self {
            f,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<F, In, Out> Op for Map<F, In>
where
    F: Fn(In) -> Out + Send + Sync,
    In: Send + Sync,
    Out: Send + Sync,
{
    type Input = In;
    type Output = Out;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input)
    }
}

pub fn map<F, In, Out>(f: F) -> Map<F, In>
where
    F: Fn(In) -> Out + Send + Sync,
    In: Send + Sync,
    Out: Send + Sync,
{
    Map::new(f)
}

pub struct Then<F, In> {
    f: F,
    _pd: std::marker::PhantomData<In>,
}

impl<F, In> Then<F, In> {
    #[inline(always)]
    fn new(f: F) -> Self {
        Self {
            f,
            _pd: std::marker::PhantomData,
        }
    }
}

impl<F, In, Fut> Op for Then<F, In>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: Send + Sync,
    Fut: Future + Send,
    Fut::Output: Send + Sync,
{
    type Input = In;
    type Output = Fut::Output;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        (self.f)(input).await
    }
}

pub fn then<F, In, Fut>(f: F) -> Then<F, In>
where
    F: Fn(In) -> Fut + Send + Sync,
    In: Send + Sync,
    Fut: Future + Send,
    Fut::Output: Send + Sync,
{
    Then::new(f)
}

/// Identity node – forwards the input unchanged.
pub struct Passthrough<T>(std::marker::PhantomData<T>);

impl<T> Passthrough<T> {
    #[inline(always)]
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: Send + Sync> Op for Passthrough<T> {
    type Input = T;
    type Output = T;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        input
    }
}

pub fn passthrough<T: Send + Sync>() -> Passthrough<T> {
    Passthrough::new()
}

/* --------------------------------------------------------------------------
 * Tests
 * ----------------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sequential_runs_in_order() {
        let pipeline = map(|x: i32| x + 1)
            .map(|x| x * 2)
            .then(|x| async move { x * 3 });

        assert_eq!(pipeline.call(1).await, 12);
    }

    #[tokio::test]
    async fn batch_processing() {
        let op = map(|x: i32| x + 1);
        let data = vec![1, 2, 3, 4];

        let out = op.batch_call(2, data).await;
        assert_eq!(out, vec![2, 3, 4, 5]);
    }
}
