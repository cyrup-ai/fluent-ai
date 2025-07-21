// ============================================================================
// File: src/pipeline/try_ops.rs
// ----------------------------------------------------------------------------
// Fallible (“try-”) combinators for the async `Op` abstraction.  Mirrors the
// ergonomics of `Result` while preserving structured, ergonomic error flow
// across pipelined operations.
// ============================================================================

use fluent_ai_http3::async_task::AsyncStream;

use futures::stream::{self, StreamExt, TryStreamExt};
use futures::{join, try_join};

use super::{Map, Op, Then};

// ================================================================
// 0. Core trait – any `Op` that yields a `Result` automatically
//    implements `TryOp` via the blanket impl further below.
// ================================================================
pub trait TryOp: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;
    type Error: Send + Sync;

    fn try_call(
        &self,
        input: Self::Input,
    ) -> AsyncStream<Result<Self::Output, Self::Error>>;

    // ---------- fan-out with bounded parallelism ----------
    #[inline]
    fn try_batch_call<I>(
        &self,
        concurrency: usize,
        input: I,
    ) -> AsyncStream<Result<Vec<Self::Output>, Self::Error>>
    where
        I: IntoIterator<Item = Self::Input> + Send,
        I::IntoIter: Send,
        Self: Sized + Clone,
    {
        let (tx, stream) = AsyncStream::channel();
        let inputs: Vec<_> = input.into_iter().collect();
        let op = self.clone();
        
        tokio::spawn(async move {
            let mut results = Vec::with_capacity(inputs.len());
            let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(concurrency));
            
            let mut handles = Vec::new();
            
            for input_item in inputs {
                let permit = semaphore.clone().acquire_owned().await.unwrap();
                let op_clone = op.clone();
                
                let handle = tokio::spawn(async move {
                    let mut call_stream = op_clone.try_call(input_item);
                    let result = call_stream.next();
                    drop(permit);
                    result
                });
                handles.push(handle);
            }
            
            let mut final_results = Vec::new();
            let mut has_error = None;
            
            for handle in handles {
                match handle.await {
                    Ok(Some(Ok(value))) => final_results.push(value),
                    Ok(Some(Err(error))) => {
                        has_error = Some(error);
                        break;
                    }
                    _ => continue,
                }
            }
            
            let result = match has_error {
                Some(error) => Err(error),
                None => Ok(final_results),
            };
            let _ = tx.send(result);
        });
        
        stream
    }

    // ---------- success-branch combinators ----------
    #[inline]
    fn map_ok<F, O>(self, f: F) -> MapOk<Self, Map<F, Self::Output>>
    where
        F: Fn(Self::Output) -> O + Send + Sync,
        O: Send + Sync,
        Self: Sized,
    {
        MapOk::new(self, Map::new(f))
    }

    #[inline]
    fn and_then<F, Fut, O>(self, f: F) -> AndThen<Self, Then<F, Self::Output>>
    where
        F: Fn(Self::Output) -> Fut + Send + Sync,
        Fut: Future<Output = Result<O, Self::Error>> + Send + Sync,
        O: Send + Sync,
        Self: Sized,
    {
        AndThen::new(self, Then::new(f))
    }

    #[inline]
    fn chain_ok<N>(self, op: N) -> TrySequential<Self, N>
    where
        N: Op<Input = Self::Output>,
        Self: Sized,
    {
        TrySequential::new(self, op)
    }

    // ---------- error-branch combinators ----------
    #[inline]
    fn map_err<F, E2>(self, f: F) -> MapErr<Self, Map<F, Self::Error>>
    where
        F: Fn(Self::Error) -> E2 + Send + Sync,
        E2: Send + Sync,
        Self: Sized,
    {
        MapErr::new(self, Map::new(f))
    }

    #[inline]
    fn or_else<F, Fut, E2>(self, f: F) -> OrElse<Self, Then<F, Self::Error>>
    where
        F: Fn(Self::Error) -> Fut + Send + Sync,
        Fut: Future<Output = Result<Self::Output, E2>> + Send + Sync,
        E2: Send + Sync,
        Self: Sized,
    {
        OrElse::new(self, Then::new(f))
    }
}

// blanket-impl: plug any `Op<Output = Result<_, _>>` directly into the
// `TryOp` ecosystem.
impl<O, T, E> TryOp for O
where
    O: Op<Output = Result<T, E>>,
    T: Send + Sync,
    E: Send + Sync,
{
    type Input = O::Input;
    type Output = T;
    type Error = E;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        self.call(input).await
    }
}

// ================================================================
// 1. Combinator implementations
// ================================================================
macro_rules! ok_err_wrapper {
    // generate MapOk / MapErr / AndThen / OrElse bodies succinctly
    ($name:ident, $ok:block, $err:block) => {
        pub struct $name<A, B> {
            prev: A,
            op: B,
        }
        impl<A, B> $name<A, B> {
            #[inline]
            fn new(prev: A, op: B) -> Self {
                Self { prev, op }
            }
        }

        impl<A, B> Op for $name<A, B>
        where
            A: TryOp,
            B: Op<Input = A::Output>,
        {
            type Input = A::Input;
            type Output = Result<B::Output, A::Error>;

            #[inline]
            async fn call(&self, input: Self::Input) -> Self::Output {
                match self.prev.try_call(input).await {
                    Ok(v) => $ok,
                    Err(e) => $err,
                }
            }
        }
    };
}

// ok_err_wrapper!(MapOk, { Ok(self.op.call(v).await) }, { Err(e) });

pub struct MapOk<A, B> {
    prev: A,
    op: B,
}

impl<A, B> MapOk<A, B> {
    #[inline]
    fn new(prev: A, op: B) -> Self {
        Self { prev, op }
    }
}

impl<A, B> Op for MapOk<A, B>
where
    A: TryOp,
    B: Op<Input = A::Output>,
{
    type Input = A::Input;
    type Output = Result<B::Output, A::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(v) => Ok(self.op.call(v).await),
            Err(e) => Err(e),
        }
    }
}

pub struct MapErr<A, B> {
    prev: A,
    op: B,
}
impl<A, B> MapErr<A, B> {
    #[inline]
    fn new(prev: A, op: B) -> Self {
        Self { prev, op }
    }
}
impl<A, B> Op for MapErr<A, B>
where
    A: TryOp,
    B: Op<Input = A::Error>,
{
    type Input = A::Input;
    type Output = Result<A::Output, B::Output>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(v) => Ok(v),
            Err(e) => Err(self.op.call(e).await),
        }
    }
}

pub struct AndThen<A, B> {
    prev: A,
    op: B,
}
impl<A, B> AndThen<A, B> {
    #[inline]
    fn new(prev: A, op: B) -> Self {
        Self { prev, op }
    }
}
impl<A, B> Op for AndThen<A, B>
where
    A: TryOp,
    B: TryOp<Input = A::Output, Error = A::Error>,
{
    type Input = A::Input;
    type Output = Result<B::Output, A::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        let v = self.prev.try_call(input).await?;
        self.op.try_call(v).await
    }
}

pub struct OrElse<A, B> {
    prev: A,
    op: B,
}
impl<A, B> OrElse<A, B> {
    #[inline]
    fn new(prev: A, op: B) -> Self {
        Self { prev, op }
    }
}
impl<A, B> Op for OrElse<A, B>
where
    A: TryOp,
    B: TryOp<Input = A::Error>,
{
    type Input = A::Input;
    type Output = Result<A::Output, B::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(v) => Ok(v),
            Err(e) => self.op.try_call(e).await,
        }
    }
}

pub struct TrySequential<A, B> {
    prev: A,
    op: B,
}
impl<A, B> TrySequential<A, B> {
    #[inline]
    fn new(prev: A, op: B) -> Self {
        Self { prev, op }
    }
}
impl<A, B> Op for TrySequential<A, B>
where
    A: TryOp,
    B: Op<Input = A::Output>,
{
    type Input = A::Input;
    type Output = Result<B::Output, A::Error>;

    #[inline]
    async fn call(&self, input: Self::Input) -> Self::Output {
        match self.prev.try_call(input).await {
            Ok(v) => Ok(self.op.call(v).await),
            Err(e) => Err(e),
        }
    }
}

// ================================================================
// 2. Concurrency – short-circuiting parallel execution
// ================================================================
pub struct TryParallel<A, B> {
    left: A,
    right: B,
}
impl<A, B> TryParallel<A, B> {
    #[inline]
    pub fn new(left: A, right: B) -> Self {
        Self { left, right }
    }
}

impl<A, B> TryOp for TryParallel<A, B>
where
    A: TryOp,
    A::Input: Clone,
    B: TryOp<Input = A::Input, Error = A::Error>,
{
    type Input = A::Input;
    type Output = (A::Output, B::Output);
    type Error = A::Error;

    #[inline]
    async fn try_call(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        try_join!(
            self.left.try_call(input.clone()),
            self.right.try_call(input)
        )
    }
}

// ================================================================
// 3. Tests (compile-time + runtime behaviour)
// ================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{map, then};

    #[tokio::test]
    async fn map_ok_works() {
        let pipe = map(|x| Ok::<_, &str>(x)).map_ok(|v| v + 1);
        assert_eq!(pipe.try_call(1).await, Ok(2));
    }

    #[tokio::test]
    async fn error_flow_preserved() {
        let pipe = map(|_: i32| Err::<i32, &str>("boom")).map_ok(|v| v + 1);
        assert_eq!(pipe.try_call(0).await, Err("boom"));
    }

    #[tokio::test]
    async fn and_then_chains() {
        let pipe = map(|x| Ok::<_, &str>(x)).and_then(|x| async move { Ok::<_, &str>(x * 2) });
        assert_eq!(pipe.try_call(3).await, Ok(6));
    }

    #[tokio::test]
    async fn or_else_recovers() {
        let pipe = map(|_: i32| Err::<i32, &str>("fail"))
            .or_else(|e| async move { Ok::<i32, &str>(if e == "fail" { 42 } else { 0 }) });
        assert_eq!(pipe.try_call(0).await, Ok(42));
    }

    #[tokio::test]
    async fn try_parallel_ok() {
        let left = map(|x| Ok::<_, &str>(x + 1));
        let right = map(|x| Ok::<_, &str>(x * 2));
        let para = TryParallel::new(left, right);

        assert_eq!(para.try_call(2).await, Ok((3, 4)));
    }

    #[tokio::test]
    async fn try_parallel_short_circuit() {
        let left = map(|_: i32| Err::<i32, &str>("nope"));
        let right = map(|x| Ok::<_, &str>(x * 2));
        let para = TryParallel::new(left, right);

        assert_eq!(para.try_call(2).await, Err("nope"));
    }
}
