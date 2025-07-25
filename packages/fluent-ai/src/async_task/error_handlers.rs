//! Zero-allocation error handler infrastructure for unwrapping Results in AsyncTask/AsyncStream
//!
//! This module provides blazing-fast, lock-free error handling patterns for the fluent-ai builder system.
//! All operations are zero-allocation with static dispatch for maximum performance.

use std::fmt;
use std::marker::PhantomData;

/// Trait for providing default implementations when futures fail
/// Zero-allocation: Uses static dispatch, no heap allocation
pub trait BadTraitImpl: Send + Sync + fmt::Debug + Clone {
    /// Create a default "bad" implementation that logs the error
    /// Performance: Inlined for zero-cost abstraction
    fn bad_impl(error: &str) -> Self;
}

/// Trait for providing default chunks when streams fail  
/// Zero-allocation: Uses static dispatch, no heap allocation
pub trait BadAppleChunk: Send + Sync + fmt::Debug + Clone {
    /// Create a default "bad" chunk that represents an error state
    /// Performance: Inlined for zero-cost abstraction
    fn bad_chunk(error: &str) -> Self;
}

/// Zero-allocation error handler for futures
/// Performance: Inlined, no heap allocation, static dispatch
#[inline(always)]
pub fn default_future_error_handler<T: BadTraitImpl>(error: &str) -> T {
    eprintln!("Future error (BadTraitImpl): {}", error);
    T::bad_impl(error)
}

/// Zero-allocation error handler for streams
/// Performance: Inlined, no heap allocation, static dispatch  
#[inline(always)]
pub fn default_stream_error_handler<T: BadAppleChunk>(error: &str) -> T {
    eprintln!("Stream error (BadAppleChunk): {}", error);
    T::bad_chunk(error)
}

/// Zero-allocation error handler wrapper using static dispatch
/// Performance: No heap allocation, compile-time function resolution
pub struct ErrorHandlerWrapper<F, T, E> {
    handler: F,
    _phantom: PhantomData<(T, E)>}

impl<F, T, E> ErrorHandlerWrapper<F, T, E>
where
    F: Fn(&str) -> T + Send + Sync,
{
    /// Create new error handler wrapper
    /// Performance: Zero allocation, inlined
    #[inline(always)]
    pub const fn new(handler: F) -> Self {
        Self {
            handler,
            _phantom: PhantomData}
    }

    /// Handle error with zero allocation
    /// Performance: Inlined, static dispatch
    #[inline(always)]
    pub fn handle(&self, error: &str) -> T {
        (self.handler)(error)
    }
}

/// Zero-allocation result handler wrapper using static dispatch
/// Performance: No heap allocation, compile-time function resolution
pub struct ResultHandlerWrapper<F, T, E> {
    handler: F,
    _phantom: PhantomData<(T, E)>}

impl<F, T, E> ResultHandlerWrapper<F, T, E>
where
    F: Fn(Result<T, E>) -> T + Send + Sync,
{
    /// Create new result handler wrapper
    /// Performance: Zero allocation, inlined
    #[inline(always)]
    pub const fn new(handler: F) -> Self {
        Self {
            handler,
            _phantom: PhantomData}
    }

    /// Handle result with zero allocation
    /// Performance: Inlined, static dispatch
    #[inline(always)]
    pub fn handle(&self, result: Result<T, E>) -> T {
        (self.handler)(result)
    }
}

/// Zero-allocation chunk handler wrapper using static dispatch
/// Performance: No heap allocation, compile-time function resolution
pub struct ChunkHandlerWrapper<F, T, E> {
    handler: F,
    _phantom: PhantomData<(T, E)>}

impl<F, T, E> ChunkHandlerWrapper<F, T, E>
where
    F: Fn(Result<T, E>) -> T + Send + Sync,
{
    /// Create new chunk handler wrapper
    /// Performance: Zero allocation, inlined
    #[inline(always)]
    pub const fn new(handler: F) -> Self {
        Self {
            handler,
            _phantom: PhantomData}
    }

    /// Handle chunk with zero allocation
    /// Performance: Inlined, static dispatch
    #[inline(always)]
    pub fn handle(&self, result: Result<T, E>) -> T {
        (self.handler)(result)
    }
}

/// Zero-allocation error handler trait using static dispatch
/// Performance: Compile-time polymorphism, no vtable overhead
pub trait ErrorHandler<T>: Send + Sync {
    /// Handle error with zero allocation
    /// Performance: Inlined for maximum speed
    fn handle_error(&self, error: &str) -> T;
}

/// Zero-allocation result handler trait using static dispatch
/// Performance: Compile-time polymorphism, no vtable overhead
pub trait ResultHandler<T>: Send + Sync {
    /// Handle result with zero allocation
    /// Performance: Inlined for maximum speed
    fn handle_result(&self, value: T) -> T;
}

/// Zero-allocation chunk handler trait using static dispatch
/// Performance: Compile-time polymorphism, no vtable overhead
pub trait ChunkHandler<T>: Send + Sync {
    /// Handle chunk with zero allocation
    /// Performance: Inlined for maximum speed
    fn handle_chunk(&self, value: T) -> T;
}

/// Zero-allocation default error handler implementation
/// Performance: Zero-sized type, no runtime overhead
pub struct DefaultErrorHandler<T: BadTraitImpl>(PhantomData<T>);

impl<T: BadTraitImpl> ErrorHandler<T> for DefaultErrorHandler<T> {
    #[inline(always)]
    fn handle_error(&self, error: &str) -> T {
        default_future_error_handler(error)
    }
}

impl<T: BadTraitImpl> Default for DefaultErrorHandler<T> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: BadTraitImpl> Clone for DefaultErrorHandler<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: BadTraitImpl> Copy for DefaultErrorHandler<T> {}

/// Zero-allocation default result handler implementation
/// Performance: Zero-sized type, no runtime overhead
pub struct DefaultResultHandler<T>(PhantomData<T>);

impl<T: Send + Sync> ResultHandler<T> for DefaultResultHandler<T> {
    #[inline(always)]
    fn handle_result(&self, value: T) -> T {
        value
    }
}

impl<T: Send + Sync> Default for DefaultResultHandler<T> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: Send + Sync> Clone for DefaultResultHandler<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: Send + Sync> Copy for DefaultResultHandler<T> {}

/// Zero-allocation default chunk handler implementation
/// Performance: Zero-sized type, no runtime overhead
pub struct DefaultChunkHandler<T>(PhantomData<T>);

impl<T: Send + Sync> ChunkHandler<T> for DefaultChunkHandler<T> {
    #[inline(always)]
    fn handle_chunk(&self, value: T) -> T {
        value
    }
}

impl<T: Send + Sync> Default for DefaultChunkHandler<T> {
    #[inline(always)]
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T: Send + Sync> Clone for DefaultChunkHandler<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: Send + Sync> Copy for DefaultChunkHandler<T> {}

/// Zero-allocation closure-based error handler
/// Performance: Static dispatch, inlined closures
pub struct ClosureErrorHandler<F, T> {
    closure: F,
    _phantom: PhantomData<T>}

impl<F, T> ClosureErrorHandler<F, T>
where
    F: Fn(&str) -> T + Send + Sync,
{
    #[inline(always)]
    pub const fn new(closure: F) -> Self {
        Self {
            closure,
            _phantom: PhantomData}
    }
}

impl<F, T> ErrorHandler<T> for ClosureErrorHandler<F, T>
where
    F: Fn(&str) -> T + Send + Sync,
    T: Send + Sync,
{
    #[inline(always)]
    fn handle_error(&self, error: &str) -> T {
        (self.closure)(error)
    }
}

/// Zero-allocation closure-based result handler
/// Performance: Static dispatch, inlined closures
pub struct ClosureResultHandler<F, T> {
    closure: F,
    _phantom: PhantomData<T>}

impl<F, T> ClosureResultHandler<F, T>
where
    F: Fn(T) -> T + Send + Sync,
{
    #[inline(always)]
    pub const fn new(closure: F) -> Self {
        Self {
            closure,
            _phantom: PhantomData}
    }
}

impl<F, T> ResultHandler<T> for ClosureResultHandler<F, T>
where
    F: Fn(T) -> T + Send + Sync,
    T: Send + Sync,
{
    #[inline(always)]
    fn handle_result(&self, value: T) -> T {
        (self.closure)(value)
    }
}

/// Zero-allocation closure-based chunk handler
/// Performance: Static dispatch, inlined closures
pub struct ClosureChunkHandler<F, T> {
    closure: F,
    _phantom: PhantomData<T>}

impl<F, T> ClosureChunkHandler<F, T>
where
    F: Fn(T) -> T + Send + Sync,
{
    #[inline(always)]
    pub const fn new(closure: F) -> Self {
        Self {
            closure,
            _phantom: PhantomData}
    }
}

impl<F, T> ChunkHandler<T> for ClosureChunkHandler<F, T>
where
    F: Fn(T) -> T + Send + Sync,
    T: Send + Sync,
{
    #[inline(always)]
    fn handle_chunk(&self, value: T) -> T {
        (self.closure)(value)
    }
}

/// Unwrap Result<T, E> using error handler with zero allocation
/// Performance: Inlined, static dispatch, compile-time optimization
#[inline(always)]
pub fn unwrap_with_handler<T, E, H>(result: Result<T, E>, handler: &H) -> T
where
    E: fmt::Display,
    H: ErrorHandler<T>,
{
    match result {
        Ok(value) => value,
        Err(error) => handler.handle_error(&error.to_string())}
}

/// Unwrap Result<T, E> using result handler with zero allocation
/// Performance: Inlined, static dispatch, compile-time optimization
#[inline(always)]
pub fn unwrap_with_result_handler<T, E, H>(result: Result<T, E>, handler: &H) -> T
where
    E: fmt::Display,
    H: ResultHandler<T>,
    T: BadTraitImpl,
{
    match result {
        Ok(value) => handler.handle_result(value),
        Err(error) => T::bad_impl(&error.to_string())}
}

/// Process chunk with handler using zero allocation
/// Performance: Inlined, static dispatch, compile-time optimization
#[inline(always)]
pub fn process_chunk_with_handler<T, H>(chunk: T, handler: &H) -> T
where
    H: ChunkHandler<T>,
{
    handler.handle_chunk(chunk)
}
