//! Error handler infrastructure for unwrapping Results in AsyncTask/AsyncStream
//! 
//! This module provides the BadTraitImpl and BadAppleChunk patterns for default error handling
//! when users don't provide custom error handlers in the builder pattern.

use std::fmt;

/// Trait for providing default implementations when futures fail
pub trait BadTraitImpl: Send + Sync + fmt::Debug + Clone {
    /// Create a default "bad" implementation that logs the error
    fn bad_impl(error: String) -> Self;
}

/// Trait for providing default chunks when streams fail
pub trait BadAppleChunk: Send + Sync + fmt::Debug + Clone {
    /// Create a default "bad" chunk that represents an error state
    fn bad_chunk(error: String) -> Self;
}

/// Default error handler for futures - logs error and returns BadTraitImpl
pub fn default_future_error_handler<T: BadTraitImpl>(error: String) -> T {
    eprintln!("Future error (using BadTraitImpl): {}", error);
    T::bad_impl(error)
}

/// Default error handler for streams - logs error and returns BadAppleChunk
pub fn default_stream_error_handler<T: BadAppleChunk>(error: String) -> T {
    eprintln!("Stream error (using BadAppleChunk): {}", error);
    T::bad_chunk(error)
}

/// Error handler wrapper for on_result closures
pub struct ResultHandlerWrapper<T, E> {
    handler: Box<dyn FnOnce(Result<T, E>) -> T + Send + 'static>,
}

impl<T, E> ResultHandlerWrapper<T, E> {
    pub fn new<F>(handler: F) -> Self
    where
        F: FnOnce(Result<T, E>) -> T + Send + 'static,
    {
        Self {
            handler: Box::new(handler),
        }
    }
    
    pub fn handle(self, result: Result<T, E>) -> T {
        (self.handler)(result)
    }
}

/// Error handler wrapper for on_chunk closures
pub struct ChunkHandlerWrapper<T, E> {
    handler: Box<dyn FnMut(Result<T, E>) -> T + Send + 'static>,
}

impl<T, E> ChunkHandlerWrapper<T, E> {
    pub fn new<F>(handler: F) -> Self
    where
        F: FnMut(Result<T, E>) -> T + Send + 'static,
    {
        Self {
            handler: Box::new(handler),
        }
    }
    
    pub fn handle(&mut self, result: Result<T, E>) -> T {
        (self.handler)(result)
    }
}

/// Convenience macro for creating default error handlers
#[macro_export]
macro_rules! default_error_handler {
    (future, $type:ty) => {
        |error: String| -> $type {
            $crate::async_task::error_handlers::default_future_error_handler(error)
        }
    };
    (stream, $type:ty) => {
        |error: String| -> $type {
            $crate::async_task::error_handlers::default_stream_error_handler(error)
        }
    };
}

/// Generic error handler for builders - handles string errors
pub trait ErrorHandler<T>
where
{
    fn handle_error(&self, error: String) -> T;
}

/// Generic result handler for builders - handles successful values and transforms them
pub trait ResultHandler<T>
where
{
    fn handle_result(&self, value: T) -> T;
}

/// Generic chunk handler for builders - handles individual chunks and transforms them
pub trait ChunkHandler<T>
where
{
    fn handle_chunk(&mut self, value: T) -> T;
}

/// Default error handler implementation
pub struct DefaultErrorHandler<T: BadTraitImpl>(std::marker::PhantomData<T>);

impl<T: BadTraitImpl> ErrorHandler<T> for DefaultErrorHandler<T> {
    fn handle_error(&self, error: String) -> T {
        default_future_error_handler(error)
    }
}

impl<T: BadTraitImpl> Default for DefaultErrorHandler<T> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

/// Default result handler implementation - just passes through successful values
pub struct DefaultResultHandler<T: BadTraitImpl>(std::marker::PhantomData<T>);

impl<T: BadTraitImpl> ResultHandler<T> for DefaultResultHandler<T> {
    fn handle_result(&self, value: T) -> T {
        // Default implementation just passes through the value
        value
    }
}

impl<T: BadTraitImpl> Default for DefaultResultHandler<T> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

/// Default chunk handler implementation - just passes through chunk values
pub struct DefaultChunkHandler<T: BadAppleChunk>(std::marker::PhantomData<T>);

impl<T: BadAppleChunk> ChunkHandler<T> for DefaultChunkHandler<T> {
    fn handle_chunk(&mut self, value: T) -> T {
        // Default implementation just passes through the value
        value
    }
}

impl<T: BadAppleChunk> Default for DefaultChunkHandler<T> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}