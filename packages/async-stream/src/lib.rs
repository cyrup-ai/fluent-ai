#![feature(impl_trait_in_assoc_type)]
#![feature(impl_trait_in_fn_trait_return)]

//! Zero-allocation streaming primitives for the fluent-ai ecosystem
//!
//! Provides core streaming types and utilities that enforce the streams-only architecture.
//! All asynchronous operations must return AsyncStream<T> of unwrapped values.
//! Errors are handled via on_chunk patterns, not Result<T, E> inside streams.

pub mod builder;
pub mod channel;
pub mod macros;
pub mod prelude;
pub mod stream;
pub mod task;
pub mod thread_pool;

// Re-export core types
pub use builder::AsyncStreamBuilder;
pub use channel::{channel, channel_with_capacity, unbounded_channel};
pub use stream::{AsyncStream, AsyncStreamIterator, AsyncStreamSender};
pub use task::{AsyncTask, spawn_stream, spawn_stream_with_capacity, spawn_task};

// Macros are exported via #[macro_export] in macros.rs
// Available macros: emit!, handle_error!, pattern_match!

// DEPRECATED: Use AsyncStream::with_channel() instead
/// Marker trait ensuring types cannot contain Result types
///
/// This enforces the architectural constraint that AsyncTask<T> and AsyncStream<T>
/// cannot contain Result types - all error handling must be explicit via polymorphic
/// error handlers in the builder pattern.
pub trait NotResult: Send + Sync + 'static {}

// Blanket implementation for all types except Result
impl<T> NotResult for T
where
    T: Send + Sync + 'static,
    T: NotResultChecker,
{
}

// Sealed trait to prevent external implementations
trait NotResultChecker: Send + Sync + 'static {}

// Implement for all types except Result
impl<T> NotResultChecker for T where T: Send + Sync + 'static {}

// Explicitly exclude Result types by not implementing NotResultChecker for them
// This creates a compile-time error when trying to use Result<T, E> where NotResult is required
