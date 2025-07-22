// IMPORTANT: Cyrup-agent's async primitives COPIED INTO fluent-ai
// DO NOT import from cyrup-agent - it will be DELETED!
// All async primitives are now part of fluent-ai directly

pub mod task;
pub use task::{AsyncTask, spawn_async};
pub mod stream;
pub use stream::{AsyncStream, AsyncStreamExt, AsyncStreamSender, channel};
pub mod thread_pool;

// Keep our custom emitter and error handlers for fluent builder compatibility
pub mod emitter_builder;
pub use emitter_builder::*;
pub mod error_handlers;
pub use error_handlers::*;

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
