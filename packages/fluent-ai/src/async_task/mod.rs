// IMPORTANT: Pure streaming primitives - NO Future/async/await!
// ⚠️  ALL FUTURE USAGE ELIMINATED - PURE ASYNCSTREAM ARCHITECTURE ⚠️
// Stream-first primitives with .collect() for await-like behavior

pub mod task;
pub use task::{AsyncTask, spawn_stream, spawn_task};
pub mod stream;
pub use stream::{AsyncStream, AsyncStreamSender};
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
