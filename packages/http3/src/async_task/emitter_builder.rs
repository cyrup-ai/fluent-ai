//! EmitterBuilder - builds AsyncStream with error handling
//! NO FUTURES - pure streaming architecture

use super::AsyncStream;

/// Builder that emits AsyncStream after handling Result
/// NO FUTURES - pure streaming architecture
pub struct EmitterBuilder<T> {
    inner: Box<dyn EmitterImpl<T>>,
}

/// Hidden trait for implementation - NO FUTURES
pub trait EmitterImpl<T>: Send {
    /// Execute and return streaming result
    /// NO FUTURES - pure streaming architecture
    fn execute(self: Box<Self>) -> Result<Vec<T>, Box<dyn std::error::Error + Send>>;
}

impl<T: Send + 'static> EmitterBuilder<T> {
    /// Create a new EmitterBuilder
    pub fn new(inner: Box<dyn EmitterImpl<T>>) -> Self {
        Self { inner }
    }

    /// Execute with error handling - returns pure stream
    /// NO FUTURES - clients call .collect() for await-like behavior
    pub fn emit<FOk, FErr>(self, on_ok: FOk, on_err: FErr) -> AsyncStream<T>
    where
        T: crate::async_task::NotResult,
        FOk: FnOnce(Vec<T>) -> Vec<T> + Send + 'static,
        FErr: FnOnce(Box<dyn std::error::Error + Send>) + Send + 'static,
    {
        // Execute synchronously and create stream from results
        match self.inner.execute() {
            Ok(items) => {
                let processed_items = on_ok(items);
                let (sender, stream) = AsyncStream::channel();

                // Emit all items to stream
                for item in processed_items {
                    if sender.try_send(item).is_err() {
                        break; // Stream closed
                    }
                }

                stream
            }
            Err(e) => {
                on_err(e);
                AsyncStream::empty() // Return empty stream on error
            }
        }
    }
}

// Pure streaming functionality - no exposed macros
// NO FUTURES - clients get AsyncStream and call .collect() for await behavior
