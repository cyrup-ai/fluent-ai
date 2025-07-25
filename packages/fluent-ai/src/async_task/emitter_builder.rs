//! EmitterBuilder - builds AsyncStream with error handling - NO Future usage!

use super::AsyncStream;

/// Builder that emits AsyncStream after handling Result
pub struct EmitterBuilder<T> {
    inner: Box<dyn EmitterImpl<T>>}

/// Hidden trait for implementation - NO async/Future usage!
pub trait EmitterImpl<T>: Send {
    fn execute(self: Box<Self>) -> Result<Vec<T>, Box<dyn std::error::Error + Send>>;
}

impl<T: Send + 'static> EmitterBuilder<T> {
    /// Create a new EmitterBuilder
    pub fn new(inner: Box<dyn EmitterImpl<T>>) -> Self {
        Self { inner }
    }

    /// Execute with error handling - NO async/Future/.await usage!
    pub fn emit<FOk, FErr>(self, on_ok: FOk, on_err: FErr) -> AsyncStream<T>
    where
        T: crate::async_task::NotResult,
        FOk: FnOnce(Vec<T>) -> Vec<T> + Send + 'static,
        FErr: FnOnce(Box<dyn std::error::Error + Send>) + Send + 'static,
    {
        let (sender, stream) = AsyncStream::channel();

        std::thread::spawn(move || match self.inner.execute() {
            Ok(items) => {
                for item in on_ok(items) {
                    if sender.try_send(item).is_err() {
                        break;
                    }
                }
            }
            Err(e) => on_err(e)});

        stream
    }
}

// Emit functionality moved to proper method implementations
// No exposed macros
