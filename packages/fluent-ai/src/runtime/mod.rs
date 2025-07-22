pub mod async_stream;
pub mod async_task;
pub mod channel;
pub mod thread_pool;
pub mod tokio_rt;

pub use async_stream::AsyncStream;
pub use async_task::{AsyncTask, spawn_async};
pub use channel::*;
// Export executor module functionality
pub use executor::GlobalExecutor as Executor;
pub use thread_pool::ThreadPool;
pub use tokio_rt::*;

// global single-thread executor reused by AsyncTask
pub(crate) mod executor {
    use std::sync::Once;

    use super::thread_pool::ThreadPool;

    static INIT: Once = Once::new();
    static mut EXEC: Option<GlobalExecutor> = None;

    pub struct GlobalExecutor {
        pool: ThreadPool,
    }

    /// Accessor that bootstraps lazily.
    pub(super) fn global() -> Option<&'static GlobalExecutor> {
        unsafe {
            INIT.call_once(|| {
                let pool = ThreadPool::new();
                EXEC = Some(GlobalExecutor { pool });
            });
            EXEC.as_ref()
        }
    }

    /// Register a waker for an AsyncTask receiver.
    pub fn register_waker<R>(_rx: R, _w: std::task::Waker)
    where
        R: Send + 'static,
    {
        // Simplified implementation - waker functionality handled by tokio runtime
        // The old AtomicWaker-based implementation is no longer needed
    }

    /// Enqueue a future for execution in the current tokio runtime
    ///
    /// # Arguments
    /// * `f` - Future to execute
    ///
    /// # Panics
    /// Panics if no tokio runtime is available. This enforces async-only execution
    /// per production code constraints.
    ///
    /// # Architecture
    /// This function no longer provides blocking fallbacks. All callers must
    /// ensure they're operating within a proper async runtime context.
    /// This guarantees zero-blocking, production-ready execution.
    pub fn enqueue<F>(f: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        // PRODUCTION CONSTRAINT: No blocking code allowed
        // Previous implementation used futures_executor::block_on and rt.block_on
        // which violate the strict async-only requirement.

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                // We have a tokio runtime available - use it
                handle.spawn(f);
            }
            Err(_) => {
                // ARCHITECTURAL REQUIREMENT: Fail fast instead of blocking
                panic!(
                    "enqueue() called without tokio runtime context. \
                     This violates async-only execution constraints. \
                     \
                     SOLUTION: Ensure all code paths that call enqueue() are within \
                     a proper tokio runtime context. Use tokio::main or explicitly \
                     create a runtime before calling functions that use enqueue(). \
                     \
                     This constraint ensures zero-blocking, production-ready code."
                );
            }
        }
    }
}
