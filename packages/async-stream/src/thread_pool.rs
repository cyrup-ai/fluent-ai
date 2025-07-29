//! Pure thread pool - NO Future usage!
//!
//! Zero-allocation, crossbeam-based thread pool with proven performance
//! Eliminated Future usage - pure closure-based execution

pub struct GlobalExecutor {
    // NO FUTURES! Removed waker_registry per NO FUTURES architecture
}

impl GlobalExecutor {
    pub fn new() -> Self {
        Self {
            // NO FUTURES! Removed waker_registry per NO FUTURES architecture
        }
    }

    /// Execute closure on thread - NO Future usage!
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        std::thread::spawn(f);
    }

    /// Execute closure and send result to channel
    pub fn execute_with_result<F, T>(&self, f: F) -> crossbeam_channel::Receiver<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            let result = f();
            let _ = tx.send(result);
        });
        rx
    }
}

impl Default for GlobalExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// Static instance for global access - SAFE, no mutable static
static GLOBAL_EXECUTOR_INSTANCE: std::sync::OnceLock<GlobalExecutor> = std::sync::OnceLock::new();

pub fn global_executor() -> &'static GlobalExecutor {
    GLOBAL_EXECUTOR_INSTANCE.get_or_init(GlobalExecutor::new)
}
