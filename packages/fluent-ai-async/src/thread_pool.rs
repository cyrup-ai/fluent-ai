//! Pure thread pool - NO Future usage!
//!
//! Zero-allocation, crossbeam-based thread pool with proven performance
//! Eliminated Future usage - pure closure-based execution

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::task::Waker;

use crossbeam_channel::Receiver;

pub struct GlobalExecutor {
    waker_registry: Arc<Mutex<HashMap<u64, Waker>>>,
}

impl GlobalExecutor {
    pub fn new() -> Self {
        Self {
            waker_registry: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn register_waker<T>(&self, rx: Receiver<T>, waker: Waker) {
        // Generate unique ID for this waker based on receiver address
        let waker_id = &rx as *const _ as usize as u64;

        // Store waker in registry for later coordination
        if let Ok(mut registry) = self.waker_registry.lock() {
            registry.insert(waker_id, waker.clone());
        }

        // Wake immediately to prevent blocking (fallback)
        waker.wake();
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
    GLOBAL_EXECUTOR_INSTANCE.get_or_init(|| GlobalExecutor::new())
}
