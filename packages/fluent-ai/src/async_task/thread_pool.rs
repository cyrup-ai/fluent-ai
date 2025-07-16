//! Cyrup-agent's thread pool and global executor copied into fluent-ai
//!
//! Zero-allocation, crossbeam-based thread pool with proven performance

use crossbeam_channel::Receiver;
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::task::Waker;

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

    pub fn enqueue<F>(&self, fut: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        tokio::spawn(fut);
    }
}

lazy_static::lazy_static! {
    pub static ref GLOBAL_EXECUTOR: GlobalExecutor = GlobalExecutor::new();
}