//! Domain initialization and configuration

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use crossbeam_queue::SegQueue;
use fluent_ai_memory::MemoryConfig;
use once_cell::sync::Lazy;

use crate::core::DomainInitError;
use crate::memory::SurrealDBMemoryManager;

// Global connection pool with lazy initialization
static CONNECTION_POOL: Lazy<Arc<SegQueue<Arc<SurrealDBMemoryManager>>>>> = Lazy::new(|| {
    let queue = SegQueue::new();
    Arc::new(queue)
});

// Global configuration with atomic reference counting
static CONFIG: Lazy<ArcSwap<MemoryConfig>> = Lazy::new(|| {
    ArcSwap::from_pointee(MemoryConfig::default())
});

/// Initialize the domain with default configuration
pub async fn initialize_domain() -> Result<Arc<SurrealDBMemoryManager>, DomainInitError> {
    let config = get_default_memory_config();
    initialize_domain_with_config(config).await
}

/// Initialize domain with custom configuration
pub async fn initialize_domain_with_config(
    config: MemoryConfig,
) -> Result<Arc<SurrealDBMemoryManager>, DomainInitError> {
    // Update global config
    CONFIG.store(Arc::new(config.clone()));
    
    // Initialize memory manager
    let memory = Arc::new(SurrealDBMemoryManager::new(config).await?);
    
    // Add to connection pool
    CONNECTION_POOL.push(memory.clone());
    
    Ok(memory)
}

/// Get default memory configuration
pub fn get_default_memory_config() -> MemoryConfig {
    (*CONFIG.load()).clone()
}

/// Get a memory manager from the connection pool
pub fn get_from_pool() -> Option<Arc<SurrealDBMemoryManager>> {
    CONNECTION_POOL.pop().ok()
}

/// Return a memory manager to the connection pool
pub fn return_to_pool(memory: Arc<SurrealDBMemoryManager>) {
    CONNECTION_POOL.push(memory);
}

/// Get the current connection pool size
pub fn pool_size() -> usize {
    CONNECTION_POOL.len()
}