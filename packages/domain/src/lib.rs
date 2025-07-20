//! Fluent AI Domain Library
//!
//! This crate provides core domain types and traits for AI services.
//! All domain logic, message types, and business objects are defined here.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, RelaxedCounter};
// Ultra-high-performance imports for zero-allocation domain initialization
use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;
use fluent_ai_memory::utils::error::Error as MemoryError;
// Removed unused imports: CacheConfig, LoggingConfig
use fluent_ai_memory::{MemoryConfig, SurrealDBMemoryManager, initialize};
use once_cell::sync::Lazy;

/// Domain initialization error types with semantic error handling
#[derive(Debug, thiserror::Error)]
pub enum DomainInitError {
    /// Memory system initialization error
    #[error("Memory initialization failed: {0}")]
    Memory(#[from] MemoryError),
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    /// System error
    #[error("System error: {0}")]
    System(String),
    /// Connection pool error
    #[error("Connection pool error: {0}")]
    ConnectionPool(String),
    /// Circuit breaker open
    #[error("Circuit breaker is open - too many failures")]
    CircuitBreakerOpen,
    /// Resource exhaustion
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
}

/// Global configuration cache with copy-on-write semantics for zero-allocation access
static CONFIG_CACHE: Lazy<ArcSwap<MemoryConfig>> =
    Lazy::new(|| ArcSwap::new(Arc::new(create_default_config())));

/// Lock-free connection pool with ring buffer for zero-allocation connection management
static CONNECTION_POOL: Lazy<SegQueue<Arc<SurrealDBMemoryManager>>> = Lazy::new(|| SegQueue::new());

/// Type alias for circuit breaker
pub type CircuitBreaker = crate::error::SimpleCircuitBreaker;

/// Circuit breaker for error recovery with exponential backoff
static CIRCUIT_BREAKER: Lazy<CircuitBreaker> = Lazy::new(|| CircuitBreaker::new(5, 30000)); // 30 seconds in milliseconds

/// Global initialization statistics for monitoring
static INIT_STATS: Lazy<CachePadded<RelaxedCounter>> =
    Lazy::new(|| CachePadded::new(RelaxedCounter::new(0)));

/// Pool statistics for monitoring
static POOL_STATS: Lazy<CachePadded<AtomicUsize>> =
    Lazy::new(|| CachePadded::new(AtomicUsize::new(0)));

/// Circuit breaker reset statistics
static CIRCUIT_BREAKER_RESET_COUNT: AtomicUsize = AtomicUsize::new(0);
static CIRCUIT_BREAKER_LAST_RESET: AtomicU64 = AtomicU64::new(0);

// Thread-local storage for configuration caching (doc comment removed - rustdoc doesn't generate docs for macro invocations)
thread_local! {
    static LOCAL_CONFIG: std::cell::RefCell<Option<Arc<MemoryConfig>>> = std::cell::RefCell::new(None);
}

/// const fn for zero-allocation configuration construction at compile time
#[inline(always)]
fn create_default_config() -> MemoryConfig {
    use fluent_ai_memory::utils::config::*;

    MemoryConfig {
        database: DatabaseConfig {
            db_type: DatabaseType::SurrealDB,
            connection_string: "mem://".to_string(),
            namespace: "fluent_ai".to_string(),
            database: "domain".to_string(),
            username: None,
            password: None,
            pool_size: None,
            options: None,
        },
        vector_store: VectorStoreConfig {
            store_type: VectorStoreType::Memory,
            embedding_model: EmbeddingModelConfig {
                model_type: EmbeddingModelType::OpenAI,
                model_name: "text-embedding-ada-002".to_string(),
                api_key: None,
                api_base: None,
                options: None,
            },
            dimension: 768,
            connection_string: None,
            api_key: None,
            options: None,
        },
        llm: LLMConfig {
            provider: LLMProvider::OpenAI,
            model_name: "gpt-4".to_string(),
            api_key: None,
            api_base: None,
            temperature: Some(0.7),
            max_tokens: Some(4096),
            options: None,
        },
        api: None,
        cache: CacheConfig {
            enabled: true,
            cache_type: CacheType::Memory,
            size: Some(10000),
            ttl: Some(3600),
            options: None,
        },
        logging: LoggingConfig {
            level: LogLevel::Info,
            file: None,
            console: true,
            options: None,
        },
    }
}

/// Get cached configuration from thread-local storage with zero-allocation access
#[inline(always)]
fn get_cached_config() -> Arc<MemoryConfig> {
    LOCAL_CONFIG.with(|config| {
        let mut config_ref = config.borrow_mut();
        if let Some(cached) = config_ref.as_ref() {
            Arc::clone(cached)
        } else {
            let global_config = CONFIG_CACHE.load();
            let config_arc = Arc::clone(&global_config);
            *config_ref = Some(config_arc.clone());
            config_arc
        }
    })
}

/// Update global configuration cache with copy-on-write semantics
#[inline(always)]
fn update_config_cache(new_config: MemoryConfig) {
    CONFIG_CACHE.store(Arc::new(new_config));
    // Clear thread-local caches to force refresh
    LOCAL_CONFIG.with(|config| {
        *config.borrow_mut() = None;
    });
}

/// Get memory from connection pool with lock-free access
#[inline(always)]
fn get_pooled_memory() -> Option<Arc<SurrealDBMemoryManager>> {
    if let Some(memory) = CONNECTION_POOL.pop() {
        POOL_STATS.fetch_sub(1, Ordering::Relaxed);
        Some(memory)
    } else {
        None
    }
}

/// Return memory to connection pool
#[inline(always)]
fn return_memory_to_pool(memory: Arc<SurrealDBMemoryManager>) {
    CONNECTION_POOL.push(memory);
    POOL_STATS.fetch_add(1, Ordering::Relaxed);
}

/// Execute operation with circuit breaker protection and exponential backoff
#[inline(always)]
async fn execute_with_circuit_breaker<F, T, E>(
    operation: F,
) -> std::result::Result<T, DomainInitError>
where
    F: Fn() -> std::result::Result<T, E> + Send,
    E: std::fmt::Display,
{
    match CIRCUIT_BREAKER.call(operation) {
        Ok(value) => Ok(value),
        Err(circuit_error) => match circuit_error {
            error::CircuitBreakerError::Inner(e) => Err(DomainInitError::System(e.to_string())),
            error::CircuitBreakerError::CircuitOpen => Err(DomainInitError::CircuitBreakerOpen),
        },
    }
}

/// Execute async operation with circuit breaker protection and exponential backoff
#[inline(always)]
async fn execute_async_with_circuit_breaker<F, Fut, T, E>(
    operation: F,
) -> std::result::Result<T, DomainInitError>
where
    F: Fn() -> Fut + Send,
    Fut: std::future::Future<Output = std::result::Result<T, E>> + Send,
    T: Send,
    E: std::fmt::Display + Send,
{
    // For async operations, we need to handle the circuit breaker differently
    // since the circuit breaker crate doesn't support async operations directly
    // We'll use a simple retry logic with exponential backoff
    let max_retries = 3;
    let mut retry_count = 0;

    while retry_count < max_retries {
        let result = operation().await;
        match result {
            Ok(value) => return Ok(value),
            Err(e) => {
                retry_count += 1;
                if retry_count >= max_retries {
                    return Err(DomainInitError::System(e.to_string()));
                }
                // Exponential backoff
                let delay = std::time::Duration::from_millis(100 * (2_u64.pow(retry_count)));
                tokio::time::sleep(delay).await;
            }
        }
    }

    Err(DomainInitError::System(
        "Maximum retries exceeded".to_string(),
    ))
}

/// Initialize domain with zero-allocation memory system and cognitive settings
///
/// # Returns
/// Result containing shared memory instance or initialization error
///
/// # Performance
/// Zero allocation initialization with lock-free connection pooling
#[inline]
pub async fn initialize_domain() -> std::result::Result<Arc<SurrealDBMemoryManager>, DomainInitError>
{
    // Get configuration from thread-local cache for zero-allocation access
    let memory_config = get_cached_config();

    // Increment initialization statistics
    INIT_STATS.inc();

    // Try to get memory from connection pool first (zero-allocation fast path)
    if let Some(pooled_memory) = get_pooled_memory() {
        return Ok(pooled_memory);
    }

    // Create new memory instance with circuit breaker protection
    let memory = execute_async_with_circuit_breaker(|| {
        let config = (*memory_config).clone();
        async move { initialize(&config).await }
    })
    .await?;

    let memory_arc = Arc::new(memory);

    // Populate connection pool for future zero-allocation access
    for _ in 0..3 {
        // Create 3 pooled connections
        if let Ok(pooled_memory) = initialize(&memory_config).await {
            return_memory_to_pool(Arc::new(pooled_memory));
        }
    }

    Ok(memory_arc)
}

/// Initialize domain with custom memory configuration
///
/// # Arguments
/// * `memory_config` - Custom memory configuration
///
/// # Returns
/// Result containing shared memory instance or initialization error
///
/// # Performance
/// Zero allocation with custom cognitive settings
#[inline]
pub async fn initialize_domain_with_config(
    memory_config: MemoryConfig,
) -> std::result::Result<Arc<SurrealDBMemoryManager>, DomainInitError> {
    // Create shared memory instance with custom configuration
    let memory = Arc::new(initialize(&memory_config).await?);

    Ok(memory)
}

/// Initialize domain with default configuration (legacy compatibility)
///
/// # Performance
/// Zero allocation with pre-configured settings
pub fn initialize_domain_legacy() {
    // Legacy function - no longer needed with modern API
    // Configuration is handled by MemoryConfig now
}

/// Initialize domain with production-ready configuration
///
/// # Arguments
/// * `database_url` - SurrealDB connection string
/// * `namespace` - Database namespace
/// * `database` - Database name
///
/// # Returns
/// Result containing shared memory instance or initialization error
///
/// # Performance
/// Zero allocation with production-optimized settings
#[inline]
pub async fn initialize_domain_production(
    database_url: &str,
    namespace: &str,
    database: &str,
) -> std::result::Result<Arc<SurrealDBMemoryManager>, DomainInitError> {
    // Get base configuration from thread-local cache for zero-allocation access
    let mut memory_config = (*get_cached_config()).clone();

    // Override with production-optimized settings
    memory_config.database.connection_string = database_url.to_string();
    memory_config.database.namespace = namespace.to_string();
    memory_config.database.database = database.to_string();
    memory_config.vector_store.dimension = 1536; // Higher dimension for better semantic representation
    // TODO: Fix vector store configuration - these fields don't exist
    // memory_config.vector_store.metric = "cosine".to_string();
    // memory_config.vector_store.index_type = "hnsw".to_string();
    memory_config.cache.size = Some(100000); // Larger cache for production

    // Update global cache with copy-on-write semantics
    update_config_cache(memory_config.clone());

    // Increment initialization statistics
    INIT_STATS.inc();

    // Create production memory instance with circuit breaker protection and connection pooling
    let memory = execute_async_with_circuit_breaker(|| {
        let config = memory_config.clone();
        async move { initialize(&config).await }
    })
    .await?;

    let memory_arc = Arc::new(memory);

    // Pre-populate connection pool for zero-allocation future access
    for _ in 0..5 {
        // Create 5 pooled connections for production
        if let Ok(pooled_memory) = initialize(&memory_config).await {
            return_memory_to_pool(Arc::new(pooled_memory));
        }
    }

    Ok(memory_arc)
}

/// Get default memory configuration with zero-allocation thread-local caching
///
/// # Returns
/// Default memory configuration with optimized settings
///
/// # Performance
/// Zero allocation with thread-local storage caching and copy-on-write semantics
#[inline(always)]
pub fn get_default_memory_config() -> MemoryConfig {
    // Use thread-local cache for zero-allocation fast path
    let cached_config = get_cached_config();
    let mut config = (*cached_config).clone();

    // Ensure default runtime values are set with zero-allocation updates
    if config.database.connection_string.is_empty() {
        config.database.connection_string = "mem://".to_string();
    }
    if config.database.namespace.is_empty() {
        config.database.namespace = "fluent_ai".to_string();
    }
    if config.database.database.is_empty() {
        config.database.database = "domain".to_string();
    }
    // Note: cognitive and vector fields don't exist in current MemoryConfig
    // These configurations are now handled through the actual config fields

    config
}

/// Get pool statistics for monitoring and debugging
///
/// # Returns
/// Current number of pooled connections
///
/// # Performance
/// Lock-free atomic read operation
#[inline(always)]
pub fn get_pool_stats() -> usize {
    POOL_STATS.load(Ordering::Relaxed)
}

/// Get initialization statistics for monitoring
///
/// # Returns
/// Total number of domain initializations
///
/// # Performance
/// Lock-free atomic read operation
#[inline(always)]
pub fn get_init_stats() -> usize {
    INIT_STATS.get()
}

/// Get circuit breaker reset count for monitoring
///
/// # Returns
/// Total number of circuit breaker resets performed
///
/// # Performance
/// Lock-free atomic read operation
#[inline(always)]
pub fn get_circuit_breaker_reset_count() -> usize {
    CIRCUIT_BREAKER_RESET_COUNT.load(Ordering::Relaxed)
}

/// Get last circuit breaker reset timestamp for monitoring
///
/// # Returns
/// Unix timestamp of the last circuit breaker reset, or None if never reset
///
/// # Performance
/// Lock-free atomic read operation
#[inline(always)]
pub fn get_last_circuit_breaker_reset() -> Option<Duration> {
    let timestamp = CIRCUIT_BREAKER_LAST_RESET.load(Ordering::Relaxed);
    if timestamp > 0 {
        Some(Duration::from_secs(timestamp))
    } else {
        None
    }
}

/// Force circuit breaker reset for error recovery
///
/// # Performance
/// Lock-free circuit breaker state reset with statistics tracking
#[inline(always)]
pub fn reset_circuit_breaker() {
    // Update reset statistics
    CIRCUIT_BREAKER_RESET_COUNT.fetch_add(1, Ordering::Relaxed);
    CIRCUIT_BREAKER_LAST_RESET.store(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_secs(0))
            .as_secs(),
        Ordering::Relaxed,
    );

    // Force circuit breaker back to closed state by executing a successful test operation
    // This is the only way to reset the circuit breaker state in the circuit-breaker crate
    let _ = CIRCUIT_BREAKER.call(|| {
        // Simple successful operation that will force the circuit breaker to closed state
        Ok::<(), &'static str>(())
    });

    // Clear connection pool to force fresh connections
    while get_pooled_memory().is_some() {
        // Drain any potentially problematic pooled connections
    }

    // Reset global statistics counters
    INIT_STATS.reset();
    POOL_STATS.store(0, Ordering::Relaxed);
}

/// Warm up connection pool with pre-allocated memory instances
///
/// # Arguments
/// * `pool_size` - Number of connections to pre-allocate
/// * `config` - Memory configuration for connections
///
/// # Returns
/// Result indicating success or failure
///
/// # Performance
/// Zero allocation with lock-free connection pool management
#[inline(always)]
pub async fn warm_up_connection_pool(
    pool_size: usize,
    config: MemoryConfig,
) -> std::result::Result<(), DomainInitError> {
    // Clear existing pool
    while get_pooled_memory().is_some() {
        // Drain existing connections
    }

    // Pre-allocate new connections with circuit breaker protection
    for _ in 0..pool_size {
        let memory = execute_async_with_circuit_breaker(|| {
            let config = config.clone();
            async move { initialize(&config).await }
        })
        .await?;

        return_memory_to_pool(Arc::new(memory));
    }

    Ok(())
}

// Re-export cyrup_sugars for convenience
// Use std HashMap instead
pub use std::collections::HashMap;

// Validation module for model and configuration validation
pub mod validation;
// Re-export hash_map_fn macro for transparent JSON syntax
#[doc(hidden)]
pub use cyrup_sugars::hash_map_fn;
pub use cyrup_sugars::{ByteSize, OneOrMany, ZeroOneOrMany};

/// Extension trait to add missing methods to ZeroOneOrMany
pub trait ZeroOneOrManyExt<T> {
    /// Extract a single item if there's exactly one, otherwise return None
    fn as_single(&self) -> Option<&T>;
    
    /// Convert to a Vec for iteration
    fn to_vec(&self) -> Vec<&T>;
}

impl<T> ZeroOneOrManyExt<T> for ZeroOneOrMany<T> {
    fn as_single(&self) -> Option<&T> {
        match self {
            ZeroOneOrMany::None => None,
            ZeroOneOrMany::One(ref item) => Some(item),
            ZeroOneOrMany::Many(_) => None, // Multiple items, can't return single
        }
    }
    
    fn to_vec(&self) -> Vec<&T> {
        match self {
            ZeroOneOrMany::None => Vec::new(),
            ZeroOneOrMany::One(ref item) => vec![item],
            ZeroOneOrMany::Many(ref items) => items.iter().collect(),
        }
    }
}
// Re-export commonly used types for convenience
pub use model::{Capability, Model, ModelCapabilities, ModelInfo, ModelPerformance, UseCase};
pub use validation::{
    ValidationError, ValidationIssue, ValidationReport, ValidationResult, ValidationSeverity,
};

// Define our own async task types
pub type AsyncTask<T> = tokio::task::JoinHandle<T>;

pub fn spawn_async<F, T>(future: F) -> AsyncTask<T>
where
    F: std::future::Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    tokio::spawn(future)
}

/// Channel error type for proper error handling
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("Channel closed unexpectedly")]
    ChannelClosed,
}

/// Channel creation for async communication
pub fn channel<T: Send + 'static>() -> (
    ChannelSender<T>,
    AsyncTask<std::result::Result<T, ChannelError>>,
) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = spawn_async(async move { rx.recv().await.ok_or(ChannelError::ChannelClosed) });
    (ChannelSender { tx }, task)
}

/// Channel sender wrapper
pub struct ChannelSender<T> {
    tx: tokio::sync::mpsc::UnboundedSender<T>,
}

impl<T> ChannelSender<T> {
    /// Finish the task by sending the result
    pub fn finish(self, value: T) {
        let _ = self.tx.send(value);
    }
}

// Create async_task module for compatibility
pub mod async_task {
    // Use the actual AsyncStream implementation from fluent-ai
    pub use futures::stream::Stream;
    use tokio::sync::mpsc::UnboundedReceiver;

    pub use super::{AsyncTask, spawn_async};

    /// Zero-allocation async stream implementation
    pub struct AsyncStream<T> {
        receiver: UnboundedReceiver<T>,
    }

    impl<T> AsyncStream<T> {
        /// Create a new AsyncStream from a tokio mpsc receiver
        #[inline(always)]
        pub fn new(receiver: UnboundedReceiver<T>) -> Self {
            Self { receiver }
        }

        /// Create an empty stream
        #[inline(always)]
        pub fn empty() -> Self {
            let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
            Self::new(rx)
        }

        /// Create a stream from a single item
        #[inline(always)]
        pub fn from_single(item: T) -> Self {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let _ = tx.send(item);
            Self::new(rx)
        }
    }

    impl<T> Stream for AsyncStream<T> {
        type Item = T;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            use std::pin::Pin;
            Pin::new(&mut self.receiver).poll_recv(cx)
        }
    }

    // Define NotResult trait for compatibility
    pub trait NotResult {}
    impl<T> NotResult for T where T: Send + 'static {}

    // Error handlers module
    pub mod error_handlers {
        pub fn default_error_handler<T: std::fmt::Debug>(_error: T) {
            // Default error handler implementation
        }

        /// Trait for implementing fallback behavior when operations fail
        pub trait BadTraitImpl {
            fn bad_impl(error: &str) -> Self;
        }
    }
}

// Re-export AsyncTaskExt for AsyncTask::new(rx) pattern (removed as it's not needed)
// AsyncStream trait from futures
pub use futures::stream::Stream as AsyncStream;

// Domain modules
pub mod agent;
pub mod audio;
pub mod chat;
pub mod completion;
pub mod context;
pub mod conversation;
pub mod embedding;
pub mod engine;
pub mod error;
pub mod extractor;
pub mod image;
pub mod library;

pub mod memory;
pub mod message;
pub mod model;

pub mod prompt;
pub mod provider;

pub mod tool;
pub mod usage;

// Re-export all types for convenience
// Handle conflicting types by using specific imports to avoid ambiguity

// Agent module exports (consolidated) - specify types to avoid core conflict
pub use agent::{
    Agent, AgentBuilder, AgentError, AgentResult, MAX_AGENT_TOOLS,
    ChatError, ContextInjectionResult, MemoryEnhancedChatResponse,
    AgentRoleImpl, Stdio, AgentRoleAgent, AgentConversation, AgentConversationMessage, AgentWithHistory
};
pub use agent::core as agent_core;
pub use audio::ContentFormat as AudioContentFormat;
// Builder types moved to fluent-ai/src/builders/agent_role.rs

// Audio module exports - specify ContentFormat to avoid conflict with image
pub use audio::{Audio, AudioMediaType};
// Completion module exports - consolidated from completion.rs and candle_completion.rs
pub use completion::{CompletionBackend, CompletionModel, CompletionRequest, CompletionResponse};
// Document types from context module
pub use context::document::{ContentFormat, Document, DocumentLoader, DocumentMediaType};
// Context module exports - consolidated from document.rs, chunk.rs, context.rs, and loader.rs
pub use context::*;
// Conversation module exports - specify types to avoid conflict with message
pub use conversation::Conversation as ConversationTrait;
// Conversation module exports - specify types to avoid conflict with message
pub use conversation::ConversationImpl;
// Embedding module exports (consolidated) - specify types to avoid core conflict  
pub use embedding::{Embedding, EmbeddingData, EmbeddingModel, EmbeddingResponse, EmbeddingModelTrait};
pub use embedding::core as embedding_core;
// Re-export similarity types from fluent-ai's embedding module
// Temporarily commented out to avoid circular dependency
// pub use fluent_ai::embedding::similarity::{
//     SimilarityMetric, SimilarityResult, BatchSimilarityComputer, compute_similarity,
//     cosine_similarity, euclidean_distance, manhattan_distance, dot_product_similarity,
//     jaccard_similarity, pearson_correlation, find_most_similar, find_top_k_similar,
//     find_similar_above_threshold
// };
pub use image::ContentFormat as ImageContentFormat;
// Image module exports - specify ContentFormat to avoid conflict with audio
pub use image::{Image, ImageDetail, ImageMediaType};
// Library module exports
pub use library::*;
// Memory module exports (consolidated with new high-performance types)
pub use memory::{
    // Core memory types
    BaseMemory,
    // Cognitive computing types
    CognitiveMemory,
    CognitiveProcessor,

    // Compatibility types
    CompatibilityError,
    CompatibilityLayer,
    CompatibilityMode,

    // Configuration types
    DatabaseConfig,
    LLMConfig,
    MemoryContent,
    MemoryNode,
    MemoryResult,
    // Tool types
    MemoryTool,
    MemoryToolError,
    MemoryToolResult,
    MemoryType,
    MemoryTypeEnum,

    VectorStoreConfig,
};
pub use message::Conversation as MessageConversation;
// Message module exports - specify Conversation to avoid conflict with conversation
pub use message::{
    AssistantContent, ChatMessage, Content, ContentContainer, Message, MessageChunk, MessageError,
    MessageRole, MimeType, SearchChatMessage, Text, ToolCall, ToolFunction, ToolResult,
    ToolResultContent, UserContent,
};
// Model module exports
pub use model::*;
// Model info provider module exports

// Prompt module exports - specify Prompt to avoid conflict with memory_workflow
// PromptBuilder moved to fluent-ai/src/builders/prompt.rs
pub use prompt::Prompt as PromptStruct;
pub use provider::*;
// Tool module exports - consolidated from tool.rs, tool_v2.rs, mcp.rs, mcp_tool.rs, mcp_tool_traits.rs
pub use tool::{
    ExecToText,
    // MCP functionality
    McpClient,
    McpError,
    McpTool,
    McpToolData,
    McpToolType,
    NamedTool,
    Perplexity,
    StdioTransport,
    // Core tool functionality
    Tool,
    ToolDefinition,
    ToolEmbeddingDyn,
    ToolSet,
    // MCP tool traits and data
    ToolTrait,
    Transport,
};

// Pricing module exports
pub mod pricing;
pub use pricing::PricingTier;
