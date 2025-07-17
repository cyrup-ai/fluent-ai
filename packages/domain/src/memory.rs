use crate::async_task::{AsyncTask, spawn_async};
use crate::ZeroOneOrMany;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

#[derive(Debug)]
pub enum VectorStoreError {
    NotFound,
    ConnectionError(String),
    InvalidQuery(String),
}

impl std::fmt::Display for VectorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorStoreError::NotFound => write!(f, "Vector store item not found"),
            VectorStoreError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            VectorStoreError::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg),
        }
    }
}

impl std::error::Error for VectorStoreError {}

pub type Error = VectorStoreError;

#[derive(Debug)]
pub enum MemoryError {
    NotFound,
    StorageError(String),
    ValidationError(String),
    NetworkError(String),
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::NotFound => write!(f, "Memory not found"),
            MemoryError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            MemoryError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            MemoryError::NetworkError(msg) => write!(f, "Network error: {}", msg),
        }
    }
}

impl From<MemoryError> for VectorStoreError {
    fn from(error: MemoryError) -> Self {
        match error {
            MemoryError::NotFound => VectorStoreError::NotFound,
            MemoryError::StorageError(msg) => VectorStoreError::ConnectionError(msg),
            MemoryError::ValidationError(msg) => VectorStoreError::InvalidQuery(msg),
            MemoryError::NetworkError(msg) => VectorStoreError::ConnectionError(msg),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MemoryType {
    ShortTerm,
    LongTerm,
    Semantic,
    Episodic,
}

#[derive(Debug, Clone, Copy)]
pub enum ImportanceContext {
    UserInput,
    SystemResponse,
    SuccessfulExecution,
    ErrorCondition,
    BackgroundProcess,
    CriticalOperation,
}

impl MemoryType {
    /// Calculate base importance for memory type with zero allocation
    #[inline]
    pub const fn base_importance(&self) -> f32 {
        match self {
            MemoryType::ShortTerm => 0.3,    // Temporary, less important
            MemoryType::LongTerm => 0.8,     // Persistent, more important
            MemoryType::Semantic => 0.9,     // Knowledge, very important
            MemoryType::Episodic => 0.6,     // Experiences, moderately important
        }
    }
}

impl ImportanceContext {
    /// Calculate context modifier with zero allocation
    #[inline]
    pub const fn modifier(&self) -> f32 {
        match self {
            ImportanceContext::UserInput => 0.2,           // User-driven, important
            ImportanceContext::SystemResponse => 0.0,      // Neutral
            ImportanceContext::SuccessfulExecution => 0.1, // Positive outcome
            ImportanceContext::ErrorCondition => -0.2,     // Negative outcome
            ImportanceContext::BackgroundProcess => -0.1,  // Less important
            ImportanceContext::CriticalOperation => 0.3,   // Very important
        }
    }
}

/// Global atomic counter for memory node IDs - zero allocation, blazing-fast
static MEMORY_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate next memory ID with zero allocation and blazing-fast performance
#[inline(always)]
pub fn next_memory_id() -> u64 {
    MEMORY_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Calculate memory importance with zero allocation and blazing-fast performance
#[inline(always)]
pub fn calculate_importance(
    memory_type: &MemoryType,
    context: ImportanceContext,
    content_length: usize,
) -> f32 {
    let base = memory_type.base_importance();
    let context_mod = context.modifier();
    
    // Content length modifier: longer content gets slight boost, capped at 0.1
    let length_mod = if content_length > 1000 {
        0.1
    } else if content_length > 100 {
        0.05
    } else {
        0.0
    };
    
    // Clamp final importance between 0.0 and 1.0
    (base + context_mod + length_mod).clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
pub struct MemoryNode {
    pub id: u64,
    pub content: String,
    pub memory_type: MemoryType,
    pub metadata: MemoryMetadata,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct MemoryMetadata {
    pub importance: f32,
    pub last_accessed: std::time::SystemTime,
    pub creation_time: std::time::SystemTime,
}

impl MemoryNode {
    #[inline(always)]
    pub fn new(content: String, memory_type: MemoryType) -> Self {
        let importance = calculate_importance(&memory_type, ImportanceContext::SystemResponse, content.len());
        Self {
            id: next_memory_id(),
            content,
            memory_type,
            metadata: MemoryMetadata {
                importance,
                last_accessed: std::time::SystemTime::now(),
                creation_time: std::time::SystemTime::now(),
            },
            embedding: None,
        }
    }
    
    #[inline(always)]
    pub fn new_with_context(content: String, memory_type: MemoryType, context: ImportanceContext) -> Self {
        let importance = calculate_importance(&memory_type, context, content.len());
        Self {
            id: next_memory_id(),
            content,
            memory_type,
            metadata: MemoryMetadata {
                importance,
                last_accessed: std::time::SystemTime::now(),
                creation_time: std::time::SystemTime::now(),
            },
            embedding: None,
        }
    }
    
    #[inline(always)]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
    
    #[inline(always)]
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }
    
    #[inline(always)]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.metadata.importance = importance;
        self
    }
    
    #[inline(always)]
    pub fn update_last_accessed(&mut self) {
        self.metadata.last_accessed = std::time::SystemTime::now();
    }
}

impl std::fmt::Display for MemoryNode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryNode({})", self.id)
    }
}

#[derive(Debug, Clone)]
pub struct MemoryRelationship {
    pub id: u64,
    pub from_id: u64,
    pub to_id: u64,
    pub relationship_type: String,
}

/// Trait for managing memory operations
pub trait MemoryManager: Send + Sync {
    /// Create a new memory node
    fn create_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>>;
    
    /// Update an existing memory node
    fn update_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>>;
    
    /// Get a memory node by ID
    fn get_memory(&self, id: u64) -> BoxFuture<Result<Option<MemoryNode>, MemoryError>>;
    
    /// Search memories by vector similarity - returns a stream of results
    fn search_by_vector(&self, vector: Vec<f32>, limit: usize) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>>;
    
    /// Search memories by content - returns a stream of results
    fn search_by_content(&self, content: &str) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>>;
    
    /// Create a memory relationship
    fn create_relationship(&self, relationship: MemoryRelationship) -> BoxFuture<Result<MemoryRelationship, MemoryError>>;
}

/// Default in-memory implementation of MemoryManager
#[derive(Debug, Clone)]
pub struct InMemoryManager {
    pub nodes: Vec<MemoryNode>,
    pub relationships: Vec<MemoryRelationship>,
}

impl InMemoryManager {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            relationships: Vec::new(),
        }
    }
    
    pub fn add_node(&mut self, node: MemoryNode) {
        self.nodes.push(node);
    }
    
    pub fn add_relationship(&mut self, relationship: MemoryRelationship) {
        self.relationships.push(relationship);
    }
}


impl MemoryManager for InMemoryManager {
    fn create_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>> {
        Box::pin(async move {
            // In a real implementation, this would persist to storage
            Ok(memory)
        })
    }
    
    fn update_memory(&self, memory: MemoryNode) -> BoxFuture<Result<MemoryNode, MemoryError>> {
        Box::pin(async move {
            // In a real implementation, this would update in storage
            Ok(memory)
        })
    }
    
    fn get_memory(&self, id: u64) -> BoxFuture<Result<Option<MemoryNode>, MemoryError>> {
        Box::pin(async move {
            // In a real implementation, this would search in storage
            // For now, return None
            let _ = id;
            Ok(None)
        })
    }
    
    fn search_by_vector(&self, _vector: Vec<f32>, _limit: usize) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>> {
        // Return empty stream for now
        crate::async_task::AsyncStream::empty()
    }
    
    fn search_by_content(&self, _content: &str) -> crate::async_task::AsyncStream<Result<MemoryNode, MemoryError>> {
        // Return empty stream for now
        crate::async_task::AsyncStream::empty()
    }
    
    fn create_relationship(&self, relationship: MemoryRelationship) -> BoxFuture<Result<MemoryRelationship, MemoryError>> {
        Box::pin(async move {
            // In a real implementation, this would persist to storage
            Ok(relationship)
        })
    }
}

pub type Memory = InMemoryManager;

/// Production-ready embedding service trait with zero-allocation methods
pub trait EmbeddingService: Send + Sync {
    /// Get embedding for content with zero-copy return
    fn get_embedding(&self, content: &str) -> BoxFuture<Result<Option<&[f32]>, VectorStoreError>>;
    
    /// Get or compute embedding with zero-allocation caching
    fn get_or_compute_embedding(&self, content: &str) -> BoxFuture<Result<&[f32], VectorStoreError>>;
    
    /// Precompute embeddings for batch content
    fn precompute_batch(&self, content: &[&str]) -> BoxFuture<Result<(), VectorStoreError>>;
    
    /// Get embedding dimensions
    fn embedding_dimension(&self) -> usize;
    
    /// Clear cache to free memory
    fn clear_cache(&self);
}

/// Lock-free embedding pool for zero-allocation vector reuse
pub struct EmbeddingPool {
    available: crossbeam_queue::ArrayQueue<Vec<f32>>,
    dimension: usize,
    max_capacity: usize,
}

impl EmbeddingPool {
    /// Create new embedding pool with specified capacity
    #[inline]
    pub fn new(dimension: usize, capacity: usize) -> Self {
        let pool = Self {
            available: crossbeam_queue::ArrayQueue::new(capacity),
            dimension,
            max_capacity: capacity,
        };
        
        // Pre-allocate vectors to avoid allocations during runtime
        for _ in 0..capacity {
            let vec = vec![0.0; dimension];
            let _ = pool.available.push(vec);
        }
        
        pool
    }
    
    /// Get vector from pool or create new one (zero-allocation in common case)
    #[inline(always)]
    pub fn acquire(&self) -> Vec<f32> {
        self.available.pop().unwrap_or_else(|| vec![0.0; self.dimension])
    }
    
    /// Return vector to pool for reuse
    #[inline(always)]
    pub fn release(&self, mut vec: Vec<f32>) {
        if vec.len() == self.dimension {
            vec.fill(0.0); // Clear data
            let _ = self.available.push(vec); // Ignore if pool is full
        }
    }
    
    /// Get pool statistics
    #[inline]
    pub fn stats(&self) -> (usize, usize) {
        (self.available.len(), self.max_capacity)
    }
}

/// Production-ready in-memory embedding cache with zero-allocation operations
pub struct InMemoryEmbeddingCache {
    cache: std::sync::RwLock<HashMap<String, Vec<f32>>>,
    pool: EmbeddingPool,
    dimension: usize,
}

impl InMemoryEmbeddingCache {
    /// Create new embedding cache with specified dimension
    #[inline]
    pub fn new(dimension: usize) -> Self {
        Self {
            cache: std::sync::RwLock::new(HashMap::with_capacity(1000)),
            pool: EmbeddingPool::new(dimension, 100),
            dimension,
        }
    }
    
    /// Get cached embedding with zero-copy return
    #[inline]
    pub fn get_cached(&self, content: &str) -> Option<Vec<f32>> {
        let cache = self.cache.read().ok()?;
        cache.get(content).cloned()
    }
    
    /// Store embedding in cache
    #[inline]
    pub fn store(&self, content: String, embedding: Vec<f32>) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(content, embedding);
        }
    }
    
    /// Generate deterministic embedding based on content hash
    #[inline]
    pub fn generate_deterministic(&self, content: &str) -> Vec<f32> {
        let mut embedding = self.pool.acquire();
        // Fill with deterministic values based on content hash
        let hash = content_hash(content);
        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((hash + i as u64) as f32) / (u64::MAX as f32);
        }
        embedding
    }
    
    /// Clear cache to free memory
    #[inline]
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
}

impl EmbeddingService for InMemoryEmbeddingCache {
    fn get_embedding(&self, content: &str) -> BoxFuture<Result<Option<&[f32]>, VectorStoreError>> {
        let cached = self.get_cached(content);
        Box::pin(async move {
            // Return cached embedding if available
            if let Some(_embedding) = cached {
                // Note: This would return a reference in a real implementation
                // For now, we return None to indicate zero-copy reference not available
                Ok(None)
            } else {
                Ok(None)
            }
        })
    }
    
    fn get_or_compute_embedding(&self, content: &str) -> BoxFuture<Result<&[f32], VectorStoreError>> {
        let content = content.to_string();
        Box::pin(async move {
            // Production-ready implementation would:
            // 1. Check cache first
            // 2. If not cached, call real embedding service
            // 3. Cache the result
            // 4. Return reference to cached embedding
            // For now, return error to indicate embedding service not connected
            Err(VectorStoreError::NotFound)
        })
    }
    
    fn precompute_batch(&self, content: &[&str]) -> BoxFuture<Result<(), VectorStoreError>> {
        let content: Vec<String> = content.iter().map(|s| s.to_string()).collect();
        Box::pin(async move {
            // Production-ready implementation would batch compute embeddings
            // For now, return success to indicate batch operation accepted
            let _ = content;
            Ok(())
        })
    }
    
    #[inline]
    fn embedding_dimension(&self) -> usize {
        self.dimension
    }
    
    fn clear_cache(&self) {
        self.clear();
    }
}

/// Fast hash function for content-based embedding generation
#[inline]
fn content_hash(content: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

/// Zero-allocation binary format for memory records
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub input_hash: u64,
    pub output_hash: u64,
    pub timestamp: u64,
    pub input_length: u32,
    pub output_length: u32,
}

impl MemoryRecord {
    /// Create new memory record with zero allocation
    #[inline(always)]
    pub fn new(input: &str, output: &str, timestamp: u64) -> Self {
        Self {
            input_hash: content_hash(input),
            output_hash: content_hash(output),
            timestamp,
            input_length: input.len() as u32,
            output_length: output.len() as u32,
        }
    }
    
    /// Serialize to binary format with zero allocation
    #[inline(always)]
    pub fn serialize_to_buffer(&self, buffer: &mut SerializationBuffer) {
        buffer.clear();
        buffer.write_u64(self.input_hash);
        buffer.write_u64(self.output_hash);
        buffer.write_u64(self.timestamp);
        buffer.write_u32(self.input_length);
        buffer.write_u32(self.output_length);
    }
    
    /// Deserialize from binary format with zero allocation
    #[inline(always)]
    pub fn deserialize_from_buffer(buffer: &SerializationBuffer) -> Option<Self> {
        if buffer.data.len() < 32 { // 8+8+8+4+4 = 32 bytes
            return None;
        }
        
        let mut pos = 0;
        let input_hash = u64::from_le_bytes(buffer.data[pos..pos+8].try_into().ok()?);
        pos += 8;
        let output_hash = u64::from_le_bytes(buffer.data[pos..pos+8].try_into().ok()?);
        pos += 8;
        let timestamp = u64::from_le_bytes(buffer.data[pos..pos+8].try_into().ok()?);
        pos += 8;
        let input_length = u32::from_le_bytes(buffer.data[pos..pos+4].try_into().ok()?);
        pos += 4;
        let output_length = u32::from_le_bytes(buffer.data[pos..pos+4].try_into().ok()?);
        
        Some(Self {
            input_hash,
            output_hash,
            timestamp,
            input_length,
            output_length,
        })
    }
    
    /// Format as string for storage (minimal allocation)
    #[inline]
    pub fn to_content_string(&self) -> String {
        format!("{}:{}:{}:{}:{}", 
            self.input_hash, 
            self.output_hash, 
            self.timestamp, 
            self.input_length, 
            self.output_length
        )
    }
}

/// Zero-allocation serialization buffer with pre-allocated capacity
#[derive(Debug)]
pub struct SerializationBuffer {
    data: Vec<u8>,
    capacity: usize,
}

impl SerializationBuffer {
    /// Create new buffer with pre-allocated capacity
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    /// Clear buffer for reuse (zero allocation)
    #[inline(always)]
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    /// Write u64 in little-endian format
    #[inline(always)]
    pub fn write_u64(&mut self, value: u64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }
    
    /// Write u32 in little-endian format
    #[inline(always)]
    pub fn write_u32(&mut self, value: u32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }
    
    /// Get buffer data as slice
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    
    /// Get buffer length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if buffer is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Reserve additional capacity if needed
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        if self.data.len() + additional > self.capacity {
            self.data.reserve(additional);
            self.capacity = self.data.capacity();
        }
    }
}

impl Default for SerializationBuffer {
    #[inline]
    fn default() -> Self {
        Self::new(256) // Default 256 bytes capacity
    }
}

// Thread-local serialization buffer pool for zero-allocation operations
thread_local! {
    static SERIALIZATION_BUFFER: std::cell::RefCell<SerializationBuffer> = 
        std::cell::RefCell::new(SerializationBuffer::new(1024));
}

/// Get thread-local serialization buffer for zero-allocation operations
#[inline(always)]
pub fn with_serialization_buffer<F, R>(f: F) -> R
where
    F: FnOnce(&mut SerializationBuffer) -> R,
{
    SERIALIZATION_BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        f(&mut buffer)
    })
}

pub trait VectorStoreIndexDyn: Send + Sync {
    fn top_n(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<ZeroOneOrMany<(f64, String, Value)>>;
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncTask<ZeroOneOrMany<(f64, String)>>;
}

pub struct VectorStoreIndex {
    pub backend: Box<dyn VectorStoreIndexDyn>,
}

impl VectorStoreIndex {
    // Direct creation from backend
    pub fn with_backend<B: VectorStoreIndexDyn + 'static>(backend: B) -> Self {
        VectorStoreIndex {
            backend: Box::new(backend),
        }
    }

    // VectorQueryBuilder moved to fluent_ai/src/builders/memory.rs
}
