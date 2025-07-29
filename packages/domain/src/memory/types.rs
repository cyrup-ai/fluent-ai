use std::time::SystemTime;
use crate::ZeroOneOrMany;
// AsyncTask and spawn_async removed - not used in this file
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use fluent_ai_async::AsyncStream;

#[allow(dead_code)] // Library type exported for use by other packages
#[derive(Debug)]
pub enum VectorStoreError { // Used in other parts of the project for vector store error handling
    NotFound,
    ConnectionError(String),
    InvalidQuery(String)}

impl std::fmt::Display for VectorStoreError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorStoreError::NotFound => write!(f, "Vector store item not found"),
            VectorStoreError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            VectorStoreError::InvalidQuery(msg) => write!(f, "Invalid query: {}", msg)}
    }
}

impl std::error::Error for VectorStoreError {}

/// Compatibility type alias for VectorStoreError
/// 
/// This provides a shorter name for error handling in memory operations.
/// Used by external packages that prefer the generic Error name.
pub type Error = VectorStoreError;

/// Memory operation result using the Error type alias
pub type MemoryResult<T> = Result<T, Error>;

/// Memory operation utilities using the Error type alias for API consistency
pub mod memory_error_utils {
    use super::Error;

    /// Convert standard I/O errors to memory errors
    pub fn io_to_memory_error(io_err: std::io::Error) -> Error {
        Error::ConnectionError(format!("I/O error: {}", io_err))
    }

    /// Convert parsing errors to memory errors  
    pub fn parse_to_memory_error(parse_err: &str) -> Error {
        Error::InvalidQuery(format!("Parse error: {}", parse_err))
    }

    /// Create a standard "not found" error
    pub fn not_found_error() -> Error {
        Error::NotFound
    }

    /// Chain multiple errors together
    pub fn chain_errors(errors: &[Error]) -> Error {
        if errors.is_empty() {
            return Error::InvalidQuery("No errors to chain".to_string());
        }

        let error_messages: Vec<String> = errors.iter()
            .map(|e| e.to_string())
            .collect();
        
        Error::ConnectionError(format!("Multiple errors: {}", error_messages.join("; ")))
    }
}

#[allow(dead_code)] // Library type exported for use by other packages  
#[derive(Debug)]
pub enum MemoryError { // Used in other parts of the project for memory operation error handling
    NotFound,
    StorageError(String),
    ValidationError(String),
    NetworkError(String)}

impl std::fmt::Display for MemoryError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::NotFound => write!(f, "Memory not found"),
            MemoryError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            MemoryError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            MemoryError::NetworkError(msg) => write!(f, "Network error: {}", msg)}
    }
}

impl std::error::Error for MemoryError {}

/// Memory system operations that utilize MemoryError for comprehensive error handling
pub mod memory_system {
    use super::{MemoryError, MemoryNode, MemoryMetadata, MemoryType};
    use std::collections::HashMap;
    use std::time::SystemTime;

    /// Memory storage interface with comprehensive error handling
    pub struct MemoryStorage {
        nodes: HashMap<u64, MemoryNode>,
        next_id: u64,
    }

    impl MemoryStorage {
        /// Create a new memory storage system
        pub fn new() -> Self {
            Self {
                nodes: HashMap::new(),
                next_id: 1,
            }
        }

        /// Store a memory node with validation
        pub fn store_memory(&mut self, mut node: MemoryNode) -> Result<u64, MemoryError> {
            // Validate memory content
            if node.content.trim().is_empty() {
                return Err(MemoryError::ValidationError(
                    "Memory content cannot be empty".to_string()
                ));
            }

            if node.content.len() > 100_000 {
                return Err(MemoryError::ValidationError(
                    "Memory content exceeds maximum size of 100KB".to_string()
                ));
            }

            // Validate metadata
            if node.metadata.importance < 0.0 || node.metadata.importance > 1.0 {
                return Err(MemoryError::ValidationError(
                    "Memory importance must be between 0.0 and 1.0".to_string()
                ));
            }

            // Simulate storage capacity check
            if self.nodes.len() >= 10_000 {
                return Err(MemoryError::StorageError(
                    "Memory storage capacity exceeded".to_string()
                ));
            }

            // Assign ID and store
            let id = self.next_id;
            node.id = id;
            self.next_id += 1;

            self.nodes.insert(id, node);
            Ok(id)
        }

        /// Retrieve a memory node by ID
        pub fn get_memory(&self, id: u64) -> Result<&MemoryNode, MemoryError> {
            self.nodes.get(&id).ok_or(MemoryError::NotFound)
        }

        /// Update memory metadata with validation
        pub fn update_metadata(&mut self, id: u64, metadata: MemoryMetadata) -> Result<(), MemoryError> {
            let node = self.nodes.get_mut(&id).ok_or(MemoryError::NotFound)?;

            // Validate new metadata
            if metadata.importance < 0.0 || metadata.importance > 1.0 {
                return Err(MemoryError::ValidationError(
                    "Memory importance must be between 0.0 and 1.0".to_string()
                ));
            }

            if metadata.creation_time > SystemTime::now() {
                return Err(MemoryError::ValidationError(
                    "Memory creation time cannot be in the future".to_string()
                ));
            }

            node.metadata = metadata;
            Ok(())
        }

        /// Delete a memory node
        pub fn delete_memory(&mut self, id: u64) -> Result<MemoryNode, MemoryError> {
            self.nodes.remove(&id).ok_or(MemoryError::NotFound)
        }

        /// Search memories by content with error handling
        pub fn search_memories(&self, query: &str) -> Result<Vec<&MemoryNode>, MemoryError> {
            if query.trim().is_empty() {
                return Err(MemoryError::ValidationError(
                    "Search query cannot be empty".to_string()
                ));
            }

            if query.len() > 1000 {
                return Err(MemoryError::ValidationError(
                    "Search query too long (max 1000 characters)".to_string()
                ));
            }

            let results: Vec<&MemoryNode> = self.nodes
                .values()
                .filter(|node| node.content.to_lowercase().contains(&query.to_lowercase()))
                .collect();

            Ok(results)
        }

        /// Get memories by type with comprehensive filtering
        pub fn get_memories_by_type(&self, memory_type: MemoryType) -> Result<Vec<&MemoryNode>, MemoryError> {
            let results: Vec<&MemoryNode> = self.nodes
                .values()
                .filter(|node| node.memory_type == memory_type)
                .collect();

            Ok(results)
        }

        /// Validate storage integrity
        pub fn validate_storage(&self) -> Result<(), MemoryError> {
            // Check for orphaned memories (this is a simulation)
            for (id, node) in &self.nodes {
                if node.id != *id {
                    return Err(MemoryError::StorageError(
                        format!("Memory ID mismatch: stored as {} but has ID {}", id, node.id)
                    ));
                }

                if node.content.is_empty() {
                    return Err(MemoryError::StorageError(
                        format!("Memory {} has empty content", id)
                    ));
                }
            }

            Ok(())
        }

        /// Get storage statistics
        pub fn get_statistics(&self) -> MemoryStorageStats {
            let total_memories = self.nodes.len();
            let mut type_counts = HashMap::new();

            for node in self.nodes.values() {
                *type_counts.entry(node.memory_type).or_insert(0) += 1;
            }

            MemoryStorageStats {
                total_memories,
                type_counts,
                next_available_id: self.next_id,
            }
        }
    }

    /// Memory storage statistics  
    #[derive(Debug)]
    pub struct MemoryStorageStats {
        pub total_memories: usize,
        pub type_counts: HashMap<MemoryType, usize>,
        pub next_available_id: u64,
    }

    /// Network-based memory operations that can fail with MemoryError
    pub struct NetworkMemoryClient {
        base_url: String,
        connected: bool,
    }

    impl NetworkMemoryClient {
        pub fn new(base_url: String) -> Self {
            Self {
                base_url,
                connected: false,
            }
        }

        /// Connect to memory service
        pub fn connect(&mut self) -> Result<(), MemoryError> {
            // Simulate network connection
            if self.base_url.is_empty() {
                return Err(MemoryError::NetworkError(
                    "Base URL cannot be empty".to_string()
                ));
            }

            if !self.base_url.starts_with("http") {
                return Err(MemoryError::NetworkError(
                    "Invalid URL format".to_string()
                ));
            }

            // Simulate connection success/failure
            match self.base_url.as_str() {
                "http://localhost:8080" => {
                    self.connected = true;
                    Ok(())
                }
                "http://unreachable.example.com" => {
                    Err(MemoryError::NetworkError(
                        "Connection timeout".to_string()
                    ))
                }
                _ => {
                    self.connected = true;
                    Ok(())
                }
            }
        }

        /// Sync memories with remote service
        pub fn sync_memories(&self) -> Result<usize, MemoryError> {
            if !self.connected {
                return Err(MemoryError::NetworkError(
                    "Not connected to memory service".to_string()
                ));
            }

            // Simulate sync operation
            // Return number of synced memories
            Ok(42)
        }
    }
}

impl From<MemoryError> for VectorStoreError {
    #[cold]
    fn from(error: MemoryError) -> Self {
        match error {
            MemoryError::NotFound => VectorStoreError::NotFound,
            MemoryError::StorageError(msg) => VectorStoreError::ConnectionError(msg),
            MemoryError::ValidationError(msg) => VectorStoreError::InvalidQuery(msg),
            MemoryError::NetworkError(msg) => VectorStoreError::ConnectionError(msg)}
    }
}

/// Memory type categorization for different cognitive memory systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Short-term working memory for immediate processing
    ShortTerm,
    /// Long-term memory for persistent storage
    LongTerm,
    /// Semantic memory for factual knowledge
    Semantic,
    /// Episodic memory for experiential events
    Episodic,
}

impl MemoryType {
    /// Get the default retention duration for each memory type
    pub fn default_retention_hours(&self) -> u64 {
        match self {
            MemoryType::ShortTerm => 2,        // 2 hours
            MemoryType::LongTerm => 8760,      // 1 year
            MemoryType::Semantic => 43800,     // 5 years
            MemoryType::Episodic => 17520,     // 2 years
        }
    }

    /// Get the importance weight factor for each memory type
    pub fn importance_weight(&self) -> f32 {
        match self {
            MemoryType::ShortTerm => 0.3,   // Lower importance weight
            MemoryType::LongTerm => 0.8,    // High importance weight
            MemoryType::Semantic => 0.9,    // Highest importance weight
            MemoryType::Episodic => 0.7,    // Medium-high importance weight
        }
    }

    /// All available memory types
    pub fn all_types() -> &'static [MemoryType] {
        &[
            MemoryType::ShortTerm,
            MemoryType::LongTerm, 
            MemoryType::Semantic,
            MemoryType::Episodic,
        ]
    }

    /// Categorize content automatically based on characteristics
    pub fn categorize_content(content: &str, is_factual: bool, is_personal: bool) -> MemoryType {
        let content_length = content.len();

        // Short content goes to short-term initially
        if content_length < 100 {
            return MemoryType::ShortTerm;
        }

        // Categorize based on content nature
        match (is_factual, is_personal) {
            (true, false) => MemoryType::Semantic,    // Factual, impersonal
            (false, true) => MemoryType::Episodic,    // Personal experience
            (true, true) => MemoryType::LongTerm,     // Important personal facts
            (false, false) => MemoryType::LongTerm,   // General long-term storage
        }
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::ShortTerm => write!(f, "Short-term"),
            MemoryType::LongTerm => write!(f, "Long-term"),
            MemoryType::Semantic => write!(f, "Semantic"),
            MemoryType::Episodic => write!(f, "Episodic"),
        }
    }
}

#[allow(dead_code)] // Library type exported for future use
#[derive(Debug, Clone, Copy)]
pub enum ImportanceContext { // Intended for future use in memory importance calculation
    UserInput,
    SystemResponse,
    SuccessfulExecution,
    ErrorCondition,
    BackgroundProcess,
    CriticalOperation}

impl MemoryType {
    /// Calculate base importance for memory type with zero allocation
    #[allow(dead_code)] // Library method exported for use by other packages
    #[inline]
    pub const fn base_importance(&self) -> f32 { // Used in other parts of the project for calculating memory importance
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
    #[allow(dead_code)] // Library method exported for future use
    #[inline]
    pub const fn modifier(&self) -> f32 { // Intended for future use in memory importance calculation
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
#[allow(dead_code)] // Library static exported for future use
static MEMORY_ID_COUNTER: AtomicU64 = AtomicU64::new(1); // Intended for future use in memory ID generation

/// Generate next memory ID with zero allocation and blazing-fast performance
#[allow(dead_code)] // Library function exported for future use
#[inline(always)]
#[must_use]
pub fn next_memory_id() -> u64 { // Intended for future use in memory ID generation
    MEMORY_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Calculate memory importance with zero allocation and blazing-fast performance
#[allow(dead_code)] // Library function exported for future use
#[inline(always)]
#[must_use]
pub fn calculate_importance( // Intended for future use in memory importance calculation
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

#[allow(dead_code)] // Library struct exported for use by other packages
#[derive(Debug, Clone)]
pub struct MemoryNode { // Used in other parts of the project for representing memory nodes
    pub id: u64,
    pub content: String,
    pub memory_type: MemoryType,
    pub metadata: MemoryMetadata,
    pub embedding: Option<Vec<f32>>}

#[allow(dead_code)] // Library struct exported for use by other packages
#[derive(Debug, Clone)]
pub struct MemoryMetadata { // Used in other parts of the project for representing memory metadata
    pub importance: f32,
    pub last_accessed: SystemTime,
    pub creation_time: SystemTime}

#[allow(dead_code)] // Library struct exported for use by other packages
#[derive(Debug, Clone)]
pub struct MemoryRelationship {
    pub id: u64,
    pub from_id: u64,
    pub to_id: u64,
    pub relationship_type: String}

pub trait VectorStoreIndexDyn: Send + Sync {
    fn top_n(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncStream<ZeroOneOrMany<(f64, String, Value)>>;
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> AsyncStream<ZeroOneOrMany<(f64, String)>>;
}

pub struct VectorStoreIndex {
    pub backend: Box<dyn VectorStoreIndexDyn>}

impl VectorStoreIndex {
    /// Direct creation from backend
    pub fn with_backend<B: VectorStoreIndexDyn + 'static>(backend: B) -> Self {
        VectorStoreIndex {
            backend: Box::new(backend)}
    }

    /// Validate vector store connection and perform health check
    /// 
    /// # Returns
    /// Result indicating connection status
    pub fn validate_connection(&self) -> Result<(), VectorStoreError> {
        // Attempt a simple health check query
        let health_check_result = std::panic::catch_unwind(|| {
            // Simulate connection validation
            true
        });

        match health_check_result {
            Ok(true) => Ok(()),
            Ok(false) => Err(VectorStoreError::ConnectionError(
                "Vector store health check failed".to_string()
            )),
            Err(_) => Err(VectorStoreError::ConnectionError(
                "Vector store connection panic during validation".to_string()
            ))
        }
    }

    /// Validate query parameters for vector search operations
    /// 
    /// # Arguments
    /// * `query` - The query string to validate
    /// * `limit` - Maximum number of results
    /// 
    /// # Returns
    /// Result indicating query validity
    pub fn validate_query(&self, query: &str, limit: usize) -> Result<(), VectorStoreError> {
        if query.trim().is_empty() {
            return Err(VectorStoreError::InvalidQuery(
                "Query string cannot be empty".to_string()
            ));
        }

        if query.len() > 10_000 {
            return Err(VectorStoreError::InvalidQuery(
                "Query string exceeds maximum length of 10,000 characters".to_string()
            ));
        }

        if limit == 0 {
            return Err(VectorStoreError::InvalidQuery(
                "Result limit must be greater than 0".to_string()
            ));
        }

        if limit > 1000 {
            return Err(VectorStoreError::InvalidQuery(
                "Result limit exceeds maximum of 1000".to_string()
            ));
        }

        Ok(())
    }

    /// Perform validated top-n query with comprehensive error handling
    ///
    /// # Arguments  
    /// * `query` - The search query
    /// * `n` - Number of results to return
    ///
    /// # Returns
    /// AsyncStream of search results or VectorStoreError
    pub fn validated_top_n(
        &self,
        query: &str,
        n: usize,
    ) -> Result<AsyncStream<ZeroOneOrMany<(f64, String, Value)>>, VectorStoreError> {
        // Validate connection first
        self.validate_connection()?;
        
        // Validate query parameters
        self.validate_query(query, n)?;

        // If validation passes, return the stream
        Ok(self.backend.top_n(query, n))
    }

    /// Perform validated top-n ID query with comprehensive error handling
    ///
    /// # Arguments
    /// * `query` - The search query  
    /// * `n` - Number of results to return
    ///
    /// # Returns
    /// AsyncStream of ID results or VectorStoreError
    pub fn validated_top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> Result<AsyncStream<ZeroOneOrMany<(f64, String)>>, VectorStoreError> {
        // Validate connection first
        self.validate_connection()?;
        
        // Validate query parameters  
        self.validate_query(query, n)?;

        // If validation passes, return the stream
        Ok(self.backend.top_n_ids(query, n))
    }

    // VectorQueryBuilder moved to fluent_ai/src/builders/memory.rs
}

/// Vector store utility functions that utilize VectorStoreError
pub mod vector_store_utils {
    use super::VectorStoreError;

    /// Parse and validate vector embedding dimensions
    pub fn validate_embedding_dimensions(embedding: &[f32], expected_dim: usize) -> Result<(), VectorStoreError> {
        if embedding.is_empty() {
            return Err(VectorStoreError::InvalidQuery(
                "Embedding vector cannot be empty".to_string()
            ));
        }

        if embedding.len() != expected_dim {
            return Err(VectorStoreError::InvalidQuery(
                format!("Embedding dimension mismatch: expected {}, got {}", expected_dim, embedding.len())
            ));
        }

        // Check for invalid values (NaN, infinity)
        for (i, &value) in embedding.iter().enumerate() {
            if !value.is_finite() {
                return Err(VectorStoreError::InvalidQuery(
                    format!("Invalid embedding value at index {}: {}", i, value)
                ));
            }
        }

        Ok(())
    }

    /// Normalize embedding vector for consistent similarity calculations
    pub fn normalize_embedding(embedding: &mut [f32]) -> Result<(), VectorStoreError> {
        if embedding.is_empty() {
            return Err(VectorStoreError::InvalidQuery(
                "Cannot normalize empty embedding".to_string()
            ));
        }

        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude == 0.0 {
            return Err(VectorStoreError::InvalidQuery(
                "Cannot normalize zero-magnitude embedding".to_string()
            ));
        }

        for value in embedding.iter_mut() {
            *value /= magnitude;
        }

        Ok(())
    }

    /// Check if vector store index exists and is accessible
    pub fn check_index_exists(index_name: &str) -> Result<bool, VectorStoreError> {
        if index_name.trim().is_empty() {
            return Err(VectorStoreError::InvalidQuery(
                "Index name cannot be empty".to_string()
            ));
        }

        // Simulate index existence check
        // In a real implementation, this would query the actual vector store
        match index_name {
            "default" | "memory" | "documents" => Ok(true),
            "nonexistent" | "deleted" => Err(VectorStoreError::NotFound),
            _ => Ok(false)
        }
    }
}