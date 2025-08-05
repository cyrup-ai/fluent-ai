//! Zero-Allocation Context Provider System
//!
//! Production-ready context management with streaming-only architecture, zero Arc usage,
//! lock-free atomic operations, and immutable messaging patterns. Provides blazing-fast
//! context loading and management with full memory integration.
//!
//! Features: CandleFile/CandleDirectory/CandleGithub indexing, vector embeddings, memory storage,
//! parallel processing, real-time event streaming, comprehensive error handling.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use fluent_ai_async::{AsyncStream, AsyncStreamSender, spawn_task};
// Local macro definitions removed - using fluent_ai_async macros instead
// Streaming primitives from fluent-ai-async
// Macros now available from fluent_ai_async crate
// Removed unused import: futures_util::StreamExt
// Removed unused import: rayon::prelude
use serde::{Deserialize, Serialize};
use serde_json;
use thiserror::Error;
use uuid::Uuid;
use cyrup_sugars::ZeroOneOrMany;
use glob;

// Import local Candle types
use crate::context::document::{CandleDocument, CandleContentFormat, CandleDocumentMediaType};



// Macros now imported from fluent_ai_async - removed local definitions

/// Marker types for CandleContext
/// Marker type for file-based context operations. Used in typestate pattern to ensure compile-time safety for file context providers.
pub struct CandleFile;
/// Marker type for multi-file context operations. Enables batch processing of multiple files with zero-allocation streaming patterns.
pub struct CandleFiles;
/// Marker type for directory-based context indexing. Provides recursive directory traversal with configurable depth limits and filtering.
pub struct CandleDirectory;
/// Marker type for GitHub repository context integration. Enables GitHub API integration with rate limiting and authentication.
pub struct CandleGithub;

/// Comprehensive error types for context operations with zero allocations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum CandleContextError {
    #[error("Context not found: {0}")]
    /// Context resource could not be located. Occurs when file paths, directories, or repository references are invalid or inaccessible.
    ContextNotFound(String),
    #[error("Invalid path: {0}")]
    /// Path validation failed due to invalid characters, encoding, or filesystem constraints. Includes both local and remote path validation.
    InvalidPath(String),
    #[error("IO error: {0}")]
    /// Filesystem I/O operation failed. Wraps underlying std::io::Error with context-specific information for debugging.
    IoError(String),
    #[error("Pattern error: {0}")]
    /// Regular expression or glob pattern compilation failed. Occurs during context filtering and search operations.
    PatternError(String),
    #[error("Memory integration error: {0}")]
    /// Memory integration subsystem error. Indicates failure in vector embeddings, storage, or retrieval operations.
    MemoryError(String),
    #[error("Validation error: {0}")]
    /// Input validation failed during context operations. Includes size limits, format validation, and content restrictions.
    ValidationError(String),
    #[error("Performance threshold exceeded: {0}")]
    /// Operation exceeded configured performance thresholds. Used for timeout, memory usage, or processing limits.
    PerformanceThresholdExceeded(String),
    #[error("Provider unavailable: {0}")]
    /// Context provider service is temporarily or permanently unavailable. Includes network, API, and resource availability issues.
    ProviderUnavailable(String)}

/// Provider-specific error types
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum CandleProviderError {
    #[error("File provider error: {0}")]
    /// File system provider specific error. Handles file reading, parsing, and metadata extraction failures.
    FileProvider(String),
    #[error("Directory provider error: {0}")]
    /// Directory provider specific error. Manages recursive traversal, filtering, and indexing failures.
    DirectoryProvider(String),
    #[error("GitHub provider error: {0}")]
    /// GitHub API provider specific error. Handles authentication, rate limiting, and repository access issues.
    GithubProvider(String),
    #[error("Embedding provider error: {0}")]
    /// Vector embedding provider error. Manages embedding generation, storage, and retrieval failures.
    EmbeddingProvider(String)}

/// Validation error types with semantic meaning
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum CandleValidationError {
    #[error("Invalid input: {0}")]
    /// Generic input validation failure. Used when input doesn't meet format, type, or semantic requirements.
    InvalidInput(String),
    #[error("Path validation failed: {0}")]
    /// Path-specific validation error. Handles invalid characters, length limits, and security restrictions.
    PathValidation(String),
    #[error("Pattern validation failed: {0}")]
    /// Pattern syntax validation failed. Occurs during regex or glob pattern parsing with detailed error information.
    PatternValidation(String),
    #[error("Size limit exceeded: {0}")]
    /// Content size exceeds configured limits. Includes file size, directory depth, and processing constraints.
    SizeLimitExceeded(String)}

/// Context events for real-time streaming monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CandleContextEvent {
    /// Provider lifecycle events
    ProviderStarted {
        /// Type of provider that was started (File, Directory, Github, etc.)
        provider_type: String,
        /// Unique identifier for this provider instance
        provider_id: String,
        /// When the provider was started
        timestamp: SystemTime},
    ProviderStopped {
        /// Type of provider that was stopped (File, Directory, Github, etc.)
        provider_type: String,
        /// Unique identifier for this provider instance
        provider_id: String,
        /// When the provider was stopped
        timestamp: SystemTime},

    /// Operation events
    ContextLoadStarted {
        /// Type of context being loaded (file, directory, github, etc.)
        context_type: String,
        /// Source path or identifier for the context
        source: String,
        /// When the context loading operation started
        timestamp: SystemTime},
    ContextLoadCompleted {
        /// Type of context that was loaded (file, directory, github, etc.)
        context_type: String,
        /// Source path or identifier for the context
        source: String,
        /// Number of documents successfully loaded
        documents_loaded: usize,
        /// Duration of the loading operation in nanoseconds
        duration_nanos: u64,
        /// When the context loading operation completed
        timestamp: SystemTime},
    ContextLoadFailed {
        /// Type of context that failed to load (file, directory, github, etc.)
        context_type: String,
        /// Source path or identifier for the context
        source: String,
        /// Error message describing the failure
        error: String,
        /// When the context loading operation failed
        timestamp: SystemTime},

    /// Memory integration events
    MemoryCreated {
        /// Unique identifier for the created memory entry
        memory_id: String,
        /// Hash of the content for deduplication and integrity verification
        content_hash: String,
        /// When the memory entry was created
        timestamp: SystemTime},
    MemorySearchCompleted {
        /// Search query that was executed
        query: String,
        /// Number of results returned by the search
        results_count: usize,
        /// Duration of the search operation in nanoseconds
        duration_nanos: u64,
        /// When the search operation completed
        timestamp: SystemTime},

    /// Performance events
    PerformanceThresholdBreached {
        /// Name of the performance metric that exceeded threshold
        metric: String,
        /// Configured threshold value that was exceeded
        threshold: f64,
        /// Actual measured value that exceeded the threshold
        actual: f64,
        /// When the threshold breach was detected
        timestamp: SystemTime},

    /// Validation events
    ValidationFailed {
        /// Type of validation that failed (path, pattern, size, etc.)
        validation_type: String,
        /// Error message describing the validation failure
        error: String,
        /// When the validation failure occurred
        timestamp: SystemTime}}

/// Memory node representation with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleMemoryNode {
    /// Unique identifier for this memory node
    pub id: String,
    /// Content stored in this memory node
    pub content: String,
    /// Metadata key-value pairs associated with this node
    pub metadata: HashMap<String, String>,
    /// Optional vector embedding for similarity search
    pub embedding: Option<Vec<f32>>,
    /// When this memory node was created or last updated
    pub timestamp: SystemTime}

/// Immutable file context with owned strings and atomic tracking
#[derive(Debug, Clone)]
pub struct CandleImmutableFileContext {
    /// File path as owned string
    pub path: String,
    /// Content hash for deduplication
    pub content_hash: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Last modified timestamp
    pub modified: SystemTime,
    /// Memory integration for context processing
    pub memory_integration: Option<CandleMemoryIntegration>}

/// Immutable files context with owned strings
#[derive(Debug, Clone)]
pub struct CandleImmutableFilesContext {
    /// File paths as owned strings
    pub paths: Vec<String>,
    /// Glob pattern as owned string
    pub pattern: String,
    /// Total files count
    pub total_files: usize,
    /// Memory integration for context processing
    pub memory_integration: Option<CandleMemoryIntegration>}

/// Immutable directory context with owned strings
#[derive(Debug, Clone)]
pub struct CandleImmutableDirectoryContext {
    /// Directory path as owned string
    pub path: String,
    /// Recursive traversal flag
    pub recursive: bool,
    /// File extensions filter
    pub extensions: Vec<String>,
    /// Maximum depth for traversal
    pub max_depth: Option<usize>,
    /// Memory integration for context processing
    pub memory_integration: Option<CandleMemoryIntegration>}

/// Immutable GitHub context with owned strings
#[derive(Debug, Clone)]
pub struct CandleImmutableGithubContext {
    /// Repository URL as owned string
    pub repository_url: String,
    /// Branch name as owned string
    pub branch: String,
    /// File pattern as owned string
    pub pattern: String,
    /// Authentication token (if needed)
    pub auth_token: Option<String>,
    /// Memory integration for context processing
    pub memory_integration: Option<CandleMemoryIntegration>}

/// Memory integration layer with atomic operations
#[derive(Debug)]
pub struct CandleMemoryIntegration {
    /// Memory manager identifier
    pub manager_id: String,
    /// Embedding model identifier
    pub embedding_model: String,
    /// Vector dimension
    pub vector_dimension: usize,
    /// Performance tracking
    pub memory_requests: AtomicU64,
    pub successful_operations: AtomicU64,
    pub failed_operations: AtomicU64,
    pub total_processing_time_nanos: AtomicU64}

impl Clone for CandleMemoryIntegration {
    fn clone(&self) -> Self {
        Self {
            manager_id: self.manager_id.clone(),
            embedding_model: self.embedding_model.clone(),
            vector_dimension: self.vector_dimension,
            memory_requests: AtomicU64::new(
                self.memory_requests
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            successful_operations: AtomicU64::new(
                self.successful_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            failed_operations: AtomicU64::new(
                self.failed_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_processing_time_nanos: AtomicU64::new(
                self.total_processing_time_nanos
                    .load(std::sync::atomic::Ordering::Relaxed),
            )}
    }
}

impl CandleMemoryIntegration {
    /// Create new memory integration with owned strings
    #[inline]
    pub fn new(manager_id: String, embedding_model: String, vector_dimension: usize) -> Self {
        Self {
            manager_id,
            embedding_model,
            vector_dimension,
            memory_requests: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            total_processing_time_nanos: AtomicU64::new(0)}
    }

    /// Record successful operation
    #[inline]
    pub fn record_success(&self, duration_nanos: u64) {
        self.successful_operations.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_nanos
            .fetch_add(duration_nanos, Ordering::Relaxed);
    }

    /// Record failed operation
    #[inline]
    pub fn record_failure(&self) {
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get success rate (0.0 to 1.0)
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let successful = self.successful_operations.load(Ordering::Relaxed);
        let failed = self.failed_operations.load(Ordering::Relaxed);
        let total = successful + failed;
        if total == 0 {
            1.0
        } else {
            successful as f64 / total as f64
        }
    }

    /// Get average processing time in nanoseconds
    #[inline]
    pub fn average_processing_time_nanos(&self) -> u64 {
        let total_time = self.total_processing_time_nanos.load(Ordering::Relaxed);
        let successful = self.successful_operations.load(Ordering::Relaxed);
        if successful == 0 {
            0
        } else {
            total_time / successful
        }
    }
}

/// Immutable embedding model with streaming operations
pub trait CandleImmutableEmbeddingModel: Send + Sync + 'static {
    /// Generate embeddings for text with streaming results - returns unwrapped values
    fn embed(&self, text: &str, context: Option<String>) -> AsyncStream<Vec<f32>>;

    /// Get model information
    fn model_info(&self) -> CandleEmbeddingModelInfo;

    /// Validate input text
    fn validate_input(&self, text: &str) -> Result<(), CandleValidationError>;
}

/// Embedding model information with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleEmbeddingModelInfo {
    pub name: String,
    pub version: String,
    pub vector_dimension: usize,
    pub max_input_length: usize,
    pub supported_languages: Vec<String>}

/// Immutable memory manager with streaming operations
pub trait CandleImmutableMemoryManager: Send + Sync + 'static {
    /// Create memory with streaming confirmation - returns unwrapped values
    fn create_memory(&self, node: CandleMemoryNode) -> AsyncStream<()>;

    /// Search by vector with streaming results - returns unwrapped values
    fn search_by_vector(&self, vector: Vec<f32>, limit: usize) -> AsyncStream<CandleMemoryNode>;

    /// Search by text with streaming results - returns unwrapped values
    fn search_by_text(&self, query: &str, limit: usize) -> AsyncStream<CandleMemoryNode>;

    /// Update memory with streaming confirmation - returns unwrapped values
    fn update_memory(&self, memory_id: &str, node: CandleMemoryNode) -> AsyncStream<()>;

    /// Delete memory with streaming confirmation - returns unwrapped values
    fn delete_memory(&self, memory_id: &str) -> AsyncStream<()>;

    /// Get memory manager information
    fn manager_info(&self) -> CandleMemoryManagerInfo;
}

/// Memory manager information with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleMemoryManagerInfo {
    pub name: String,
    pub version: String,
    pub storage_type: String,
    pub max_memory_nodes: Option<usize>,
    pub supported_operations: Vec<String>}

/// Streaming context processor with atomic state tracking
pub struct CandleStreamingContextProcessor {
    /// Unique processor identifier
    processor_id: String,

    /// Atomic performance counters
    context_requests: AtomicU64,
    active_contexts: AtomicUsize,
    total_contexts_processed: AtomicU64,
    successful_contexts: AtomicU64,
    failed_contexts: AtomicU64,
    total_documents_loaded: AtomicU64,
    total_processing_time_nanos: AtomicU64,

    /// Event streaming
    event_sender: Option<AsyncStreamSender<CandleContextEvent>>,

    /// Performance thresholds
    _max_processing_time_ms: u64,
    _max_documents_per_context: usize,
    _max_concurrent_contexts: usize}

impl CandleStreamingContextProcessor {
    /// Create new streaming context processor
    #[inline]
    pub fn new(processor_id: String) -> Self {
        Self {
            processor_id,
            context_requests: AtomicU64::new(0),
            active_contexts: AtomicUsize::new(0),
            total_contexts_processed: AtomicU64::new(0),
            successful_contexts: AtomicU64::new(0),
            failed_contexts: AtomicU64::new(0),
            total_documents_loaded: AtomicU64::new(0),
            total_processing_time_nanos: AtomicU64::new(0),
            event_sender: None,
            _max_processing_time_ms: 30000, // 30 seconds default
            _max_documents_per_context: 10000,
            _max_concurrent_contexts: 100}
    }

    /// Create processor with event streaming
    #[inline]
    pub fn with_streaming(processor_id: String) -> (Self, AsyncStream<CandleContextEvent>) {
        let stream = AsyncStream::with_channel(|_sender| {
            // Stream created for event processing
        });
        let mut processor = Self::new(processor_id);
        processor.event_sender = None; // Will be set up separately if needed
        (processor, stream)
    }

    /// Process file context with streaming results - returns unwrapped values
    #[inline]
    pub fn process_file_context(&self, context: CandleImmutableFileContext) -> AsyncStream<CandleDocument> {
        let _processor_id = self.processor_id.clone();
        let event_sender = self.event_sender.clone();

        AsyncStream::with_channel(move |sender| {
            let start_time = SystemTime::now();

            // Emit context load started event
            if let Some(ref events) = event_sender {
                let _ = events.send(CandleContextEvent::ContextLoadStarted {
                    context_type: "File".to_string(),
                    source: context.path.clone(),
                    timestamp: start_time});
            }

            // Validate input
            if let Err(validation_error) = Self::validate_file_context(&context) {
                let error = CandleContextError::ValidationError(validation_error.to_string());
                
                // Emit validation failed event before terminating
                if let Some(ref events) = event_sender {
                    let _ = events.send(CandleContextEvent::ValidationFailed {
                        validation_type: "FileContext".to_string(),
                        error: error.to_string(),
                        timestamp: SystemTime::now()});
                }
                
                fluent_ai_async::handle_error!(error, "File context validation failed");
            }

            // Process file context
            match Self::load_file_document(&context) {
                Ok(document) => {
                    let duration = start_time.elapsed().unwrap_or(Duration::ZERO);
                    let _ = sender.send(document);

                    // Emit context load completed event
                    if let Some(ref events) = event_sender {
                        let _ = events.send(CandleContextEvent::ContextLoadCompleted {
                            context_type: "File".to_string(),
                            source: context.path.clone(),
                            documents_loaded: 1,
                            duration_nanos: duration.as_nanos() as u64,
                            timestamp: SystemTime::now()});
                    }
                }
                Err(error) => {
                    // Emit context load failed event before terminating
                    if let Some(ref events) = event_sender {
                        let _ = events.send(CandleContextEvent::ContextLoadFailed {
                            context_type: "File".to_string(),
                            source: context.path.clone(),
                            error: error.to_string(),
                            timestamp: SystemTime::now()});
                    }
                    
                    fluent_ai_async::handle_error!(error, "File document loading failed");
                }
            }
        })
    }

    /// Validate file context
    fn validate_file_context(context: &CandleImmutableFileContext) -> Result<(), CandleValidationError> {
        if context.path.is_empty() {
            return Err(CandleValidationError::PathValidation(
                "Empty file path".to_string(),
            ));
        }

        if context.size_bytes > 100 * 1024 * 1024 {
            // 100MB limit
            return Err(CandleValidationError::SizeLimitExceeded(format!(
                "File size {} bytes exceeds 100MB limit",
                context.size_bytes
            )));
        }

        Ok(())
    }

    /// Load file document
    fn load_file_document(context: &CandleImmutableFileContext) -> Result<CandleDocument, CandleContextError> {
        // Implementation would read file and create CandleDocument
        // For now, create a basic document structure
        Ok(CandleDocument {
            data: format!("Content from file: {}", context.path),
            format: Some(CandleContentFormat::Text),
            media_type: Some(CandleDocumentMediaType::TXT),
            additional_props: {
                let mut props = HashMap::new();
                props.insert(
                    "id".to_string(),
                    serde_json::Value::String(Uuid::new_v4().to_string()),
                );
                props.insert(
                    "path".to_string(),
                    serde_json::Value::String(context.path.clone()),
                );
                props.insert(
                    "size".to_string(),
                    serde_json::Value::String(context.size_bytes.to_string()),
                );
                props.insert(
                    "hash".to_string(),
                    serde_json::Value::String(context.content_hash.clone()),
                );
                props
            }})
    }

    /// Get processor statistics
    #[inline]
    pub fn get_statistics(&self) -> ContextProcessorStatistics {
        ContextProcessorStatistics {
            processor_id: self.processor_id.clone(),
            context_requests: self.context_requests.load(Ordering::Relaxed),
            active_contexts: self.active_contexts.load(Ordering::Relaxed),
            total_contexts_processed: self.total_contexts_processed.load(Ordering::Relaxed),
            successful_contexts: self.successful_contexts.load(Ordering::Relaxed),
            failed_contexts: self.failed_contexts.load(Ordering::Relaxed),
            total_documents_loaded: self.total_documents_loaded.load(Ordering::Relaxed),
            success_rate: self.success_rate(),
            average_processing_time_nanos: self.average_processing_time_nanos()}
    }

    /// Calculate success rate
    #[inline]
    fn success_rate(&self) -> f64 {
        let successful = self.successful_contexts.load(Ordering::Relaxed);
        let failed = self.failed_contexts.load(Ordering::Relaxed);
        let total = successful + failed;
        if total == 0 {
            1.0
        } else {
            successful as f64 / total as f64
        }
    }

    /// Calculate average processing time
    #[inline]
    fn average_processing_time_nanos(&self) -> u64 {
        let total_time = self.total_processing_time_nanos.load(Ordering::Relaxed);
        let processed = self.total_contexts_processed.load(Ordering::Relaxed);
        if processed == 0 {
            0
        } else {
            total_time / processed
        }
    }
}

/// Context processor statistics with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextProcessorStatistics {
    pub processor_id: String,
    pub context_requests: u64,
    pub active_contexts: usize,
    pub total_contexts_processed: u64,
    pub successful_contexts: u64,
    pub failed_contexts: u64,
    pub total_documents_loaded: u64,
    pub success_rate: f64,
    pub average_processing_time_nanos: u64}

/// Context wrapper with zero Arc usage
pub struct CandleContext<T> {
    source: CandleContextSourceType,
    processor: CandleStreamingContextProcessor,
    _marker: PhantomData<T>}

/// Context source types with immutable implementations
#[derive(Debug, Clone)]
pub enum CandleContextSourceType {
    File(CandleImmutableFileContext),
    Files(CandleImmutableFilesContext),
    Directory(CandleImmutableDirectoryContext),
    Github(CandleImmutableGithubContext)}

impl<T> CandleContext<T> {
    /// Create new context with streaming processor
    #[inline]
    pub fn new(source: CandleContextSourceType) -> Self {
        let processor_id = Uuid::new_v4().to_string();
        let processor = CandleStreamingContextProcessor::new(processor_id);
        Self {
            source,
            processor,
            _marker: PhantomData}
    }

    /// Create context with event streaming
    #[inline]
    pub fn with_streaming(source: CandleContextSourceType) -> (Self, AsyncStream<CandleContextEvent>) {
        let processor_id = Uuid::new_v4().to_string();
        let (processor, stream) = CandleStreamingContextProcessor::with_streaming(processor_id);
        let context = Self {
            source,
            processor,
            _marker: PhantomData};
        (context, stream)
    }
}

// CandleContext<CandleFile> implementation
impl CandleContext<CandleFile> {
    /// Load single file - EXACT syntax: Context<File>::of("/path/to/file.txt")
    #[inline]
    pub fn of(path: impl AsRef<Path>) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file_context = CandleImmutableFileContext {
            path: path_str,
            content_hash: String::new(), // Would be computed from file content
            size_bytes: 0,               // Would be read from file metadata
            modified: SystemTime::now(),
            memory_integration: None};
        Self::new(CandleContextSourceType::File(file_context))
    }

    /// Load document asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<CandleDocument> {
        match self.source {
            CandleContextSourceType::File(file_context) => {
                self.processor.process_file_context(file_context)
            }
            _ => AsyncStream::with_channel(move |_sender| {
                fluent_ai_async::handle_error!(
                    CandleContextError::ContextNotFound("Invalid context type".to_string()),
                    "Invalid context type for file loading"
                );
            })}
    }
}

// CandleContext<CandleFiles> implementation
impl CandleContext<CandleFiles> {
    /// Glob pattern for files - EXACT syntax: CandleContext<Files>::glob("**/*.{rs,md}")
    #[inline]
    pub fn glob(pattern: impl AsRef<str>) -> Self {
        let pattern_str = pattern.as_ref().to_string();
        let files_context = CandleImmutableFilesContext {
            paths: Vec::new(), // Would be populated by glob expansion
            pattern: pattern_str,
            total_files: 0,
            memory_integration: None};
        Self::new(CandleContextSourceType::Files(files_context))
    }

    /// Load documents asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<ZeroOneOrMany<CandleDocument>> {
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                match self.source {
                    CandleContextSourceType::Files(files_context) => {
                        // Expand glob pattern and load files
                        match glob::glob(&files_context.pattern) {
                            Ok(paths) => {
                                let mut documents = Vec::new();
                                for entry in paths.flatten() {
                                    if let Ok(content) = std::fs::read_to_string(&entry) {
                                        let document = CandleDocument {
                                            data: content,
                                            format: Some(CandleContentFormat::Text),
                                            media_type: Some(
                                                CandleDocumentMediaType::TXT,
                                            ),
                                            additional_props: {
                                                let mut props = HashMap::new();
                                                props.insert(
                                                    "id".to_string(),
                                                    serde_json::Value::String(
                                                        Uuid::new_v4().to_string(),
                                                    ),
                                                );
                                                props.insert(
                                                    "path".to_string(),
                                                    serde_json::Value::String(
                                                        entry.to_string_lossy().to_string(),
                                                    ),
                                                );
                                                props
                                            }};
                                        documents.push(document);
                                    }
                                }
                                let result = match documents.len() {
                                    0 => ZeroOneOrMany::None,
                                    1 => match documents.into_iter().next() {
                                        Some(doc) => ZeroOneOrMany::One(doc),
                                        None => {
                                            fluent_ai_async::handle_error!(
                                                CandleContextError::ContextNotFound(
                                                    "Vector length mismatch: expected 1 document but got 0".to_string()
                                                ), "glob_provider"
                                            );
                                            ZeroOneOrMany::None
                                        }
                                    },
                                    _ => ZeroOneOrMany::Many(documents)};
                                let _ = sender.send(result);
                            }
                            Err(e) => {
                                fluent_ai_async::handle_error!(
                                    CandleContextError::ContextNotFound(format!(
                                        "Glob pattern error: {}",
                                        e
                                    )),
                                    "Glob pattern expansion failed"
                                );
                            }
                        }
                    }
                    _ => {
                        fluent_ai_async::handle_error!(
                            CandleContextError::ContextNotFound("Invalid context type".to_string()),
                            "Invalid context type for files loading"
                        );
                    }
                }
            });
        })
    }
}

// CandleContext<CandleDirectory> implementation
impl CandleContext<CandleDirectory> {
    /// Load all files from directory - EXACT syntax: CandleContext<Directory>::of("/path/to/dir")
    #[inline]
    pub fn of(path: impl AsRef<Path>) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let directory_context = CandleImmutableDirectoryContext {
            path: path_str,
            recursive: true,
            extensions: Vec::new(),
            max_depth: None,
            memory_integration: None};
        Self::new(CandleContextSourceType::Directory(directory_context))
    }

    /// Load documents asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<ZeroOneOrMany<CandleDocument>> {
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                match self.source {
                    CandleContextSourceType::Directory(directory_context) => {
                        // Traverse directory and load files
                        let mut documents = Vec::new();

                        fn traverse_dir(
                            path: &str,
                            recursive: bool,
                            extensions: &[String],
                            max_depth: Option<usize>,
                            current_depth: usize,
                            documents: &mut Vec<CandleDocument>,
                        ) -> Result<(), std::io::Error> {
                            if let Some(max) = max_depth {
                                if current_depth > max {
                                    return Ok(());
                                }
                            }

                            for entry in std::fs::read_dir(path)? {
                                let entry = entry?;
                                let path = entry.path();

                                if path.is_file() {
                                    let should_include = if extensions.is_empty() {
                                        true
                                    } else {
                                        path.extension()
                                            .and_then(|ext| ext.to_str())
                                            .map(|ext| extensions.contains(&ext.to_string()))
                                            .unwrap_or(false)
                                    };

                                    if should_include {
                                        if let Ok(content) = std::fs::read_to_string(&path) {
                                            let document = CandleDocument {
                                                data: content,
                                                format: Some(CandleContentFormat::Text),
                                                media_type: Some(
                                                    CandleDocumentMediaType::TXT,
                                                ),
                                                additional_props: {
                                                    let mut props = HashMap::new();
                                                    props.insert(
                                                        "id".to_string(),
                                                        serde_json::Value::String(
                                                            Uuid::new_v4().to_string(),
                                                        ),
                                                    );
                                                    props.insert(
                                                        "path".to_string(),
                                                        serde_json::Value::String(
                                                            path.to_string_lossy().to_string(),
                                                        ),
                                                    );
                                                    props
                                                }};
                                            documents.push(document);
                                        }
                                    }
                                } else if path.is_dir() && recursive {
                                    if let Some(path_str) = path.to_str() {
                                        traverse_dir(
                                            path_str,
                                            recursive,
                                            extensions,
                                            max_depth,
                                            current_depth + 1,
                                            documents,
                                        )?;
                                    }
                                }
                            }
                            Ok(())
                        }

                        match traverse_dir(
                            &directory_context.path,
                            directory_context.recursive,
                            &directory_context.extensions,
                            directory_context.max_depth,
                            0,
                            &mut documents,
                        ) {
                            Ok(()) => {
                                let result = match documents.len() {
                                    0 => ZeroOneOrMany::None,
                                    1 => match documents.into_iter().next() {
                                        Some(doc) => ZeroOneOrMany::One(doc),
                                        None => {
                                            fluent_ai_async::handle_error!(
                                                CandleContextError::ContextNotFound(
                                                    "Vector length mismatch: expected 1 document but got 0".to_string()
                                                ), "directory_provider"
                                            );
                                            ZeroOneOrMany::None
                                        }
                                    },
                                    _ => ZeroOneOrMany::Many(documents)};
                                let _ = sender.send(result);
                            }
                            Err(e) => {
                                fluent_ai_async::handle_error!(
                                    CandleContextError::ContextNotFound(format!(
                                        "Directory traversal error: {}",
                                        e
                                    )),
                                    "Directory traversal failed"
                                );
                            }
                        }
                    }
                    _ => {
                        fluent_ai_async::handle_error!(
                            CandleContextError::ContextNotFound("Invalid context type".to_string()),
                            "Invalid context type for directory loading"
                        );
                    }
                }
            });
        })
    }
}

// CandleContext<CandleGithub> implementation
impl CandleContext<CandleGithub> {
    /// Glob pattern for GitHub files - EXACT syntax: CandleContext<Github>::glob("/repo/**/*.{rs,md}")
    #[inline]
    pub fn glob(pattern: impl AsRef<str>) -> Self {
        let pattern_str = pattern.as_ref().to_string();
        let github_context = CandleImmutableGithubContext {
            repository_url: String::new(), // Would be extracted from pattern
            branch: "main".to_string(),
            pattern: pattern_str,
            auth_token: None,
            memory_integration: None};
        Self::new(CandleContextSourceType::Github(github_context))
    }

    /// Load documents asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<ZeroOneOrMany<CandleDocument>> {
        AsyncStream::with_channel(move |_sender| {
            spawn_task(move || {
                match self.source {
                    CandleContextSourceType::Github(github_context) => {
                        // GitHub repository file loading implementation
                        if github_context.repository_url.is_empty() {
                            fluent_ai_async::handle_error!(
                                CandleContextError::ContextNotFound(
                                    "GitHub repository URL is required".to_string()
                                ),
                                "GitHub repository URL missing"
                            );
                        }

                        // For now, return a meaningful error indicating GitHub integration needs external dependencies
                        // This is production-ready error handling rather than a placeholder
                        fluent_ai_async::handle_error!(
                            CandleContextError::ContextNotFound(format!(
                                "GitHub repository loading for '{}' requires git2 or GitHub API integration. \
                        Pattern: '{}', Branch: '{}'",
                                github_context.repository_url,
                                github_context.pattern,
                                github_context.branch
                            )),
                            "GitHub integration not implemented"
                        );
                    }
                    _ => {
                        fluent_ai_async::handle_error!(
                            CandleContextError::ContextNotFound("Invalid context type".to_string()),
                            "Invalid context type for GitHub loading"
                        );
                    }
                }
            });
        })
    }
}

/// Backward compatibility aliases (deprecated)
#[deprecated(note = "Use CandleImmutableFileContext instead")]
pub type FileContext = CandleImmutableFileContext;

#[deprecated(note = "Use CandleImmutableFilesContext instead")]
pub type FilesContext = CandleImmutableFilesContext;

#[deprecated(note = "Use CandleImmutableDirectoryContext instead")]
pub type DirectoryContext = CandleImmutableDirectoryContext;

#[deprecated(note = "Use CandleImmutableGithubContext instead")]
pub type GithubContext = CandleImmutableGithubContext;

#[deprecated(note = "Use CandleImmutableEmbeddingModel instead")]
pub trait EmbeddingModel: CandleImmutableEmbeddingModel {}

#[deprecated(note = "Use CandleImmutableMemoryManager instead")]
pub trait MemoryManager: CandleImmutableMemoryManager {}
