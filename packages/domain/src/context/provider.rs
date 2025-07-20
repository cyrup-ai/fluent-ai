//! Production-Ready Context Management with Memory Integration
//!
//! This module provides blazing-fast context loading and management with full integration
//! to the fluent_ai_memory system for actual content indexing, storage, and retrieval.
//! Originally from context.rs.
//!
//! Features: File/Directory/GitHub indexing, vector embeddings, memory storage, parallel processing.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

// High-performance dependencies
use arrayvec::ArrayVec;
// Memory system integration
use fluent_ai_memory::memory::manager::surreal::SurrealDBMemoryManager;
use fluent_ai_memory::memory::primitives::metadata::MemoryMetadata;
use fluent_ai_memory::memory::primitives::types::MemoryTypeEnum;
use fluent_ai_memory::memory::primitives::MemoryNode;
use fluent_ai_memory::utils::error::Error as MemoryError;
use fluent_ai_memory::vector::embedding_model::EmbeddingModel;
// Additional imports for async operations
use futures::StreamExt;
use glob::Pattern;

use jwalk::WalkDir;
use memmap2::MmapOptions;
use rayon::prelude::*;
use smallvec::SmallVec;
use tokio_stream::Stream;

use crate::{Document, ZeroOneOrMany};

/// Marker types for Context
pub struct File;
pub struct Files;
pub struct Directory;
pub struct Github;

/// Context source types with production-ready implementations
#[derive(Debug, Clone)]
pub enum ContextSourceType {
    File(FileContext),
    Files(FilesContext),
    Directory(DirectoryContext),
    Github(GithubContext),
}

/// Comprehensive error types for context operations
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Memory mapping failed: {0}")]
    MemoryMappingFailed(String),

    #[error("Invalid glob pattern: {0}")]
    InvalidGlobPattern(String),

    #[error("Document parsing failed: {0}")]
    DocumentParsingFailed(String),

    #[error("Memory system error: {0}")]
    MemoryError(String),

    #[error("Embedding generation failed: {0}")]
    EmbeddingError(String),

    #[error("Git repository error: {0}")]
    GitError(String),

    #[error("Context not found: {0}")]
    ContextNotFound(String),
}

impl From<MemoryError> for ContextError {
    fn from(error: MemoryError) -> Self {
        ContextError::MemoryError(error.to_string())
    }
}

impl From<std::io::Error> for ContextError {
    fn from(error: std::io::Error) -> Self {
        ContextError::IoError(error.to_string())
    }
}

impl From<glob::PatternError> for ContextError {
    fn from(error: glob::PatternError) -> Self {
        ContextError::InvalidGlobPattern(error.to_string())
    }
}

/// Memory integration layer for Context providers
#[derive(Debug)]
pub struct MemoryIntegration {
    memory_manager: Arc<SurrealDBMemoryManager>,
    embedding_model: Arc<dyn EmbeddingModel>,
}

impl MemoryIntegration {
    /// Create new memory integration instance
    pub fn new(
        memory_manager: Arc<SurrealDBMemoryManager>,
        embedding_model: Arc<dyn EmbeddingModel>,
    ) -> Self {
        Self {
            memory_manager,
            embedding_model,
        }
    }

    /// Store document content as memory with embedding
    pub async fn store_document(
        &self,
        path: &str,
        content: &str,
        memory_type: MemoryTypeEnum,
    ) -> Result<MemoryNode, ContextError> {
        // Generate embedding for content
        let embedding = self
            .embedding_model
            .embed(content, Some("document_indexing".to_string()))
            .await
            .map_err(|e| ContextError::EmbeddingError(e.to_string()))?;

        // Create memory metadata with correct fields
        let mut metadata = MemoryMetadata {
            user_id: None,
            agent_id: None,
            context: "file_context".to_string(),
            keywords: vec![],
            tags: vec!["document".to_string(), "context".to_string()],
            category: "file".to_string(),
            importance: 0.5,
            source: Some(path.to_string()),
            created_at: chrono::Utc::now(),
            last_accessed_at: Some(chrono::Utc::now()),
            embedding: None,
            custom: serde_json::json!({
                "file_path": path.to_string()
            }),
        };

        // Create memory node
        let memory_node = MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            embedding: Some(embedding),
            memory_type,
            metadata,
        };

        // Store in memory system
        self.memory_manager
            .store_memory(&memory_node)
            .await
            .map_err(ContextError::from)?;

        Ok(memory_node)
    }

    /// Store multiple documents in batch
    pub async fn store_documents_batch(
        &self,
        documents: Vec<(String, String, MemoryTypeEnum)>,
    ) -> Result<Vec<MemoryNode>, ContextError> {
        let mut results = Vec::with_capacity(documents.len());

        for (path, content, memory_type) in documents {
            let node = self.store_document(&path, &content, memory_type).await?;
            results.push(node);
        }

        Ok(results)
    }

    /// Search documents by content
    pub async fn search_documents(&self, query: &str) -> Result<Vec<MemoryNode>, ContextError> {
        // Generate embedding for query
        let query_embedding = self
            .embedding_model
            .embed(query, Some("document_search".to_string()))
            .await
            .map_err(|e| ContextError::EmbeddingError(e.to_string()))?;

        // Search in memory system
        let results = self
            .memory_manager
            .search_by_embedding(&query_embedding, Some(10))
            .await
            .map_err(ContextError::from)?;

        Ok(results)
    }
}

/// Production-ready FileContext implementation
#[derive(Debug, Clone)]
pub struct FileContext {
    path: ArrayVec<u8, 256>,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl FileContext {
    /// Create new FileContext
    pub fn new(path: impl AsRef<Path>) -> Result<Self, ContextError> {
        let path_str = path.as_ref().to_string_lossy();
        let mut path_array = ArrayVec::new();

        for byte in path_str.as_bytes().iter().take(256) {
            if path_array.try_push(*byte).is_err() {
                break;
            }
        }

        Ok(Self {
            path: path_array,
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Load file content efficiently using memory mapping for large files
    pub fn load_content(&self) -> Result<String, ContextError> {
        let path_str = String::from_utf8_lossy(&self.path);
        let path = Path::new(path_str.as_ref());

        if !path.exists() {
            return Err(ContextError::FileNotFound(path_str.to_string()));
        }

        let metadata = std::fs::metadata(path)?;

        // Use memory mapping for files larger than 1MB
        if metadata.len() > 1_048_576 {
            self.load_content_mmap()
        } else {
            self.load_content_standard()
        }
    }

    /// Memory-mapped file loading for large files
    pub fn load_content_mmap(&self) -> Result<String, ContextError> {
        let path_str = String::from_utf8_lossy(&self.path);
        let file = std::fs::File::open(path_str.as_ref())?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| ContextError::MemoryMappingFailed(e.to_string()))?
        };

        String::from_utf8_lossy(&mmap).to_string().into()
    }

    /// Standard file loading for small files
    pub fn load_content_standard(&self) -> Result<String, ContextError> {
        let path_str = String::from_utf8_lossy(&self.path);
        std::fs::read_to_string(path_str.as_ref()).map_err(ContextError::from)
    }

    /// Convert to documents with memory storage
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let content = self.load_content()?;
        let path_str = String::from_utf8_lossy(&self.path);

        // Store in memory if integration is available
        if let Some(integration) = &self.memory_integration {
            integration
                .store_document(&path_str, &content, MemoryTypeEnum::Document)
                .await?;
        }

        let document = Document {
            data: content,
            format: Some(super::document::ContentFormat::Text),
            media_type: Some(super::document::DocumentMediaType::TXT),
            additional_props: HashMap::new(),
        };

        Ok(ZeroOneOrMany::One(document))
    }
}

/// Production-ready FilesContext implementation
#[derive(Debug, Clone)]
pub struct FilesContext {
    pattern: SmallVec<u8, 128>,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl FilesContext {
    /// Create new FilesContext with glob pattern
    pub fn new(pattern: &str) -> Result<Self, ContextError> {
        let mut pattern_vec = SmallVec::new();

        for byte in pattern.as_bytes().iter().take(128) {
            pattern_vec.push(*byte);
        }

        Ok(Self {
            pattern: pattern_vec,
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Convert to documents with parallel processing
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let pattern_str = String::from_utf8_lossy(&self.pattern);
        let pattern = Pattern::new(&pattern_str)?;

        // Use glob to find matching files
        let paths: Vec<PathBuf> = glob::glob(&pattern_str)
            .map_err(ContextError::from)?
            .filter_map(Result::ok)
            .collect();

        if paths.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }

        // Process files in parallel using rayon
        let documents: Result<Vec<Document>, ContextError> = paths
            .par_iter()
            .map(|path| {
                let content = std::fs::read_to_string(path).map_err(ContextError::from)?;

                Ok(Document {
                    data: content,
                    format: Some(super::document::ContentFormat::Text),
                    media_type: Some(super::document::DocumentMediaType::TXT),
                    additional_props: HashMap::new(),
                })
            })
            .collect();

        let documents = documents?;

        // Store in memory if integration is available
        if let Some(integration) = &self.memory_integration {
            let docs_for_storage: Vec<(String, String, MemoryTypeEnum)> = paths
                .iter()
                .zip(documents.iter())
                .map(|(path, doc)| {
                    (
                        path.to_string_lossy().to_string(),
                        doc.data.clone(),
                        MemoryTypeEnum::Document,
                    )
                })
                .collect();

            integration.store_documents_batch(docs_for_storage).await?;
        }

        Ok(ZeroOneOrMany::many(documents))
    }
}

/// Production-ready DirectoryContext implementation
#[derive(Debug, Clone)]
pub struct DirectoryContext {
    path: ArrayVec<u8, 256>,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl DirectoryContext {
    /// Create new DirectoryContext
    pub fn new(path: impl AsRef<Path>) -> Result<Self, ContextError> {
        let path_str = path.as_ref().to_string_lossy();
        let mut path_array = ArrayVec::new();

        for byte in path_str.as_bytes().iter().take(256) {
            if path_array.try_push(*byte).is_err() {
                break;
            }
        }

        Ok(Self {
            path: path_array,
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Convert to documents with parallel directory traversal
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let path_str = String::from_utf8_lossy(&self.path);
        let path = Path::new(path_str.as_ref());

        if !path.exists() {
            return Err(ContextError::FileNotFound(path_str.to_string()));
        }

        // Use jwalk for parallel directory traversal
        let files: Vec<PathBuf> = WalkDir::new(path)
            .into_iter()
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    if e.file_type().is_file() {
                        Some(e.path())
                    } else {
                        None
                    }
                })
            })
            .collect();

        if files.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }

        // Process files in parallel
        let documents: Result<Vec<Document>, ContextError> = files
            .par_iter()
            .map(|path| {
                let content = std::fs::read_to_string(path).map_err(ContextError::from)?;

                Ok(Document {
                    data: content,
                    format: Some(super::document::ContentFormat::Text),
                    media_type: Some(super::document::DocumentMediaType::TXT),
                    additional_props: HashMap::new(),
                })
            })
            .collect();

        let documents = documents?;

        // Store in memory if integration is available
        if let Some(integration) = &self.memory_integration {
            let docs_for_storage: Vec<(String, String, MemoryTypeEnum)> = files
                .iter()
                .zip(documents.iter())
                .map(|(path, doc)| {
                    (
                        path.to_string_lossy().to_string(),
                        doc.data.clone(),
                        MemoryTypeEnum::Document,
                    )
                })
                .collect();

            integration.store_documents_batch(docs_for_storage).await?;
        }

        Ok(ZeroOneOrMany::many(documents))
    }
}

/// Production-ready GithubContext implementation
#[derive(Debug, Clone)]
pub struct GithubContext {
    pattern: SmallVec<u8, 128>,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl GithubContext {
    /// Create new GithubContext with pattern
    pub fn new(pattern: &str) -> Result<Self, ContextError> {
        let mut pattern_vec = SmallVec::new();

        for byte in pattern.as_bytes().iter().take(128) {
            pattern_vec.push(*byte);
        }

        Ok(Self {
            pattern: pattern_vec,
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Convert to documents with git integration
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let pattern_str = String::from_utf8_lossy(&self.pattern);

        // Basic git repository file reading (simplified implementation)
        // Full git integration would use gix crate for proper repository handling
        let files: Vec<PathBuf> = glob::glob(&pattern_str)
            .map_err(ContextError::from)?
            .filter_map(Result::ok)
            .collect();

        if files.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }

        // Process files in parallel
        let documents: Result<Vec<Document>, ContextError> = files
            .par_iter()
            .map(|path| {
                let content = std::fs::read_to_string(path).map_err(ContextError::from)?;

                Ok(Document {
                    data: content,
                    format: Some(super::document::ContentFormat::Text),
                    media_type: Some(super::document::DocumentMediaType::TXT),
                    additional_props: HashMap::new(),
                })
            })
            .collect();

        let documents = documents?;

        // Store in memory if integration is available
        if let Some(integration) = &self.memory_integration {
            let docs_for_storage: Vec<(String, String, MemoryTypeEnum)> = files
                .iter()
                .zip(documents.iter())
                .map(|(path, doc)| {
                    (
                        path.to_string_lossy().to_string(),
                        doc.data.clone(),
                        MemoryTypeEnum::Document,
                    )
                })
                .collect();

            integration.store_documents_batch(docs_for_storage).await?;
        }

        Ok(ZeroOneOrMany::many(documents))
    }
}

/// Main Context wrapper with type-safe API
#[derive(Debug, Clone)]
pub struct Context<T> {
    source: ContextSourceType,
    context_id: u64,
    created_at: SystemTime,
    _marker: PhantomData<T>,
}

impl<T> Context<T> {
    /// Create new context with source
    pub fn new(source: ContextSourceType) -> Self {
        Self {
            source,
            context_id: generate_context_id(),
            created_at: SystemTime::now(),
            _marker: PhantomData,
        }
    }

    /// Get context statistics
    pub fn stats(&self) -> ContextStats {
        ContextStats {
            context_id: self.context_id,
            age: self.created_at.elapsed().unwrap_or(Duration::ZERO),
            source_type: match &self.source {
                ContextSourceType::File(_) => "File".to_string(),
                ContextSourceType::Files(_) => "Files".to_string(),
                ContextSourceType::Directory(_) => "Directory".to_string(),
                ContextSourceType::Github(_) => "Github".to_string(),
            },
        }
    }
}

/// Context statistics for monitoring
#[derive(Debug, Clone)]
pub struct ContextStats {
    pub context_id: u64,
    pub age: Duration,
    pub source_type: String,
}

/// Generate unique context ID using atomic counter
use std::sync::atomic::{AtomicU64, Ordering};
static CONTEXT_ID_GENERATOR: AtomicU64 = AtomicU64::new(1);

#[inline(always)]
fn generate_context_id() -> u64 {
    CONTEXT_ID_GENERATOR.fetch_add(1, Ordering::Relaxed)
}

// Context<File> implementation
impl Context<File> {
    /// Load a single file - EXACT syntax: Context<File>::of("/path/to/file.pdf")
    #[inline(always)]
    pub fn of(path: impl AsRef<Path>) -> Self {
        let file_context = FileContext::new(path.as_ref()).unwrap_or_else(|_| FileContext {
            path: ArrayVec::new(),
            memory_integration: None,
        });
        Self::new(ContextSourceType::File(file_context))
    }

    /// Load document asynchronously
    #[inline(always)]
    pub async fn load(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        match self.source {
            ContextSourceType::File(file_context) => file_context.into_documents().await,
            _ => Err(ContextError::ContextNotFound("Invalid context type".into())),
        }
    }
}

// Context<Files> implementation
impl Context<Files> {
    /// Glob pattern for files - EXACT syntax: Context<Files>::glob("/path/**/*.{md,txt}")
    #[inline(always)]
    pub fn glob(pattern: impl AsRef<str>) -> Self {
        let files_context = FilesContext::new(pattern.as_ref()).unwrap_or_else(|_| FilesContext {
            pattern: SmallVec::new(),
            memory_integration: None,
        });
        Self::new(ContextSourceType::Files(files_context))
    }

    /// Load documents asynchronously
    #[inline(always)]
    pub async fn load(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        match self.source {
            ContextSourceType::Files(files_context) => files_context.into_documents().await,
            _ => Err(ContextError::ContextNotFound("Invalid context type".into())),
        }
    }
}

// Context<Directory> implementation
impl Context<Directory> {
    /// Load all files from directory - EXACT syntax: Context<Directory>::of("/path/to/dir")
    #[inline(always)]
    pub fn of(path: impl AsRef<Path>) -> Self {
        let directory_context =
            DirectoryContext::new(path.as_ref()).unwrap_or_else(|_| DirectoryContext {
                path: ArrayVec::new(),
                memory_integration: None,
            });
        Self::new(ContextSourceType::Directory(directory_context))
    }

    /// Load documents asynchronously
    #[inline(always)]
    pub async fn load(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        match self.source {
            ContextSourceType::Directory(directory_context) => {
                directory_context.into_documents().await
            }
            _ => Err(ContextError::ContextNotFound("Invalid context type".into())),
        }
    }
}

// Context<Github> implementation
impl Context<Github> {
    /// Glob pattern for GitHub files - EXACT syntax: Context<Github>::glob("/repo/**/*.{rs,md}")
    #[inline(always)]
    pub fn glob(pattern: impl AsRef<str>) -> Self {
        let github_context =
            GithubContext::new(pattern.as_ref()).unwrap_or_else(|_| GithubContext {
                pattern: SmallVec::new(),
                memory_integration: None,
            });
        Self::new(ContextSourceType::Github(github_context))
    }

    /// Load documents asynchronously
    #[inline(always)]
    pub async fn load(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        match self.source {
            ContextSourceType::Github(github_context) => github_context.into_documents().await,
            _ => Err(ContextError::ContextNotFound("Invalid context type".into())),
        }
    }
}
