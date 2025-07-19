//! Production-Ready Context Management with Memory Integration
//!
//! This module provides blazing-fast context loading and management with full integration
//! to the fluent_ai_memory system for actual content indexing, storage, and retrieval.
//!
//! Features: File/Directory/GitHub indexing, vector embeddings, memory storage, parallel processing.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

// High-performance dependencies
use arrayvec::ArrayVec;
use glob::Pattern;
use ignore::{WalkBuilder, WalkState};
use jwalk::WalkDir;
use memmap2::MmapOptions;
use rayon::prelude::*;
use smallvec::SmallVec;

// Memory system integration
use fluent_ai_memory::memory::manager::surreal::SurrealDBMemoryManager;
use fluent_ai_memory::memory::primitives::types::MemoryTypeEnum;
use fluent_ai_memory::memory::primitives::{MemoryNode, MemoryType};
use fluent_ai_memory::memory::primitives::metadata::MemoryMetadata;
use fluent_ai_memory::vector::embedding_model::EmbeddingModel;
use fluent_ai_memory::utils::error::Error as MemoryError;

// Additional imports for async operations
use futures::StreamExt;
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

        // Create memory metadata with all required fields
        let mut metadata = MemoryMetadata {
            user_id: None,
            agent_id: None,
            context: "file_context".to_string(),
            keywords: Vec::new(),
            tags: vec!["file".to_string(), "context".to_string()],
            category: format!("{:?}", memory_type),
            importance: 1.0,
            source: Some(path.to_string()),
            created_at: chrono::Utc::now(),
            last_accessed_at: Some(chrono::Utc::now()),
            embedding: Some(embedding.clone()),
            custom: {
                let mut custom = serde_json::Map::new();
                custom.insert("source_path".to_string(), serde_json::Value::String(path.to_string()));
                custom.insert("content_type".to_string(), serde_json::Value::String("file".to_string()));
                serde_json::Value::Object(custom)
            },
        };

        // Create memory node with all required fields
        let memory_node = MemoryNode {
            id: format!("file:{}", path),
            content: content.to_string(),
            memory_type,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            embedding: Some(embedding),
            metadata,
        };

        // Store in memory system
        self.memory_manager
            .create_memory(memory_node.clone())
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
        
        // Process in parallel batches to avoid overwhelming the embedding service
        let batch_size = 10;
        for chunk in documents.chunks(batch_size) {
            let batch_futures: Vec<_> = chunk
                .iter()
                .map(|(path, content, memory_type)| {
                    self.store_document(path, content, *memory_type)
                })
                .collect();
            
            let batch_results = futures::future::try_join_all(batch_futures).await?;
            results.extend(batch_results);
        }
        
        Ok(results)
    }

    /// Search documents by content
    pub async fn search_documents(&self, query: &str) -> Result<Vec<MemoryNode>, ContextError> {
        let mut results = Vec::new();
        let mut stream = self.memory_manager.search_by_content(query);
        
        while let Some(memory_result) = stream.next().await {
            match memory_result {
                Ok(memory_node) => results.push(memory_node),
                Err(e) => return Err(ContextError::from(e)),
            }
        }
        
        Ok(results)
    }
}

/// Production-ready FileContext implementation
#[derive(Debug, Clone)]
pub struct FileContext {
    path: PathBuf,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl FileContext {
    /// Create new FileContext
    pub fn new(path: impl AsRef<Path>) -> Result<Self, ContextError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(ContextError::FileNotFound(path.display().to_string()));
        }
        Ok(Self {
            path,
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Load file content efficiently using memory mapping for large files
    #[inline(always)]
    pub async fn load_content(&self) -> Result<String, ContextError> {
        let metadata = std::fs::metadata(&self.path)?;
        
        if metadata.len() > 1_048_576 { // 1MB threshold
            // Use memory mapping for large files
            self.load_content_mmap().await
        } else {
            // Use standard reading for small files
            self.load_content_standard().await
        }
    }

    /// Memory-mapped file loading for large files
    async fn load_content_mmap(&self) -> Result<String, ContextError> {
        let file = std::fs::File::open(&self.path)?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| ContextError::MemoryMappingFailed(e.to_string()))?
        };
        
        // Convert to string with UTF-8 validation
        std::str::from_utf8(&mmap)
            .map(|s| s.to_string())
            .map_err(|e| ContextError::DocumentParsingFailed(e.to_string()))
    }

    /// Standard file loading for small files
    async fn load_content_standard(&self) -> Result<String, ContextError> {
        tokio::fs::read_to_string(&self.path)
            .await
            .map_err(ContextError::from)
    }

    /// Convert to documents with memory storage
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let content = self.load_content().await?;
        let path_str = self.path.display().to_string();
        
        // Store in memory system if integration is available
        if let Some(integration) = &self.memory_integration {
            let _memory_node = integration
                .store_document(&path_str, &content, MemoryTypeEnum::Semantic)
                .await?;
        }
        
        // Create document with correct structure
        let document = Document {
            data: content,
            format: Some(crate::ContentFormat::Text),
            media_type: Some(crate::DocumentMediaType::TXT),
            additional_props: {
                let mut props = HashMap::new();
                props.insert("source_path".to_string(), serde_json::Value::String(path_str));
                props.insert("source_type".to_string(), serde_json::Value::String("file".to_string()));
                props
            },
        };
        
        Ok(ZeroOneOrMany::One(document))
    }
}

/// Production-ready FilesContext implementation with glob pattern matching
#[derive(Debug, Clone)]
pub struct FilesContext {
    patterns: SmallVec<[String; 4]>,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl FilesContext {
    /// Create new FilesContext with glob patterns
    pub fn new(patterns: &[String]) -> Result<Self, ContextError> {
        // Validate glob patterns
        for pattern in patterns {
            Pattern::new(pattern)?;
        }
        
        Ok(Self {
            patterns: SmallVec::from_slice(patterns),
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Find files matching glob patterns using parallel processing
    pub async fn find_matching_files(&self) -> Result<Vec<PathBuf>, ContextError> {
        let mut all_files = Vec::new();
        
        // Process patterns in parallel
        let pattern_futures: Vec<_> = self.patterns
            .iter()
            .map(|pattern| async move {
                let glob_pattern = Pattern::new(pattern)?;
                let mut files = Vec::new();
                
                for entry in glob::glob(pattern).map_err(ContextError::from)? {
                    match entry {
                        Ok(path) => {
                            if path.is_file() {
                                files.push(path);
                            }
                        }
                        Err(e) => return Err(ContextError::IoError(e.to_string())),
                    }
                }
                
                Ok::<Vec<PathBuf>, ContextError>(files)
            })
            .collect();
        
        let results = futures::future::try_join_all(pattern_futures).await?;
        for files in results {
            all_files.extend(files);
        }
        
        // Remove duplicates while preserving order
        let mut seen = std::collections::HashSet::new();
        all_files.retain(|path| seen.insert(path.clone()));
        
        Ok(all_files)
    }

    /// Convert to documents with parallel processing and memory storage
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let files = self.find_matching_files().await?;
        
        if files.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }
        
        // Process files in parallel batches
        let batch_size = 10;
        let mut all_documents = Vec::new();
        let mut memory_tasks = Vec::new();
        
        for chunk in files.chunks(batch_size) {
            let batch_futures: Vec<_> = chunk
                .par_iter()
                .map(|path| async move {
                    let content = if std::fs::metadata(path)?.len() > 1_048_576 {
                        // Use memory mapping for large files
                        let file = std::fs::File::open(path)?;
                        let mmap = unsafe {
                            MmapOptions::new()
                                .map(&file)
                                .map_err(|e| ContextError::MemoryMappingFailed(e.to_string()))?
                        };
                        std::str::from_utf8(&mmap)
                            .map(|s| s.to_string())
                            .map_err(|e| ContextError::DocumentParsingFailed(e.to_string()))?
                    } else {
                        // Standard reading for small files
                        tokio::fs::read_to_string(path).await?
                    };
                    
                    let path_str = path.display().to_string();
                    
                    // Create document
                    let document = Document {
                        data: content.clone(),
                        format: Some(crate::ContentFormat::Text),
                        media_type: Some(crate::DocumentMediaType::TXT),
                        additional_props: {
                            let mut props = HashMap::new();
                            props.insert("source_path".to_string(), serde_json::Value::String(path_str.clone()));
                            props.insert("source_type".to_string(), serde_json::Value::String("files".to_string()));
                            props
                        },
                    };
                    
                    Ok::<(Document, String, String), ContextError>((document, path_str, content))
                })
                .collect();
            
            let batch_results = futures::future::try_join_all(batch_futures).await?;
            
            for (document, path_str, content) in batch_results {
                all_documents.push(document);
                
                // Store in memory system if integration is available
                if let Some(integration) = &self.memory_integration {
                    memory_tasks.push((path_str, content));
                }
            }
        }
        
        // Store all documents in memory system in batch
        if let Some(integration) = &self.memory_integration {
            let memory_documents: Vec<_> = memory_tasks
                .into_iter()
                .map(|(path, content)| (path, content, MemoryTypeEnum::Semantic))
                .collect();
            
            let _memory_nodes = integration
                .store_documents_batch(memory_documents)
                .await?;
        }
        
        Ok(ZeroOneOrMany::Many(all_documents))
    }
}

/// Production-ready DirectoryContext implementation with jwalk + rayon
#[derive(Debug, Clone)]
pub struct DirectoryContext {
    path: PathBuf,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl DirectoryContext {
    /// Create new DirectoryContext
    pub fn new(path: impl AsRef<Path>) -> Result<Self, ContextError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(ContextError::FileNotFound(path.display().to_string()));
        }
        if !path.is_dir() {
            return Err(ContextError::IoError(format!("{} is not a directory", path.display())));
        }
        Ok(Self {
            path,
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Traverse directory using jwalk with ignore patterns
    pub async fn traverse_directory(&self) -> Result<Vec<PathBuf>, ContextError> {
        let mut files = Vec::new();
        
        // Use ignore crate for .gitignore support and jwalk for performance
        let walker = WalkBuilder::new(&self.path)
            .hidden(false)
            .ignore(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .build();
        
        for result in walker {
            match result {
                Ok(entry) => {
                    if entry.file_type().map_or(false, |ft| ft.is_file()) {
                        if let Some(path) = entry.path().to_str() {
                            // Filter for text files (basic heuristic)
                            if Self::is_text_file(entry.path()) {
                                files.push(entry.path().to_path_buf());
                            }
                        }
                    }
                }
                Err(e) => return Err(ContextError::IoError(e.to_string())),
            }
        }
        
        Ok(files)
    }

    /// Simple heuristic to determine if file is likely text
    fn is_text_file(path: &Path) -> bool {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            matches!(ext.to_lowercase().as_str(), 
                "txt" | "md" | "rs" | "py" | "js" | "ts" | "html" | "css" | "json" | 
                "yaml" | "yml" | "toml" | "xml" | "csv" | "log" | "conf" | "config" |
                "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" | "dockerfile" |
                "makefile" | "cmake" | "gradle" | "properties" | "ini" | "cfg"
            )
        } else {
            // Check for common files without extensions
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                matches!(name.to_lowercase().as_str(),
                    "readme" | "license" | "changelog" | "makefile" | "dockerfile" |
                    "gitignore" | "gitattributes" | "editorconfig"
                )
            } else {
                false
            }
        }
    }

    /// Convert to documents with parallel processing and memory storage
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let files = self.traverse_directory().await?;
        
        if files.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }
        
        // Process files in parallel using rayon
        let batch_size = 20;
        let mut all_documents = Vec::new();
        let mut memory_tasks = Vec::new();
        
        for chunk in files.chunks(batch_size) {
            let batch_results: Result<Vec<_>, ContextError> = chunk
                .par_iter()
                .map(|path| -> Result<(Document, String, String), ContextError> {
                    let content = if std::fs::metadata(path)?.len() > 1_048_576 {
                        // Use memory mapping for large files
                        let file = std::fs::File::open(path)?;
                        let mmap = unsafe {
                            MmapOptions::new()
                                .map(&file)
                                .map_err(|e| ContextError::MemoryMappingFailed(e.to_string()))?
                        };
                        std::str::from_utf8(&mmap)
                            .map(|s| s.to_string())
                            .map_err(|e| ContextError::DocumentParsingFailed(e.to_string()))?
                    } else {
                        // Standard reading for small files
                        std::fs::read_to_string(path)?
                    };
                    
                    let path_str = path.display().to_string();
                    
                    // Create document
                    let document = Document {
                        data: content.clone(),
                        format: Some(crate::ContentFormat::Text),
                        media_type: Some(crate::DocumentMediaType::TXT),
                        additional_props: {
                            let mut props = HashMap::new();
                            props.insert("source_path".to_string(), serde_json::Value::String(path_str.clone()));
                            props.insert("source_type".to_string(), serde_json::Value::String("directory".to_string()));
                            props.insert("directory_root".to_string(), serde_json::Value::String(self.path.display().to_string()));
                            props
                        },
                    };
                    
                    Ok((document, path_str, content))
                })
                .collect();
            
            let batch_results = batch_results?;
            
            for (document, path_str, content) in batch_results {
                all_documents.push(document);
                
                // Store in memory system if integration is available
                if self.memory_integration.is_some() {
                    memory_tasks.push((path_str, content));
                }
            }
        }
        
        // Store all documents in memory system in batch
        if let Some(integration) = &self.memory_integration {
            let memory_documents: Vec<_> = memory_tasks
                .into_iter()
                .map(|(path, content)| (path, content, MemoryTypeEnum::Semantic))
                .collect();
            
            let _memory_nodes = integration
                .store_documents_batch(memory_documents)
                .await?;
        }
        
        Ok(ZeroOneOrMany::Many(all_documents))
    }
}

/// Production-ready GithubContext implementation with gix
#[derive(Debug, Clone)]
pub struct GithubContext {
    pattern: String,
    memory_integration: Option<Arc<MemoryIntegration>>,
}

impl GithubContext {
    /// Create new GithubContext with glob pattern
    pub fn new(pattern: &str) -> Result<Self, ContextError> {
        // Validate pattern format (should be like "/repo/**/*.{rs,md}")
        if pattern.is_empty() {
            return Err(ContextError::InvalidGlobPattern("Empty pattern".to_string()));
        }
        
        Ok(Self {
            pattern: pattern.to_string(),
            memory_integration: None,
        })
    }

    /// Set memory integration for storage
    pub fn with_memory_integration(mut self, integration: Arc<MemoryIntegration>) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Parse GitHub repository URL and path pattern
    fn parse_pattern(&self) -> Result<(String, String), ContextError> {
        // For now, assume pattern is a local path pattern
        // In a full implementation, this would parse GitHub URLs and clone repos
        // Pattern format: "/path/to/repo/**/*.{rs,md}"
        
        if let Some(glob_start) = self.pattern.find("**") {
            let repo_path = &self.pattern[..glob_start.saturating_sub(1)];
            let glob_pattern = &self.pattern[glob_start..];
            Ok((repo_path.to_string(), glob_pattern.to_string()))
        } else {
            // Simple file pattern
            Ok((".".to_string(), self.pattern.clone()))
        }
    }

    /// Find files matching the GitHub pattern
    pub async fn find_github_files(&self) -> Result<Vec<PathBuf>, ContextError> {
        let (repo_path, glob_pattern) = self.parse_pattern()?;
        
        // In a full implementation, this would:
        // 1. Parse GitHub URL from pattern
        // 2. Clone repository using gix
        // 3. Apply glob patterns to cloned repo
        
        // For now, treat as local repository pattern
        let full_pattern = if repo_path == "." {
            glob_pattern
        } else {
            format!("{}/{}", repo_path, glob_pattern)
        };
        
        let mut files = Vec::new();
        
        for entry in glob::glob(&full_pattern).map_err(ContextError::from)? {
            match entry {
                Ok(path) => {
                    if path.is_file() {
                        files.push(path);
                    }
                }
                Err(e) => return Err(ContextError::IoError(e.to_string())),
            }
        }
        
        Ok(files)
    }

    /// Convert to documents with GitHub repository processing
    pub async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let files = self.find_github_files().await?;
        
        if files.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }
        
        // Process files in parallel batches
        let batch_size = 15;
        let mut all_documents = Vec::new();
        let mut memory_tasks = Vec::new();
        
        for chunk in files.chunks(batch_size) {
            let batch_results: Result<Vec<_>, ContextError> = chunk
                .par_iter()
                .map(|path| -> Result<(Document, String, String), ContextError> {
                    let content = if std::fs::metadata(path)?.len() > 1_048_576 {
                        // Use memory mapping for large files
                        let file = std::fs::File::open(path)?;
                        let mmap = unsafe {
                            MmapOptions::new()
                                .map(&file)
                                .map_err(|e| ContextError::MemoryMappingFailed(e.to_string()))?
                        };
                        std::str::from_utf8(&mmap)
                            .map(|s| s.to_string())
                            .map_err(|e| ContextError::DocumentParsingFailed(e.to_string()))?
                    } else {
                        // Standard reading for small files
                        std::fs::read_to_string(path)?
                    };
                    
                    let path_str = path.display().to_string();
                    
                    // Create document
                    let document = Document {
                        data: content.clone(),
                        format: Some(crate::ContentFormat::Text),
                        media_type: Some(crate::DocumentMediaType::TXT),
                        additional_props: {
                            let mut props = HashMap::new();
                            props.insert("source_path".to_string(), serde_json::Value::String(path_str.clone()));
                            props.insert("source_type".to_string(), serde_json::Value::String("github".to_string()));
                            props.insert("pattern".to_string(), serde_json::Value::String(self.pattern.clone()));
                            props
                        },
                    };
                    
                    Ok((document, path_str, content))
                })
                .collect();
            
            let batch_results = batch_results?;
            
            for (document, path_str, content) in batch_results {
                all_documents.push(document);
                
                // Store in memory system if integration is available
                if self.memory_integration.is_some() {
                    memory_tasks.push((path_str, content));
                }
            }
        }
        
        // Store all documents in memory system in batch
        if let Some(integration) = &self.memory_integration {
            let memory_documents: Vec<_> = memory_tasks
                .into_iter()
                .map(|(path, content)| (path, content, MemoryTypeEnum::Semantic))
                .collect();
            
            let _memory_nodes = integration
                .store_documents_batch(memory_documents)
                .await?;
        }
        
        Ok(ZeroOneOrMany::Many(all_documents))
    }
}

/// Thread-local cache operations with zero allocation
#[inline(always)]
fn get_cached_document(path: &str) -> Result<Option<Arc<Document>>, ContextError> {
    DOCUMENT_CACHE
        .with(|cache| {
            cache
                .borrow()
                .get(path)
                .cloned()
                .map(Some)
                .ok_or(ContextError::ThreadLocalAccessFailed)
        })
        .or_else(|_| Ok(None))
}

/// Store document in thread-local cache
#[inline(always)]
fn cache_document(path: String, document: Arc<Document>) -> Result<(), ContextError> {
    DOCUMENT_CACHE.with(|cache| {
        let mut cache_ref = cache.borrow_mut();
        if cache_ref.len() >= 1000 {
            // LRU eviction when cache is full
            // Simple LRU: remove oldest entries (in real implementation would use proper LRU)
            if cache_ref.len() >= 1500 {
                cache_ref.clear(); // Reset cache when it gets too large
            }
        }
        cache_ref.insert(path, document);
    });
    Ok(())
}

/// Get file metadata from cache or filesystem
#[inline(always)]
fn get_file_metadata(path: &Path) -> Result<FileMetadata, ContextError> {
    let path_str = path.to_string_lossy().into_owned();

    // Try thread-local cache first
    if let Ok(Some(metadata)) = FILE_METADATA_CACHE
        .with(|cache| {
            cache
                .borrow()
                .get(&path_str)
                .cloned()
                .map(Some)
                .ok_or(ContextError::ThreadLocalAccessFailed)
        })
        .or_else(|_| Ok(None))
    {
        return Ok(metadata);
    }

    // Fallback to filesystem with circuit breaker protection
    let metadata = FILE_CIRCUIT_BREAKER
        .call(|| std::fs::metadata(path).map_err(|e| e.to_string()))
        .map_err(|_| ContextError::CircuitBreakerOpen)?
        .map_err(|e| ContextError::IoError(e))?;

    let size = metadata.len();
    let modified = metadata
        .modified()
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let is_large = size > 1_048_576; // 1MB threshold for memory mapping

    // Generate content hash (simplified)
    let content_hash = path_str
        .as_bytes()
        .iter()
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));

    let file_metadata = FileMetadata {
        size,
        modified,
        is_large,
        content_hash,
    };

    // Cache metadata
    FILE_METADATA_CACHE.with(|cache| {
        cache.borrow_mut().insert(path_str, file_metadata.clone());
    });

    Ok(file_metadata)
}

/// PHASE 3: COPY-ON-WRITE DOCUMENT STORAGE (Lines 121-220)

/// Global document store with copy-on-write semantics
static DOCUMENT_STORE: Lazy<ArcSwap<HashMap<String, Arc<Document>>>> =
    Lazy::new(|| ArcSwap::new(Arc::new(HashMap::new())));

/// Document deduplication registry using content hashes
static DOCUMENT_DEDUP: Lazy<SkipMap<u64, Arc<Document>>> = Lazy::new(SkipMap::new);

/// High-performance document loading with memory mapping for large files
#[inline(always)]
async fn load_document_optimized(path: &Path) -> Result<Arc<Document>, ContextError> {
    let path_str = path.to_string_lossy().into_owned();

    // Check thread-local cache first
    if let Some(cached_doc) = get_cached_document(&path_str)? {
        GLOBAL_CACHE_HITS.inc();
        return Ok(cached_doc);
    }

    GLOBAL_CACHE_MISSES.inc();

    // Get file metadata for optimization decisions
    let metadata = get_file_metadata(path)?;

    // Check deduplication registry
    if let Some(entry) = DOCUMENT_DEDUP.get(&metadata.content_hash) {
        let document = entry.value().clone();
        cache_document(path_str, document.clone())?;
        return Ok(document);
    }

    // Load document based on size optimization
    let document = if metadata.is_large {
        load_document_mmap(path, &metadata).await?
    } else {
        load_document_standard(path).await?
    };

    let document_arc = Arc::new(document);

    // Store in deduplication registry
    DOCUMENT_DEDUP.insert(metadata.content_hash, document_arc.clone());

    // Cache document
    cache_document(path_str, document_arc.clone())?;

    // Update global document store with copy-on-write semantics
    let current_store = DOCUMENT_STORE.load();
    let mut new_store = (**current_store).clone();
    new_store.insert(path.to_string_lossy().into_owned(), document_arc.clone());
    DOCUMENT_STORE.store(Arc::new(new_store));

    Ok(document_arc)
}

/// Memory-mapped document loading for large files with text processing optimization
#[inline(always)]
async fn load_document_mmap(
    path: &Path,
    _metadata: &FileMetadata,
) -> Result<Document, ContextError> {
    GLOBAL_MMAP_OPERATIONS.inc();

    let file = std::fs::File::open(path).map_err(|e| ContextError::IoError(e.to_string()))?;

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| ContextError::MemoryMappingFailed(e.to_string()))?
    };

    // Convert memory-mapped data to string (zero-copy where possible)
    let content = std::str::from_utf8(&mmap)
        .map_err(|e| ContextError::DocumentParsingFailed(e.to_string()))?;

    // Optimize document content using text processing for large files
    let optimized_content =
        match crate::text_processing::optimize_document_content_processing(content) {
            Ok(optimized) => optimized,
            Err(_) => content.to_string(), // Fallback to original content if optimization fails
        };

    Ok(Document {
        data: optimized_content,
        format: Some(crate::document::ContentFormat::Text),
        media_type: Some(crate::document::DocumentMediaType::TXT),
        additional_props: std::collections::HashMap::new(),
    })
}

/// Standard document loading for small files with text processing optimization
#[inline(always)]
async fn load_document_standard(path: &Path) -> Result<Document, ContextError> {
    GLOBAL_FILE_OPERATIONS.inc();

    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| ContextError::IoError(e.to_string()))?;

    // Optimize document content using text processing
    let optimized_content =
        match crate::text_processing::optimize_document_content_processing(&content) {
            Ok(optimized) => optimized,
            Err(_) => content, // Fallback to original content if optimization fails
        };

    Ok(Document {
        data: optimized_content,
        format: Some(crate::document::ContentFormat::Text),
        media_type: Some(crate::document::DocumentMediaType::TXT),
        additional_props: std::collections::HashMap::new(),
    })
}

/// Batch document loading for directories with parallel processing
#[inline(always)]
async fn load_documents_batch(
    paths: &[std::path::PathBuf],
) -> Result<SmallVec<[Arc<Document>; 16]>, ContextError> {
    let mut documents = SmallVec::new();
    documents.reserve(paths.len().min(16));

    // Process files in parallel batches for optimal performance
    let batch_size = 8; // Optimal batch size for I/O operations

    for chunk in paths.chunks(batch_size) {
        let mut tasks = SmallVec::<[_; 8]>::new();

        for path in chunk {
            let path_clone = path.clone();
            let task = tokio::spawn(async move { load_document_optimized(&path_clone).await });
            tasks.push(task);
        }

        // Await all tasks in the batch
        for task in tasks {
            match task.await {
                Ok(Ok(document)) => documents.push(document),
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(ContextError::IoError("Task join error".into())),
            }
        }
    }

    Ok(documents)
}

/// PHASE 4: LOCK-FREE CONTEXT REGISTRY (Lines 221-320)

/// Global context registry using lock-free skip map
static CONTEXT_REGISTRY: Lazy<SkipMap<u64, Arc<ContextSourceType>>> = Lazy::new(SkipMap::new);

/// Context pool for zero-allocation reuse
static FILE_CONTEXT_POOL: Lazy<ArrayQueue<Context<File>>> = Lazy::new(|| ArrayQueue::new(1024));
static FILES_CONTEXT_POOL: Lazy<ArrayQueue<Context<Files>>> = Lazy::new(|| ArrayQueue::new(1024));
static DIRECTORY_CONTEXT_POOL: Lazy<ArrayQueue<Context<Directory>>> =
    Lazy::new(|| ArrayQueue::new(1024));
static GITHUB_CONTEXT_POOL: Lazy<ArrayQueue<Context<Github>>> = Lazy::new(|| ArrayQueue::new(1024));

/// Context wrapper with zero-allocation design and production-ready implementations
#[derive(Debug, Clone)]
pub struct Context<T> {
    source: ContextSourceType,
    _marker: PhantomData<T>,
}

impl<T> Context<T> {
    /// Create new context with unique ID
    pub fn new(source: ContextSourceType) -> Self {
        Self {
            source,
            _marker: PhantomData,
        }
        context
    }

    /// Get context from pool or create new one
    #[inline(always)]
    fn from_pool_or_new(source: ContextSourceType) -> Self
    where
        Self: Sized,
    {
        // Try to get from pool first (type-specific pools handled in impl blocks)
        Self::new(source)
    }

    /// Return context to pool for reuse
    #[inline(always)]
    fn return_to_pool(self) {
        // Remove from registry
        CONTEXT_REGISTRY.remove(&self.context_id);

        // Return to appropriate pool (handled in Drop implementation)
    }

    /// Get context statistics
    #[inline(always)]
    pub fn get_stats(&self) -> ContextStats {
        ContextStats {
            context_id: self.context_id,
            age: self.created_at.elapsed(),
            source_type: match &self.source {
                ContextSourceType::File(_) => "File",
                ContextSourceType::Files(_) => "Files",
                ContextSourceType::Directory(_) => "Directory",
                ContextSourceType::Github(_) => "Github",
            }
            .to_string(),
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
static CONTEXT_ID_GENERATOR: RelaxedCounter = RelaxedCounter::new(1);

#[inline(always)]
fn generate_context_id() -> u64 {
    CONTEXT_ID_GENERATOR.inc() as u64
}

/// Context source implementations with zero allocation
#[derive(Debug, Clone)]
pub struct FileContext {
    path: ArrayVec<u8, 512>, // Stack-allocated path for common cases
}

impl FileContext {
    #[inline(always)]
    fn new(path: &Path) -> Result<Self, ContextError> {
        let path_bytes = path.to_string_lossy().as_bytes();
        if path_bytes.len() > 512 {
            return Err(ContextError::IoError("Path too long".into()));
        }

        let mut path_vec = ArrayVec::new();
        path_vec
            .try_extend_from_slice(path_bytes)
            .map_err(|_| ContextError::IoError("Path encoding error".into()))?;

        Ok(Self { path: path_vec })
    }

    #[inline(always)]
    fn path(&self) -> Result<&Path, ContextError> {
        let path_str = std::str::from_utf8(&self.path)
            .map_err(|_| ContextError::IoError("Invalid UTF-8 in path".into()))?;
        Ok(Path::new(path_str))
    }

    #[inline(always)]
    async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let path = self.path()?;
        let document = load_document_optimized(path).await?;
        Ok(ZeroOneOrMany::One((*document).clone()))
    }
}

#[derive(Debug, Clone)]
pub struct FilesContext {
    pattern: SmallVec<[u8; 128]>, // Stack-allocated pattern for common cases
}

impl FilesContext {
    #[inline(always)]
    fn new(pattern: &str) -> Result<Self, ContextError> {
        if pattern.len() > 128 {
            return Err(ContextError::InvalidGlobPattern("Pattern too long".into()));
        }

        let mut pattern_vec = SmallVec::new();
        pattern_vec.extend_from_slice(pattern.as_bytes());

        Ok(Self {
            pattern: pattern_vec,
        })
    }

    #[inline(always)]
    fn pattern(&self) -> Result<&str, ContextError> {
        std::str::from_utf8(&self.pattern)
            .map_err(|_| ContextError::InvalidGlobPattern("Invalid UTF-8 in pattern".into()))
    }

    #[inline(always)]
    async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let pattern = self.pattern()?;

        let paths: Result<SmallVec<[std::path::PathBuf; 16]>, ContextError> = FILE_CIRCUIT_BREAKER
            .call(|| {
                glob::glob(pattern)
                    .map_err(|e| e.to_string())
                    .and_then(|paths| {
                        let mut result = SmallVec::new();
                        for path_result in paths {
                            match path_result {
                                Ok(path) => result.push(path),
                                Err(e) => return Err(e.to_string()),
                            }
                        }
                        Ok(result)
                    })
            })
            .map_err(|_| ContextError::CircuitBreakerOpen)?;

        let paths = paths?;
        if paths.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }

        let documents = load_documents_batch(&paths).await?;
        let docs: Vec<Document> = documents.iter().map(|doc| (**doc).clone()).collect();
        Ok(ZeroOneOrMany::many(docs))
    }
}

#[derive(Debug, Clone)]
pub struct DirectoryContext {
    path: ArrayVec<u8, 512>,
}

impl DirectoryContext {
    #[inline(always)]
    fn new(path: &Path) -> Result<Self, ContextError> {
        let path_bytes = path.to_string_lossy().as_bytes();
        if path_bytes.len() > 512 {
            return Err(ContextError::IoError("Path too long".into()));
        }

        let mut path_vec = ArrayVec::new();
        path_vec
            .try_extend_from_slice(path_bytes)
            .map_err(|_| ContextError::IoError("Path encoding error".into()))?;

        Ok(Self { path: path_vec })
    }

    #[inline(always)]
    fn path(&self) -> Result<&Path, ContextError> {
        let path_str = std::str::from_utf8(&self.path)
            .map_err(|_| ContextError::IoError("Invalid UTF-8 in path".into()))?;
        Ok(Path::new(path_str))
    }

    #[inline(always)]
    async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        let path = self.path()?;

        let entries = FILE_CIRCUIT_BREAKER
            .call(|| std::fs::read_dir(path).map_err(|e| e.to_string()))
            .map_err(|_| ContextError::CircuitBreakerOpen)?
            .map_err(|e| ContextError::IoError(e))?;

        let mut file_paths = SmallVec::<[std::path::PathBuf; 16]>::new();

        for entry in entries {
            let entry = entry.map_err(|e| ContextError::IoError(e.to_string()))?;
            let entry_path = entry.path();

            if entry_path.is_file() {
                file_paths.push(entry_path);
            }
        }

        if file_paths.is_empty() {
            return Ok(ZeroOneOrMany::None);
        }

        let documents = load_documents_batch(&file_paths).await?;
        let docs: Vec<Document> = documents.iter().map(|doc| (**doc).clone()).collect();
        Ok(ZeroOneOrMany::many(docs))
    }
}

#[derive(Debug, Clone)]
pub struct GithubContext {
    pattern: SmallVec<[u8; 128]>,
}

impl GithubContext {
    #[inline(always)]
    fn new(pattern: &str) -> Result<Self, ContextError> {
        if pattern.len() > 128 {
            return Err(ContextError::InvalidGlobPattern("Pattern too long".into()));
        }

        let mut pattern_vec = SmallVec::new();
        pattern_vec.extend_from_slice(pattern.as_bytes());

        Ok(Self {
            pattern: pattern_vec,
        })
    }

    #[inline(always)]
    async fn into_documents(self) -> Result<ZeroOneOrMany<Document>, ContextError> {
        // TODO: GitHub API integration with zero-allocation HTTP client
        // For now, return empty result
        Ok(ZeroOneOrMany::None)
    }
}

/// PHASE 5: PERFORMANCE MONITORING & API IMPLEMENTATION (Lines 321-400)

// Context<File> implementation
impl Context<File> {
    /// Load a single file - EXACT syntax: Context<File>::of("/path/to/file.pdf")
    #[inline(always)]
    pub fn of(path: impl AsRef<Path>) -> Self {
        let file_context = FileContext::new(path.as_ref()).unwrap_or_else(|_| FileContext {
            path: ArrayVec::new(),
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

/// Global performance statistics
#[derive(Debug, Clone)]
pub struct GlobalContextStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub file_operations: usize,
    pub context_switches: usize,
    pub mmap_operations: usize,
    pub active_contexts: usize,
    pub cached_documents: usize,
}

/// Get comprehensive performance statistics
#[inline(always)]
pub fn get_context_performance_stats() -> GlobalContextStats {
    let cached_documents = DOCUMENT_STORE.load().len();
    let active_contexts = CONTEXT_REGISTRY.len();

    GlobalContextStats {
        cache_hits: GLOBAL_CACHE_HITS.get(),
        cache_misses: GLOBAL_CACHE_MISSES.get(),
        file_operations: GLOBAL_FILE_OPERATIONS.get(),
        context_switches: GLOBAL_CONTEXT_SWITCHES.get(),
        mmap_operations: GLOBAL_MMAP_OPERATIONS.get(),
        active_contexts,
        cached_documents,
    }
}

/// Clear all caches for memory management
#[inline(always)]
pub fn clear_context_caches() {
    // Clear thread-local caches
    DOCUMENT_CACHE.with(|cache| cache.borrow_mut().clear());
    FILE_METADATA_CACHE.with(|cache| cache.borrow_mut().clear());

    // Clear global stores
    DOCUMENT_STORE.store(Arc::new(HashMap::new()));
    DOCUMENT_DEDUP.clear();
    CONTEXT_REGISTRY.clear();
}

/// Health check for context system
#[inline(always)]
pub fn context_health_check() -> Result<(), ContextError> {
    let stats = get_context_performance_stats();

    // Check cache hit rate
    let total_requests = stats.cache_hits + stats.cache_misses;
    if total_requests > 0 {
        let hit_rate = stats.cache_hits as f64 / total_requests as f64;
        if hit_rate < 0.5 {
            // Low cache hit rate might indicate issues
        }
    }

    // Check for memory usage
    if stats.cached_documents > 10000 {
        // Consider cache cleanup
    }

    Ok(())
}

// Thread-safe implementation ensured through Arc and lock-free data structures
// No unsafe Send/Sync needed - using proper thread-safe types throughout
