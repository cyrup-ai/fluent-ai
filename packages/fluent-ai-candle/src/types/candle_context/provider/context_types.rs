//! Context Data Structures with Zero Arc Usage
//!
//! Immutable context types with owned strings and atomic tracking.

use std::marker::PhantomData;
use std::path::Path;
use std::time::SystemTime;

use super::memory::MemoryIntegration;
use super::processor::StreamingContextProcessor;

/// Marker types for Context
pub struct File;
pub struct Files;
pub struct Directory;
pub struct Github;

/// Immutable file context with owned strings and atomic tracking
#[derive(Debug, Clone)]
pub struct ImmutableFileContext {
    /// File path as owned string
    pub path: String,
    /// Content hash for deduplication
    pub content_hash: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Last modified timestamp
    pub modified: SystemTime,
    /// Memory integration layer
    pub memory_integration: Option<MemoryIntegration>,
}

impl ImmutableFileContext {
    /// Create new file context
    pub fn new(path: String) -> Self {
        Self {
            path,
            content_hash: String::new(),
            size_bytes: 0,
            modified: SystemTime::now(),
            memory_integration: None,
        }
    }

    /// Set content hash
    pub fn with_content_hash(mut self, hash: String) -> Self {
        self.content_hash = hash;
        self
    }

    /// Set file size
    pub fn with_size(mut self, size_bytes: u64) -> Self {
        self.size_bytes = size_bytes;
        self
    }

    /// Set modified timestamp
    pub fn with_modified(mut self, modified: SystemTime) -> Self {
        self.modified = modified;
        self
    }

    /// Add memory integration
    pub fn with_memory_integration(mut self, integration: MemoryIntegration) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Get file extension
    pub fn extension(&self) -> Option<String> {
        Path::new(&self.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_string())
    }

    /// Get file name
    pub fn filename(&self) -> Option<String> {
        Path::new(&self.path)
            .file_name()
            .and_then(|name| name.to_str())
            .map(|s| s.to_string())
    }
}

/// Immutable files context with owned strings
#[derive(Debug, Clone)]
pub struct ImmutableFilesContext {
    /// File paths as owned strings
    pub paths: Vec<String>,
    /// Glob pattern as owned string
    pub pattern: String,
    /// Total files count
    pub total_files: usize,
    /// Memory integration layer
    pub memory_integration: Option<MemoryIntegration>,
}

impl ImmutableFilesContext {
    /// Create new files context
    pub fn new(pattern: String) -> Self {
        Self {
            paths: Vec::new(),
            pattern,
            total_files: 0,
            memory_integration: None,
        }
    }

    /// Add file path
    pub fn add_path(mut self, path: String) -> Self {
        self.paths.push(path);
        self.total_files = self.paths.len();
        self
    }

    /// Add multiple paths
    pub fn with_paths(mut self, paths: Vec<String>) -> Self {
        self.paths = paths;
        self.total_files = self.paths.len();
        self
    }

    /// Add memory integration
    pub fn with_memory_integration(mut self, integration: MemoryIntegration) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Get file count
    pub fn file_count(&self) -> usize {
        self.total_files
    }
}

/// Immutable directory context with owned strings
#[derive(Debug, Clone)]
pub struct ImmutableDirectoryContext {
    /// Directory path as owned string
    pub path: String,
    /// Recursive traversal flag
    pub recursive: bool,
    /// File extensions filter
    pub extensions: Vec<String>,
    /// Maximum depth for traversal
    pub max_depth: Option<usize>,
    /// Memory integration layer
    pub memory_integration: Option<MemoryIntegration>,
}

impl ImmutableDirectoryContext {
    /// Create new directory context
    pub fn new(path: String) -> Self {
        Self {
            path,
            recursive: true,
            extensions: Vec::new(),
            max_depth: None,
            memory_integration: None,
        }
    }

    /// Set recursive traversal
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Add file extension filter
    pub fn with_extension(mut self, extension: String) -> Self {
        self.extensions.push(extension);
        self
    }

    /// Set multiple extensions
    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.extensions = extensions;
        self
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Add memory integration
    pub fn with_memory_integration(mut self, integration: MemoryIntegration) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Check if extension is allowed
    pub fn allows_extension(&self, extension: &str) -> bool {
        self.extensions.is_empty() || self.extensions.contains(&extension.to_string())
    }
}

/// Immutable GitHub context with owned strings
#[derive(Debug, Clone)]
pub struct ImmutableGithubContext {
    /// Repository URL as owned string
    pub repository_url: String,
    /// Branch name as owned string
    pub branch: String,
    /// File pattern as owned string
    pub pattern: String,
    /// Authentication token (if needed)
    pub auth_token: Option<String>,
    /// Memory integration layer
    pub memory_integration: Option<MemoryIntegration>,
}

impl ImmutableGithubContext {
    /// Create new GitHub context
    pub fn new(repository_url: String, pattern: String) -> Self {
        Self {
            repository_url,
            branch: "main".to_string(),
            pattern,
            auth_token: None,
            memory_integration: None,
        }
    }

    /// Set branch
    pub fn with_branch(mut self, branch: String) -> Self {
        self.branch = branch;
        self
    }

    /// Set authentication token
    pub fn with_auth_token(mut self, token: String) -> Self {
        self.auth_token = Some(token);
        self
    }

    /// Add memory integration
    pub fn with_memory_integration(mut self, integration: MemoryIntegration) -> Self {
        self.memory_integration = Some(integration);
        self
    }

    /// Check if has authentication
    pub fn has_auth(&self) -> bool {
        self.auth_token.is_some()
    }
}

/// Context source types with immutable implementations
#[derive(Debug, Clone)]
pub enum ContextSourceType {
    File(ImmutableFileContext),
    Files(ImmutableFilesContext),
    Directory(ImmutableDirectoryContext),
    Github(ImmutableGithubContext),
}

impl ContextSourceType {
    /// Get source type name
    pub fn type_name(&self) -> &'static str {
        match self {
            ContextSourceType::File(_) => "file",
            ContextSourceType::Files(_) => "files",
            ContextSourceType::Directory(_) => "directory",
            ContextSourceType::Github(_) => "github",
        }
    }

    /// Get memory integration if present
    pub fn memory_integration(&self) -> Option<&MemoryIntegration> {
        match self {
            ContextSourceType::File(ctx) => ctx.memory_integration.as_ref(),
            ContextSourceType::Files(ctx) => ctx.memory_integration.as_ref(),
            ContextSourceType::Directory(ctx) => ctx.memory_integration.as_ref(),
            ContextSourceType::Github(ctx) => ctx.memory_integration.as_ref(),
        }
    }
}

/// Context wrapper with zero Arc usage
pub struct Context<T> {
    pub(crate) source: ContextSourceType,
    pub(crate) processor: StreamingContextProcessor,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> Context<T> {
    /// Create new context with streaming processor
    #[inline]
    pub fn new(source: ContextSourceType) -> Self {
        let processor_id = uuid::Uuid::new_v4().to_string();
        let processor = StreamingContextProcessor::new(processor_id);
        Self {
            source,
            processor,
            _marker: PhantomData,
        }
    }

    /// Create context with event streaming
    #[inline]
    pub fn with_streaming(source: ContextSourceType) -> (Self, fluent_ai_async::AsyncStream<super::events::ContextEvent>) {
        let processor_id = uuid::Uuid::new_v4().to_string();
        let (processor, stream) = StreamingContextProcessor::with_streaming(processor_id);
        let context = Self {
            source,
            processor,
            _marker: PhantomData,
        };
        (context, stream)
    }

    /// Get context source type
    pub fn source(&self) -> &ContextSourceType {
        &self.source
    }

    /// Get processor statistics
    pub fn statistics(&self) -> super::processor::ContextProcessorStatistics {
        self.processor.get_statistics()
    }
}

/// Backward compatibility aliases (deprecated)
#[deprecated(note = "Use ImmutableFileContext instead")]
pub type FileContext = ImmutableFileContext;

#[deprecated(note = "Use ImmutableFilesContext instead")]
pub type FilesContext = ImmutableFilesContext;

#[deprecated(note = "Use ImmutableDirectoryContext instead")]
pub type DirectoryContext = ImmutableDirectoryContext;

#[deprecated(note = "Use ImmutableGithubContext instead")]
pub type GithubContext = ImmutableGithubContext;