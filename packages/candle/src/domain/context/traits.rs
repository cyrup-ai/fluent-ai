//! CandleContext trait definition - mirrors fluent-ai-domain Context trait exactly
//!
//! This trait provides the core context interface for Candle-backed context implementations,
//! enabling trait composition, testability, and 'room to move' architecture benefits.

use fluent_ai_async::AsyncStream;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// CandleContext trait - mirrors fluent-ai-domain::Context exactly with Candle prefix
/// 
/// This trait enables:
/// - Trait composition for flexible context architectures  
/// - Testability with mock implementations
/// - Zero-cost abstractions via static dispatch
/// - 'Room to move' for future context providers
pub trait CandleContext: Send + Sync + 'static {
    /// Load context content from the source
    ///
    /// # Returns
    /// AsyncStream containing context content chunks
    fn load_content(&self) -> AsyncStream<CandleContextChunk>;
    
    /// Get context metadata and information
    ///
    /// # Returns
    /// Context metadata for introspection
    fn get_metadata(&self) -> CandleContextMetadata;
    
    /// Refresh context content if it has changed
    ///
    /// # Returns
    /// AsyncStream indicating whether refresh was successful
    fn refresh(&self) -> AsyncStream<bool>;
    
    /// Get context capabilities and supported operations
    ///
    /// # Returns
    /// AsyncStream containing context capabilities
    fn get_capabilities(&self) -> AsyncStream<CandleContextCapabilities>;
}

/// Context content chunk for streaming large contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleContextChunk {
    /// Chunk content
    pub content: String,
    /// Content type (text, code, data, etc.)
    pub content_type: CandleContextType,
    /// Chunk metadata
    pub metadata: Option<HashMap<String, String>>,
    /// Whether this is the last chunk
    pub is_final: bool,
    /// Chunk sequence number
    pub sequence: u64,
}

impl CandleContextChunk {
    /// Create new context chunk
    pub fn new(content: impl Into<String>, content_type: CandleContextType) -> Self {
        Self {
            content: content.into(),
            content_type,
            metadata: None,
            is_final: false,
            sequence: 0,
        }
    }
    
    /// Mark chunk as final
    pub fn with_final(mut self) -> Self {
        self.is_final = true;
        self
    }
    
    /// Set chunk sequence number
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = sequence;
        self
    }
    
    /// Add metadata to chunk
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        if let Some(ref mut metadata) = self.metadata {
            metadata.insert(key.into(), value.into());
        }
        self
    }
}

/// Context content type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CandleContextType {
    /// Plain text content
    Text,
    /// Code content with language
    Code(String),
    /// Markdown content
    Markdown,
    /// JSON data
    Json,
    /// XML data
    Xml,
    /// Binary data (base64 encoded)
    Binary,
    /// Directory listing
    Directory,
    /// File metadata
    FileInfo,
    /// Unknown content type
    Unknown,
}

impl Default for CandleContextType {
    fn default() -> Self {
        CandleContextType::Text
    }
}

/// Context metadata for introspection and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleContextMetadata {
    /// Context name or identifier
    pub name: String,
    /// Context description
    pub description: String,
    /// Context source type (file, directory, url, etc.)
    pub source_type: CandleContextSource,
    /// Source location or path
    pub source_location: String,
    /// Content size in bytes (if known)
    pub content_size: Option<u64>,
    /// Last modified timestamp
    pub last_modified: Option<u64>,
    /// Context version or revision
    pub version: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CandleContextMetadata {
    /// Create new context metadata
    pub fn new(name: impl Into<String>, source_type: CandleContextSource, location: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            source_type,
            source_location: location.into(),
            content_size: None,
            last_modified: None,
            version: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
    
    /// Set content size
    pub fn with_size(mut self, size: u64) -> Self {
        self.content_size = Some(size);
        self
    }
    
    /// Set last modified timestamp
    pub fn with_last_modified(mut self, timestamp: u64) -> Self {
        self.last_modified = Some(timestamp);
        self
    }
    
    /// Add metadata field
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Context source type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CandleContextSource {
    /// File system file
    File,
    /// File system directory
    Directory,
    /// Git repository
    Git,
    /// HTTP URL
    Http,
    /// Database query
    Database,
    /// In-memory content
    Memory,
    /// Environment variables
    Environment,
    /// Command output
    Command,
    /// Unknown source
    Unknown,
}

/// Context capabilities and supported operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleContextCapabilities {
    /// Whether context supports real-time updates
    pub supports_realtime_updates: bool,
    /// Whether context can be refreshed
    pub supports_refresh: bool,
    /// Whether context supports streaming
    pub supports_streaming: bool,
    /// Whether context supports search/filtering
    pub supports_search: bool,
    /// Maximum content size supported
    pub max_content_size: Option<u64>,
    /// Supported content types
    pub supported_content_types: Vec<CandleContextType>,
    /// Required permissions or capabilities
    pub required_permissions: Vec<String>,
}

impl Default for CandleContextCapabilities {
    fn default() -> Self {
        Self {
            supports_realtime_updates: false,
            supports_refresh: true,
            supports_streaming: true,
            supports_search: false,
            max_content_size: Some(10 * 1024 * 1024), // 10MB
            supported_content_types: vec![CandleContextType::Text],
            required_permissions: Vec::new(),
        }
    }
}

/// File context implementation
#[derive(Debug, Clone)]
pub struct CandleFileContext {
    /// File path
    pub path: PathBuf,
    /// Context metadata
    pub metadata: CandleContextMetadata,
}

impl CandleFileContext {
    /// Create new file context
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path_buf = path.into();
        let metadata = CandleContextMetadata::new(
            path_buf.file_name().unwrap_or_default().to_string_lossy(),
            CandleContextSource::File,
            path_buf.to_string_lossy(),
        );
        
        Self {
            path: path_buf,
            metadata,
        }
    }
    
    /// Create file context with description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.metadata = self.metadata.with_description(description);
        self
    }
}

impl CandleContext for CandleFileContext {
    fn load_content(&self) -> AsyncStream<CandleContextChunk> {
        let path = self.path.clone();
        
        AsyncStream::with_channel(move |sender| {
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    let content_type = match extension {
                        "rs" | "py" | "js" | "ts" | "go" | "java" | "cpp" | "c" | "h" => {
                            CandleContextType::Code(extension.to_string())
                        }
                        "md" => CandleContextType::Markdown,
                        "json" => CandleContextType::Json,
                        "xml" | "html" => CandleContextType::Xml,
                        _ => CandleContextType::Text,
                    };
                    
                    let chunk = CandleContextChunk::new(content, content_type)
                        .with_final()
                        .with_sequence(0)
                        .with_metadata("file_path", path.to_string_lossy());
                        
                    let _ = sender.send(chunk);
                }
                Err(_) => {
                    let error_chunk = CandleContextChunk::new(
                        format!("Error reading file: {}", path.display()),
                        CandleContextType::Text,
                    ).with_final();
                    let _ = sender.send(error_chunk);
                }
            }
        })
    }
    
    fn get_metadata(&self) -> CandleContextMetadata {
        self.metadata.clone()
    }
    
    fn refresh(&self) -> AsyncStream<bool> {
        AsyncStream::with_channel(move |sender| {
            // For files, always return true since file system access is always "fresh"
            let _ = sender.send(true);
        })
    }
    
    fn get_capabilities(&self) -> AsyncStream<CandleContextCapabilities> {
        AsyncStream::with_channel(move |sender| {
            let capabilities = CandleContextCapabilities {
                supports_realtime_updates: false, // Would need file watching
                supports_refresh: true,
                supports_streaming: true,
                supports_search: false,
                max_content_size: Some(100 * 1024 * 1024), // 100MB
                supported_content_types: vec![
                    CandleContextType::Text,
                    CandleContextType::Code("*".to_string()),
                    CandleContextType::Markdown,
                    CandleContextType::Json,
                    CandleContextType::Xml,
                ],
                required_permissions: vec!["file_read".to_string()],
            };
            let _ = sender.send(capabilities);
        })
    }
}

/// Mock context implementation for testing
#[derive(Debug, Clone)]
pub struct MockCandleContext {
    /// Mock content to return
    pub mock_content: String,
    /// Context metadata
    pub metadata: CandleContextMetadata,
    /// Simulate loading delay
    pub load_delay_ms: u64,
    /// Simulate error for testing
    pub simulate_error: bool,
}

impl MockCandleContext {
    /// Create new mock context
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        let metadata = CandleContextMetadata::new(
            name,
            CandleContextSource::Memory,
            "mock://context",
        );
        
        Self {
            mock_content: content.into(),
            metadata,
            load_delay_ms: 0,
            simulate_error: false,
        }
    }
    
    /// Set loading delay
    pub fn with_delay(mut self, delay_ms: u64) -> Self {
        self.load_delay_ms = delay_ms;
        self
    }
    
    /// Enable error simulation
    pub fn with_error(mut self) -> Self {
        self.simulate_error = true;
        self
    }
}

impl CandleContext for MockCandleContext {
    fn load_content(&self) -> AsyncStream<CandleContextChunk> {
        let content = self.mock_content.clone();
        let delay = self.load_delay_ms;
        let error = self.simulate_error;
        
        AsyncStream::with_channel(move |sender| {
            // Simulate loading delay
            if delay > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }
            
            if error {
                let error_chunk = CandleContextChunk::new(
                    "Simulated context error",
                    CandleContextType::Text,
                ).with_final();
                let _ = sender.send(error_chunk);
            } else {
                let chunk = CandleContextChunk::new(content, CandleContextType::Text)
                    .with_final()
                    .with_sequence(0)
                    .with_metadata("source", "mock");
                let _ = sender.send(chunk);
            }
        })
    }
    
    fn get_metadata(&self) -> CandleContextMetadata {
        self.metadata.clone()
    }
    
    fn refresh(&self) -> AsyncStream<bool> {
        let simulate_error = self.simulate_error;
        AsyncStream::with_channel(move |sender| {
            let _ = sender.send(!simulate_error);
        })
    }
    
    fn get_capabilities(&self) -> AsyncStream<CandleContextCapabilities> {
        AsyncStream::with_channel(move |sender| {
            let capabilities = CandleContextCapabilities {
                supports_realtime_updates: true,
                supports_refresh: true,
                supports_streaming: true,
                supports_search: true,
                max_content_size: Some(1024 * 1024), // 1MB
                supported_content_types: vec![CandleContextType::Text],
                required_permissions: vec!["testing".to_string()],
            };
            let _ = sender.send(capabilities);
        })
    }
}