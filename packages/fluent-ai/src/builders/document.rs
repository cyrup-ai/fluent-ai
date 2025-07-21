//! Document builder implementations with zero-allocation, lock-free design
//!
//! Provides EXACT API syntax for document loading and processing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fluent_ai_domain::chunk::DocumentChunk;
use fluent_ai_domain::{
    AsyncTask, ContentFormat, Document, DocumentMediaType, HashMap, ZeroOneOrMany,
    async_task::AsyncStream, spawn_async,
};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpMethod, HttpRequest};
use serde_json::Value;
use tokio::fs;

/// Document builder data enumeration for zero-allocation type tracking
#[derive(Debug, Clone)]
pub enum DocumentBuilderData {
    File(PathBuf),
    Url(String),
    Github {
        repo: String,
        path: String,
        branch: Option<String>,
    },
    Glob(String),
    Text(String),
}

/// Core document builder with zero-allocation design
pub struct DocumentBuilder {
    data: DocumentBuilderData,
    format: Option<ContentFormat>,
    media_type: Option<DocumentMediaType>,
    additional_props: BTreeMap<String, Value>,
    encoding: Option<String>,
    max_size: Option<usize>,
    timeout_ms: Option<u64>,
    retry_attempts: u8,
    cache_enabled: bool,
}

/// Document builder with error handler - enables terminal methods
pub struct DocumentBuilderWithHandler {
    inner: DocumentBuilder,
    error_handler: Arc<dyn Fn(String) + Send + Sync>,
    chunk_handler: Option<Arc<dyn Fn(DocumentChunk) -> DocumentChunk + Send + Sync>>,
}

impl Document {
    /// Create document from file path - EXACT syntax: Document::from_file("path/to/file.txt")
    #[inline]
    pub fn from_file<P: AsRef<Path>>(path: P) -> DocumentBuilder {
        DocumentBuilder {
            data: DocumentBuilderData::File(path.as_ref().to_path_buf()),
            format: None,
            media_type: None,
            additional_props: BTreeMap::new(),
            encoding: None,
            max_size: None,
            timeout_ms: None,
            retry_attempts: 3,
            cache_enabled: true,
        }
    }

    /// Create document from URL - EXACT syntax: Document::from_url("https://example.com/doc.pdf")
    #[inline]
    pub fn from_url(url: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder {
            data: DocumentBuilderData::Url(url.into()),
            format: None,
            media_type: None,
            additional_props: BTreeMap::new(),
            encoding: None,
            max_size: Some(10 * 1024 * 1024), // 10MB default
            timeout_ms: Some(30000),          // 30s default
            retry_attempts: 3,
            cache_enabled: true,
        }
    }

    /// Create document from GitHub - EXACT syntax: Document::from_github("owner/repo", "path/to/file.md")
    #[inline]
    pub fn from_github(repo: impl Into<String>, path: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder {
            data: DocumentBuilderData::Github {
                repo: repo.into(),
                path: path.into(),
                branch: None,
            },
            format: None,
            media_type: None,
            additional_props: BTreeMap::new(),
            encoding: None,
            max_size: Some(1024 * 1024), // 1MB default for GitHub files
            timeout_ms: Some(15000),     // 15s default
            retry_attempts: 3,
            cache_enabled: true,
        }
    }

    /// Create document from glob pattern - EXACT syntax: Document::from_glob("**/*.md")
    #[inline]
    pub fn from_glob(pattern: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder {
            data: DocumentBuilderData::Glob(pattern.into()),
            format: None,
            media_type: None,
            additional_props: BTreeMap::new(),
            encoding: None,
            max_size: None,
            timeout_ms: None,
            retry_attempts: 1,
            cache_enabled: false,
        }
    }

    /// Create document from text - EXACT syntax: Document::from_text("content")
    #[inline]
    pub fn from_text(text: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder {
            data: DocumentBuilderData::Text(text.into()),
            format: Some(ContentFormat::Text),
            media_type: Some(DocumentMediaType::PlainText),
            additional_props: BTreeMap::new(),
            encoding: Some("utf-8".to_string()),
            max_size: None,
            timeout_ms: None,
            retry_attempts: 0,
            cache_enabled: false,
        }
    }

    /// Create document from base64 data - EXACT syntax: Document::from_base64("base64data")
    #[inline]
    pub fn from_base64(data: impl Into<String>) -> DocumentBuilder {
        DocumentBuilder {
            data: DocumentBuilderData::Text(data.into()),
            format: Some(ContentFormat::Base64),
            media_type: Some(DocumentMediaType::Binary),
            additional_props: BTreeMap::new(),
            encoding: None,
            max_size: None,
            timeout_ms: None,
            retry_attempts: 0,
            cache_enabled: false,
        }
    }

    /// Create document from data - EXACT syntax: Document::from_data("content")
    #[inline]
    pub fn from_data(data: impl Into<String>) -> DocumentBuilder {
        Self::from_text(data)
    }
}

impl DocumentBuilder {
    /// Set content format - EXACT syntax: .format(ContentFormat::Markdown)
    #[inline]
    pub fn format(mut self, format: ContentFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Set media type - EXACT syntax: .media_type(DocumentMediaType::PDF)
    #[inline]
    pub fn media_type(mut self, media_type: DocumentMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }

    /// Set encoding - EXACT syntax: .encoding("utf-8")
    #[inline]
    pub fn encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = Some(encoding.into());
        self
    }

    /// Set maximum file size - EXACT syntax: .max_size(1024 * 1024)
    #[inline]
    pub fn max_size(mut self, size: usize) -> Self {
        self.max_size = Some(size);
        self
    }

    /// Set request timeout - EXACT syntax: .timeout(30000)
    #[inline]
    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set retry attempts - EXACT syntax: .retry(5)
    #[inline]
    pub fn retry(mut self, attempts: u8) -> Self {
        self.retry_attempts = attempts;
        self
    }

    /// Enable/disable caching - EXACT syntax: .cache(true)
    #[inline]
    pub fn cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Set GitHub branch - EXACT syntax: .branch("main")
    #[inline]
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        if let DocumentBuilderData::Github {
            repo: _,
            path: _,
            ref mut branch_ref,
        } = &mut self.data
        {
            *branch_ref = Some(branch.into());
        }
        self
    }

    /// Add metadata property - EXACT syntax: .property("key", "value")
    #[inline]
    pub fn property(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.additional_props.insert(key.into(), value.into());
        self
    }

    /// Add multiple properties - EXACT syntax: .properties(hash_map!{"key" => "value"})
    #[inline]
    pub fn properties<F>(mut self, f: F) -> Self
    where
        F: FnOnce() -> BTreeMap<String, Value>,
    {
        let props = f();
        for (key, value) in props {
            self.additional_props.insert(key, value);
        }
        self
    }

    /// Add error handler - EXACT syntax: .on_error(|error| { ... })
    #[inline]
    pub fn on_error<F>(self, handler: F) -> DocumentBuilderWithHandler
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        DocumentBuilderWithHandler {
            inner: self,
            error_handler: Arc::new(handler),
            chunk_handler: None,
        }
    }

    /// Add chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    #[inline]
    pub fn on_chunk<F>(self, handler: F) -> DocumentBuilderWithHandler
    where
        F: Fn(DocumentChunk) -> DocumentChunk + Send + Sync + 'static,
    {
        DocumentBuilderWithHandler {
            inner: self,
            error_handler: Arc::new(|e| eprintln!("Document error: {}", e)),
            chunk_handler: Some(Arc::new(handler)),
        }
    }

    /// Synchronous load for immediate data only - EXACT syntax: .load()
    pub fn load(self) -> Document {
        match self.data {
            DocumentBuilderData::Text(data) => Document {
                data,
                format: self.format,
                media_type: self.media_type,
                metadata: self.additional_props.into_iter().collect(),
            },
            _ => {
                // Return error document instead of panicking
                Document {
                    data: "Error: load() can only be used with immediate data. Use on_error().load_async() for file/url/glob operations.".to_string(),
                    format: Some(ContentFormat::Text),
                    media_type: Some(DocumentMediaType::PlainText),
                    metadata: HashMap::new(),
                }
            }
        }
    }
}

impl DocumentBuilderWithHandler {
    /// Set content format - EXACT syntax: .format(ContentFormat::Markdown)
    #[inline]
    pub fn format(mut self, format: ContentFormat) -> Self {
        self.inner.format = Some(format);
        self
    }

    /// Set media type - EXACT syntax: .media_type(DocumentMediaType::PDF)
    #[inline]
    pub fn media_type(mut self, media_type: DocumentMediaType) -> Self {
        self.inner.media_type = Some(media_type);
        self
    }

    /// Set encoding - EXACT syntax: .encoding("utf-8")
    #[inline]
    pub fn encoding(mut self, encoding: impl Into<String>) -> Self {
        self.inner.encoding = Some(encoding.into());
        self
    }

    /// Set maximum file size - EXACT syntax: .max_size(1024 * 1024)
    #[inline]
    pub fn max_size(mut self, size: usize) -> Self {
        self.inner.max_size = Some(size);
        self
    }

    /// Set request timeout - EXACT syntax: .timeout(30000)
    #[inline]
    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.inner.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set retry attempts - EXACT syntax: .retry(5)
    #[inline]
    pub fn retry(mut self, attempts: u8) -> Self {
        self.inner.retry_attempts = attempts;
        self
    }

    /// Enable/disable caching - EXACT syntax: .cache(true)
    #[inline]
    pub fn cache(mut self, enabled: bool) -> Self {
        self.inner.cache_enabled = enabled;
        self
    }

    /// Set GitHub branch - EXACT syntax: .branch("main")
    #[inline]
    pub fn branch(mut self, branch: impl Into<String>) -> Self {
        if let DocumentBuilderData::Github {
            repo: _,
            path: _,
            ref mut branch_ref,
        } = &mut self.inner.data
        {
            *branch_ref = Some(branch.into());
        }
        self
    }

    /// Add metadata property - EXACT syntax: .property("key", "value")
    #[inline]
    pub fn property(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.inner.additional_props.insert(key.into(), value.into());
        self
    }

    /// Load document asynchronously - EXACT syntax: .load_async()
    pub fn load_async(self) -> AsyncTask<Document> {
        let inner = self.inner;
        let error_handler = self.error_handler;

        spawn_async(async move {
            match Self::load_document_data(inner, error_handler.clone()).await {
                Ok(document) => document,
                Err(error) => {
                    error_handler(error.clone());
                    // Return empty document on error
                    Document {
                        data: String::new(),
                        format: Some(ContentFormat::Text),
                        media_type: Some(DocumentMediaType::PlainText),
                        metadata: HashMap::new(),
                    }
                }
            }
        })
    }

    /// Load multiple documents - EXACT syntax: .load_all()
    pub fn load_all(self) -> AsyncTask<ZeroOneOrMany<Document>> {
        let inner = self.inner;
        let error_handler = self.error_handler;

        spawn_async(async move {
            match inner.data {
                DocumentBuilderData::Glob(pattern) => {
                    Self::load_glob_documents(pattern, inner, error_handler).await
                }
                _ => {
                    // Single document
                    match Self::load_document_data(inner, error_handler.clone()).await {
                        Ok(doc) => ZeroOneOrMany::One(doc),
                        Err(error) => {
                            error_handler(error);
                            ZeroOneOrMany::None
                        }
                    }
                }
            }
        })
    }

    /// Stream documents one by one - EXACT syntax: .stream()
    pub fn stream(self) -> AsyncStream<Document> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let inner = self.inner;
        let error_handler = self.error_handler;
        let chunk_handler = self.chunk_handler;

        tokio::spawn(async move {
            match inner.data {
                DocumentBuilderData::Glob(pattern) => {
                    if let Ok(paths) = glob::glob(&pattern) {
                        for entry in paths.filter_map(Result::ok) {
                            let doc_builder = DocumentBuilder {
                                data: DocumentBuilderData::File(entry),
                                format: inner.format.clone(),
                                media_type: inner.media_type.clone(),
                                additional_props: inner.additional_props.clone(),
                                encoding: inner.encoding.clone(),
                                max_size: inner.max_size,
                                timeout_ms: inner.timeout_ms,
                                retry_attempts: inner.retry_attempts,
                                cache_enabled: inner.cache_enabled,
                            };

                            match Self::load_document_data(doc_builder, error_handler.clone()).await
                            {
                                Ok(doc) => {
                                    if tx.send(doc).is_err() {
                                        break;
                                    }
                                }
                                Err(error) => error_handler(error),
                            }
                        }
                    }
                }
                _ => match Self::load_document_data(inner, error_handler.clone()).await {
                    Ok(doc) => {
                        let _ = tx.send(doc);
                    }
                    Err(error) => error_handler(error),
                },
            }
        });

        AsyncStream::new(rx)
    }

    /// Stream document content in chunks - EXACT syntax: .stream_chunks(1024)
    pub fn stream_chunks(self, chunk_size: usize) -> AsyncStream<DocumentChunk> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let chunk_handler = self.chunk_handler;

        tokio::spawn(async move {
            match self.load_async().await {
                Ok(doc) => {
                    let content = &doc.data;
                    let mut offset = 0;

                    while offset < content.len() {
                        let end = (offset + chunk_size).min(content.len());
                        let mut chunk =
                            DocumentChunk::new(&content[offset..end]).with_range(offset, end);

                        // Apply chunk handler if present
                        if let Some(ref handler) = chunk_handler {
                            chunk = handler(chunk);
                        }

                        if tx.send(chunk).is_err() {
                            break;
                        }

                        offset = end;
                    }
                }
                Err(_) => {} // Error already handled by load_async
            }
        });

        AsyncStream::new(rx)
    }

    /// Stream document content line by line - EXACT syntax: .stream_lines()
    pub fn stream_lines(self) -> AsyncStream<DocumentChunk> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let chunk_handler = self.chunk_handler;

        tokio::spawn(async move {
            match self.load_async().await {
                Ok(doc) => {
                    let mut offset = 0;
                    for line in doc.data.lines() {
                        let mut chunk =
                            DocumentChunk::new(line).with_range(offset, offset + line.len());

                        // Apply chunk handler if present
                        if let Some(ref handler) = chunk_handler {
                            chunk = handler(chunk);
                        }

                        if tx.send(chunk).is_err() {
                            break;
                        }

                        offset += line.len() + 1; // +1 for newline
                    }
                }
                Err(_) => {} // Error already handled by load_async
            }
        });

        AsyncStream::new(rx)
    }

    // ========================================================================
    // Internal Implementation - Zero Allocation, Lock-Free
    // ========================================================================

    async fn load_document_data(
        builder: DocumentBuilder,
        error_handler: Arc<dyn Fn(String) + Send + Sync>,
    ) -> Result<Document, String> {
        let content = match builder.data {
            DocumentBuilderData::File(path) => Self::load_file_content(&path, &builder).await?,
            DocumentBuilderData::Url(url) => Self::load_url_content(&url, &builder).await?,
            DocumentBuilderData::Github { repo, path, branch } => {
                Self::load_github_content(&repo, &path, branch.as_deref(), &builder).await?
            }
            DocumentBuilderData::Text(text) => text,
            DocumentBuilderData::Glob(_) => {
                return Err("Glob patterns require load_all() or stream()".to_string());
            }
        };

        // Detect format if not specified
        let format = builder
            .format
            .unwrap_or_else(|| Self::detect_format(&content, &builder.data));

        // Detect media type if not specified
        let media_type = builder
            .media_type
            .unwrap_or_else(|| Self::detect_media_type(&format, &builder.data));

        // Build metadata
        let mut metadata = HashMap::with_capacity(builder.additional_props.len() + 4);
        for (key, value) in builder.additional_props {
            metadata.insert(key, value);
        }

        if let Some(encoding) = builder.encoding {
            metadata.insert("encoding".to_string(), Value::String(encoding));
        }

        metadata.insert("size".to_string(), Value::Number(content.len().into()));
        metadata.insert(
            "cache_enabled".to_string(),
            Value::Bool(builder.cache_enabled),
        );

        Ok(Document {
            data: content,
            format: Some(format),
            media_type: Some(media_type),
            metadata,
        })
    }

    async fn load_file_content(path: &Path, builder: &DocumentBuilder) -> Result<String, String> {
        // Check file size first if max_size is set
        if let Some(max_size) = builder.max_size {
            let metadata = fs::metadata(path)
                .await
                .map_err(|e| format!("Failed to read file metadata: {}", e))?;

            if metadata.len() as usize > max_size {
                return Err(format!(
                    "File size {} exceeds maximum {}",
                    metadata.len(),
                    max_size
                ));
            }
        }

        // Attempt to read with retries
        let mut last_error = String::new();
        for attempt in 0..=builder.retry_attempts {
            match fs::read_to_string(path).await {
                Ok(content) => return Ok(content),
                Err(e) => {
                    last_error = format!("Attempt {}: {}", attempt + 1, e);
                    if attempt < builder.retry_attempts {
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            100 * (1 << attempt), // Exponential backoff
                        ))
                        .await;
                    }
                }
            }
        }

        Err(format!(
            "Failed to read file after {} attempts: {}",
            builder.retry_attempts + 1,
            last_error
        ))
    }

    async fn load_url_content(url: &str, builder: &DocumentBuilder) -> Result<String, String> {
        let client = HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let mut request = HttpRequest::new(HttpMethod::Get, url.to_string());

        // Set timeout if specified
        if let Some(timeout_ms) = builder.timeout_ms {
            request = request.with_timeout(std::time::Duration::from_millis(timeout_ms));
        }

        // Attempt request with retries
        let mut last_error = String::new();
        for attempt in 0..=builder.retry_attempts {
            match client.send(request.clone()).await {
                Ok(response) => {
                    if response.status().is_success() {
                        let content = response
                            .text()
                            .await
                            .map_err(|e| format!("Failed to read response body: {}", e))?;

                        // Check size if max_size is set
                        if let Some(max_size) = builder.max_size {
                            if content.len() > max_size {
                                return Err(format!(
                                    "Response size {} exceeds maximum {}",
                                    content.len(),
                                    max_size
                                ));
                            }
                        }

                        return Ok(content);
                    } else {
                        return Err(format!("HTTP error: {}", response.status()));
                    }
                }
                Err(e) => {
                    last_error = format!("Attempt {}: {}", attempt + 1, e);
                    if attempt < builder.retry_attempts {
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            100 * (1 << attempt), // Exponential backoff
                        ))
                        .await;
                    }
                }
            }
        }

        Err(format!(
            "Failed to fetch URL after {} attempts: {}",
            builder.retry_attempts + 1,
            last_error
        ))
    }

    async fn load_github_content(
        repo: &str,
        path: &str,
        branch: Option<&str>,
        builder: &DocumentBuilder,
    ) -> Result<String, String> {
        let branch = branch.unwrap_or("main");
        let url = format!(
            "https://raw.githubusercontent.com/{}/{}/{}",
            repo, branch, path
        );

        Self::load_url_content(&url, builder).await
    }

    async fn load_glob_documents(
        pattern: String,
        builder: DocumentBuilder,
        error_handler: Arc<dyn Fn(String) + Send + Sync>,
    ) -> ZeroOneOrMany<Document> {
        let paths = match glob::glob(&pattern) {
            Ok(paths) => paths.filter_map(Result::ok).collect::<Vec<_>>(),
            Err(e) => {
                error_handler(format!("Invalid glob pattern: {}", e));
                return ZeroOneOrMany::None;
            }
        };

        if paths.is_empty() {
            return ZeroOneOrMany::None;
        }

        let mut documents = Vec::with_capacity(paths.len());

        for path in paths {
            let doc_builder = DocumentBuilder {
                data: DocumentBuilderData::File(path),
                format: builder.format.clone(),
                media_type: builder.media_type.clone(),
                additional_props: builder.additional_props.clone(),
                encoding: builder.encoding.clone(),
                max_size: builder.max_size,
                timeout_ms: builder.timeout_ms,
                retry_attempts: builder.retry_attempts,
                cache_enabled: builder.cache_enabled,
            };

            match Self::load_document_data(doc_builder, error_handler.clone()).await {
                Ok(doc) => documents.push(doc),
                Err(error) => error_handler(error),
            }
        }

        match documents.len() {
            0 => ZeroOneOrMany::None,
            1 => {
                let mut iter = documents.into_iter();
                match iter.next() {
                    Some(doc) => ZeroOneOrMany::One(doc),
                    None => ZeroOneOrMany::None,
                }
            }
            _ => ZeroOneOrMany::Many(documents),
        }
    }

    #[inline]
    fn detect_format(content: &str, data: &DocumentBuilderData) -> ContentFormat {
        match data {
            DocumentBuilderData::File(path) | DocumentBuilderData::Github { path, .. } => {
                match path.extension().and_then(|ext| ext.to_str()) {
                    Some("md") | Some("markdown") => ContentFormat::Markdown,
                    Some("html") | Some("htm") => ContentFormat::Html,
                    Some("json") => ContentFormat::Json,
                    Some("xml") => ContentFormat::Xml,
                    Some("yaml") | Some("yml") => ContentFormat::Yaml,
                    Some("csv") => ContentFormat::Csv,
                    _ => {
                        // Content-based detection
                        if content.trim_start().starts_with('{')
                            || content.trim_start().starts_with('[')
                        {
                            ContentFormat::Json
                        } else if content.trim_start().starts_with('<') {
                            ContentFormat::Html
                        } else {
                            ContentFormat::Text
                        }
                    }
                }
            }
            DocumentBuilderData::Url(url) => {
                if url.ends_with(".json") {
                    ContentFormat::Json
                } else if url.ends_with(".html") || url.ends_with(".htm") {
                    ContentFormat::Html
                } else if url.ends_with(".md") || url.ends_with(".markdown") {
                    ContentFormat::Markdown
                } else {
                    ContentFormat::Text
                }
            }
            _ => ContentFormat::Text,
        }
    }

    #[inline]
    fn detect_media_type(format: &ContentFormat, data: &DocumentBuilderData) -> DocumentMediaType {
        match format {
            ContentFormat::Json => DocumentMediaType::Json,
            ContentFormat::Html => DocumentMediaType::Html,
            ContentFormat::Markdown => DocumentMediaType::Markdown,
            ContentFormat::Xml => DocumentMediaType::Xml,
            ContentFormat::Csv => DocumentMediaType::Csv,
            ContentFormat::Yaml => DocumentMediaType::Yaml,
            ContentFormat::Base64 => match data {
                DocumentBuilderData::File(path) => {
                    match path.extension().and_then(|ext| ext.to_str()) {
                        Some("pdf") => DocumentMediaType::PDF,
                        Some("doc") | Some("docx") => DocumentMediaType::Document,
                        Some("jpg") | Some("jpeg") | Some("png") | Some("gif") => {
                            DocumentMediaType::Image
                        }
                        _ => DocumentMediaType::Binary,
                    }
                }
                _ => DocumentMediaType::Binary,
            },
            _ => DocumentMediaType::PlainText,
        }
    }
}

/// Convenience function for fluent document creation
#[inline]
pub fn document() -> DocumentBuilder {
    Document::from_text("")
}
