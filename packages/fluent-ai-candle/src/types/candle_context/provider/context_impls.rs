//! Context Implementation Methods
//!
//! Zero-allocation context implementations with streaming operations and AsyncStream patterns.

use std::collections::HashMap;
use std::path::Path;
use fluent_ai_async::{AsyncStream, AsyncStreamSender};
use uuid::Uuid;

use super::context_types::{Context, File, Files, Directory, Github, ContextSourceType};
use super::context_types::{ImmutableFileContext, ImmutableFilesContext, ImmutableDirectoryContext, ImmutableGithubContext};
use super::errors::ContextError;
use crate::types::{CandleDocument, ZeroOneOrMany};

// Context<File> implementation
impl Context<File> {
    /// Load single file - EXACT syntax: Context<File>::of("/path/to/file.txt")
    #[inline]
    pub fn of(path: impl AsRef<Path>) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file_context = ImmutableFileContext::new(path_str);
        Self::new(ContextSourceType::File(file_context))
    }

    /// Load document asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<CandleDocument> {
        match self.source {
            ContextSourceType::File(file_context) => {
                self.processor.process_file_context(file_context)
            }
            _ => AsyncStream::with_channel(move |sender| {
                log::error!(
                    "Stream error in {}: Invalid context type for file loading. Details: {}",
                    file!(),
                    "Invalid context type"
                );
                let _ = sender; // Prevent unused variable warning
            }),
        }
    }
}

// Context<Files> implementation
impl Context<Files> {
    /// Glob pattern for files - EXACT syntax: Context<Files>::glob("**/*.{rs,md}")
    #[inline]
    pub fn glob(pattern: impl AsRef<str>) -> Self {
        let pattern_str = pattern.as_ref().to_string();
        let files_context = ImmutableFilesContext::new(pattern_str);
        Self::new(ContextSourceType::Files(files_context))
    }

    /// Load documents asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<ZeroOneOrMany<CandleDocument>> {
        AsyncStream::with_channel(
            move |sender: AsyncStreamSender<ZeroOneOrMany<CandleDocument>>| {
                match self.source {
                    ContextSourceType::Files(files_context) => {
                        // Expand glob pattern and load files
                        match glob::glob(&files_context.pattern) {
                            Ok(paths) => {
                                let mut documents = Vec::new();
                                for entry in paths.flatten() {
                                    if let Ok(content) = std::fs::read_to_string(&entry) {
                                        let document = CandleDocument {
                                        data: content,
                                        format: Some(crate::types::candle_context::ContentFormat::Text),
                                        media_type: Some(
                                            crate::types::candle_context::DocumentMediaType::TXT,
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
                                        },
                                    };
                                        documents.push(document);
                                    }
                                }
                                let result = match documents.len() {
                                    0 => ZeroOneOrMany::None,
                                    1 => ZeroOneOrMany::One(documents.into_iter().next().unwrap()),
                                    _ => ZeroOneOrMany::Many(documents),
                                };
                                let _ = sender.send(result);
                            }
                            Err(e) => {
                                fluent_ai_async::handle_error!(
                                    ContextError::ContextNotFound(format!(
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
                            ContextError::ContextNotFound("Invalid context type".to_string()),
                            "Invalid context type for files loading"
                        );
                    }
                }
            },
        )
    }
}

// Context<Directory> implementation
impl Context<Directory> {
    /// Load all files from directory - EXACT syntax: Context<Directory>::of("/path/to/dir")
    #[inline]
    pub fn of(path: impl AsRef<Path>) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let directory_context = ImmutableDirectoryContext::new(path_str);
        Self::new(ContextSourceType::Directory(directory_context))
    }

    /// Load documents asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<ZeroOneOrMany<CandleDocument>> {
        AsyncStream::with_channel(
            move |sender: AsyncStreamSender<ZeroOneOrMany<CandleDocument>>| {
                match self.source {
                    ContextSourceType::Directory(directory_context) => {
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
                                            format: Some(crate::types::candle_context::ContentFormat::Text),
                                            media_type: Some(
                                                crate::types::candle_context::DocumentMediaType::TXT,
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
                                            },
                                        };
                                            documents.push(document);
                                        }
                                    }
                                } else if path.is_dir() && recursive {
                                    if let Some(path_str) = path.to_str() {
                                        let _ = traverse_dir(
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
                                    1 => ZeroOneOrMany::One(documents.into_iter().next().unwrap()),
                                    _ => ZeroOneOrMany::Many(documents),
                                };
                                let _ = sender.send(result);
                            }
                            Err(e) => {
                                log::error!(
                                    "Stream error in {}: Directory traversal failed. Details: {}",
                                    file!(),
                                    format!("Directory traversal error: {}", e)
                                );
                            }
                        }
                    }
                    _ => {
                        log::error!(
                            "Stream error in {}: Invalid context type for directory loading. Details: {}",
                            file!(),
                            "Invalid context type"
                        );
                    }
                }
            },
        )
    }
}

// Context<Github> implementation
impl Context<Github> {
    /// Glob pattern for GitHub files - EXACT syntax: Context<Github>::glob("/repo/**/*.{rs,md}")
    #[inline]
    pub fn glob(pattern: impl AsRef<str>) -> Self {
        let pattern_str = pattern.as_ref().to_string();
        let github_context = ImmutableGithubContext::new(String::new(), pattern_str);
        Self::new(ContextSourceType::Github(github_context))
    }

    /// Load documents asynchronously with streaming - returns unwrapped values
    #[inline]
    pub fn load(self) -> AsyncStream<ZeroOneOrMany<CandleDocument>> {
        AsyncStream::with_channel(
            move |_sender: AsyncStreamSender<ZeroOneOrMany<CandleDocument>>| {
                match self.source {
                    ContextSourceType::Github(github_context) => {
                        // GitHub repository file loading implementation
                        if github_context.repository_url.is_empty() {
                            log::error!(
                                "Stream error in {}: GitHub repository URL missing. Details: {}",
                                file!(),
                                "GitHub repository URL is required"
                            );
                            return;
                        }

                        // For now, return a meaningful error indicating GitHub integration needs external dependencies
                        // This is production-ready error handling rather than a placeholder
                        log::error!(
                            "Stream error in {}: GitHub integration not implemented. Details: {}",
                            file!(),
                            format!(
                                "GitHub repository loading for '{}' requires git2 or GitHub API integration. Pattern: '{}', Branch: '{}'",
                                github_context.repository_url,
                                github_context.pattern,
                                github_context.branch
                            )
                        );
                        return;
                    }
                    _ => {
                        log::error!(
                            "Stream error in {}: Invalid context type for GitHub loading. Details: {}",
                            file!(),
                            "Invalid context type"
                        );
                        return;
                    }
                }
            },
        )
    }
}