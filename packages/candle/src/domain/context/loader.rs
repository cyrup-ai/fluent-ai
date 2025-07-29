//! Core File Loading Interface and Implementation
//!
//! This module provides the core file loading interface and implementation
//! for pattern-based file loading and streaming operations.
//! Originally from loader.rs.

use std::fmt;
use std::path::PathBuf;

use fluent_ai_async::AsyncStream;
use fluent_ai_async::AsyncTask;

use cyrup_sugars::ZeroOneOrMany;

/// Trait defining the core file loading interface
pub trait Loader<T>: Send + Sync + fmt::Debug + Clone
where
    T: Send + Sync + fmt::Debug + Clone + 'static,
{
    /// Get the current file pattern
    fn pattern(&self) -> Option<&str>;

    /// Get the recursive setting
    fn recursive(&self) -> bool;

    /// Load all files matching the criteria
    fn load_all(&self) -> AsyncTask<ZeroOneOrMany<T>>
    where
        T: fluent_ai_async::NotResult;

    /// Stream files one by one
    fn stream_files(&self) -> AsyncStream<T>
    where
        T: fluent_ai_async::NotResult;

    /// Process each file with a processor function
    fn process_each<F, U>(&self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + fluent_ai_async::NotResult;

    /// Create new loader with pattern
    fn new(pattern: impl Into<String>) -> Self;

    /// Set recursive loading
    fn with_recursive(self, recursive: bool) -> Self;

    /// Apply filter to results
    fn with_filter<F>(self, filter: F) -> Self
    where
        F: Fn(&T) -> bool + Send + Sync + 'static;
}

/// Implementation of the Loader trait for PathBuf
pub struct LoaderImpl<T: Send + Sync + fmt::Debug + Clone + 'static> {
    #[allow(dead_code)] // TODO: Use for file glob pattern matching and directory traversal
    pattern: Option<String>,
    #[allow(dead_code)] // TODO: Use for recursive directory loading configuration
    recursive: bool,
    #[allow(dead_code)] // TODO: Use for custom file iteration and processing
    iterator: Option<Box<dyn Iterator<Item = T> + Send + Sync>>,
    #[allow(dead_code)] // TODO: Use for file filtering and selection criteria
    filter_fn: Option<Box<dyn Fn(&T) -> bool + Send + Sync>>}

// LoaderImpl implements NotResult since it contains no Result types

impl<T: Send + Sync + fmt::Debug + Clone + 'static> std::fmt::Debug for LoaderImpl<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoaderImpl")
            .field("pattern", &self.pattern)
            .field("recursive", &self.recursive)
            .field("iterator", &"<opaque>")
            .field("filter_fn", &"<function>")
            .finish()
    }
}

impl<T: Send + Sync + fmt::Debug + Clone + 'static> Clone for LoaderImpl<T> {
    fn clone(&self) -> Self {
        Self {
            pattern: self.pattern.clone(),
            recursive: self.recursive,
            iterator: None,  // Can't clone trait objects
            filter_fn: None, // Can't clone function pointers
        }
    }
}

impl Loader<PathBuf> for LoaderImpl<PathBuf> {
    fn pattern(&self) -> Option<&str> {
        self.pattern.as_deref()
    }

    fn recursive(&self) -> bool {
        self.recursive
    }

    fn load_all(&self) -> AsyncTask<ZeroOneOrMany<PathBuf>>
    where
        PathBuf: fluent_ai_async::NotResult,
    {
        let pattern = self.pattern.clone();
        fluent_ai_async::spawn_task(move || {
            let results: Vec<PathBuf> = match pattern {
                Some(p) => {
                    match glob::glob(&p) {
                        Ok(paths) => paths.filter_map(Result::ok).collect(),
                        Err(_) => Vec::new(), // Return empty on pattern error
                    }
                }
                None => Vec::new()};

            // Convert Vec<PathBuf> to ZeroOneOrMany<PathBuf> without unwrap
            match results.len() {
                0 => ZeroOneOrMany::None,
                1 => {
                    let mut iter = results.into_iter();
                    if let Some(item) = iter.next() {
                        ZeroOneOrMany::One(item)
                    } else {
                        ZeroOneOrMany::None
                    }
                }
                _ => ZeroOneOrMany::many(results)}
        })
    }

    fn stream_files(&self) -> AsyncStream<PathBuf>
    where
        PathBuf: fluent_ai_async::NotResult,
    {
        let pattern = self.pattern.clone();

        AsyncStream::with_channel(move |sender| {
            if let Some(p) = pattern {
                if let Ok(paths) = glob::glob(&p) {
                    for path in paths.filter_map(Result::ok) {
                        if sender.try_send(path).is_err() {
                            break;
                        }
                    }
                }
            }
        })
    }

    fn process_each<F, U>(&self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&PathBuf) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + fluent_ai_async::NotResult,
    {
        let load_task = self.load_all();
        fluent_ai_async::spawn_task(move || {
            let paths = load_task.collect();
            let results: Vec<U> = match paths {
                ZeroOneOrMany::None => Vec::new(),
                ZeroOneOrMany::One(path) => vec![processor(&path)],
                ZeroOneOrMany::Many(paths) => paths.iter().map(|p| processor(p)).collect()};

            // Convert Vec<U> to ZeroOneOrMany<U> without unwrap
            match results.len() {
                0 => ZeroOneOrMany::None,
                1 => {
                    let mut iter = results.into_iter();
                    if let Some(item) = iter.next() {
                        ZeroOneOrMany::One(item)
                    } else {
                        ZeroOneOrMany::None
                    }
                }
                _ => ZeroOneOrMany::many(results)}
        })
    }

    fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: Some(pattern.into()),
            recursive: false,
            iterator: None,
            filter_fn: None}
    }

    fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: Fn(&PathBuf) -> bool + Send + Sync + 'static,
    {
        self.filter_fn = Some(Box::new(filter));
        self
    }
}

// Generic implementation for other types
impl<T: Send + Sync + fmt::Debug + Clone + 'static> LoaderImpl<T> {
    /// Create loader with custom iterator
    pub fn with_iterator<I>(iterator: I) -> Self
    where
        I: Iterator<Item = T> + Send + Sync + 'static,
    {
        Self {
            pattern: None,
            recursive: false,
            iterator: Some(Box::new(iterator)),
            filter_fn: None}
    }
}

// Builder implementations moved to fluent_ai/src/builders/loader.rs

// Type alias for convenience
pub type DefaultLoader<T> = LoaderImpl<T>;

// Backward compatibility aliases
pub type FileLoader<T> = LoaderImpl<T>;
