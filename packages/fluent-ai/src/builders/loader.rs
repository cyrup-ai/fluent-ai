//! Loader builder implementations
//!
//! All loader construction logic and builder patterns.

use std::fmt;
use std::path::PathBuf;

use fluent_ai_domain::loader::{Loader, LoaderImpl};
use fluent_ai_domain::{AsyncTask, ZeroOneOrMany, spawn_async};

/// Builder for creating Loader instances
pub struct LoaderBuilder<T: Send + Sync + fmt::Debug + Clone + 'static> {
    pattern: Option<String>,
    recursive: bool,
    iterator: Option<Box<dyn Iterator<Item = T> + Send + Sync>>}

/// Builder with error handler for polymorphic error handling
pub struct LoaderBuilderWithHandler<T: Send + Sync + fmt::Debug + Clone + 'static> {
    #[allow(dead_code)] // TODO: Use for file glob pattern matching and directory traversal
    pattern: Option<String>,
    #[allow(dead_code)] // TODO: Use for recursive directory loading configuration
    recursive: bool,
    #[allow(dead_code)] // TODO: Use for custom file iteration and processing
    iterator: Option<Box<dyn Iterator<Item = T> + Send + Sync>>,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during loading operations
    error_handler: Box<dyn Fn(String) + Send + Sync>,
    #[allow(dead_code)] // TODO: Use for loading result processing and transformation
    result_handler: Option<Box<dyn FnOnce(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static>>,
    #[allow(dead_code)] // TODO: Use for streaming loading chunk processing
    chunk_handler: Option<Box<dyn FnMut(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static>>}

impl LoaderImpl<PathBuf> {
    // Semantic entry point
    pub fn files_matching(pattern: &str) -> LoaderBuilder<PathBuf> {
        LoaderBuilder {
            pattern: Some(pattern.to_string()),
            recursive: false,
            iterator: None}
    }
}

impl<T: Send + Sync + fmt::Debug + Clone + 'static> LoaderBuilder<T> {
    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn filter<F>(self, _f: F) -> Self
    where
        F: Fn(&T) -> bool + 'static,
    {
        // For now, store filter for later application
        // Full implementation would need to modify the iterator
        self
    }

    pub fn map<U, F>(self, _f: F) -> LoaderBuilder<U>
    where
        F: Fn(T) -> U + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static,
    {
        LoaderBuilder {
            pattern: self.pattern,
            recursive: self.recursive,
            iterator: None, // Would need to transform iterator
        }
    }

    // Error handling - required before terminal methods
    pub fn on_error<F>(self, handler: F) -> LoaderBuilderWithHandler<T>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        LoaderBuilderWithHandler {
            pattern: self.pattern,
            recursive: self.recursive,
            iterator: self.iterator,
            error_handler: Box::new(handler),
            result_handler: None,
            chunk_handler: None}
    }

    pub fn on_result<F>(self, handler: F) -> LoaderBuilderWithHandler<T>
    where
        F: FnOnce(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
    {
        LoaderBuilderWithHandler {
            pattern: self.pattern,
            recursive: self.recursive,
            iterator: self.iterator,
            error_handler: Box::new(|e| eprintln!("Loader error: {}", e)),
            result_handler: Some(Box::new(handler)),
            chunk_handler: None}
    }

    pub fn on_chunk<F>(self, handler: F) -> LoaderBuilderWithHandler<T>
    where
        F: FnMut(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
    {
        LoaderBuilderWithHandler {
            pattern: self.pattern,
            recursive: self.recursive,
            iterator: self.iterator,
            error_handler: Box::new(|e| eprintln!("Loader chunk error: {}", e)),
            result_handler: None,
            chunk_handler: Some(Box::new(handler))}
    }
}

impl<T: Send + Sync + fmt::Debug + Clone + 'static> LoaderBuilderWithHandler<T> {
    // Terminal method - returns impl Loader
    pub fn build(self) -> impl Loader<T>
    where
        LoaderImpl<T>: Loader<T>,
    {
        LoaderImpl {
            pattern: self.pattern,
            recursive: self.recursive,
            iterator: self.iterator,
            filter_fn: None}
    }

    // Terminal method - async build
    pub fn build_async(self) -> AsyncTask<impl Loader<T>>
    where
        LoaderImpl<T>: Loader<T> + fluent_ai_domain::async_task::NotResult,
    {
        spawn_async(async move { self.build() })
    }

    // Terminal method - load files immediately
    pub fn load_files(self) -> AsyncTask<ZeroOneOrMany<T>>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        let loader = self.build();
        loader.load_all()
    }

    // Terminal method - stream files immediately
    pub fn stream(self) -> fluent_ai_domain::async_task::AsyncStream<T>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        let loader = self.build();
        loader.stream_files()
    }

    // Legacy terminal methods for backward compatibility
    pub fn load(self) -> AsyncTask<ZeroOneOrMany<T>>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        self.load_files()
    }

    pub fn process<F, U>(self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + fluent_ai_domain::async_task::NotResult,
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        let loader = self.build();
        loader.process_each(processor)
    }

    pub fn on_each<F>(self, handler: F) -> AsyncTask<()>
    where
        F: Fn(&T) + Send + Sync + 'static,
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        let load_task = self.load_files();
        spawn_async(async move {
            let items = match load_task.await {
                Ok(items) => items,
                Err(_) => return, // Handle JoinError
            };
            match items {
                ZeroOneOrMany::None => {}
                ZeroOneOrMany::One(item) => handler(&item),
                ZeroOneOrMany::Many(items) => {
                    for item in &items {
                        handler(item);
                    }
                }
            }
        })
    }
}

// Type aliases for convenience
pub type DefaultLoader<T> = LoaderImpl<T>;
pub type FileLoader<T> = LoaderImpl<T>;
pub type FileLoaderBuilder<T> = LoaderBuilder<T>;
