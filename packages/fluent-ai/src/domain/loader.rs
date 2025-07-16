use crate::{AsyncStream, ZeroOneOrMany};
use crate::async_task::{AsyncTask, spawn_async};
use std::fmt;
use std::path::PathBuf;

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
        T: crate::async_task::NotResult;
    
    /// Stream files one by one
    fn stream_files(&self) -> AsyncStream<T>
    where
        T: crate::async_task::NotResult;
    
    /// Process each file with a processor function
    fn process_each<F, U>(&self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + crate::async_task::NotResult;
    
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
    filter_fn: Option<Box<dyn Fn(&T) -> bool + Send + Sync>>,
}

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
            iterator: None, // Can't clone trait objects
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
        PathBuf: crate::async_task::NotResult,
    {
        let pattern = self.pattern.clone();
        spawn_async(async move {
            let results: Vec<PathBuf> = match pattern {
                Some(p) => {
                    match glob::glob(&p) {
                        Ok(paths) => paths.filter_map(Result::ok).collect(),
                        Err(_) => Vec::new(), // Return empty on pattern error
                    }
                }
                None => Vec::new(),
            };
            
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
                },
                _ => ZeroOneOrMany::from_vec(results),
            }
        })
    }
    
    fn stream_files(&self) -> AsyncStream<PathBuf>
    where
        PathBuf: crate::async_task::NotResult,
    {
        let pattern = self.pattern.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        tokio::spawn(async move {
            if let Some(p) = pattern {
                if let Ok(paths) = glob::glob(&p) {
                    for path in paths.filter_map(Result::ok) {
                        if tx.send(path).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        
        AsyncStream::new(rx)
    }
    
    fn process_each<F, U>(&self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&PathBuf) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + crate::async_task::NotResult,
    {
        let load_task = self.load_all();
        spawn_async(async move {
            let paths = load_task.await; // AsyncTask now returns T directly, not Result<T, E>
            let results: Vec<U> = match paths {
                ZeroOneOrMany::None => Vec::new(),
                ZeroOneOrMany::One(path) => vec![processor(&path)],
                ZeroOneOrMany::Many(paths) => paths.iter().map(|p| processor(p)).collect(),
            };
            
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
                },
                _ => ZeroOneOrMany::from_vec(results),
            }
        })
    }
    
    fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: Some(pattern.into()),
            recursive: false,
            iterator: None,
            filter_fn: None,
        }
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
            filter_fn: None,
        }
    }
}

/// Builder for creating Loader instances
pub struct LoaderBuilder<T: Send + Sync + fmt::Debug + Clone + 'static> {
    pattern: Option<String>,
    recursive: bool,
    iterator: Option<Box<dyn Iterator<Item = T> + Send + Sync>>,
}

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
    chunk_handler: Option<Box<dyn FnMut(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static>>,
}

impl LoaderImpl<PathBuf> {
    // Semantic entry point
    pub fn files_matching(pattern: &str) -> LoaderBuilder<PathBuf> {
        LoaderBuilder {
            pattern: Some(pattern.to_string()),
            recursive: false,
            iterator: None,
        }
    }
}

impl<T: Send + Sync + fmt::Debug + Clone + 'static>
    LoaderBuilder<T>
{
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
            chunk_handler: None,
        }
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
            chunk_handler: None,
        }
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
            chunk_handler: Some(Box::new(handler)),
        }
    }
}

impl<T: Send + Sync + fmt::Debug + Clone + 'static>
    LoaderBuilderWithHandler<T>
{
    // Terminal method - returns impl Loader
    pub fn build(self) -> impl Loader<T> where LoaderImpl<T>: Loader<T> {
        LoaderImpl {
            pattern: self.pattern,
            recursive: self.recursive,
            iterator: self.iterator,
            filter_fn: None,
        }
    }
    
    // Terminal method - async build
    pub fn build_async(self) -> AsyncTask<impl Loader<T>> 
    where 
        LoaderImpl<T>: Loader<T> + crate::async_task::NotResult,
    {
        spawn_async(async move {
            self.build()
        })
    }
    
    // Terminal method - load files immediately
    pub fn load_files(self) -> AsyncTask<ZeroOneOrMany<T>> 
    where 
        LoaderImpl<T>: Loader<T>,
        T: crate::async_task::NotResult,
    {
        let loader = self.build();
        loader.load_all()
    }
    
    // Terminal method - stream files immediately  
    pub fn stream(self) -> AsyncStream<T> 
    where 
        LoaderImpl<T>: Loader<T>,
        T: crate::async_task::NotResult,
    {
        let loader = self.build();
        loader.stream_files()
    }

    // Legacy terminal methods for backward compatibility
    pub fn load(self) -> AsyncTask<ZeroOneOrMany<T>> 
    where 
        LoaderImpl<T>: Loader<T>,
        T: crate::async_task::NotResult,
    {
        self.load_files()
    }

    pub fn process<F, U>(self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + crate::async_task::NotResult,
        LoaderImpl<T>: Loader<T>,
        T: crate::async_task::NotResult,
    {
        let loader = self.build();
        loader.process_each(processor)
    }

    pub fn on_each<F>(self, handler: F) -> AsyncTask<()>
    where
        F: Fn(&T) + Send + Sync + 'static,
        LoaderImpl<T>: Loader<T>,
        T: crate::async_task::NotResult,
    {
        let load_task = self.load_files();
        spawn_async(async move {
            let items = load_task.await; // AsyncTask now returns T directly, not Result<T, E>
            match items {
                ZeroOneOrMany::None => {},
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

// Type alias for convenience
pub type DefaultLoader<T> = LoaderImpl<T>;

// Backward compatibility aliases
pub type FileLoader<T> = LoaderImpl<T>;
pub type FileLoaderBuilder<T> = LoaderBuilder<T>;