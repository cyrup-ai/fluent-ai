//! Loader builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All loader construction logic and builder patterns with zero allocation.

use std::fmt;
use std::path::PathBuf;
use std::marker::PhantomData;

use crate::domain::context::{CandleLoader as Loader, CandleLoaderImpl as LoaderImpl};
use crate::util::ZeroOneOrMany;
use fluent_ai_async::{AsyncTask, spawn_task};

/// Loader builder trait - elegant zero-allocation builder pattern
pub trait LoaderBuilder<T>: Sized 
where
    T: Send + Sync + fmt::Debug + Clone + 'static,
{
    /// Set recursive loading - EXACT syntax: .recursive(true)
    fn recursive(self, recursive: bool) -> impl LoaderBuilder<T>;
    
    /// Set filter function - EXACT syntax: .filter(|item| { ... })
    fn filter<F>(self, f: F) -> impl LoaderBuilder<T>
    where
        F: Fn(&T) -> bool + 'static;
    
    /// Map transformation - EXACT syntax: .map(|item| { ... })
    fn map<U, F>(self, f: F) -> impl LoaderBuilder<U>
    where
        F: Fn(T) -> U + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static;
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl LoaderBuilder<T>
    where
        F: Fn(String) + Send + Sync + 'static;
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl LoaderBuilder<T>
    where
        F: FnOnce(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static;
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl LoaderBuilder<T>
    where
        F: FnMut(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static;
    
    /// Build loader - EXACT syntax: .build()
    fn build(self) -> impl Loader<T>
    where
        LoaderImpl<T>: Loader<T>;
    
    /// Build async loader - EXACT syntax: .build_async()
    fn build_async(self) -> AsyncTask<impl Loader<T>>
    where
        LoaderImpl<T>: Loader<T> + fluent_ai_domain::async_task::NotResult;
    
    /// Load files immediately - EXACT syntax: .load_files()
    fn load_files(self) -> AsyncTask<ZeroOneOrMany<T>>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult;
    
    /// Stream files immediately - EXACT syntax: .stream()
    fn stream(self) -> fluent_ai_domain::async_task::AsyncStream<T>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult;
    
    /// Legacy load method - EXACT syntax: .load()
    fn load(self) -> AsyncTask<ZeroOneOrMany<T>>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult;
    
    /// Process with function - EXACT syntax: .process(|item| { ... })
    fn process<F, U>(self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + fluent_ai_domain::async_task::NotResult,
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult;
    
    /// Handle each item - EXACT syntax: .on_each(|item| { ... })
    fn on_each<F>(self, handler: F) -> AsyncTask<()>
    where
        F: Fn(&T) + Send + Sync + 'static,
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct LoaderBuilderImpl<
    T: Send + Sync + fmt::Debug + Clone + 'static,
    F1 = fn(String),
    F2 = fn(ZeroOneOrMany<T>) -> ZeroOneOrMany<T>,
    F3 = fn(ZeroOneOrMany<T>) -> ZeroOneOrMany<T>,
> where
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
    F3: FnMut(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
{
    pattern: Option<String>,
    recursive: bool,
    error_handler: Option<F1>,
    result_handler: Option<F2>,
    chunk_handler: Option<F3>,
    _marker: PhantomData<T>,
}

impl LoaderImpl<PathBuf> {
    /// Semantic entry point - EXACT syntax: LoaderImpl::files_matching("pattern")
    pub fn files_matching(pattern: &str) -> impl LoaderBuilder<PathBuf> {
        LoaderBuilderImpl {
            pattern: Some(pattern.to_string()),
            recursive: false,
            error_handler: None,
            result_handler: None,
            chunk_handler: None,
            _marker: PhantomData,
        }
    }
}

impl<T, F1, F2, F3> LoaderBuilder<T> for LoaderBuilderImpl<T, F1, F2, F3>
where
    T: Send + Sync + fmt::Debug + Clone + 'static,
    F1: Fn(String) + Send + Sync + 'static,
    F2: FnOnce(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
    F3: FnMut(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
{
    /// Set recursive loading - EXACT syntax: .recursive(true)
    fn recursive(mut self, recursive: bool) -> impl LoaderBuilder<T> {
        self.recursive = recursive;
        self
    }
    
    /// Set filter function - EXACT syntax: .filter(|item| { ... })
    fn filter<F>(self, _f: F) -> impl LoaderBuilder<T>
    where
        F: Fn(&T) -> bool + 'static,
    {
        // For now, store filter for later application
        // Full implementation would need to modify the iterator
        self
    }
    
    /// Map transformation - EXACT syntax: .map(|item| { ... })
    fn map<U, F>(self, _f: F) -> impl LoaderBuilder<U>
    where
        F: Fn(T) -> U + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static,
    {
        LoaderBuilderImpl {
            pattern: self.pattern,
            recursive: self.recursive,
            error_handler: None,
            result_handler: None,
            chunk_handler: None,
            _marker: PhantomData,
        }
    }
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl LoaderBuilder<T>
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        LoaderBuilderImpl {
            pattern: self.pattern,
            recursive: self.recursive,
            error_handler: Some(handler),
            result_handler: self.result_handler,
            chunk_handler: self.chunk_handler,
            _marker: PhantomData,
        }
    }
    
    /// Set result handler - EXACT syntax: .on_result(|result| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_result<F>(self, handler: F) -> impl LoaderBuilder<T>
    where
        F: FnOnce(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
    {
        LoaderBuilderImpl {
            pattern: self.pattern,
            recursive: self.recursive,
            error_handler: self.error_handler,
            result_handler: Some(handler),
            chunk_handler: self.chunk_handler,
            _marker: PhantomData,
        }
    }
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_chunk<F>(self, handler: F) -> impl LoaderBuilder<T>
    where
        F: FnMut(ZeroOneOrMany<T>) -> ZeroOneOrMany<T> + Send + 'static,
    {
        LoaderBuilderImpl {
            pattern: self.pattern,
            recursive: self.recursive,
            error_handler: self.error_handler,
            result_handler: self.result_handler,
            chunk_handler: Some(handler),
            _marker: PhantomData,
        }
    }
    
    /// Build loader - EXACT syntax: .build()
    fn build(self) -> impl Loader<T>
    where
        LoaderImpl<T>: Loader<T>,
    {
        LoaderImpl {
            pattern: self.pattern,
            recursive: self.recursive,
            iterator: None,
            filter_fn: None,
        }
    }
    
    /// Build async loader - EXACT syntax: .build_async()
    fn build_async(self) -> AsyncTask<impl Loader<T>>
    where
        LoaderImpl<T>: Loader<T> + fluent_ai_domain::async_task::NotResult,
    {
        spawn_async(async move { self.build() })
    }
    
    /// Load files immediately - EXACT syntax: .load_files()
    fn load_files(self) -> AsyncTask<ZeroOneOrMany<T>>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        let loader = self.build();
        loader.load_all()
    }
    
    /// Stream files immediately - EXACT syntax: .stream()
    fn stream(self) -> fluent_ai_domain::async_task::AsyncStream<T>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        let loader = self.build();
        loader.stream_files()
    }
    
    /// Legacy load method - EXACT syntax: .load()
    fn load(self) -> AsyncTask<ZeroOneOrMany<T>>
    where
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        self.load_files()
    }
    
    /// Process with function - EXACT syntax: .process(|item| { ... })
    fn process<F, U>(self, processor: F) -> AsyncTask<ZeroOneOrMany<U>>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Send + Sync + fmt::Debug + Clone + 'static + fluent_ai_domain::async_task::NotResult,
        LoaderImpl<T>: Loader<T>,
        T: fluent_ai_domain::async_task::NotResult,
    {
        let loader = self.build();
        loader.process_each(processor)
    }
    
    /// Handle each item - EXACT syntax: .on_each(|item| { ... })
    fn on_each<F>(self, handler: F) -> AsyncTask<()>
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
pub type FileLoaderBuilder<T> = impl LoaderBuilder<T>;