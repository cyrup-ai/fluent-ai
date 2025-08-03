//! JSONPath streaming functionality for HTTP responses
//!
//! Provides JsonPathStream for transforming raw HTTP byte streams into
//! individual JSON objects based on JSONPath expressions.

use std::marker::PhantomData;

use fluent_ai_async::thread_pool::global_executor;
use fluent_ai_async::{AsyncStream, AsyncStreamSender, emit, handle_error};
use futures_util::StreamExt;
use serde::de::DeserializeOwned;

use crate::json_path::{JsonArrayStream, JsonPathError};
use crate::{HttpChunk, HttpError, HttpStream};

/// JSONPath streaming wrapper for HTTP responses
///
/// Transforms raw HTTP byte streams into individual JSON objects based on JSONPath expressions.
/// Maintains compatibility with existing `on_chunk` error handling patterns.
pub struct JsonPathStream<T> {
    http_stream: HttpStream,
    jsonpath_expr: String,
    chunk_processors: Vec<Box<dyn FnMut(Result<T, JsonPathError>) -> T + Send>>,
    _phantom: PhantomData<T>,
}

impl<T> JsonPathStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Create new JSONPath streaming wrapper
    ///
    /// # Arguments
    ///
    /// * `http_stream` - Raw HTTP response stream
    /// * `jsonpath_expr` - JSONPath expression for element selection
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::builder::streaming::JsonPathStream;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let stream = JsonPathStream::<User>::new(http_stream, "$.users[*]".to_string());
    /// ```
    pub fn new(http_stream: HttpStream, jsonpath_expr: String) -> Self {
        Self {
            http_stream,
            jsonpath_expr,
            chunk_processors: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Process each deserialized object with a handler function
    ///
    /// Maintains compatibility with existing `on_chunk` pattern while operating
    /// on deserialized objects instead of raw HTTP chunks.
    ///
    /// # Arguments
    ///
    /// * `handler` - Function to process each deserialized object
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use fluent_ai_http3::builder::streaming::JsonPathStream;
    ///
    /// let processed_stream = stream.on_chunk(|result| {
    ///     match result {
    ///         Ok(model) => {
    ///             println!("Received: {:?}", model);
    ///             model
    ///         }
    ///         Err(_) => Default::default()
    ///     }
    /// });
    /// ```
    pub fn on_chunk<F>(mut self, mut handler: F) -> Self
    where
        F: FnMut(Result<T, JsonPathError>) -> T + Send + 'static,
    {
        let processor = Box::new(move |result: Result<T, JsonPathError>| -> T { handler(result) });

        self.chunk_processors.push(processor);
        self
    }

    /// Collect all streaming objects into a Vec
    ///
    /// Processes the entire HTTP response stream through JSONPath deserialization
    /// and returns all matching objects as a vector.
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let users: Vec<User> = Http3Builder::json()
    ///     .array_stream("$.users[*]")
    ///     .get("https://api.example.com/users")
    ///     .collect();
    /// ```
    pub fn collect(self) -> Vec<T> {
        self.collect_or_else(|_| Vec::new())
    }

    /// Collect streaming objects with error handling
    ///
    /// Processes the HTTP response stream and returns all successfully deserialized
    /// objects. Calls the error handler if stream processing fails.
    ///
    /// # Arguments
    ///
    /// * `error_handler` - Function to call on stream errors
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, Default)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let users: Vec<User> = Http3Builder::json()
    ///     .array_stream("$.users[*]")
    ///     .get("https://api.example.com/users")
    ///     .collect_or_else(|error| {
    ///         log::error!("Stream error: {:?}", error);
    ///         vec![User::default()]
    ///     });
    /// ```
    pub fn collect_or_else<F>(self, error_handler: F) -> Vec<T>
    where
        F: Fn(HttpError) -> Vec<T> + Send + Sync + 'static + Clone,
    {
        // Extract data before moving into closures
        let jsonpath_expr = self.jsonpath_expr.clone();
        let http_stream = self.http_stream;
        let mut processors = self.chunk_processors;

        // Create AsyncStream using pure async-stream pattern - NO TOKIO
        let stream =
            AsyncStream::<T, 1024>::with_channel(move |sender: AsyncStreamSender<T, 1024>| {
                // Use global_executor for HTTP polling encapsulation with crossbeam bridge
                let chunks_rx = global_executor().execute_with_result(move || {
                    // Create crossbeam channel for synchronous HttpStream consumption
                    let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded();

                    // Spawn dedicated thread for async HttpStream polling
                    std::thread::spawn(move || {
                        // Minimal tokio runtime ONLY for HttpStream polling - isolated async work
                        let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| {
                            log::error!("Failed to create HTTP polling runtime: {}", e);
                            panic!("HTTP polling runtime required for futures::Stream consumption");
                        });
                        rt.block_on(async move {
                            let mut http_stream = http_stream;
                            while let Some(chunk_result) = http_stream.next().await {
                                if chunk_tx.send(chunk_result).is_err() {
                                    break; // Receiver dropped
                                }
                            }
                        });
                    });

                    // Synchronous consumption in main thread - NO FUTURES
                    let mut chunks = Vec::new();
                    while let Ok(chunk_result) = chunk_rx.recv() {
                        chunks.push(chunk_result);
                    }
                    chunks
                });

                // Synchronous processing with JsonArrayStream
                let mut json_array_stream = JsonArrayStream::new(&jsonpath_expr);

                // Collect chunks from global_executor - pure crossbeam pattern
                let chunks = match chunks_rx.recv() {
                    Ok(chunks) => chunks,
                    Err(_) => {
                        log::error!("Failed to receive HTTP chunks from global executor");
                        return;
                    }
                };

                for chunk_result in chunks {
                    match chunk_result {
                        Ok(HttpChunk::Body(bytes)) => {
                            // Process bytes through JSONPath deserializer synchronously
                            let objects_stream = json_array_stream.process_chunk(bytes);
                            let objects = objects_stream.collect(); // Collect all objects from AsyncStream

                            for obj in objects {
                                // Apply chunk processors - objects are already unwrapped from AsyncStream
                                let mut final_obj = obj;
                                for processor in &mut processors {
                                    final_obj = processor(Ok(final_obj));
                                }

                                emit!(sender, final_obj);
                            }
                        }
                        Ok(HttpChunk::Error(e)) | Err(e) => {
                            handle_error!(e, "HTTP chunk error in JSONPath stream");
                        }
                        Ok(_) => {
                            // Ignore other chunk types (Head, Deserialized) in JSONPath streaming
                            continue;
                        }
                    }
                }
            });

        // Collect from AsyncStream
        let results = stream.collect();

        if results.is_empty() {
            error_handler(HttpError::Generic("No data received".to_string()))
        } else {
            results
        }
    }

    /// Collect the first object from the stream
    ///
    /// Processes the HTTP response stream and returns the first matching
    /// JSON object, or None if no objects are found.
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let user: Option<User> = Http3Builder::json()
    ///     .array_stream("$.users[0]")
    ///     .get("https://api.example.com/users")
    ///     .collect_one();
    /// ```
    pub fn collect_one(self) -> Option<T> {
        self.collect().into_iter().next()
    }

    /// Collect the first object with error handling
    ///
    /// Processes the HTTP response stream and returns the first matching
    /// JSON object. Calls the error handler if no objects are found or
    /// if stream processing fails.
    ///
    /// # Arguments
    ///
    /// * `error_handler` - Function to call on errors or empty results
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, Default)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let user: User = Http3Builder::json()
    ///     .array_stream("$.users[0]")
    ///     .get("https://api.example.com/users")
    ///     .collect_one_or_else(|error| {
    ///         log::error!("Failed to get user: {:?}", error);
    ///         User::default()
    ///     });
    /// ```
    pub fn collect_one_or_else<F>(self, error_handler: F) -> T
    where
        F: Fn(HttpError) -> T + Send + Sync + 'static + Clone,
    {
        let error_handler_clone = error_handler.clone();
        let vec_error_handler = move |e| vec![error_handler_clone(e)];
        self.collect_or_else(vec_error_handler)
            .pop()
            .unwrap_or_else(|| error_handler(HttpError::Generic("No data received".to_string())))
    }
}
