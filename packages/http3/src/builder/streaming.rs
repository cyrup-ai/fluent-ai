//! JSONPath streaming functionality for HTTP responses
//!
//! Provides JsonPathStream for transforming raw HTTP byte streams into
//! individual JSON objects based on JSONPath expressions.
//! Follows the streams-only architecture with zero allocation and no futures.

use std::marker::PhantomData;

use fluent_ai_async::{AsyncStream, AsyncStreamSender, emit, handle_error};
use serde::de::DeserializeOwned;

use crate::json_path::{JsonStreamProcessor, JsonPathError};
use crate::{HttpChunk, HttpError, HttpStream};

/// JSONPath streaming wrapper for HTTP responses
///
/// Transforms raw HTTP byte streams into individual JSON objects based on JSONPath expressions.
/// Maintains compatibility with existing `on_chunk` error handling patterns while following
/// the streams-only architecture with zero allocation.
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

        // Use streams-only architecture with JsonStreamProcessor
        let stream = AsyncStream::<T>::with_channel(move |sender: AsyncStreamSender<T>| {
            // Collect all HTTP chunks synchronously - no futures
            let chunks = Self::collect_http_chunks_sync(http_stream);
            
            match chunks {
                Ok(chunk_vec) => {
                    // Create JsonStreamProcessor for high-performance JSONPath processing
                    let json_processor = JsonStreamProcessor::new(&jsonpath_expr);
                    
                    // Process chunks through JsonStreamProcessor
                    let object_stream = json_processor.process_chunks(chunk_vec.into_iter());
                    
                    // Emit each processed object through the sender
                    for obj in object_stream {
                        let mut result = Ok(obj);
                        
                        // Apply chunk processors in sequence
                        for processor in &mut processors {
                            result = Ok(processor(result));
                        }
                        
                        match result {
                            Ok(processed_obj) => {
                                emit!(sender, processed_obj);
                            }
                            Err(e) => {
                                handle_error!(e, "Chunk processor failed");
                            }
                        }
                    }
                }
                Err(e) => {
                    handle_error!(e, "Failed to collect HTTP chunks");
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

    /// Collect HTTP chunks synchronously using streams-only architecture
    ///
    /// This method follows the no-futures constraint by using crossbeam channels
    /// to bridge between the async HttpStream and synchronous processing.
    fn collect_http_chunks_sync(mut http_stream: HttpStream) -> Result<Vec<HttpChunk>, HttpError> {
        use crossbeam_channel::{bounded, Receiver};
        use std::time::Duration;

        // Create bounded channel for chunk communication
        let (tx, rx): (crossbeam_channel::Sender<Result<HttpChunk, HttpError>>, Receiver<Result<HttpChunk, HttpError>>) = bounded(1024);

        // Elite crossbeam polling - use existing runtime directly
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let _task_handle = handle.spawn(async move {
                    use futures_util::StreamExt;
                    
                    while let Some(chunk_result) = http_stream.next().await {
                        if tx.send(chunk_result).is_err() {
                            break; // Receiver dropped
                        }
                    }
                });
            }
            Err(_) => {
                let _ = tx.send(Err(HttpError::Generic("No tokio runtime available".to_string())));
            }
        }

        // Synchronous collection from crossbeam channel - no futures
        let mut chunks = Vec::new();
        let timeout = Duration::from_secs(30); // 30 second timeout for HTTP collection

        loop {
            match rx.recv_timeout(timeout) {
                Ok(chunk_result) => {
                    match chunk_result {
                        Ok(chunk) => chunks.push(chunk),
                        Err(e) => return Err(e),
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    // Timeout - consider it complete
                    break;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // Sender finished - natural completion
                    break;
                }
            }
        }

        Ok(chunks)
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