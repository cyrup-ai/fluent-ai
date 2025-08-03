//! Request execution and response handling functionality
//!
//! Provides extension traits and implementations for executing HTTP requests
//! and processing responses with streaming support and error handling.

use fluent_ai_async::thread_pool::global_executor;
use fluent_ai_async::{AsyncStream, AsyncStreamSender, emit};
use futures_util::StreamExt;
use serde::de::DeserializeOwned;

use crate::{HttpChunk, HttpError, HttpStream};

/// Extension trait for collecting HTTP streams into Vec of deserialized types
///
/// Provides convenient methods for collecting and processing HTTP response streams
/// with automatic deserialization and error handling.
pub trait HttpStreamExt {
    /// Collect the entire HTTP stream into a Vec of deserialized types
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each response into (must implement Default)
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::{Http3Builder, builder::execution::HttpStreamExt};
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, Default)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let users: Vec<User> = Http3Builder::json()
    ///     .get("https://api.example.com/users")
    ///     .collect();
    /// ```
    fn collect<T: DeserializeOwned + Default + Send + 'static>(self) -> Vec<T>;

    /// Collect the entire HTTP stream into a vector, calling error handler on failure
    ///
    /// # Arguments
    /// * `error_handler` - Function to call when collection fails
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize each response into
    /// * `F` - Error handler function type
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::{Http3Builder, builder::execution::HttpStreamExt};
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let users: Vec<User> = Http3Builder::json()
    ///     .get("https://api.example.com/users")
    ///     .collect_or_else(|error| {
    ///         log::error!("Failed to collect users: {:?}", error);
    ///         Vec::new()
    ///     });
    /// ```
    fn collect_or_else<
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> Vec<T> + Send + Sync + 'static + Clone,
    >(
        self,
        f: F,
    ) -> Vec<T>;

    /// Collect the first item from the HTTP stream, returns None if empty
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize the response into (must implement Default)
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::{Http3Builder, builder::execution::HttpStreamExt};
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, Default)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let user: Option<User> = Http3Builder::json()
    ///     .get("https://api.example.com/users/123")
    ///     .collect_one();
    /// ```
    fn collect_one<T: DeserializeOwned + Default + Send + 'static>(self) -> Option<T>;

    /// Collect the first item from the HTTP stream, calling error handler on failure
    ///
    /// # Arguments
    /// * `error_handler` - Function to call when collection fails or returns empty
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize the response into
    /// * `F` - Error handler function type
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::{Http3Builder, builder::execution::HttpStreamExt};
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, Default)]
    /// struct User {
    ///     id: u64,
    ///     name: String,
    /// }
    ///
    /// let user: User = Http3Builder::json()
    ///     .get("https://api.example.com/users/123")
    ///     .collect_one_or_else(|error| {
    ///         log::error!("Failed to get user: {:?}", error);
    ///         User::default()
    ///     });
    /// ```
    fn collect_one_or_else<
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + Sync + 'static + Clone,
    >(
        self,
        f: F,
    ) -> T;

    /// Process each chunk with a handler function supporting pattern matching syntax
    ///
    /// Allows custom processing of each HTTP chunk as it arrives, enabling
    /// real-time processing and transformation of response data.
    ///
    /// # Arguments
    /// * `handler` - Function to process each chunk result
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::{Http3Builder, builder::execution::HttpStreamExt, HttpChunk};
    ///
    /// let stream = Http3Builder::json()
    ///     .get("https://api.example.com/stream")
    ///     .on_chunk(|chunk_result| {
    ///         match chunk_result {
    ///             Ok(HttpChunk::Body(bytes)) => {
    ///                 println!("Received {} bytes", bytes.len());
    ///                 HttpChunk::Body(bytes)
    ///             }
    ///             Ok(HttpChunk::Error(e)) => {
    ///                 log::error!("Chunk error: {:?}", e);
    ///                 HttpChunk::Error(e)
    ///             }
    ///             Ok(chunk) => chunk,
    ///             Err(e) => {
    ///                 log::error!("Stream error: {:?}", e);
    ///                 HttpChunk::Error(e)
    ///             }
    ///         }
    ///     });
    /// ```
    fn on_chunk<F>(self, handler: F) -> Self
    where
        F: FnMut(Result<HttpChunk, crate::HttpError>) -> HttpChunk + Send + 'static;
}

impl HttpStreamExt for HttpStream {
    #[inline(always)]
    fn collect<T: DeserializeOwned + Default + Send + 'static>(self) -> Vec<T> {
        self.collect_internal()
    }

    #[inline(always)]
    fn collect_or_else<T, F>(self, f: F) -> Vec<T>
    where
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> Vec<T> + Send + Sync + 'static + Clone,
    {
        self.collect_or_else_impl(f)
    }

    #[inline(always)]
    fn collect_one<T: DeserializeOwned + Default + Send + 'static>(self) -> Option<T> {
        HttpStreamExt::collect(self).into_iter().next()
    }

    #[inline(always)]
    fn collect_one_or_else<T, F>(self, f: F) -> T
    where
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> T + Send + Sync + 'static + Clone,
    {
        let error_handler_clone = f.clone();
        let error_handler = move |e| vec![f(e)];
        let mut results = self.collect_or_else(error_handler);
        if let Some(item) = results.pop() {
            item
        } else {
            error_handler_clone(HttpError::Generic("No data received".to_string()))
        }
    }

    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: FnMut(Result<HttpChunk, crate::HttpError>) -> HttpChunk + Send + 'static,
    {
        // Create a processor that passes HttpChunk wrapped in Result to handler
        let mut handler = handler;
        let processor = Box::new(move |chunk: HttpChunk| -> HttpChunk {
            // Wrap the chunk in Ok and pass to handler - handler can unwrap and process
            let result = Ok(chunk);
            handler(result)
        });

        // Add the processor to the stream
        self.add_processor(processor);
        self
    }
}

impl HttpStream {
    /// Internal collection implementation using pure AsyncStream pattern
    ///
    /// NO FUTURES - Pure AsyncStream pattern for HTTP response collection
    pub(crate) fn collect_internal<T: DeserializeOwned + Send + 'static>(self) -> T
    where
        T: Default,
    {
        use futures_util::StreamExt;

        // Convert HttpStream to AsyncStream using pure streaming architecture
        let stream =
            AsyncStream::<T, 1024>::with_channel(move |sender: AsyncStreamSender<T, 1024>| {
                // Use global_executor for HTTP polling encapsulation with crossbeam bridge
                let response_rx = global_executor().execute_with_result(move || {
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
                            let mut http_stream = self;
                            while let Some(chunk_result) = http_stream.next().await {
                                if chunk_tx.send(chunk_result).is_err() {
                                    break; // Receiver dropped
                                }
                            }
                        });
                    });

                    // Synchronous consumption in main thread - NO FUTURES
                    let mut all_bytes = Vec::new();
                    let mut status_code = None;
                    let mut processed_value = None;

                    while let Ok(chunk_result) = chunk_rx.recv() {
                        match chunk_result {
                            Ok(HttpChunk::Body(bytes)) => {
                                all_bytes.extend_from_slice(&bytes);
                            }
                            Ok(HttpChunk::Head(status, _)) => {
                                status_code = Some(status);
                                log::debug!("HTTP Response Status: {}", status.as_u16());
                            }
                            Ok(HttpChunk::Deserialized(json_value)) => {
                                // Handle pre-deserialized data from on_chunk processors
                                match serde_json::from_value::<T>(json_value) {
                                    Ok(value) => {
                                        processed_value = Some(value);
                                        break;
                                    }
                                    Err(e) => {
                                        log::error!("Failed to convert deserialized chunk: {}", e);
                                        processed_value = Some(T::default());
                                        break;
                                    }
                                }
                            }
                            Ok(HttpChunk::Error(http_error)) => {
                                log::error!("HttpChunk error: {}", http_error);
                                processed_value = Some(T::default());
                                break;
                            }
                            Err(e) => {
                                log::error!("Error receiving chunk: {}", e);
                                processed_value = Some(T::default());
                                break;
                            }
                        }
                    }

                    // Return processed value or attempt deserialization
                    if let Some(value) = processed_value {
                        value
                    } else if status_code.map_or(false, |s| s.as_u16() == 204)
                        || all_bytes.is_empty()
                    {
                        // If we got a 204 No Content, return default
                        T::default()
                    } else {
                        // Try to deserialize the response
                        match serde_json::from_slice(&all_bytes) {
                            Ok(value) => value,
                            Err(e) => {
                                log::error!("Failed to deserialize response: {}", e);
                                T::default()
                            }
                        }
                    }
                });

                // Synchronous processing - receive result from global_executor
                match response_rx.recv() {
                    Ok(value) => {
                        emit!(sender, value);
                    }
                    Err(_) => {
                        log::error!("Failed to receive HTTP response from global executor");
                    }
                }
            });

        // Return single collected item using streams-only architecture
        stream
            .collect()
            .into_iter()
            .next()
            .unwrap_or_else(T::default)
    }

    /// Internal collection implementation with error handling
    pub(crate) fn collect_or_else_impl<T, F>(self, error_handler: F) -> Vec<T>
    where
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> Vec<T> + Send + Sync + 'static + Clone,
    {
        // Convert HttpStream to AsyncStream for error-aware collection using pure streaming
        let stream =
            AsyncStream::<T, 1024>::with_channel(move |sender: AsyncStreamSender<T, 1024>| {
                // Use global_executor for HTTP polling encapsulation with crossbeam bridge
                let response_rx = global_executor().execute_with_result(move || {
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
                            let mut http_stream = self;
                            while let Some(chunk_result) = http_stream.next().await {
                                if chunk_tx.send(chunk_result).is_err() {
                                    break; // Receiver dropped
                                }
                            }
                        });
                    });

                    // Synchronous consumption in main thread - NO FUTURES
                    let mut all_bytes = Vec::new();
                    let mut status_code = None;
                    let mut processed_value = None;
                    let mut error_result = None;

                    while let Ok(chunk_result) = chunk_rx.recv() {
                        match chunk_result {
                            Ok(HttpChunk::Body(bytes)) => {
                                all_bytes.extend_from_slice(&bytes);
                            }
                            Ok(HttpChunk::Head(status, _)) => {
                                status_code = Some(status);
                                log::debug!("HTTP Response Status: {}", status.as_u16());
                            }
                            Ok(HttpChunk::Deserialized(json_value)) => {
                                // Handle pre-deserialized data from on_chunk processors
                                match serde_json::from_value::<T>(json_value) {
                                    Ok(value) => {
                                        processed_value = Some(vec![value]);
                                        break;
                                    }
                                    Err(e) => {
                                        log::error!("Failed to convert deserialized chunk: {}", e);
                                        error_result = Some(HttpError::Generic(
                                            "Stream processing failed".to_string(),
                                        ));
                                        break;
                                    }
                                }
                            }
                            Ok(HttpChunk::Error(http_error)) => {
                                log::error!("HttpChunk error: {}", http_error);
                                error_result = Some(http_error);
                                break;
                            }
                            Err(e) => {
                                log::error!("Error receiving chunk: {}", e);
                                error_result = Some(e);
                                break;
                            }
                        }
                    }

                    // Return result structure
                    (processed_value, all_bytes, status_code, error_result)
                });

                // Synchronous processing - receive result from global_executor and handle errors
                let (processed_value, all_bytes, status_code, error_result) =
                    match response_rx.recv() {
                        Ok(result) => result,
                        Err(_) => {
                            log::error!("Failed to receive HTTP response from global executor");
                            for item in error_handler(HttpError::Generic(
                                "Global executor communication failed".to_string(),
                            )) {
                                emit!(sender, item);
                            }
                            return;
                        }
                    };

                // Handle error cases first
                if let Some(error) = error_result {
                    for item in error_handler(error) {
                        emit!(sender, item);
                    }
                    return;
                }

                // Handle pre-processed values
                if let Some(values) = processed_value {
                    for value in values {
                        emit!(sender, value);
                    }
                    return;
                }

                // Handle empty response or 204 No Content
                if status_code.map_or(false, |s| s.as_u16() == 204) || all_bytes.is_empty() {
                    // 204 No Content is not an error, just empty
                    return;
                }

                // Try to deserialize the response
                match serde_json::from_slice(&all_bytes) {
                    Ok(value) => {
                        emit!(sender, value);
                    }
                    Err(e) => {
                        log::error!("Failed to deserialize response: {}", e);
                        for item in error_handler(HttpError::Generic(format!(
                            "Deserialization failed: {}",
                            e
                        ))) {
                            emit!(sender, item);
                        }
                    }
                }
            });

        // Return collected items
        stream.collect()
    }
}
