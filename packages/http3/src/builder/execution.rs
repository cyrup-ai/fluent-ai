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
    fn collect<T: DeserializeOwned + Default + Send + 'static + From<crate::BadChunk>>(self) -> Vec<T>;

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
    fn collect_one<T: DeserializeOwned + Default + Send + 'static + From<crate::BadChunk>>(self) -> Option<T>;

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


}

impl HttpStreamExt for HttpStream {
    #[inline(always)]
    fn collect<T: DeserializeOwned + Default + Send + 'static + From<crate::BadChunk>>(self) -> Vec<T> {
        use std::any::TypeId;
        
        // Check if T is Vec<u8> using TypeId - safer than string comparison
        if TypeId::of::<T>() == TypeId::of::<Vec<u8>>() {
            // Safe cast for Vec<u8> - return raw bytes without JSON deserialization
            let bytes = self.collect_bytes_internal();
            // Safe cast using Any trait with proper error handling
            let any_bytes: Box<dyn std::any::Any> = Box::new(bytes);
            let typed_bytes = any_bytes.downcast::<Vec<u8>>().map_err(|_| {
                // This should never happen due to TypeId check, but handle gracefully
                log::error!("Failed to downcast bytes to Vec<u8> despite TypeId match");
                vec![]
            });
            
            match typed_bytes {
                Ok(bytes_box) => {
                    let result: Box<dyn std::any::Any> = bytes_box;
                    match result.downcast::<T>() {
                        Ok(final_result) => vec![*final_result],
                        Err(_) => {
                            log::error!("Failed to downcast Vec<u8> to target type T despite TypeId match");
                            vec![]
                        }
                    }
                }
                Err(empty_vec) => empty_vec,
            }
        } else {
            vec![self.collect_internal()]
        }
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
    fn collect_one<T: DeserializeOwned + Default + Send + 'static + From<crate::BadChunk>>(self) -> Option<T> {
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


}

impl HttpStream {
    /// Internal collection implementation for raw bytes using pure AsyncStream pattern
    ///
    /// Returns raw HTTP response bytes without JSON deserialization
    pub(crate) fn collect_bytes_internal(self) -> Vec<u8> {
        use futures_util::StreamExt;

        log::debug!("🔍 collect_bytes_internal: Starting HTTP raw bytes collection");

        // Convert HttpStream to AsyncStream using pure streaming architecture
        let stream =
            AsyncStream::<Vec<u8>, 1024>::with_channel(move |sender: AsyncStreamSender<Vec<u8>, 1024>| {
                log::debug!("🔍 collect_bytes_internal: Inside AsyncStream channel closure");
                
                // Use global_executor for HTTP polling encapsulation with crossbeam bridge
                log::debug!("🔍 collect_bytes_internal: Calling global_executor().execute_with_result()");
                let response_rx = global_executor().execute_with_result(move || {
                    log::debug!("🔍 collect_bytes_internal: Inside global_executor closure - START");
                    // Create crossbeam channel for synchronous HttpStream consumption
                    log::debug!("🔍 collect_bytes_internal: Creating crossbeam channels");
                    let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded();

                    // Spawn dedicated thread for async HttpStream polling
                    log::debug!("🔍 collect_bytes_internal: Spawning tokio thread for HTTP polling");
                    std::thread::spawn(move || {
                        log::debug!("🔍 collect_bytes_internal: Inside tokio thread - creating runtime");
                        // Minimal tokio runtime ONLY for HttpStream polling - isolated async work
                        let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| {
                            log::error!("Failed to create HTTP polling runtime: {}", e);
                            panic!("HTTP polling runtime required for futures::Stream consumption");
                        });
                        log::debug!("🔍 collect_bytes_internal: Tokio runtime created, starting block_on");
                        rt.block_on(async move {
                            log::debug!("🔍 collect_bytes_internal: Inside async block, starting HTTP stream polling");
                            let mut http_stream = self;
                            let mut chunk_count = 0;
                            while let Some(chunk_result) = http_stream.next().await {
                                chunk_count += 1;
                                log::debug!("🔍 collect_bytes_internal: Received chunk #{} in tokio thread", chunk_count);
                                if chunk_tx.send(chunk_result).is_err() {
                                    log::debug!("🔍 collect_bytes_internal: Receiver dropped, breaking");
                                    break; // Receiver dropped
                                }
                            }
                            log::debug!("🔍 collect_bytes_internal: HTTP stream polling completed, {} chunks processed", chunk_count);
                        });
                        log::debug!("🔍 collect_bytes_internal: Tokio thread finished");
                    });

                    // Synchronous consumption in main thread - NO FUTURES
                    log::debug!("🔍 collect_bytes_internal: Starting synchronous chunk consumption in main thread");
                    let mut all_bytes = Vec::new();
                    let mut chunk_count = 0;

                    while let Ok(chunk_result) = chunk_rx.recv() {
                        chunk_count += 1;
                        log::debug!("🔍 collect_bytes_internal: Processing chunk #{} in main thread", chunk_count);
                        match chunk_result {
                            Ok(HttpChunk::Body(bytes)) => {
                                log::debug!("Received body chunk: {} bytes", bytes.len());
                                all_bytes.extend_from_slice(&bytes);
                            }
                            Ok(HttpChunk::Head(status, _)) => {
                                log::debug!("HTTP Response Status: {}", status.as_u16());
                            }
                            Ok(HttpChunk::Deserialized(_)) => {
                                log::debug!("Ignoring deserialized chunk for raw bytes collection");
                            }
                            Ok(HttpChunk::Error(http_error)) => {
                                log::error!("HttpChunk error: {}", http_error);
                                panic!("HTTP request failed: {} - check network connectivity and server response", http_error);
                            }
                            Err(e) => {
                                log::error!("Error receiving chunk: {}", e);
                                panic!("HTTP stream processing failed: {} - check network connectivity", e);
                            }
                        }
                    }

                    // Return collected raw bytes
                    log::debug!("🔍 collect_bytes_internal: Finished chunk processing - {} chunks total, {} bytes collected", chunk_count, all_bytes.len());
                    all_bytes
                });

                // Synchronous processing - receive result from global_executor
                log::debug!("🔍 collect_bytes_internal: Waiting for response from global_executor");
                match response_rx.recv() {
                    Ok(bytes) => {
                        log::debug!("🔍 collect_bytes_internal: Successfully received {} bytes from global_executor, emitting to AsyncStream", bytes.len());
                        emit!(sender, bytes);
                    }
                    Err(e) => {
                        log::error!("🔍 collect_bytes_internal: Failed to receive HTTP response from global executor: {:?}", e);
                    }
                }
                log::debug!("🔍 collect_bytes_internal: AsyncStream channel closure completed");
            });

        // Return single collected bytes using streams-only architecture
        log::debug!("🔍 collect_bytes_internal: Calling stream.collect() to get final result");
        let result = stream
            .collect()
            .into_iter()
            .next()
            .unwrap_or_else(Vec::new);
        log::debug!("🔍 collect_bytes_internal: Final result obtained, returning {} bytes", result.len());
        result
    }

    /// Internal collection implementation using pure AsyncStream pattern
    ///
    /// NO FUTURES - Pure AsyncStream pattern for HTTP response collection with JSON deserialization
    pub(crate) fn collect_internal<T: DeserializeOwned + Send + 'static>(self) -> T
    where
        T: Default + From<crate::BadChunk>,
    {
        use futures_util::StreamExt;

        log::debug!("🔍 collect_internal: Starting HTTP response collection");

        // Convert HttpStream to AsyncStream using pure streaming architecture
        let stream =
            AsyncStream::<T, 1024>::with_channel(move |sender: AsyncStreamSender<T, 1024>| {
                log::debug!("🔍 collect_internal: Inside AsyncStream channel closure");
                
                // Use global_executor for HTTP polling encapsulation with crossbeam bridge
                log::debug!("🔍 collect_internal: Calling global_executor().execute_with_result()");
                let response_rx = global_executor().execute_with_result(move || {
                    log::debug!("🔍 collect_internal: Inside global_executor closure - START");
                    // Create crossbeam channel for synchronous HttpStream consumption
                    log::debug!("🔍 collect_internal: Creating crossbeam channels");
                    let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded();

                    // Spawn dedicated thread for async HttpStream polling
                    log::debug!("🔍 collect_internal: Spawning tokio thread for HTTP polling");
                    std::thread::spawn(move || {
                        log::debug!("🔍 collect_internal: Inside tokio thread - creating runtime");
                        // Minimal tokio runtime ONLY for HttpStream polling - isolated async work
                        let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| {
                            log::error!("Failed to create HTTP polling runtime: {}", e);
                            panic!("HTTP polling runtime required for futures::Stream consumption");
                        });
                        log::debug!("🔍 collect_internal: Tokio runtime created, starting block_on");
                        rt.block_on(async move {
                            log::debug!("🔍 collect_internal: Inside async block, starting HTTP stream polling");
                            let mut http_stream = self;
                            let mut chunk_count = 0;
                            while let Some(chunk_result) = http_stream.next().await {
                                chunk_count += 1;
                                log::debug!("🔍 collect_internal: Received chunk #{} in tokio thread", chunk_count);
                                if chunk_tx.send(chunk_result).is_err() {
                                    log::debug!("🔍 collect_internal: Receiver dropped, breaking");
                                    break; // Receiver dropped
                                }
                            }
                            log::debug!("🔍 collect_internal: HTTP stream polling completed, {} chunks processed", chunk_count);
                        });
                        log::debug!("🔍 collect_internal: Tokio thread finished");
                    });

                    // Synchronous consumption in main thread - NO FUTURES
                    log::debug!("🔍 collect_internal: Starting synchronous chunk consumption in main thread");
                    let mut all_bytes = Vec::new();
                    let mut status_code = None;
                    let mut processed_value = None;
                    let mut chunk_count = 0;

                    while let Ok(chunk_result) = chunk_rx.recv() {
                        chunk_count += 1;
                        log::debug!("🔍 collect_internal: Processing chunk #{} in main thread", chunk_count);
                        match chunk_result {
                            Ok(HttpChunk::Body(bytes)) => {
                                log::debug!("Received body chunk: {} bytes", bytes.len());
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
                    log::debug!("🔍 collect_internal: Finished chunk processing - {} chunks total", chunk_count);
                    log::debug!("Processing response: all_bytes.len()={}, status_code={:?}, processed_value={}", 
                        all_bytes.len(), status_code, processed_value.is_some());
                    
                    if let Some(value) = processed_value {
                        log::debug!("Returning processed value");
                        value
                    } else if status_code.map_or(false, |s| s.as_u16() == 204)
                        || all_bytes.is_empty()
                    {
                        // HTTP request failed or returned empty response - fail fast with descriptive error
                        if let Some(status) = status_code {
                            if status.as_u16() == 204 {
                                log::error!("HTTP request returned 204 No Content, cannot deserialize to type T");
                                panic!("HTTP 204 No Content response cannot be deserialized - use Option<T> or handle empty responses explicitly");
                            }
                        }
                        log::error!("HTTP request returned empty response body, cannot deserialize to type T");
                        panic!("Empty HTTP response body cannot be deserialized - check network connectivity and server response")
                    } else {
                        // Try to deserialize the response
                        log::debug!("Attempting to deserialize {} bytes: {}", all_bytes.len(), 
                            String::from_utf8_lossy(&all_bytes[..std::cmp::min(100, all_bytes.len())]));
                        match serde_json::from_slice(&all_bytes) {
                            Ok(value) => {
                                log::debug!("Successfully deserialized response");
                                value
                            },
                            Err(e) => {
                                log::error!("Failed to deserialize HTTP response: {}", e);
                                log::error!("Response bytes (first 200): {}", 
                                    String::from_utf8_lossy(&all_bytes[..std::cmp::min(200, all_bytes.len())]));
                                // Create BadChunk and convert to T via From<BadChunk> bound
                                T::from(crate::BadChunk::from_processing_error(
                                    crate::HttpError::Generic(format!("JSON deserialization failed: {}", e)),
                                    "Response format incompatible with requested type".to_string()
                                ))
                            }
                        }
                    }
                });

                // Synchronous processing - receive result from global_executor
                log::debug!("🔍 collect_internal: Waiting for response from global_executor");
                match response_rx.recv() {
                    Ok(value) => {
                        log::debug!("🔍 collect_internal: Successfully received value from global_executor, emitting to AsyncStream");
                        emit!(sender, value);
                    }
                    Err(e) => {
                        log::error!("🔍 collect_internal: Failed to receive HTTP response from global executor: {:?}", e);
                    }
                }
                log::debug!("🔍 collect_internal: AsyncStream channel closure completed");
            });

        // Return single collected item using streams-only architecture
        log::debug!("🔍 collect_internal: Calling stream.collect() to get final result");
        let result = stream
            .collect()
            .into_iter()
            .next()
            .unwrap_or_else(T::default);
        log::debug!("🔍 collect_internal: Final result obtained, returning");
        result
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
