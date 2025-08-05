//! Request execution and response handling functionality
//!
//! Provides extension traits and implementations for executing HTTP requests
//! and processing responses with streaming support and error handling.


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
            // Elite crossbeam polling for mixed results - process ALL chunks
            log::debug!("🔍 collect: Starting mixed results collection");
            let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded();
            
            // Spawn async task on current runtime
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    let _task_handle = handle.spawn(async move {
                        log::debug!("🔍 collect: Starting HTTP stream polling");
                        let mut http_stream = self;
                        let mut chunk_count = 0;
                        while let Some(chunk_result) = http_stream.next().await {
                            chunk_count += 1;
                            log::debug!("🔍 collect: Received chunk #{}", chunk_count);
                            if chunk_tx.send(chunk_result).is_err() {
                                break; // Receiver dropped
                            }
                        }
                        log::debug!("🔍 collect: HTTP stream polling completed, {} chunks processed", chunk_count);
                        // Sender drops naturally here - channel closes
                    });
                }
                Err(_) => {
                    log::error!("🔍 collect: No tokio runtime available");
                    return Vec::new();
                }
            }
            
            // Synchronous consumption - collect mixed results (good + bad)
            log::debug!("🔍 collect: Starting synchronous mixed collection");
            let mut results = Vec::new();
            let mut all_bytes = Vec::new();
            let mut chunk_count = 0;
            
            while let Ok(chunk_result) = chunk_rx.recv() {
                chunk_count += 1;
                log::debug!("🔍 collect: Processing chunk #{}", chunk_count);
                match chunk_result {
                    Ok(HttpChunk::Body(bytes)) => {
                        log::debug!("Received body chunk: {} bytes", bytes.len());
                        all_bytes.extend_from_slice(&bytes);
                    }
                    Ok(HttpChunk::Head(status, _)) => {
                        log::debug!("HTTP Response Status: {}", status.as_u16());
                        // Process but don't add to results
                    }
                    Ok(HttpChunk::Deserialized(json_value)) => {
                        // Handle pre-deserialized data from on_chunk processors
                        match serde_json::from_value::<T>(json_value) {
                            Ok(value) => {
                                log::debug!("Successfully converted pre-deserialized chunk");
                                results.push(value);
                            }
                            Err(e) => {
                                log::error!("Failed to convert pre-deserialized chunk: {}", e);
                                let bad_chunk = crate::BadChunk::from_processing_error(
                                    crate::HttpError::Generic(format!("JSON conversion failed: {}", e)),
                                    "Pre-deserialized chunk conversion failed".to_string()
                                );
                                results.push(T::from(bad_chunk));
                            }
                        }
                    }
                    Ok(HttpChunk::Error(http_error)) => {
                        log::error!("HttpChunk error: {}", http_error);
                        let bad_chunk = crate::BadChunk::from_err(http_error);
                        results.push(T::from(bad_chunk));
                    }
                    Err(stream_error) => {
                        log::error!("Stream error: {}", stream_error);
                        let bad_chunk = crate::BadChunk::from_err(stream_error);
                        results.push(T::from(bad_chunk));
                    }
                }
            }
            
            // Try to deserialize accumulated body bytes if any
            if !all_bytes.is_empty() {
                log::debug!("🔍 collect: Attempting to deserialize {} bytes", all_bytes.len());
                match serde_json::from_slice(&all_bytes) {
                    Ok(value) => {
                        log::debug!("🔍 collect: Successfully deserialized response");
                        results.push(value);
                    }
                    Err(e) => {
                        log::error!("🔍 collect: Deserialization failed: {}", e);
                        let bad_chunk = crate::BadChunk::from_processing_error(
                            crate::HttpError::Generic(format!("JSON deserialization failed: {}", e)),
                            "Response format incompatible with requested type".to_string()
                        );
                        results.push(T::from(bad_chunk));
                    }
                }
            }
            
            log::debug!("🔍 collect: Finished mixed collection - {} total items", results.len());
            results
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

        // Elite crossbeam polling - direct implementation
        log::debug!("🔍 collect_bytes_internal: Creating crossbeam channels");
        let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded();

        // Spawn async task on current runtime
        log::debug!("🔍 collect_bytes_internal: Using elite crossbeam polling");
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let _task_handle = handle.spawn(async move {
            log::debug!("🔍 collect_bytes_internal: Starting HTTP stream polling");
            let mut http_stream = self;
            let mut chunk_count = 0;
            while let Some(chunk_result) = http_stream.next().await {
                chunk_count += 1;
                log::debug!("🔍 collect_bytes_internal: Received chunk #{}", chunk_count);
                if chunk_tx.send(chunk_result).is_err() {
                    log::debug!("🔍 collect_bytes_internal: Receiver dropped, breaking");
                    break; // Receiver dropped
                }
            }
            log::debug!("🔍 collect_bytes_internal: HTTP stream polling completed, {} chunks processed", chunk_count);
            // Sender drops naturally here - channel will close
                });
            }
            Err(_) => {
                log::error!("🔍 collect_bytes_internal: No tokio runtime available");
                return Vec::new();
            }
        }

        // Synchronous consumption in main thread - NO FUTURES
        log::debug!("🔍 collect_bytes_internal: Starting synchronous chunk consumption");
        let mut all_bytes = Vec::new();
        let mut chunk_count = 0;

        while let Ok(chunk_result) = chunk_rx.recv() {
            chunk_count += 1;
            log::debug!("🔍 collect_bytes_internal: Processing chunk #{}", chunk_count);
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
                    all_bytes.clear(); // Clear on error
                    break;
                }
                Err(_) => {
                    // Channel closed - sender dropped, stream completed naturally
                    log::debug!("🔍 collect_bytes_internal: Channel closed, stream completed");
                    break;
                }
            }
        }

        // Return collected raw bytes
        log::debug!("🔍 collect_bytes_internal: Finished - {} chunks, {} bytes", chunk_count, all_bytes.len());
        all_bytes
    }



    /// Internal collection implementation with error handling
    pub(crate) fn collect_or_else_impl<T, F>(self, error_handler: F) -> Vec<T>
    where
        T: DeserializeOwned + Send + 'static,
        F: Fn(HttpError) -> Vec<T> + Send + Sync + 'static + Clone,
    {
        use futures_util::StreamExt;

        log::debug!("🔍 collect_or_else_impl: Starting HTTP response collection with error handling");

        // Elite crossbeam polling - direct implementation
        log::debug!("🔍 collect_or_else_impl: Creating crossbeam channels");
        let (chunk_tx, chunk_rx) = crossbeam_channel::unbounded();

        // Spawn async task on current runtime
        log::debug!("🔍 collect_or_else_impl: Using elite crossbeam polling");
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let _task_handle = handle.spawn(async move {
                    log::debug!("🔍 collect_or_else_impl: Starting HTTP stream polling");
                    let mut http_stream = self;
                    let mut chunk_count = 0;
                    while let Some(chunk_result) = http_stream.next().await {
                        chunk_count += 1;
                        log::debug!("🔍 collect_or_else_impl: Received chunk #{}", chunk_count);
                        if chunk_tx.send(chunk_result).is_err() {
                            log::debug!("🔍 collect_or_else_impl: Receiver dropped, breaking");
                            break; // Receiver dropped
                        }
                    }
                    log::debug!("🔍 collect_or_else_impl: HTTP stream polling completed, {} chunks processed", chunk_count);
                    // Sender drops naturally here - channel will close
                });
            }
            Err(_) => {
                log::error!("🔍 collect_or_else_impl: No tokio runtime available");
                return error_handler(HttpError::Generic("No runtime available".to_string()));
            }
        }

        // Synchronous consumption in main thread - NO FUTURES
        log::debug!("🔍 collect_or_else_impl: Starting synchronous chunk consumption");
        let mut all_bytes = Vec::new();
        let mut status_code = None;
        let mut processed_value = None;
        let mut chunk_count = 0;

        while let Ok(chunk_result) = chunk_rx.recv() {
            chunk_count += 1;
            log::debug!("🔍 collect_or_else_impl: Processing chunk #{}", chunk_count);
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
                            processed_value = Some(vec![value]);
                            break;
                        }
                        Err(e) => {
                            log::error!("Failed to convert deserialized chunk: {}", e);
                            return error_handler(HttpError::Generic("Stream processing failed".to_string()));
                        }
                    }
                }
                Ok(HttpChunk::Error(http_error)) => {
                    log::error!("HttpChunk error: {}", http_error);
                    return error_handler(http_error);
                }
                Err(_) => {
                    // Channel closed - sender dropped, stream completed naturally
                    log::debug!("🔍 collect_or_else_impl: Channel closed, stream completed");
                    break;
                }
            }
        }

        // Handle pre-processed values first
        if let Some(values) = processed_value {
            log::debug!("🔍 collect_or_else_impl: Returning pre-processed values");
            return values;
        }

        // Handle empty response or 204 No Content
        if status_code.map_or(false, |s| s.as_u16() == 204) || all_bytes.is_empty() {
            // 204 No Content is not an error, just empty
            log::debug!("🔍 collect_or_else_impl: Empty response, returning empty vec");
            return Vec::new();
        }

        // Try to deserialize the response - if it fails, call error handler (enters ELSE clause)
        log::debug!("🔍 collect_or_else_impl: Attempting to deserialize {} bytes", all_bytes.len());
        match serde_json::from_slice(&all_bytes) {
            Ok(value) => {
                log::debug!("🔍 collect_or_else_impl: Successfully deserialized response - all chunks good");
                vec![value]
            }
            Err(e) => {
                log::error!("Deserialization failed - entering ELSE clause: {}", e);
                error_handler(HttpError::Generic(format!("Deserialization failed: {}", e)))
            }
        }
    }
}
