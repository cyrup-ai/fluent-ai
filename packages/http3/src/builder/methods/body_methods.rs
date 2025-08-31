//! This module contains terminal methods for HTTP requests that require a body:
//! POST, PUT, and PATCH operations for the BodySet builder state.

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::AsyncStream;
use serde::de::DeserializeOwned;
use http::Method;
use url::Url;

use crate::builder::core::{BodySet, Http3Builder};

/// Trait for HTTP methods that require a body
pub trait BodyMethods {
    /// Execute a POST request
    fn post<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static;

    /// Execute a PUT request
    fn put<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static;

    /// Execute a PATCH request
    fn patch<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static;
}

/// POST method implementation
pub struct PostMethod;

/// PUT method implementation
pub struct PutMethod;

/// PATCH method implementation
pub struct PatchMethod;

impl BodyMethods for Http3Builder<BodySet> {
    #[inline]
    fn post<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        let request = self.request
            .with_method(Method::POST)
            .with_url(Url::parse(url).unwrap_or_else(|_| {
                log::error!("Invalid URL: {}", url);
                Url::parse("http://invalid").unwrap()
            }));
        
        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }

    #[inline]
    fn put<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        let request = self.request
            .with_method(Method::PUT)
            .with_url(Url::parse(url).unwrap_or_else(|_| {
                log::error!("Invalid URL: {}", url);
                Url::parse("http://invalid").unwrap()
            }));
        
        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }

    #[inline]
    fn patch<T>(self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        let request = self.request
            .with_method(Method::PATCH)
            .with_url(Url::parse(url).unwrap_or_else(|_| {
                log::error!("Invalid URL: {}", url);
                Url::parse("http://invalid").unwrap()
            }));
        
        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }
}

impl Http3Builder<BodySet> {
    /// Execute a POST request
    ///
    /// # Arguments
    /// * `url` - The URL to send the POST request to
    ///
    /// # Returns
    /// `AsyncStream<T, 1024>` for streaming the response as user's type
    ///
    /// # Examples
    /// ```no_run
    /// use fluent_ai_http3::Http3Builder;
    /// use serde::{Serialize, Deserialize};
    ///
    /// #[derive(Serialize)]
    /// struct CreateUser {
    ///     name: String,
    ///     email: String,
    /// }
    ///
    /// #[derive(Deserialize, Default)]
    /// struct UserResponse {
    ///     id: u64,
    ///     status: String,
    /// }
    ///
    /// let user = CreateUser {
    ///     name: "Alice".to_string(),
    ///     email: "alice@example.com".to_string(),
    /// };
    ///
    /// let response_stream = Http3Builder::json()
    ///     .body(&user)
    ///     .post::<UserResponse>("https://api.example.com/users");
    /// ```
    #[inline]
    pub fn post<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        self.request = self.request.with_method(Method::POST);
        self.request = self.request.with_url(Url::parse(url).unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            Url::parse("http://invalid").unwrap()
        }));

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: POST {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body present");
            }
        }

        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(self.request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }

    /// Execute a PUT request
    ///
    /// # Arguments
    /// * `url` - The URL to send the PUT request to
    ///
    /// # Returns
    /// `AsyncStream<T, 1024>` for streaming the response as user's type
    #[inline]
    pub fn put<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        self.request = self.request.with_method(Method::PUT);
        self.request = self.request.with_url(Url::parse(url).unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            Url::parse("http://invalid").unwrap()
        }));

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PUT {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body present");
            }
        }

        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(self.request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }

    /// Execute a PATCH request
    ///
    /// # Arguments
    /// * `url` - The URL to send the PATCH request to
    ///
    /// # Returns
    /// `AsyncStream<T, 1024>` for streaming the response as user's type
    #[inline]
    pub fn patch<T>(mut self, url: &str) -> AsyncStream<T, 1024>
    where
        T: DeserializeOwned + MessageChunk + Default + Send + 'static,
    {
        self.request = self.request.with_method(Method::PATCH);
        self.request = self.request.with_url(Url::parse(url).unwrap_or_else(|_| {
            log::error!("Invalid URL: {}", url);
            Url::parse("http://invalid").unwrap()
        }));

        if self.debug_enabled {
            log::debug!("HTTP3 Builder: PATCH {}", url);
            if let Some(body) = self.request.body() {
                log::debug!("HTTP3 Builder: Request body present");
            }
        }

        // Execute request using canonical method and apply real-time deserialization
        let http_response = self.client.execute(self.request);
        
        // Transform HttpBodyChunk stream to typed stream with real-time deserialization
        AsyncStream::with_channel(move |sender| {
            let body_stream = http_response.into_body_stream();
            for body_chunk in body_stream {
                // Real-time JSON deserialization of each chunk
                let deserialized_obj = match serde_json::from_slice::<T>(&body_chunk.data) {
                    Ok(obj) => obj,
                    Err(_) => T::bad_chunk("JSON deserialization failed".to_string())
                };
                
                fluent_ai_async::emit!(sender, deserialized_obj);
            }
        })
    }


}
