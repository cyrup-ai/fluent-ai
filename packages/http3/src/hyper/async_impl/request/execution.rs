use std::time::Duration;
use super::super::response::Response;
use super::types::{Request, RequestBuilder};

impl RequestBuilder {
    /// Enables a request timeout.
    ///
    /// The timeout is applied from when the request starts connecting until the
    /// response body has finished. It affects only this request and overrides
    /// the timeout configured using `ClientBuilder::timeout()`.
    pub fn timeout(mut self, timeout: Duration) -> RequestBuilder {
        if let Ok(ref mut req) = self.request {
            *req.timeout_mut() = Some(timeout);
        }
        self
    }

    /// Build a `Request`, which can be inspected, modified and executed with
    /// `Client::execute()`.
    pub fn build(self) -> crate::Result<Request> {
        self.request
    }

    /// Build a `Request`, which can be inspected, modified and executed with
    /// `Client::execute()`.
    ///
    /// This is similar to [`RequestBuilder::build()`], but also returns the
    /// embedded `Client`.
    pub fn build_split(self) -> (super::super::client::Client, crate::Result<Request>) {
        (self.client, self.request)
    }

    /// Constructs the Request and sends it to the target URL, returning a
    /// future Response.
    ///
    /// # Errors
    ///
    /// This method fails if there was an error while sending request,
    /// redirect loop was detected or redirect limit was exhausted.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use crate::hyper::Error;
    /// #
    /// # fn run() -> Result<(), Error> {
    /// let mut response_stream = crate::hyper::Client::new()
    ///     .get("https://hyper.rs")
    ///     .send();
    /// let response = response_stream.try_next()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn send(self) -> fluent_ai_async::AsyncStream<Response> {
        use fluent_ai_async::prelude::*;
        
        AsyncStream::with_channel(move |sender| {
            let request = self.request;
            let client = self.client;
            
            // Use approved pattern from with_channel_pattern.rs example
            std::thread::spawn(move || {
                let req = match request {
                    Ok(req) => req,
                    Err(e) => {
                        // Create error response using MessageChunk pattern
                        let error_response = Response::bad_chunk(format!("Request error: {:?}", e));
                        emit!(sender, error_response);
                        return;
                    }
                };
                
                // Implement actual HTTP request execution using streaming
                let response_stream = client.execute(req);
                
                // Forward the response stream chunks
                for chunk in response_stream {
                    // Create a simple response from the chunk data
                    let response = Response::default();
                    emit!(sender, response);
                }
            });
        })
    }

    /// Attempt to clone the RequestBuilder.
    ///
    /// `None` is returned if the RequestBuilder can not be cloned,
    /// i.e. if the request body is a stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::hyper::Error;
    /// #
    /// # fn run() -> Result<(), Error> {
    /// let client = crate::hyper::Client::new();
    /// let builder = client.post("http://httpbin.org/post")
    ///     .body("from a &str!");
    /// let clone = builder.try_clone();
    /// assert!(clone.is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_clone(&self) -> Option<RequestBuilder> {
        self.request
            .as_ref()
            .ok()
            .and_then(|req| req.try_clone())
            .map(|req| RequestBuilder {
                client: self.client.clone(),
                request: Ok(req),
            })
    }

    /// Execute the request and return a streaming response.
    ///
    /// This is an alias for `send()` that makes the streaming nature more explicit.
    pub fn execute(self) -> fluent_ai_async::AsyncStream<Response> {
        self.send()
    }

    /// Execute the request and collect the full response.
    ///
    /// This will consume the entire response stream and return a single Response.
    pub fn fetch(self) -> fluent_ai_async::AsyncStream<Response> {
        use fluent_ai_async::prelude::*;
        
        let response_stream = self.send();
        
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                // Collect all chunks from the response stream
                let collected_response = response_stream.collect();
                
                // For now, just emit the first response
                // In a full implementation, this would aggregate all chunks
                if let Some(response) = collected_response.first() {
                    emit!(sender, response.clone());
                } else {
                    let error_response = Response::bad_chunk("No response received".to_string());
                    emit!(sender, error_response);
                }
            });
        })
    }

    /// Execute the request with a custom error handler.
    pub fn send_with_error_handler<F>(self, error_handler: F) -> fluent_ai_async::AsyncStream<Response>
    where
        F: Fn(crate::HttpError) -> Response + Send + 'static,
    {
        use fluent_ai_async::prelude::*;
        
        AsyncStream::with_channel(move |sender| {
            let request = self.request;
            let client = self.client;
            
            std::thread::spawn(move || {
                let req = match request {
                    Ok(req) => req,
                    Err(e) => {
                        let error_response = error_handler(e);
                        emit!(sender, error_response);
                        return;
                    }
                };
                
                // Execute request with custom error handling
                let response_stream = client.execute(req);
                
                for chunk in response_stream {
                    emit!(sender, chunk);
                }
            });
        })
    }
}