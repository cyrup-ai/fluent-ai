//! Request execution and building methods
//!
//! This module contains methods for building and executing HTTP requests,
//! including streaming response handling and request cloning.

use fluent_ai_async::AsyncStream;
use super::types::{Request, RequestBuilder};
use super::{Client, Response};

impl RequestBuilder {
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
    pub fn build_split(self) -> (Client, crate::Result<Request>) {
        (self.client, self.request)
    }

    /// Constructs the Request and sends it to the target URL, returning a
    /// future Response.
    ///
    /// # Errors
    ///
    /// This method fails if there was an error while sending request.
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
    pub fn send(self) -> AsyncStream<Result<Response, crate::Error>, 1024> {
        use fluent_ai_async::emit;
        
        AsyncStream::with_channel(move |sender| {
            let req = match self.request {
                Ok(req) => req,
                Err(e) => {
                    emit!(sender, Err(e));
                    return;
                }
            };
            
            let result_stream = self.client.execute_request(req);
            for result in result_stream {
                emit!(sender, result);
            }
        })
    }

    /// Attempt to clone the RequestBuilder.
    ///
    /// `None` is returned if the RequestBuilder can not be cloned.
    ///
    /// # Examples
    ///
    /// ```no_run
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
}