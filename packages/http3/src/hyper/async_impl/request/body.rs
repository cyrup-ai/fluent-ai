use serde::Serialize;

use super::super::body::Body;
use super::types::RequestBuilder;
use http::header::{CONTENT_TYPE, HeaderValue};

impl RequestBuilder {
    /// Set the request body.
    pub fn body<T: Into<Body>>(mut self, body: T) -> RequestBuilder {
        if let Ok(ref mut req) = self.request {
            *req.body_mut() = Some(body.into());
        }
        self
    }

    /// Send a form data body.
    ///
    /// Sets `Content-Type: application/x-www-form-urlencoded`.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Error;
    /// #
    /// # fn run() -> Result<(), Error> {
    /// let mut params = std::collections::HashMap::new();
    /// params.insert("lang", "rust");
    ///
    /// let client = crate::hyper::Client::new();
    /// let mut response_stream = client.post("http://httpbin.org")
    ///     .form(&params)
    ///     .send();
    /// let res = response_stream.try_next()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This method fails if the passed value cannot be serialized into
    /// url encoded format
    pub fn form<T: Serialize + ?Sized>(mut self, form: &T) -> RequestBuilder {
        let mut error = None;
        if let Ok(ref mut req) = self.request {
            match serde_urlencoded::to_string(form) {
                Ok(body) => {
                    req.headers_mut()
                        .entry(CONTENT_TYPE)
                        .or_insert(HeaderValue::from_static(
                            "application/x-www-form-urlencoded",
                        ));
                    *req.body_mut() = Some(body.into());
                }
                Err(err) => error = Some(crate::HttpError::builder(err.to_string())),
            }
        }
        if let Some(err) = error {
            self.request = Err(err);
        }
        self
    }

    /// Send a JSON body.
    ///
    /// # Optional
    ///
    /// This requires the optional `json` feature enabled.
    ///
    /// # Errors
    ///
    /// Serialization can fail if `T`'s implementation of `Serialize` decides to
    /// fail, or if `T` contains a map with non-string keys.
    #[cfg(feature = "json")]
    #[cfg_attr(docsrs, doc(cfg(feature = "json")))]
    pub fn json<T: Serialize + ?Sized>(mut self, json: &T) -> RequestBuilder {
        let mut error = None;
        if let Ok(ref mut req) = self.request {
            match serde_json::to_vec(json) {
                Ok(body) => {
                    if !req.headers().contains_key(CONTENT_TYPE) {
                        req.headers_mut()
                            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                    }
                    *req.body_mut() = Some(body.into());
                }
                Err(err) => error = Some(crate::HttpError::builder(err.to_string())),
            }
        }
        if let Some(err) = error {
            self.request = Err(err);
        }
        self
    }

    /// Sends a multipart/form-data body.
    ///
    /// ```
    /// # use crate::hyper::Error;
    /// #
    /// # fn run() -> Result<(), Error> {
    /// let client = crate::hyper::Client::new();
    /// let form = crate::hyper::multipart::Form::new()
    ///     .text("key3", "value3")
    ///     .text("key4", "value4");
    ///
    /// let mut response_stream = client.post("your url")
    ///     .multipart(form)
    ///     .send();
    /// let response = response_stream.try_next()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Optional
    ///
    /// This requires the optional `multipart` feature enabled.
    #[cfg(feature = "multipart")]
    #[cfg_attr(docsrs, doc(cfg(feature = "multipart")))]
    pub fn multipart(mut self, mut multipart: super::super::multipart::Form) -> RequestBuilder {
        let mut error = None;
        if let Ok(ref mut req) = self.request {
            match multipart.stream() {
                Ok(body) => {
                    if let Some(content_type) = multipart.content_type() {
                        req.headers_mut().insert(CONTENT_TYPE, content_type);
                    }
                    if let Some(content_length) = multipart.content_length() {
                        if let Ok(header_value) = HeaderValue::from_str(&content_length.to_string())
                        {
                            req.headers_mut()
                                .insert(crate::header::CONTENT_LENGTH, header_value);
                        }
                    }
                    *req.body_mut() = Some(Body::wrap_stream(body));
                }
                Err(err) => error = Some(crate::HttpError::builder(err.to_string())),
            }
        }
        if let Some(err) = error {
            self.request = Err(err);
        }
        self
    }

    /// Send a text body.
    pub fn text<T: Into<String>>(self, text: T) -> RequestBuilder {
        self.body(text.into()).header(CONTENT_TYPE, "text/plain")
    }

    /// Send raw bytes as body.
    pub fn bytes<T: Into<Vec<u8>>>(self, bytes: T) -> RequestBuilder {
        self.body(bytes.into())
            .header(CONTENT_TYPE, "application/octet-stream")
    }

    /// Send an empty body.
    pub fn empty_body(mut self) -> RequestBuilder {
        if let Ok(ref mut req) = self.request {
            *req.body_mut() = None;
        }
        self
    }

    /// Set body from a file path.
    ///
    /// This will read the file and set it as the request body.
    /// The content type will be guessed from the file extension.
    pub fn file<P: AsRef<std::path::Path>>(mut self, path: P) -> RequestBuilder {
        let mut error = None;
        if let Ok(ref mut req) = self.request {
            match std::fs::read(path.as_ref()) {
                Ok(bytes) => {
                    // Try to guess content type from file extension
                    if let Some(extension) = path.as_ref().extension() {
                        let content_type = match extension.to_str() {
                            Some("json") => "application/json",
                            Some("xml") => "application/xml",
                            Some("html") => "text/html",
                            Some("txt") => "text/plain",
                            Some("css") => "text/css",
                            Some("js") => "application/javascript",
                            Some("png") => "image/png",
                            Some("jpg") | Some("jpeg") => "image/jpeg",
                            Some("gif") => "image/gif",
                            Some("pdf") => "application/pdf",
                            _ => "application/octet-stream",
                        };
                        req.headers_mut()
                            .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
                    }
                    *req.body_mut() = Some(bytes.into());
                }
                Err(err) => {
                    error = Some(crate::HttpError::builder(format!(
                        "Failed to read file: {}",
                        err
                    )))
                }
            }
        }
        if let Some(err) = error {
            self.request = Err(err);
        }
        self
    }
}
