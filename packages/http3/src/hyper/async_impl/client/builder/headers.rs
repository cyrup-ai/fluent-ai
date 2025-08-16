//! Header configuration methods for ClientBuilder
//!
//! Contains methods for configuring default headers, user agent,
//! and other header-related settings.

use std::convert::TryInto;

use http::header::{HeaderMap, HeaderValue, USER_AGENT};

use super::types::ClientBuilder;

impl ClientBuilder {
    /// Sets the `User-Agent` header to be used by this client.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Name your user agent after your app?
    /// static APP_USER_AGENT: &str = concat!(
    ///     env!("CARGO_PKG_NAME"),
    ///     "/",
    ///     env!("CARGO_PKG_VERSION"),
    /// );
    ///
    /// let client = crate::hyper::Client::builder()
    ///     .user_agent(APP_USER_AGENT)
    ///     .build();
    /// let mut res_stream = client.get("https://www.rust-lang.org").send();
    /// if let Some(response) = res_stream.try_next() {
    ///     println!("Response: {:?}", response);
    /// }
    /// # // OK
    /// ```
    pub fn user_agent<V>(mut self, value: V) -> ClientBuilder
    where
        V: TryInto<HeaderValue>,
        V::Error: Into<http::Error>,
    {
        match value.try_into() {
            Ok(value) => {
                self.config.headers.insert(USER_AGENT, value);
            }
            Err(_e) => {
                self.config.error = Some(crate::HttpError::builder(
                    "Header conversion error".to_string(),
                ));
            }
        };
        self
    }

    /// Sets the default headers for every request.
    ///
    /// # Example
    ///
    /// ```rust
    /// use crate::hyper::header;
    ///
    /// let mut headers = header::HeaderMap::new();
    /// headers.insert("X-MY-HEADER", header::HeaderValue::from_static("value"));
    ///
    /// // Consider marking security-sensitive headers with `set_sensitive`.
    /// let mut auth_value = header::HeaderValue::from_static("secret");
    /// auth_value.set_sensitive(true);
    /// headers.insert(header::AUTHORIZATION, auth_value);
    ///
    /// // get a client builder
    /// let client = crate::hyper::Client::builder()
    ///     .default_headers(headers)
    ///     .build();
    /// let mut res_stream = client.get("https://www.rust-lang.org").send();
    /// if let Some(response) = res_stream.try_next() {
    ///     println!("Response: {:?}", response);
    /// }
    /// ```
    pub fn default_headers(mut self, headers: HeaderMap) -> ClientBuilder {
        for (key, value) in headers.iter() {
            self.config.headers.insert(key, value.clone());
        }
        self
    }
}
