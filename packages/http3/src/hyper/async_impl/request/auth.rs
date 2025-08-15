use std::fmt;
use crate::header::HeaderValue;
use super::types::RequestBuilder;

impl RequestBuilder {
    /// Enable HTTP basic authentication.
    ///
    /// ```rust
    /// # use crate::hyper::Error;
    ///
    /// # fn run() -> Result<(), Error> {
    /// let client = crate::hyper::Client::new();
    /// let mut response_stream = client.delete("http://httpbin.org/delete")
    ///     .basic_auth("admin", Some("good password"))
    ///     .send();
    /// let resp = response_stream.try_next()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn basic_auth<U, P>(self, username: U, password: Option<P>) -> RequestBuilder
    where
        U: fmt::Display,
        P: fmt::Display,
    {
        let header_value = crate::hyper::util::basic_auth(username, password)
            .unwrap_or_else(|_| HeaderValue::from_static(""));
        self.header_sensitive(crate::header::AUTHORIZATION, header_value, true)
    }

    /// Enable HTTP bearer authentication.
    pub fn bearer_auth<T>(self, token: T) -> RequestBuilder
    where
        T: fmt::Display,
    {
        let header_value = format!("Bearer {token}");
        self.header_sensitive(crate::header::AUTHORIZATION, header_value, true)
    }

    /// Enable HTTP digest authentication.
    pub fn digest_auth<U, P>(self, username: U, password: P) -> RequestBuilder
    where
        U: fmt::Display,
        P: fmt::Display,
    {
        // Note: Digest auth requires challenge-response, so this is a simplified version
        // In practice, digest auth needs server challenge first
        let auth_string = format!("Digest username=\"{}\", password=\"{}\"", username, password);
        self.header_sensitive(crate::header::AUTHORIZATION, auth_string, true)
    }

    /// Set a custom Authorization header.
    pub fn authorization<T>(self, value: T) -> RequestBuilder
    where
        T: fmt::Display,
    {
        self.header_sensitive(crate::header::AUTHORIZATION, value.to_string(), true)
    }

    /// Set an API key header (commonly X-API-Key).
    pub fn api_key<T>(self, key: T) -> RequestBuilder
    where
        T: fmt::Display,
    {
        self.header_sensitive("X-API-Key", key.to_string(), true)
    }

    /// Set a custom API key header with specified header name.
    pub fn api_key_header<K, V>(self, header_name: K, key: V) -> RequestBuilder
    where
        K: fmt::Display,
        V: fmt::Display,
    {
        self.header_sensitive(header_name.to_string(), key.to_string(), true)
    }

    /// Set OAuth 2.0 Bearer token.
    pub fn oauth2_bearer<T>(self, token: T) -> RequestBuilder
    where
        T: fmt::Display,
    {
        self.bearer_auth(token)
    }

    /// Set JWT token as Bearer authentication.
    pub fn jwt_bearer<T>(self, jwt_token: T) -> RequestBuilder
    where
        T: fmt::Display,
    {
        self.bearer_auth(jwt_token)
    }

    /// Set AWS Signature Version 4 authorization header.
    pub fn aws_auth<T>(self, signature: T) -> RequestBuilder
    where
        T: fmt::Display,
    {
        let auth_header = format!("AWS4-HMAC-SHA256 {}", signature);
        self.header_sensitive(crate::header::AUTHORIZATION, auth_header, true)
    }

    /// Set custom authentication scheme.
    pub fn custom_auth<S, T>(self, scheme: S, credentials: T) -> RequestBuilder
    where
        S: fmt::Display,
        T: fmt::Display,
    {
        let auth_header = format!("{} {}", scheme, credentials);
        self.header_sensitive(crate::header::AUTHORIZATION, auth_header, true)
    }
}