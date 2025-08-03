//! GET HTTP Operations Module - Streaming GET requests with conditional support

use std::collections::HashMap;

use http::{HeaderMap, HeaderName, HeaderValue, Method};

use crate::operations::HttpOperation;
use crate::{HttpResult, client::HttpClient, request::HttpRequest, stream::HttpStream};

/// GET operation implementation with streaming and conditional request support
#[derive(Clone)]
pub struct GetOperation {
    client: HttpClient,
    url: String,
    headers: HeaderMap,
    query_params: HashMap<String, String>,
}

impl GetOperation {
    /// Create a new GET operation
    ///
    /// # Arguments
    /// * `client` - The HTTP client to use for the request
    /// * `url` - The URL to send the GET request to
    ///
    /// # Returns
    /// A new `GetOperation` instance
    #[must_use]
    pub fn new(client: HttpClient, url: String) -> Self {
        Self {
            client,
            url,
            headers: HeaderMap::new(),
            query_params: HashMap::new(),
        }
    }

    /// Add a query parameter
    ///
    /// # Arguments
    /// * `key` - The query parameter key
    /// * `value` - The query parameter value
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn query_param(mut self, key: &str, value: &str) -> Self {
        self.query_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Add multiple query parameters
    ///
    /// # Arguments
    /// * `params` - A HashMap of query parameters to add
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn query_params(mut self, params: HashMap<String, String>) -> Self {
        self.query_params.extend(params);
        self
    }

    /// Add a custom header
    ///
    /// # Arguments
    /// * `key` - The header name
    /// * `value` - The header value
    ///
    /// # Returns
    /// `Result<Self, HttpError>` for method chaining
    #[must_use]
    pub fn header(mut self, key: &str, value: &str) -> HttpResult<Self> {
        let header_name = HeaderName::from_bytes(key.as_bytes())?;
        let header_value = HeaderValue::from_str(value)?;
        self.headers.insert(header_name, header_value);
        Ok(self)
    }

    /// Set headers from a HeaderMap
    ///
    /// # Arguments
    /// * `headers` - The headers to set
    ///
    /// # Returns
    /// `Self` for method chaining
    #[must_use]
    pub fn headers(mut self, headers: HeaderMap) -> Self {
        self.headers = headers;
        self
    }
}

impl HttpOperation for GetOperation {
    type Output = HttpStream;

    fn execute(&self) -> Self::Output {
        let query_params_vec: Vec<(&str, &str)> = self
            .query_params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let request = HttpRequest::new(
            self.method(),
            self.url.clone(),
            Some(self.headers.clone()),
            None,
            None,
        )
        .with_query_params(&query_params_vec);

        self.client.execute_streaming(request)
    }

    fn method(&self) -> Method {
        Method::GET
    }

    fn url(&self) -> &str {
        &self.url
    }
}
