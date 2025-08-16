use fluent_ai_async::prelude::MessageChunk;
use thiserror::Error;

/// HTTP error types
#[derive(Error, Debug, Clone)]
pub enum HttpError {
    /// Network error
    #[error("Network error: {message}")]
    NetworkError {
        /// Error message describing the network issue
        message: String,
    },

    /// Client configuration error
    #[error("Client error: {message}")]
    ClientError {
        /// Error message describing the client configuration issue
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// HTTP status error
    #[error("HTTP {status}: {message}")]
    HttpStatus {
        /// HTTP status code
        status: u16,
        /// Error message describing the HTTP error
        message: String,
        /// Response body content
        body: String,
    },

    /// Request timeout error
    #[error("Request timeout: {message}")]
    Timeout {
        /// Error message describing the timeout
        message: String,
    },

    /// SSL/TLS error
    #[error("SSL/TLS error: {message}")]
    Tls {
        /// Error message describing the TLS issue
        message: String,
    },

    /// DNS resolution error
    #[error("DNS error: {message}")]
    Dns {
        /// Error message describing the DNS issue
        message: String,
    },

    /// Connection error
    #[error("Connection error: {message}")]
    Connection {
        /// Error message describing the connection issue
        message: String,
    },

    /// Request building error
    #[error("Request error: {message}")]
    Request {
        /// Error message describing the request building issue
        message: String,
    },

    /// Response parsing error
    #[error("Response error: {message}")]
    Response {
        /// Error message describing the response parsing issue
        message: String,
    },

    /// JSON parsing error
    #[error("JSON error: {message}")]
    Json {
        /// Error message describing the JSON parsing issue
        message: String,
    },

    /// URL parsing error
    #[error("URL error: {message}")]
    Url {
        /// Error message describing the URL parsing issue
        message: String,
    },

    /// Header error
    #[error("Header error: {message}")]
    InvalidHeader {
        /// Error message describing the header issue
        message: String,
        /// Header name that caused the error
        name: String,
        /// Header value that caused the error
        value: Option<String>,
        /// Source error that caused this header error
        error_source: Option<String>,
    },

    /// File system error
    #[error("File system error: {message}")]
    FileSystemError {
        /// Error message describing the file system issue
        message: String,
    },

    /// Body error
    #[error("Body error: {message}")]
    Body {
        /// Error message describing the body issue
        message: String,
    },

    /// Redirect error
    #[error("Redirect error: {message}")]
    Redirect {
        /// Error message describing the redirect issue
        message: String,
    },

    /// Compression error
    #[error("Compression error: {message}")]
    Compression {
        /// Error message describing the compression issue
        message: String,
    },

    /// Proxy error
    #[error("Proxy error: {message}")]
    Proxy {
        /// Error message describing the proxy issue
        message: String,
    },

    /// HTTP status code error
    #[error("HTTP status code error: {code}")]
    StatusCode {
        /// HTTP status code
        code: u16,
    },

    /// Builder error
    #[error("Builder error: {message}")]
    Builder {
        /// Error message describing the builder issue
        message: String,
    },

    /// Authentication error
    #[error("Authentication error: {message}")]
    Authentication {
        /// Error message describing the authentication issue
        message: String,
    },

    /// Rate limiting error
    #[error("Rate limited: {message}")]
    RateLimit {
        /// Error message describing the rate limiting
        message: String,
    },

    /// Server error
    #[error("Server error: {message}")]
    Server {
        /// Error message describing the server issue
        message: String,
    },

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Generic error
    #[error("Error: {0}")]
    Other(String),

    /// Generic error with message field
    #[error("Generic error: {message}")]
    Generic {
        /// Error message describing the generic issue
        message: String,
    },

    /// Stream error
    #[error("Stream error: {message}")]
    StreamError {
        /// Error message describing the stream issue
        message: String,
    },

    /// TLS error
    #[error("TLS error: {message}")]
    TlsError {
        /// Error message describing the TLS issue
        message: String,
    },

    /// Invalid URL error
    #[error("Invalid URL: {message}")]
    InvalidUrl {
        /// Error message describing the URL issue
        message: String,
    },

    /// Serialization error
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message describing the serialization issue
        message: String,
    },

    /// Invalid response error
    #[error("Invalid response: {message}")]
    InvalidResponse {
        /// Error message describing the response issue
        message: String,
    },

    /// Deserialization error
    #[error("Deserialization error: {message}")]
    DeserializationError {
        /// Error message describing the deserialization issue
        message: String,
    },

    /// URL parse error
    #[error("URL parse error: {message}")]
    UrlParseError {
        /// Error message describing the URL parsing issue
        message: String,
    },

    /// Download interrupted error
    #[error("Download interrupted: {message}")]
    DownloadInterrupted {
        /// Error message describing the download interruption
        message: String,
    },

    /// Invalid content length error
    #[error("Invalid content length: {message}")]
    InvalidContentLength {
        /// Error message describing the content length issue
        message: String,
    },

    /// Chunk processing error
    #[error("Chunk processing error: {message}")]
    ChunkProcessingError {
        /// Error message describing the chunk processing issue
        message: String,
    },
}

impl MessageChunk for HttpError {
    fn bad_chunk(error: String) -> Self {
        HttpError::NetworkError { message: error }
    }

    fn error(&self) -> Option<&str> {
        None // Return None to avoid lifetime issues with temporary string
    }
}

impl Default for HttpError {
    fn default() -> Self {
        HttpError::NetworkError {
            message: "Unknown error".to_string(),
        }
    }
}

/// HTTP result type for zero-allocation error handling
pub type HttpResult<T> = Result<T, HttpError>;
