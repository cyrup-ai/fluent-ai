use super::types::HttpError;

impl HttpError {
    /// Create a new network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        HttpError::NetworkError {
            message: message.into(),
        }
    }

    /// Create a new client error
    pub fn client<S: Into<String>>(message: S) -> Self {
        HttpError::ClientError {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        HttpError::Configuration(message.into())
    }

    /// Create a new HTTP status error
    pub fn http_status(status: u16, message: String, body: String) -> Self {
        HttpError::HttpStatus {
            status,
            message,
            body,
        }
    }

    /// Create a new timeout error
    pub fn timeout<S: Into<String>>(message: S) -> Self {
        HttpError::Timeout {
            message: message.into(),
        }
    }

    /// Create a new TLS error
    pub fn tls<S: Into<String>>(message: S) -> Self {
        HttpError::Tls {
            message: message.into(),
        }
    }

    /// Create a new DNS error
    pub fn dns<S: Into<String>>(message: S) -> Self {
        HttpError::Dns {
            message: message.into(),
        }
    }

    /// Create a new connection error
    pub fn connection<S: Into<String>>(message: S) -> Self {
        HttpError::Connection {
            message: message.into(),
        }
    }

    /// Create a new request error
    pub fn request<S: Into<String>>(message: S) -> Self {
        HttpError::Request {
            message: message.into(),
        }
    }

    /// Create a new response error
    pub fn response<S: Into<String>>(message: S) -> Self {
        HttpError::Response {
            message: message.into(),
        }
    }

    /// Create a new builder error
    pub fn builder<S: Into<String>>(message: S) -> Self {
        HttpError::Generic {
            message: message.into(),
        }
    }

    /// Create a new JSON error
    pub fn json<S: Into<String>>(message: S) -> Self {
        HttpError::Json {
            message: message.into(),
        }
    }

    /// Create a new URL error
    pub fn url<S: Into<String>>(message: S) -> Self {
        HttpError::Url {
            message: message.into(),
        }
    }

    /// Create a new header error
    pub fn invalid_header<S: Into<String>, T: Into<String>>(name: S, message: T) -> Self {
        HttpError::InvalidHeader {
            name: name.into(),
            message: message.into(),
            value: None,
            error_source: None,
        }
    }

    /// Create a new body error
    pub fn body<S: Into<String>>(message: S) -> Self {
        HttpError::Body {
            message: message.into(),
        }
    }

    /// Create a new redirect error
    pub fn redirect<S: Into<String>>(message: S) -> Self {
        HttpError::Redirect {
            message: message.into(),
        }
    }

    /// Create a new compression error
    pub fn compression<S: Into<String>>(message: S) -> Self {
        HttpError::Compression {
            message: message.into(),
        }
    }

    /// Create a new proxy error
    pub fn proxy<S: Into<String>>(message: S) -> Self {
        HttpError::Proxy {
            message: message.into(),
        }
    }

    /// Create a new status code error
    pub fn status_code(code: u16) -> Self {
        HttpError::StatusCode { code }
    }

    /// Create a new authentication error
    pub fn authentication<S: Into<String>>(message: S) -> Self {
        HttpError::Authentication {
            message: message.into(),
        }
    }

    /// Create a new rate limit error
    pub fn rate_limit<S: Into<String>>(message: S) -> Self {
        HttpError::RateLimit {
            message: message.into(),
        }
    }

    /// Create a new server error
    pub fn server<S: Into<String>>(message: S) -> Self {
        HttpError::Server {
            message: message.into(),
        }
    }

    /// Create a new IO error
    pub fn io<S: Into<String>>(message: S) -> Self {
        HttpError::IoError(message.into())
    }

    /// Create a new generic error
    pub fn other<S: Into<String>>(message: S) -> Self {
        HttpError::Other(message.into())
    }

    /// Create a new generic error with message field
    pub fn generic<S: Into<String>>(message: S) -> Self {
        HttpError::Generic {
            message: message.into(),
        }
    }

    /// Create a new stream error
    pub fn stream<S: Into<String>>(message: S) -> Self {
        HttpError::StreamError {
            message: message.into(),
        }
    }

    /// Create a new TLS error (alternative constructor)
    pub fn tls_error<S: Into<String>>(message: S) -> Self {
        HttpError::TlsError {
            message: message.into(),
        }
    }

    /// Create a new invalid URL error
    pub fn invalid_url<S: Into<String>>(message: S) -> Self {
        HttpError::InvalidUrl {
            message: message.into(),
        }
    }

    /// Create a new serialization error
    pub fn serialization<S: Into<String>>(message: S) -> Self {
        HttpError::SerializationError {
            message: message.into(),
        }
    }

    /// Create a new invalid response error
    pub fn invalid_response<S: Into<String>>(message: S) -> Self {
        HttpError::InvalidResponse {
            message: message.into(),
        }
    }

    /// Create a new deserialization error
    pub fn deserialization<S: Into<String>>(message: S) -> Self {
        HttpError::DeserializationError {
            message: message.into(),
        }
    }

    /// Create a new URL parse error
    pub fn url_parse<S: Into<String>>(message: S) -> Self {
        HttpError::UrlParseError {
            message: message.into(),
        }
    }

    /// Create a new download interrupted error
    pub fn download_interrupted<S: Into<String>>(message: S) -> Self {
        HttpError::DownloadInterrupted {
            message: message.into(),
        }
    }

    /// Create a new invalid content length error
    pub fn invalid_content_length<S: Into<String>>(message: S) -> Self {
        HttpError::InvalidContentLength {
            message: message.into(),
        }
    }

    /// Create a new chunk processing error
    pub fn chunk_processing<S: Into<String>>(message: S) -> Self {
        HttpError::ChunkProcessingError {
            message: message.into(),
        }
    }

    /// Create a new file system error
    pub fn file_system<S: Into<String>>(message: S) -> Self {
        HttpError::FileSystemError {
            message: message.into(),
        }
    }
}
