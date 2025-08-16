use std::error::Error as StdError;
use std::fmt;

/// A Result alias where the Err case is `hyper::Error`.
pub type Result<T> = std::result::Result<T, Error>;

/// Represents errors that can occur handling HTTP streams.
pub struct Error {
    pub inner: Box<Inner>,
}

pub struct Inner {
    pub kind: Kind,
    pub source: Option<Box<dyn StdError + Send + Sync>>,
}

#[derive(Debug)]
pub enum Kind {
    Builder,
    Request,
    Redirect,
    #[cfg(not(target_arch = "wasm32"))]
    Status(crate::StatusCode, Option<hyper::ext::ReasonPhrase>),
    #[cfg(target_arch = "wasm32")]
    Status(crate::StatusCode),
    Body,
    Decode,
    Upgrade,
}

impl Error {
    pub(super) fn new(kind: Kind) -> Error {
        Error {
            inner: Box::new(Inner { kind, source: None }),
        }
    }

    pub(super) fn with<E: Into<Box<dyn StdError + Send + Sync>>>(mut self, source: E) -> Error {
        self.inner.source = Some(source.into());
        self
    }

    pub fn with_url(self, _url: crate::Url) -> Self {
        // For now, just return self since we don't have URL storage in our simplified structure
        // This maintains API compatibility
        self
    }

    pub(super) fn kind(&self) -> &Kind {
        &self.inner.kind
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut f = f.debug_struct("hyper::Error");

        f.field("kind", &self.inner.kind);

        if let Some(ref source) = self.inner.source {
            f.field("source", source);
        }

        f.finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            Kind::Builder => f.write_str("builder error"),
            Kind::Request => f.write_str("error sending request"),
            Kind::Body => f.write_str("request or response body error"),
            Kind::Decode => f.write_str("error decoding response body"),
            Kind::Redirect => f.write_str("error following redirect"),
            Kind::Upgrade => f.write_str("error upgrading connection"),
            #[cfg(target_arch = "wasm32")]
            Kind::Status(ref code) => {
                let prefix = if code.is_client_error() {
                    "HTTP status client error"
                } else {
                    debug_assert!(code.is_server_error());
                    "HTTP status server error"
                };
                write!(f, "{prefix} ({code})")
            }
            #[cfg(not(target_arch = "wasm32"))]
            Kind::Status(ref code, ref reason) => {
                let prefix = if code.is_client_error() {
                    "HTTP status client error"
                } else {
                    debug_assert!(code.is_server_error());
                    "HTTP status server error"
                };
                if let Some(reason) = reason {
                    write!(f, "{prefix} ({} {})", code.as_str(), reason.as_str())
                } else {
                    write!(f, "{prefix} ({code})")
                }
            }
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.inner
            .source
            .as_ref()
            .map(|err| &**err as &(dyn StdError + 'static))
    }
}
