//! HTTP Client implementation decomposed into logical modules
//!
//! This module provides a clean separation of concerns for the HTTP client:
//! - `config`: Configuration structures and defaults
//! - `builder`: ClientBuilder implementation with fluent API
//! - `core`: Core Client struct and HTTP method convenience functions
//! - `execution`: Request execution logic with AsyncStream patterns
//! - `tls_setup`: TLS connector configuration and setup
//! - `tests`: Unit tests for client functionality

pub mod config;
pub mod builder;
pub mod core;
pub mod execution;
pub mod tls_setup;

#[cfg(test)]
pub mod tests;

// Re-export main types for backward compatibility
pub use builder::ClientBuilder;
pub use core::{Client, ClientRef, SimpleHyperService};
pub use config::{Config, HttpVersionPref};

// Import necessary dependencies for the decomposed modules
use std::any::Any;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, convert::TryInto, net::SocketAddr};
use std::{fmt, str};

use fluent_ai_async::emit;

// Import hyper_util for the legacy client
use hyper_util;

use http::Uri;
use http::header::{
    ACCEPT, HeaderMap, HeaderValue, PROXY_AUTHORIZATION, USER_AGENT,
};
use http::uri::Scheme;
use http::Method;

#[cfg(feature = "http3")]
use quinn::VarInt;

use crate::hyper::async_stream_service::AsyncStreamLayer;
use fluent_ai_async::prelude::MessageChunk;
use crate::hyper::connect::{BoxedConnectorLayer, BoxedConnectorService, ConnectorBuilder};

use super::decoder::Accepts;
#[cfg(feature = "http3")]
use super::h3_client::H3Client;

use super::request::{Request, RequestBuilder};

#[cfg(feature = "__tls")]
use crate::hyper::Certificate;
#[cfg(any(feature = "native-tls", feature = "__rustls"))]
use crate::hyper::Identity;
use crate::hyper::config::{RequestConfig, RequestTimeout};
#[cfg(feature = "cookies")]
use crate::hyper::cookie;
#[cfg(feature = "hickory-dns")]
use crate::hyper::dns::hickory::HickoryDnsResolver;
use crate::hyper::dns::{DynResolver, Resolve, gai::GaiResolver};

use crate::hyper::proxy::Matcher as ProxyMatcher;
use crate::hyper::redirect;

use hyper_util::client::legacy::connect::HttpConnector as HyperUtilHttpConnector;
#[cfg(feature = "__rustls")]
use crate::hyper::tls::CertificateRevocationList;
#[cfg(feature = "__tls")]
use crate::hyper::tls::{self, TlsBackend};
#[cfg(feature = "rustls-tls-webpki-roots-no-provider")]
use webpki_roots;
#[cfg(feature = "rustls-tls-native-roots-no-provider")]
use rustls_native_certs;
use crate::hyper::{IntoUrl, Proxy};