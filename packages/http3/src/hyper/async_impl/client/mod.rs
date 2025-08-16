//! HTTP Client implementation decomposed into logical modules
//!
//! This module provides a clean separation of concerns for the HTTP client:
//! - `config`: Configuration structures and defaults
//! - `builder`: ClientBuilder implementation with fluent API
//! - `core`: Core Client struct and HTTP method convenience functions
//! - `execution`: Request execution logic with AsyncStream patterns
//! - `tls_setup`: TLS connector configuration and setup
//! - `tests`: Unit tests for client functionality

pub mod builder;
pub mod config;
pub mod core;
pub mod execution;
pub mod tls_setup;

// Re-export main types for backward compatibility
pub use core::{Client, ClientRef, SimpleHyperService};
// Import necessary dependencies for the decomposed modules
use std::any::Any;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, convert::TryInto, net::SocketAddr};
use std::{fmt, str};

pub use builder::ClientBuilder;
pub use config::{Config, HttpVersionPref};
use fluent_ai_async::emit;
use fluent_ai_async::prelude::MessageChunk;
use http::Method;
use http::Uri;
use http::header::{ACCEPT, HeaderMap, HeaderValue, PROXY_AUTHORIZATION, USER_AGENT};
use http::uri::Scheme;
// Import hyper_util for the legacy client
use hyper_util;
use hyper_util::client::legacy::connect::HttpConnector as HyperUtilHttpConnector;
#[cfg(feature = "http3")]
use quinn::VarInt;
#[cfg(feature = "rustls-tls-native-roots-no-provider")]
use rustls_native_certs;
#[cfg(feature = "rustls-tls-webpki-roots-no-provider")]
use webpki_roots;

use super::decoder::Accepts;
#[cfg(feature = "http3")]
use super::h3_client::H3Client;
use super::request::{Request, RequestBuilder};
#[cfg(feature = "__tls")]
use crate::hyper::Certificate;
#[cfg(any(feature = "native-tls", feature = "__rustls"))]
use crate::hyper::Identity;
use crate::hyper::async_stream_service::AsyncStreamLayer;
use crate::hyper::config::{RequestConfig, RequestTimeout};
use crate::hyper::connect::{BoxedConnectorLayer, BoxedConnectorService, ConnectorBuilder};
#[cfg(feature = "cookies")]
use crate::hyper::cookie;
#[cfg(feature = "hickory-dns")]
use crate::hyper::dns::hickory::HickoryDnsResolver;
use crate::hyper::dns::{DynResolver, Resolve, gai::GaiResolver};
use crate::hyper::proxy::Matcher as ProxyMatcher;
use crate::hyper::redirect;
#[cfg(feature = "__rustls")]
use crate::hyper::tls::CertificateRevocationList;
#[cfg(feature = "__tls")]
use crate::hyper::tls::{self, TlsBackend};
use crate::hyper::{IntoUrl, Proxy};
