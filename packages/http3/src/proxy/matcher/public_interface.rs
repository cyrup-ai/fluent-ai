//! Public interface for proxy matcher system
//!
//! Provides public methods and conversions for external use,
//! including trait implementations and utility functions.

use std::fmt;

use http::{HeaderMap, Uri, header::HeaderValue};

use super::super::core::{Extra, NoProxy};
use super::super::url_handling::Custom;
use super::implementation;
use super::intercept::{Intercept, Via};
use super::types::{Intercepted, Matcher, Matcher_};

impl Matcher {
    /// Create new matcher from patterns
    pub fn new(patterns: Vec<String>) -> Self {
        Self {
            inner: Matcher_::Util(implementation::Matcher::new(patterns)),
            extra: Extra::default(),
            maybe_has_http_auth: false,
            maybe_has_http_custom_headers: false,
        }
    }

    /// Create matcher from system environment
    pub fn from_system() -> Self {
        Self {
            inner: Matcher_::Util(implementation::Matcher::from_system()),
            extra: Extra::default(),
            maybe_has_http_auth: false,
            maybe_has_http_custom_headers: false,
        }
    }

    /// Check if URI should be intercepted
    pub fn intercept(&self, uri: &Uri) -> Option<Intercepted> {
        match &self.inner {
            Matcher_::Util(matcher) => matcher.intercept(uri).map(|intercept| Intercepted {
                inner: intercept,
                extra: self.extra.clone(),
            }),
            Matcher_::Custom(custom) => {
                // Custom matcher implementation
                None
            }
        }
    }
}

impl fmt::Debug for Matcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matcher")
            .field("maybe_has_http_auth", &self.maybe_has_http_auth)
            .field(
                "maybe_has_http_custom_headers",
                &self.maybe_has_http_custom_headers,
            )
            .finish()
    }
}
