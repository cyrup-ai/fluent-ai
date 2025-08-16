//! Matcher integration for proxy configuration
//!
//! Converts Proxy configuration into Matcher instances for request interception
//! with comprehensive pattern matching and no-proxy rule handling.

use super::types::{Intercept, Proxy};

impl Proxy {
    /// Convert this Proxy configuration into a Matcher for request interception
    pub(crate) fn into_matcher(self) -> super::super::matcher::Matcher {
        use super::super::matcher::Matcher_;

        let maybe_has_http_auth = match self.intercept {
            Intercept::Http(_) => true,
            Intercept::All(_) => true,
            _ => false,
        };

        let maybe_has_http_custom_headers = self.extra.misc.is_some();

        let inner = match self.intercept {
            Intercept::All(url) => {
                let mut builder = super::super::matcher::matcher::MatcherBuilder::new();
                builder = builder.all(url.to_string());
                if let Some(no_proxy) = self.no_proxy {
                    for pattern in no_proxy.inner.split(',') {
                        builder = builder.no(pattern.trim());
                    }
                }
                Matcher_::Util(builder.build())
            }
            Intercept::Http(url) => {
                let mut builder = super::super::matcher::matcher::MatcherBuilder::new();
                builder = builder.http(url.to_string());
                if let Some(no_proxy) = self.no_proxy {
                    for pattern in no_proxy.inner.split(',') {
                        builder = builder.no(pattern.trim());
                    }
                }
                Matcher_::Util(builder.build())
            }
            Intercept::Https(url) => {
                let mut builder = super::super::matcher::matcher::MatcherBuilder::new();
                builder = builder.https(url.to_string());
                if let Some(no_proxy) = self.no_proxy {
                    for pattern in no_proxy.inner.split(',') {
                        builder = builder.no(pattern.trim());
                    }
                }
                Matcher_::Util(builder.build())
            }
            Intercept::Custom(custom) => Matcher_::Custom(custom),
        };

        super::super::matcher::Matcher {
            inner,
            extra: self.extra,
            maybe_has_http_auth,
            maybe_has_http_custom_headers,
        }
    }
}
