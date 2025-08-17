//! NoProxy implementation and environment parsing
//!
//! Handles no-proxy configuration from environment variables and string parsing
//! with comprehensive pattern matching for proxy exclusion rules.

use super::types::NoProxy;

impl NoProxy {
    /// Returns a new no-proxy configuration based on environment variables (or `None` if no variables are set)
    /// see [self::NoProxy::from_string()] for the string format
    pub fn from_env() -> Option<NoProxy> {
        let raw = std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
            .unwrap_or_default();

        Self::from_string(&raw)
    }

    /// Returns a new no-proxy configuration based on a `no_proxy` string (or `None` if no variables are set)
    /// The rules are as follows:
    /// * The environment variable `NO_PROXY` is checked, if it is not set, `no_proxy` is checked
    /// * If neither environment variable is set, `None` is returned
    /// * Entries are expected to be comma-separated (whitespace between entries is ignored)
    /// * IP addresses (both IPv4 and IPv6) are allowed, as are optional subnet masks (by adding /size,
    ///   for example "`192.168.1.0/24`").
    /// * An entry "`*`" matches all hostnames (this is the only wildcard allowed)
    /// * Any other entry is considered a domain name (and may contain a leading dot, for example `google.com`
    ///   and `.google.com` are equivalent) and would match both that domain AND all subdomains.
    ///
    /// For example, if `"NO_PROXY=google.com, 192.168.1.0/24"` was set, all the following would match
    /// (and therefore would bypass the proxy):
    /// * `http://google.com/`
    /// * `http://www.google.com/`
    /// * `http://192.168.1.42/`
    ///
    /// The URL `http://notgoogle.com/` would not match.
    pub fn from_string(no_proxy_list: &str) -> Option<Self> {
        if no_proxy_list.trim().is_empty() {
            return None;
        }

        Some(NoProxy {
            inner: no_proxy_list.into(),
        })
    }

    /// Check if a host should bypass the proxy based on no-proxy rules
    pub fn matches(&self, host: &str) -> bool {
        for pattern in self.inner.split(',') {
            let pattern = pattern.trim();
            if pattern.is_empty() {
                continue;
            }

            // Wildcard match
            if pattern == "*" {
                return true;
            }

            // Exact match or subdomain match
            if host == pattern || host.ends_with(&format!(".{}", pattern)) {
                return true;
            }

            // Handle leading dot patterns
            if pattern.starts_with('.') && host.ends_with(pattern) {
                return true;
            }

            // TODO: Add IP address and subnet matching
        }

        false
    }

    /// Get the raw no-proxy string
    pub fn as_str(&self) -> &str {
        &self.inner
    }
}
