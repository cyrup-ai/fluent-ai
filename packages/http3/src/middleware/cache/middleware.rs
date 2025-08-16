//! Core cache middleware types and configuration
//!
//! Contains the main CacheMiddleware struct with configuration methods
//! and core functionality for ETag processing and cache management.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::SystemTime;

use super::date_parsing::parse_http_date_to_timestamp;
use crate::{HttpRequest, HttpResponse};

/// Cache middleware that handles ETag processing and expires computation
pub struct CacheMiddleware {
    /// Default cache duration in hours if no expires header is present
    default_expires_hours: u64,
    /// Whether to add ETags to responses that don't have them
    generate_etags: bool,
}

impl CacheMiddleware {
    /// Create a new cache middleware with default settings
    pub fn new() -> Self {
        Self {
            default_expires_hours: 24, // 24 hours default
            generate_etags: true,
        }
    }

    /// Set the default cache duration in hours
    pub fn with_default_expires_hours(mut self, hours: u64) -> Self {
        self.default_expires_hours = hours;
        self
    }

    /// Enable or disable automatic ETag generation
    pub fn with_etag_generation(mut self, generate: bool) -> Self {
        self.generate_etags = generate;
        self
    }

    /// Get the default expires hours
    pub fn default_expires_hours(&self) -> u64 {
        self.default_expires_hours
    }

    /// Check if ETag generation is enabled
    pub fn generates_etags(&self) -> bool {
        self.generate_etags
    }

    /// Compute the effective expires timestamp
    pub(super) fn compute_expires(
        &self,
        response: &HttpResponse,
        user_expires_hours: Option<u64>,
    ) -> u64 {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Parse remote expires header if present
        let remote_expires = response
            .expires()
            .and_then(|expires_str| parse_http_date_to_timestamp(expires_str));

        // Use user-provided expires if given
        let user_expires = user_expires_hours.map(|hours| now + (hours * 3600));

        // Return the greater of remote expires or user expires, with fallback to default
        match (remote_expires, user_expires) {
            (Some(remote), Some(user)) => remote.max(user),
            (Some(remote), None) => remote,
            (None, Some(user)) => user,
            (None, None) => now + (self.default_expires_hours * 3600),
        }
    }

    /// Generate an ETag for response content if none exists
    pub(super) fn generate_etag(&self, response: &HttpResponse) -> String {
        let mut hasher = DefaultHasher::new();
        response.body().hash(&mut hasher);
        response.status().as_u16().hash(&mut hasher);

        // Include content-type in hash for better cache differentiation
        if let Some(content_type) = response.content_type() {
            content_type.hash(&mut hasher);
        }

        format!("W/\"{:x}\"", hasher.finish())
    }

    /// Extract cache directives from request headers and metadata
    #[allow(dead_code)]
    pub(super) fn extract_request_cache_directives(&self, request: &HttpRequest) -> Option<u64> {
        let headers = request.headers();

        // Parse Cache-Control header for max-age directive
        if let Some(cache_control) = headers.get("cache-control") {
            if let Ok(cache_control_str) = cache_control.to_str() {
                if let Some(max_age) = self.parse_max_age_directive(cache_control_str) {
                    return Some(max_age / 3600); // Convert seconds to hours
                }
            }
        }

        // Parse custom cache expiration header
        if let Some(expires_header) = headers.get("x-cache-expires-hours") {
            if let Ok(expires_str) = expires_header.to_str() {
                if let Ok(hours) = expires_str.parse::<u64>() {
                    return Some(hours);
                }
            }
        }

        // Parse Expires header if present in request (non-standard but supported)
        if let Some(expires) = headers.get("expires") {
            if let Ok(expires_str) = expires.to_str() {
                if let Ok(expires_time) =
                    crate::common::cache::httpdate::parse_http_date(expires_str)
                {
                    if let Ok(duration) = expires_time.duration_since(std::time::SystemTime::now())
                    {
                        return Some(duration.as_secs() / 3600); // Convert to hours
                    }
                }
            }
        }

        None
    }

    /// Parse max-age directive from Cache-Control header
    fn parse_max_age_directive(&self, cache_control: &str) -> Option<u64> {
        // Parse Cache-Control directives: max-age=<seconds>
        for directive in cache_control.split(',') {
            let directive = directive.trim();
            if let Some(max_age_part) = directive.strip_prefix("max-age=") {
                if let Ok(seconds) = max_age_part.trim().parse::<u64>() {
                    return Some(seconds);
                }
            }
        }
        None
    }
}

impl Default for CacheMiddleware {
    fn default() -> Self {
        Self::new()
    }
}
