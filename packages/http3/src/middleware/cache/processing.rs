//! Cache middleware request and response processing logic
//!
//! Contains the Middleware trait implementation with request/response
//! processing functionality for cache management and ETag handling.

use super::date_formatting::format_timestamp_as_http_date;
use super::middleware::CacheMiddleware;
use crate::{HttpRequest, HttpResponse, HttpResult, Middleware};

impl Middleware for CacheMiddleware {
    fn process_request(&self, request: HttpRequest) -> HttpResult<HttpRequest> {
        // Extract cache directives from request and store in metadata
        // This could be enhanced to modify request headers based on cache directives
        HttpResult::Ok(request)
    }

    fn process_response(&self, response: HttpResponse) -> HttpResult<HttpResponse> {
        let mut headers = response.headers().clone();

        // Add or ensure ETag header exists
        if response.etag().is_none() && self.generates_etags() {
            let etag = self.generate_etag(&response);
            headers.insert("etag".to_string(), etag);
        }

        // NOTE: Request cache directives are processed in process_request method
        // This response processing uses default expiration policy
        let user_expires_hours = None;

        // Compute effective expires timestamp
        let computed_expires = self.compute_expires(&response, user_expires_hours);

        // Add computed expires as a custom header
        headers.insert(
            "x-computed-expires".to_string(),
            computed_expires.to_string(),
        );

        // Add human-readable expires if not present
        if response.expires().is_none() {
            let expires_date = format_timestamp_as_http_date(computed_expires);
            headers.insert("expires".to_string(), expires_date);
        }

        // Create new response with updated headers
        let updated_response =
            HttpResponse::from_cache(response.status(), headers, response.body().to_vec());

        HttpResult::Ok(updated_response)
    }
}
