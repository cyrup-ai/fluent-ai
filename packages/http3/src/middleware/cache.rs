//! Cache middleware for ETag processing and expires computation

use std::future::Future;
use std::pin::Pin;
use std::time::SystemTime;

use crate::{HttpResponse, HttpResult, Middleware};

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

    /// Compute the effective expires timestamp
    fn compute_expires(&self, response: &HttpResponse, user_expires_hours: Option<u64>) -> u64 {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Parse remote expires header if present
        let remote_expires = response.expires()
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
    fn generate_etag(&self, response: &HttpResponse) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        response.body().hash(&mut hasher);
        response.status().as_u16().hash(&mut hasher);
        
        // Include content-type in hash for better cache differentiation
        if let Some(content_type) = response.content_type() {
            content_type.hash(&mut hasher);
        }

        format!("W/\"{:x}\"", hasher.finish())
    }
}

impl Default for CacheMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl Middleware for CacheMiddleware {
    fn process_response(
        &self,
        response: HttpResponse,
    ) -> Pin<Box<dyn Future<Output = HttpResult<HttpResponse>> + Send + '_>> {
        Box::pin(async move {
            let mut headers = response.headers().clone();

            // Add or ensure ETag header exists
            if response.etag().is_none() && self.generate_etags {
                let etag = self.generate_etag(&response);
                headers.insert("etag".to_string(), etag);
            }

            // Parse user-provided expires from request (if any)
            // This would need to be passed through request metadata or headers
            let user_expires_hours = None; // TODO: Extract from request context

            // Compute effective expires timestamp
            let computed_expires = self.compute_expires(&response, user_expires_hours);
            
            // Add computed expires as a custom header
            headers.insert("x-computed-expires".to_string(), computed_expires.to_string());

            // Add human-readable expires if not present
            if response.expires().is_none() {
                let expires_date = format_timestamp_as_http_date(computed_expires);
                headers.insert("expires".to_string(), expires_date);
            }

            // Create new response with updated headers
            let updated_response = HttpResponse::from_cache(
                response.status(),
                headers,
                response.body().to_vec(),
            );

            Ok(updated_response)
        })
    }
}

/// Parse HTTP date string to Unix timestamp
fn parse_http_date_to_timestamp(date_str: &str) -> Option<u64> {
    // Try RFC 1123 format first: "Sun, 06 Nov 1994 08:49:37 GMT"
    if let Some(timestamp) = parse_rfc1123_to_timestamp(date_str) {
        return Some(timestamp);
    }

    // Add other date format parsers as needed
    None
}

/// Parse RFC 1123 format to Unix timestamp
fn parse_rfc1123_to_timestamp(date_str: &str) -> Option<u64> {
    // Simple parser for RFC 1123 format
    // For production, consider using chrono or time crate
    let parts: Vec<&str> = date_str.split_whitespace().collect();
    if parts.len() != 6 {
        return None;
    }

    let day: u32 = parts[1].parse().ok()?;
    let month = match parts[2] {
        "Jan" => 1, "Feb" => 2, "Mar" => 3, "Apr" => 4,
        "May" => 5, "Jun" => 6, "Jul" => 7, "Aug" => 8,
        "Sep" => 9, "Oct" => 10, "Nov" => 11, "Dec" => 12,
        _ => return None,
    };
    let year: u32 = parts[3].parse().ok()?;
    
    let time_parts: Vec<&str> = parts[4].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    
    let hour: u32 = time_parts[0].parse().ok()?;
    let minute: u32 = time_parts[1].parse().ok()?;
    let second: u32 = time_parts[2].parse().ok()?;

    // Convert to Unix timestamp (simplified calculation)
    // For production, use proper date library
    let days_since_epoch = days_since_unix_epoch(year, month, day)?;
    let seconds_in_day = (hour * 3600 + minute * 60 + second) as u64;
    Some(days_since_epoch * 86400 + seconds_in_day)
}

/// Calculate days since Unix epoch (1970-01-01)
fn days_since_unix_epoch(year: u32, month: u32, day: u32) -> Option<u64> {
    if year < 1970 {
        return None;
    }

    let mut days = 0u64;

    // Add days for complete years
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Add days for complete months in current year
    for m in 1..month {
        days += days_in_month(m, year)?;
    }

    // Add days in current month
    days += (day - 1) as u64;

    Some(days)
}

/// Check if year is a leap year
fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get number of days in a month
fn days_in_month(month: u32, year: u32) -> Option<u64> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if is_leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

/// Format Unix timestamp as HTTP date string
fn format_timestamp_as_http_date(timestamp: u64) -> String {
    // Simple formatter for HTTP date
    // For production, use proper date library
    let days_since_epoch = timestamp / 86400;
    let seconds_in_day = timestamp % 86400;
    
    let hours = seconds_in_day / 3600;
    let minutes = (seconds_in_day % 3600) / 60;
    let seconds = seconds_in_day % 60;

    // Convert days since epoch to year/month/day (simplified)
    let (year, month, day) = days_to_ymd(days_since_epoch);
    
    let month_name = match month {
        1 => "Jan", 2 => "Feb", 3 => "Mar", 4 => "Apr",
        5 => "May", 6 => "Jun", 7 => "Jul", 8 => "Aug",
        9 => "Sep", 10 => "Oct", 11 => "Nov", 12 => "Dec",
        _ => "Jan",
    };

    // RFC 1123 format: "Sun, 06 Nov 1994 08:49:37 GMT"
    // Note: Day of week calculation omitted for simplicity
    format!("Mon, {:02} {} {} {:02}:{:02}:{:02} GMT", 
            day, month_name, year, hours, minutes, seconds)
}

/// Convert days since epoch to year/month/day (simplified calculation)
fn days_to_ymd(mut days: u64) -> (u32, u32, u32) {
    let mut year = 1970u32;
    
    // Find the year
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    
    // Find the month
    let mut month = 1u32;
    for m in 1..=12 {
        if let Some(days_in_m) = days_in_month(m, year) {
            if days < days_in_m {
                month = m;
                break;
            }
            days -= days_in_m;
        }
    }
    
    // Day is remaining days + 1
    let day = (days + 1) as u32;
    
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_middleware() {
        let middleware = CacheMiddleware::new()
            .with_default_expires_hours(12);
        
        // Test would require mock HttpResponse
        // Implementation depends on test framework
    }

    #[test]
    fn test_date_parsing() {
        let date = "Sun, 06 Nov 1994 08:49:37 GMT";
        let timestamp = parse_rfc1123_to_timestamp(date);
        assert!(timestamp.is_some());
    }
}