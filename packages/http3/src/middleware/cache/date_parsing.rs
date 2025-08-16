//! HTTP date parsing utilities and timestamp conversion
//!
//! Contains functions for parsing HTTP date strings to Unix timestamps
//! with support for RFC 1123 format and calendar calculations.

/// Parse HTTP date string to Unix timestamp
pub fn parse_http_date_to_timestamp(date_str: &str) -> Option<u64> {
    // Try RFC 1123 format first: "Sun, 06 Nov 1994 08:49:37 GMT"
    if let Some(timestamp) = parse_rfc1123_to_timestamp(date_str) {
        return Some(timestamp);
    }

    // Add other date format parsers as needed
    None
}

/// Parse RFC 1123 format to Unix timestamp
fn parse_rfc1123_to_timestamp(date_str: &str) -> Option<u64> {
    // Robust parser for RFC 1123 format
    // For production, consider using chrono or time crate
    let parts: Vec<&str> = date_str.split_whitespace().collect();
    if parts.len() != 6 {
        return None;
    }

    let day: u32 = parts[1].parse().ok()?;
    let month = match parts[2] {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
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
pub(super) fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get number of days in a month
pub(super) fn days_in_month(month: u32, year: u32) -> Option<u64> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if is_leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}
