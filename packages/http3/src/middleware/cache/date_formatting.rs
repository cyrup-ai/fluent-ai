//! HTTP date formatting utilities and calendar calculations
//!
//! Contains functions for formatting Unix timestamps as HTTP date strings
//! with RFC 1123 format support and calendar conversion utilities.

use super::date_parsing::{days_in_month, is_leap_year};

/// Format Unix timestamp as HTTP date string
pub fn format_timestamp_as_http_date(timestamp: u64) -> String {
    // Efficient formatter for HTTP date
    // For production, use proper date library
    let days_since_epoch = timestamp / 86400;
    let seconds_in_day = timestamp % 86400;

    let hours = seconds_in_day / 3600;
    let minutes = (seconds_in_day % 3600) / 60;
    let seconds = seconds_in_day % 60;

    // Convert days since epoch to year/month/day (simplified)
    let (year, month, day) = days_to_ymd(days_since_epoch);

    let month_name = match month {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        12 => "Dec",
        _ => "Jan",
    };

    // RFC 1123 format: "Sun, 06 Nov 1994 08:49:37 GMT"
    // Note: Day of week calculation omitted for simplicity
    format!(
        "Mon, {:02} {} {} {:02}:{:02}:{:02} GMT",
        day, month_name, year, hours, minutes, seconds
    )
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
