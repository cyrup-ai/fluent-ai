//! # AsyncStream on_chunk Pattern
//!
//! Demonstrates .on_chunk() for unwrapping Result<T, E> into T values in streams.
//! Shows how .on_chunk() runs on each element and unwraps Results into clean stream values.

use fluent_ai_async::prelude::*;
use std::{thread, time::Duration};

#[derive(Debug, Clone, Default)]
struct LogEntry {
    timestamp: String,
    level: String,
    message: String,
    error_message: Option<String>,
}

impl MessageChunk for LogEntry {
    fn bad_chunk(error: String) -> Self {
        LogEntry {
            timestamp: "0000-00-00T00:00:00Z".to_string(),
            level: "ERROR".to_string(),
            message: "[PARSE_ERROR]".to_string(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

#[derive(Debug, Clone, Default)]
struct ApiResponse {
    name: String,
    status: String,
    data: Vec<String>,
    error_message: Option<String>,
}

impl MessageChunk for ApiResponse {
    fn bad_chunk(error: String) -> Self {
        ApiResponse {
            name: "unknown".to_string(),
            status: "error".to_string(),
            data: vec![],
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

/// Simulates parsing log lines that might succeed or fail
fn parse_log_line(line: &str) -> Result<LogEntry, String> {
    let parts: Vec<&str> = line.split(" | ").collect();

    if parts.len() != 3 {
        return Err(format!(
            "Invalid log format: expected 3 parts, got {}",
            parts.len()
        ));
    }

    if parts[0].is_empty() {
        return Err("Empty timestamp".to_string());
    }

    Ok(LogEntry {
        timestamp: parts[0].to_string(),
        level: parts[1].to_string(),
        message: parts[2].to_string(),
        error_message: None,
    })
}

/// Simulates API calls that might succeed or fail
fn make_api_call(endpoint: &str) -> Result<ApiResponse, String> {
    match endpoint {
        "users" => Ok(ApiResponse {
            name: "Users API".to_string(),
            status: "200 OK".to_string(),
            data: vec!["user1".to_string(), "user2".to_string()],
            error_message: None,
        }),
        "orders" => Ok(ApiResponse {
            name: "Orders API".to_string(),
            status: "200 OK".to_string(),
            data: vec!["order1".to_string(), "order2".to_string()],
            error_message: None,
        }),
        "invalid" => Err("API endpoint not found".to_string()),
        "timeout" => Err("Request timeout after 30s".to_string()),
        _ => Err("Unknown API error".to_string()),
    }
}

/// Log File Processing Example
///
/// Demonstrates .on_chunk() unwrapping Result<LogEntry, String> into LogEntry values.
/// Shows how the stream contains clean LogEntry values instead of Results.
fn main() {
    println!("📄 Starting comprehensive on_chunk pattern demonstration...");

    // === LOG FILE PROCESSING EXAMPLE ===
    let log_lines = vec![
        "2024-01-15T10:30:00Z | INFO | Application started successfully",
        "2024-01-15T10:30:15Z | WARN | High memory usage detected",
        "invalid log line without proper format", // This will fail to parse
        "2024-01-15T10:30:30Z | ERROR | Database connection failed",
        " | INFO | Empty timestamp", // This will fail to parse
        "2024-01-15T10:30:45Z | INFO | System recovered successfully",
    ];

    // .on_chunk() unwraps Result<LogEntry, String> into LogEntry values
    let log_stream = AsyncStream::<LogEntry, 1024>::builder()
        .on_chunk(|result: Result<LogEntry, String>| -> LogEntry {
            match result {
                Ok(entry) => {
                    println!("✅ LOG on_chunk: Parsed entry - {} {}", entry.level, entry.message);
                    entry
                }
                Err(error) => {
                    println!("❌ LOG on_chunk: Parse error - {}", error);
                    LogEntry::bad_chunk(error)
                }
            }
        })
        .with_channel(move |sender| {
            println!("📝 Processing log lines...");
            for (i, line) in log_lines.into_iter().enumerate() {
                println!("🔍 Processing log line {}: {}", i + 1, 
                    if line.len() > 50 { format!("{}...", &line[..50]) } else { line.to_string() });
                let parse_result = parse_log_line(line);
                if sender.send_result(parse_result).is_err() { break; }
                thread::sleep(Duration::from_millis(100));
            }
            println!("🏁 Log processing completed");
        });

    // Collect and process log entries
    println!("\n📊 Processing log entries...");
    let log_entries: Vec<LogEntry> = log_stream.collect();

    println!("\n📋 Log Processing Summary:");
    for (i, entry) in log_entries.iter().enumerate() {
        if entry.is_error() {
            println!("  {}. ❌ Parse Error: {}", i + 1, entry.error().unwrap_or("Unknown error"));
        } else {
            println!("  {}. ✅ Log Entry: {} - {}", i + 1, entry.level, entry.message);
        }
    }

    // === API CALLS PROCESSING EXAMPLE ===
    let api_endpoints = vec!["users", "orders", "invalid", "timeout", "products"];

    let api_stream = AsyncStream::<ApiResponse, 1024>::builder()
        .on_chunk(|result: Result<ApiResponse, String>| -> ApiResponse {
            match result {
                Ok(response) => {
                    println!("✅ API on_chunk: Success - {} {}", response.name, response.status);
                    response
                }
                Err(error) => {
                    println!("❌ API on_chunk: Error - {}", error);
                    ApiResponse::bad_chunk(error)
                }
            }
        })
        .with_channel(move |sender| {
            println!("\n🌐 Making API calls...");
            for (i, endpoint) in api_endpoints.into_iter().enumerate() {
                println!("📡 Calling API {}: {}", i + 1, endpoint);
                let api_result = make_api_call(endpoint);
                if sender.send_result(api_result).is_err() { break; }
                thread::sleep(Duration::from_millis(200));
            }
            println!("🏁 API calls completed");
        });

    // Collect and process all API responses
    println!("\n📊 Processing API responses...");
    let responses: Vec<ApiResponse> = api_stream.collect();

    println!("\n📋 API Response Summary:");
    let mut successful = 0;
    let mut failed = 0;

    for (i, response) in responses.iter().enumerate() {
        if response.is_error() {
            println!("  {}. ❌ Error: {}", i + 1, response.error().unwrap_or("Unknown error"));
            failed += 1;
        } else {
            println!("  {}. ✅ Success: {} - {} ({} data items)", 
                i + 1, response.name, response.status, response.data.len());
            successful += 1;
        }
    }

    println!("\n📈 Final Results:");
    println!("  ✅ Successful operations: {}", successful + log_entries.iter().filter(|e| !e.is_error()).count());
    println!("  ❌ Failed operations: {}", failed + log_entries.iter().filter(|e| e.is_error()).count());
    println!("  📊 Total processed: {}", responses.len() + log_entries.len());

    println!("\n💡 Key Pattern: .on_chunk() converts Result<T, E> → T");
    println!("   • Success results become clean T values");
    println!("   • Error results become T::bad_chunk(error) values"); 
    println!("   • Stream contains only T values, never Result<T,E>");
    println!("🎉 Comprehensive on_chunk pattern demonstration complete!");
}