//! # AsyncStream on_chunk Pattern
//!
//! Demonstrates .on_chunk() for unwrapping Result<T, E> into T values in streams.
//! Shows how .on_chunk() runs on each element and unwraps Results into clean stream values.

use fluent_ai_async::prelude::*;

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

/// Log File Processing Example
///
/// Demonstrates .on_chunk() unwrapping Result<LogEntry, String> into LogEntry values.
/// Shows how the stream contains clean LogEntry values instead of Results.
fn main() {
    println!("📄 Starting log file processing with on_chunk pattern...");

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
                    println!(
                        "✅ on_chunk: Parsed log entry - {} {}",
                        entry.level, entry.message
                    );
                    entry
                }
                Err(error) => {
                    println!("❌ on_chunk: Parse error - {}", error);
                    // Convert parse error to bad chunk
                    LogEntry::bad_chunk(error)
                }
            }
        })
        .with_channel(move |sender| {
            println!("📝 Processing log lines...");

            for (i, line) in log_lines.into_iter().enumerate() {
                println!(
                    "🔍 Processing line {}: {}",
                    i + 1,
                    if line.len() > 50 {
                        format!("{}...", &line[..50])
                    } else {
                        line.to_string()
                    }
                );

                // Parse the log line (might succeed or fail)
                let parse_result = parse_log_line(line);

                // Send the Result - .on_chunk() will unwrap it into LogEntry
                if sender.send_result(parse_result).is_err() {
                    break; // Stream closed
                }

                thread::sleep(Duration::from_millis(200));
            }

            println!("🏁 Log processing completed");
        });

    // Collect and process all responses
    println!("\n📊 Processing API responses...");
    let responses: Vec<ApiResponse> = api_stream.collect();

    println!("\n📋 API Response Summary:");
    let mut successful = 0;
    let mut failed = 0;

    for (i, response) in responses.iter().enumerate() {
        if response.is_error() {
            println!(
                "  {}. ❌ Error: {}",
                i + 1,
                response.error().unwrap_or("Unknown error")
            );
            failed += 1;
        } else {
            println!(
                "  {}. ✅ Success: {} - {} ({} data items)",
                i + 1,
                response.name,
                response.status,
                response.data.len()
            );
            successful += 1;
        }
    }

    println!("\n📈 Results:");
    println!("  ✅ Successful API calls: {}", successful);
    println!("  ❌ Failed API calls: {}", failed);
    println!("  📊 Total processed: {}", responses.len());

    println!("\n💡 Note: .on_chunk() converted Result<ApiResponse, String> → ApiResponse");
    println!("🎉 on_chunk pattern demonstration complete!");
}
