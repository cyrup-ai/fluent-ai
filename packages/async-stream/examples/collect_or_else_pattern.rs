//! # AsyncStream collect_or_else Pattern
//!
//! Demonstrates the collect_or_else method with proper cyrup_sugars ChunkHandler pattern.
//! Shows how to collect all good chunks OR enter error handling on the first bad chunk.

use fluent_ai_async::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ParsedData {
    name: String,
    age: u32,
    city: String,
    score: f64,
    error_message: Option<String>,
}

impl MessageChunk for ParsedData {
    fn bad_chunk(error: String) -> Self {
        ParsedData {
            name: "[ERROR]".to_string(),
            age: 0,
            city: "".to_string(),
            score: 0.0,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

/// JSON Parsing with Error Recovery Example
///
/// Shows how to handle parsing errors gracefully using on_error handlers
/// while maintaining the streams-only architecture (no Result<T, E> in streams).
fn main() {
    let json_inputs = vec![
        r#"{"name": "Alice", "age": 30, "city": "New York", "score": 95.5}"#,
        r#"{"name": "Bob", "age": 25, "city": "San Francisco", "score": 87.2}"#,
        r#"{"invalid": "json", "missing": "fields"}"#, // This will trigger error handling
        r#"{"name": "Charlie", "age": 35, "city": "Boston", "score": 92.1}"#,
        r#"invalid json syntax"#, // This will also trigger error handling
        r#"{"name": "Diana", "age": 28, "city": "Seattle", "score": 89.8}"#,
    ];

    println!("📄 Starting JSON parsing pipeline with error recovery...");
    println!("🔍 Processing {} JSON inputs", json_inputs.len());

    let parse_stream = AsyncStream::<ParsedData, 1024>::builder()
        .on_chunk(|result: Result<ParsedData, String>| -> ParsedData {
            match result {
                Ok(data) => {
                    // Process successfully parsed data
                    println!("✅ Successfully parsed: {} from {}", data.name, data.city);
                    data
                }
                Err(error) => {
                    // Handle parsing errors and return bad chunk
                    eprintln!("🔥 JSON parsing error: {}", error);
                    eprintln!("🛠️  Creating error chunk...");
                    ParsedData::bad_chunk(error)
                }
            }
        })
        .with_channel(move |sender| {
            println!("⚙️  JSON parser started");

            for (i, json_str) in json_inputs.into_iter().enumerate() {
                println!(
                    "📝 Parsing JSON {}: {}",
                    i + 1,
                    if json_str.len() > 40 {
                        format!("{}...", &json_str[..40])
                    } else {
                        json_str.to_string()
                    }
                );

                // Attempt to parse JSON
                match serde_json::from_str::<ParsedData>(json_str) {
                    Ok(parsed_data) => {
                        // Successfully parsed - emit the data
                        emit!(sender, parsed_data);
                    }
                    Err(parse_error) => {
                        // Parsing failed - emit error chunk
                        let error_msg = format!("Failed to parse JSON {}: {}", i + 1, parse_error);
                        emit!(sender, ParsedData::bad_chunk(error_msg));
                    }
                }

                // Simulate processing delay
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            println!("🎯 JSON parsing pipeline finished!");
        });

    // Use collect_or_else to collect all good chunks OR handle first error
    println!("\n📊 Using collect_or_else pattern:");

    let results: Vec<ParsedData> = parse_stream.collect_or_else(|error_msg| {
        println!(
            "🚨 collect_or_else detected first error chunk: {}",
            error_msg
        );
        println!("🛑 Entering error recovery mode instead of collecting...");

        // Return empty vec to indicate error condition
        Vec::new()
    });

    if results.is_empty() {
        println!("❌ Collection failed due to error chunk - no results collected");
        println!("💡 This demonstrates collect_or_else entering error handler on first bad chunk");
    } else {
        println!("✅ Successfully collected {} good chunks:", results.len());
        for (i, data) in results.iter().enumerate() {
            println!(
                "  {}. {} (age {}) from {} - score: {}",
                i + 1,
                data.name,
                data.age,
                data.city,
                data.score
            );
        }
    }

    println!("\n🎉 collect_or_else pattern demonstration complete!");
}
