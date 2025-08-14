//! # AsyncStream emit! Macro Pattern
//!
//! Demonstrates the emit! macro for ergonomic stream production.
//! Shows how emit! gracefully handles receiver drops and eliminates error handling boilerplate.

use fluent_ai_async::prelude::*;

#[derive(Debug, Clone, Default)]
struct SystemEvent {
    timestamp: u64,
    level: String,
    message: String,
    component: String,
    error_message: Option<String>,
}

impl MessageChunk for SystemEvent {
    fn bad_chunk(error: String) -> Self {
        SystemEvent {
            timestamp: 0,
            level: "ERROR".to_string(),
            message: "[SYSTEM_ERROR]".to_string(),
            component: "system".to_string(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl SystemEvent {
    fn new(level: &str, message: &str, component: &str) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        SystemEvent {
            timestamp,
            level: level.to_string(),
            message: message.to_string(),
            component: component.to_string(),
            error_message: None,
        }
    }
}

/// System Event Monitoring Example
///
/// Demonstrates the emit! macro for ergonomic event streaming.
/// Shows how emit! gracefully handles early termination without error handling boilerplate.
fn main() {
    println!("🖥️  Starting system event monitoring...");

    // Create event stream that might be terminated early
    let event_stream = AsyncStream::<SystemEvent, 1024>::with_channel(move |sender| {
        println!("📡 Event monitor started");

        // Simulate various system events
        let events = vec![
            ("INFO", "System startup initiated", "kernel"),
            ("INFO", "Loading configuration", "config"),
            ("WARN", "High memory usage detected", "memory"),
            ("INFO", "Network interface up", "network"),
            ("ERROR", "Database connection failed", "database"),
            ("INFO", "Retry mechanism activated", "database"),
            ("INFO", "Connection restored", "database"),
            ("INFO", "All systems operational", "system"),
        ];

        for (i, (level, message, component)) in events.into_iter().enumerate() {
            println!("📊 Generating event {}: {} - {}", i + 1, level, message);

            // Use emit! macro - gracefully handles receiver drops
            emit!(sender, SystemEvent::new(level, message, component));

            // Simulate event timing
            thread::sleep(Duration::from_millis(200));
        }

        println!("✅ Event monitoring complete");
    });

    // Collect all events - demonstrate successful streaming with emit!
    println!("🔍 Collecting all system events...");
    let events: Vec<SystemEvent> = event_stream.collect();

    println!("\n📋 System Event Log:");
    for (i, event) in events.iter().enumerate() {
        let status_icon = match event.level.as_str() {
            "INFO" => "ℹ️",
            "WARN" => "⚠️",
            "ERROR" => "❌",
            _ => "📝",
        };

        println!(
            "  {}. {} [{}] {}: {}",
            i + 1,
            status_icon,
            event.component.to_uppercase(),
            event.level,
            event.message
        );
    }

    println!("\n📊 Summary:");
    let info_count = events.iter().filter(|e| e.level == "INFO").count();
    let warn_count = events.iter().filter(|e| e.level == "WARN").count();
    let error_count = events.iter().filter(|e| e.level == "ERROR").count();

    println!("  ℹ️  INFO events: {}", info_count);
    println!("  ⚠️  WARN events: {}", warn_count);
    println!("  ❌ ERROR events: {}", error_count);
    println!("  📊 Total events: {}", events.len());

    println!("\n💡 Note: emit! macro provides clean, ergonomic streaming syntax");
    println!("🎉 emit! macro pattern demonstration complete!");
}
