//! # AsyncStream with_channel Pattern
//!
//! Demonstrates the real-world producer/consumer pattern using `with_channel`.
//! Shows how to create streams and consume them using the fluent-ai async ecosystem.

use fluent_ai_async::prelude::*;

#[derive(Debug, Clone, Default)]
struct HttpChunk {
    data: String,
    chunk_id: usize,
    is_error: bool,
}

impl MessageChunk for HttpChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            data: format!("ERROR: {}", error),
            chunk_id: 0,
            is_error: true,
        }
    }

    fn is_error(&self) -> bool {
        self.is_error
    }

    fn error(&self) -> Option<&str> {
        if self.is_error {
            Some(&self.data)
        } else {
            None
        }
    }
}

/// Real-World HTTP Response Streaming Example
///
/// Demonstrates the actual with_channel pattern used throughout fluent-ai:
/// - Producer runs in background thread with emit!()
/// - Consumer uses AsyncStream::with_channel for nested streaming
/// - Zero-allocation streaming with crossbeam primitives
/// - No external async runtimes - pure fluent-ai architecture
fn main() {
    println!("🌐 Real-World AsyncStream with_channel Pattern\n");

    let response_data = vec![
        "HTTP/1.1 200 OK\r\n",
        "Content-Type: application/json\r\n",
        "Content-Length: 58\r\n",
        "\r\n",
        "{\"status\": \"success\", \"data\": \"streaming response\"}",
    ];

    // PATTERN: AsyncStream::with_channel for producer setup
    println!("🚀 Creating HTTP response stream with background producer...");
    let http_stream = AsyncStream::<HttpChunk, 1024>::with_channel(move |sender| {
        println!("📡 Producer thread: Starting HTTP response streaming...");

        for (i, chunk_data) in response_data.into_iter().enumerate() {
            println!(
                "📤 Producer: Sending chunk {}: {}",
                i + 1,
                chunk_data.trim()
            );
            std::thread::sleep(std::time::Duration::from_millis(100)); // Simulate network delay

            let http_chunk = HttpChunk {
                data: chunk_data.to_string(),
                chunk_id: i + 1,
                is_error: false,
            };
            emit!(sender, http_chunk);
        }

        println!("✅ Producer: All chunks sent!");
    });

    // CONSUMPTION: Collect all chunks (this blocks until producer completes)
    println!("📥 Consumer: Collecting HTTP response chunks...");
    let chunks: Vec<HttpChunk> = http_stream.collect();

    println!("\n📊 Streaming Results:");
    println!("   • Total chunks received: {}", chunks.len());
    println!(
        "   • Total bytes: {}",
        chunks.iter().map(|c| c.data.len()).sum::<usize>()
    );

    println!("\n✅ Received Chunks:");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  {}. {}", i + 1, chunk.data.trim());
    }

    // PATTERN 2: Chain processing with another with_channel
    println!("\n🔄 Chaining with processing stream...");
    let processed_stream = AsyncStream::<HttpChunk, 1024>::with_channel(move |processed_sender| {
        println!("⚙️  Processing thread: Transforming chunks...");

        for (i, chunk) in chunks.into_iter().enumerate() {
            let processed_chunk = HttpChunk {
                data: format!("PROCESSED[{}]: {}", i + 1, chunk.data.trim()),
                chunk_id: i + 1,
                is_error: false,
            };

            println!("🔧 Processing: {}", processed_chunk.data);
            emit!(processed_sender, processed_chunk);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        println!("✅ Processing complete!");
    });

    // Collect processed results
    let processed_results: Vec<HttpChunk> = processed_stream.collect();

    println!("\n🎯 Final Processed Results:");
    for (i, result) in processed_results.iter().enumerate() {
        println!("  {}. {}", i + 1, result.data);
    }

    println!("\n💡 Real-World Usage Notes:");
    println!("   • with_channel creates producer threads with emit!() macro");
    println!("   • Consumers use into_iter() or .collect() for synchronous processing");
    println!("   • Chain multiple streams using nested with_channel patterns");
    println!("   • Zero-allocation streaming with crossbeam primitives");
    println!("   • No external async runtimes - pure fluent-ai architecture");
    println!("🎉 Real-world with_channel pattern demonstration complete!");
}
