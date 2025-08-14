//! # AsyncStream collect Pattern
//!
//! Demonstrates the collect() method that gathers ALL stream values (including error chunks).
//! Unlike collect_or_else(), this collects everything and lets you process mixed results.

use fluent_ai_async::prelude::*;

#[derive(Debug, Clone, Default)]
struct Embedding {
    id: usize,
    text: String,
    vector: Vec<f32>,
    model: String,
    error_message: Option<String>,
}

impl MessageChunk for Embedding {
    fn bad_chunk(error: String) -> Self {
        Embedding {
            id: 0,
            text: "[ERROR]".to_string(),
            vector: vec![],
            model: "error".to_string(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl Embedding {
    fn new(id: usize, text: &str) -> Self {
        // Mock embedding generation
        let vector = text
            .chars()
            .take(8)
            .map(|c| (c as u8 as f32) / 255.0)
            .collect();

        Self {
            id,
            text: text.to_string(),
            vector,
            model: "mock-embedding-v1".to_string(),
            error_message: None,
        }
    }
}

/// Batch Embedding Generation Example
///
/// Shows how to collect all stream results into a batch using collect().
/// This pattern is used when you need all results before proceeding,
/// such as batch processing or final aggregation.
fn main() {
    let input_texts = vec![
        "AsyncStream provides blazing-fast streaming",
        "Zero allocation in hot paths",
        "", // Empty text - will cause error
        "Producer-consumer patterns made easy",
        "This text is way too long and will exceed our mock embedding limit causing a processing error",
        "Real-world streaming solutions",
    ];

    println!("🧠 Starting batch embedding generation...");
    println!("📝 Processing {} text inputs", input_texts.len());

    let embedding_stream = AsyncStream::<Embedding, 1024>::with_channel(move |sender| {
        println!("⚙️  Embedding generator started");

        for (i, text) in input_texts.into_iter().enumerate() {
            println!(
                "🔤 Generating embedding {}: '{}'",
                i + 1,
                if text.len() > 30 {
                    format!("{}...", &text[..30])
                } else {
                    text.to_string()
                }
            );

            // Simulate embedding generation with error handling
            std::thread::sleep(Duration::from_millis(100));

            // Simulate various failure conditions
            if text.is_empty() {
                let error_msg = format!("Empty text input at position {}", i + 1);
                emit!(sender, Embedding::bad_chunk(error_msg));
            } else if text.len() > 80 {
                let error_msg =
                    format!("Text too long ({} chars) at position {}", text.len(), i + 1);
                emit!(sender, Embedding::bad_chunk(error_msg));
            } else {
                let embedding = Embedding::new(i + 1, text);
                emit!(sender, embedding);
            }
        }

        println!("✨ All embeddings generated!");
    });

    // Use collect() to get ALL values (both good and error chunks)
    println!("📊 Using collect() to gather ALL results...");
    let all_results: Vec<Embedding> = embedding_stream.collect();

    // Manually process mixed good/error results
    println!("\n🎯 Processing mixed batch results:");
    let mut successful_embeddings = Vec::new();
    let mut error_chunks = Vec::new();

    for result in all_results {
        if result.is_error() {
            println!("   ❌ Error: {}", result.error().unwrap_or("Unknown error"));
            error_chunks.push(result);
        } else {
            println!(
                "   ✅ Embedding ID {}: '{}' ({} dims)",
                result.id,
                if result.text.len() > 30 {
                    format!("{}...", &result.text[..30])
                } else {
                    result.text.clone()
                },
                result.vector.len()
            );
            successful_embeddings.push(result);
        }
    }

    // Show statistics
    println!("\n📈 Batch Statistics:");
    println!(
        "   ✅ Successful embeddings: {}",
        successful_embeddings.len()
    );
    println!("   ❌ Error chunks: {}", error_chunks.len());
    println!(
        "   📊 Total processed: {}",
        successful_embeddings.len() + error_chunks.len()
    );

    if !successful_embeddings.is_empty() {
        let avg_dims: f32 = successful_embeddings
            .iter()
            .map(|e| e.vector.len() as f32)
            .sum::<f32>()
            / successful_embeddings.len() as f32;
        println!("   📏 Average dimensions: {:.1}", avg_dims);
    }

    println!(
        "\n💡 Note: collect() gets ALL values, unlike collect_or_else() which stops on first error"
    );
    println!("🎉 Batch collection pattern demonstration complete!");
}
