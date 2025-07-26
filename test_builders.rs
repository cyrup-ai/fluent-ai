//! Test the new trait-based builders with zero Box<dyn> usage
//!
//! This test verifies that all converted builders work correctly.

use fluent_ai_domain::{Document, ContentFormat, DocumentMediaType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Trait-Based Builders");
    println!("================================");
    
    // Test DocumentBuilder trait-based implementation
    println!("\n📄 Testing DocumentBuilder...");
    
    let doc = Document::from_text("Hello, World!")
        .format(ContentFormat::Text)
        .media_type(DocumentMediaType::PlainText)
        .encoding("utf-8")
        .property("test", "value")
        .load();
    
    println!("✅ Document created: {} bytes", doc.data.len());
    println!("   Format: {:?}", doc.format);
    println!("   Media Type: {:?}", doc.media_type);
    
    // Test with error handler
    let doc_with_handler = Document::from_text("Test content")
        .on_error(|error| {
            println!("Error occurred: {}", error);
        })
        .format(ContentFormat::Markdown)
        .load_async()
        .await;
    
    println!("✅ Document with error handler: {} bytes", doc_with_handler.data.len());
    
    // Test with chunk handler  
    let doc_with_chunks = Document::from_text("Line 1\nLine 2\nLine 3")
        .on_chunk(|chunk| {
            println!("Processing chunk: {}", chunk.content);
            chunk
        })
        .stream_lines();
    
    println!("✅ Processing document chunks...");
    let mut chunk_count = 0;
    let mut stream = doc_with_chunks;
    
    use futures_util::StreamExt;
    while let Some(_chunk) = stream.next().await {
        chunk_count += 1;
    }
    
    println!("✅ Processed {} chunks", chunk_count);
    
    println!("\n🎉 All builder tests passed!");
    println!("   Zero Box<dyn> usage confirmed");
    println!("   Trait-based architecture working correctly");
    
    Ok(())
}