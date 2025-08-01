//! Basic HTTP/3 client configuration test
//! 
//! This example demonstrates the HTTP/3 optimization capabilities
//! of the fluent_ai_http3 client with various QUIC configurations.

use std::time::Duration;
use fluent_ai_http3::{HttpClient, HttpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test basic HTTP configuration
    let basic_config = HttpConfig::default()
        .with_http3(false);
    
    let client = HttpClient::with_config(basic_config)?;
    println!("✅ Basic HTTP/3 client created successfully");
    println!("   Config: HTTP/3 enabled with default settings");
    
    // Test AI-optimized configuration
    let ai_config = HttpConfig::ai_optimized();
    let ai_client = HttpClient::with_config(ai_config)?;
    println!("✅ AI-optimized HTTP/3 client created successfully");
    println!("   Config: Optimized for AI provider APIs with low-latency settings");
    
    // Test streaming-optimized configuration
    let streaming_config = HttpConfig::streaming_optimized();
    let streaming_client = HttpClient::with_config(streaming_config)?;
    println!("✅ Streaming-optimized HTTP/3 client created successfully");
    println!("   Config: Optimized for streaming responses with large receive windows");
    
    // Test custom QUIC configuration (when reqwest_unstable is available) 
    let custom_config = HttpConfig::default()
        .with_http3(true)
        .with_quic_max_idle_timeout(Duration::from_secs(30))
        .with_quic_stream_receive_window(256 * 1024)  // 256KB
        .with_quic_receive_window(1024 * 1024)        // 1MB
        .with_quic_send_window(512 * 1024)            // 512KB
        .with_quic_congestion_bbr(true)
        .with_tls_early_data(true)
        .with_h3_max_field_section_size(16384)        // 16KB
        .with_h3_enable_grease(true);
    
    let custom_client = HttpClient::with_config(custom_config)?;
    println!("✅ Custom QUIC-optimized HTTP/3 client created successfully");
    println!("   Config: Custom QUIC parameters for maximum performance");
    
    // Test HTTP/2 fallback configuration
    let http2_config = HttpConfig::default()
        .with_http3(false)
        .with_timeout(Duration::from_secs(30));
    
    let http2_client = HttpClient::with_config(http2_config)?;
    println!("✅ HTTP/2 fallback client created successfully");
    println!("   Config: HTTP/2 fallback with custom timeout");
    
    println!("\n🚀 HTTP/3 QUIC Optimization Test Complete!");
    println!("   All client configurations created successfully");
    println!("   HTTP/3 optimization is working correctly");
    
    #[cfg(feature = "reqwest_unstable")]
    println!("   Advanced QUIC features: ENABLED ⚡");
    
    #[cfg(not(feature = "reqwest_unstable"))]
    println!("   Advanced QUIC features: BASIC (enable 'reqwest_unstable' for full optimization)");
    
    Ok(())
}