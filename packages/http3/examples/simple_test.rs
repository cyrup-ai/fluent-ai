//! Simple HTTP client test - basic functionality
//! 
//! This example tests basic HTTP client creation and configuration
//! without complex features like HTTP/3 or server setup.

use std::time::Duration;
use fluent_ai_http3::{HttpClient, HttpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing basic HTTP client functionality...");
    
    // Test 1: Default client creation
    let default_client = HttpClient::default();
    println!("✅ Default HTTP client created successfully");
    
    // Test 2: Basic configuration
    let basic_config = HttpConfig::default()
        .with_timeout(Duration::from_secs(30))
        .with_connect_timeout(Duration::from_secs(10))
        .with_compression(true);
    
    let basic_client = HttpClient::with_config(basic_config)?;
    println!("✅ Basic configured HTTP client created successfully");
    
    // Test 3: AI-optimized configuration (predefined)
    let ai_config = HttpConfig::ai_optimized();
    let ai_client = HttpClient::with_config(ai_config)?;
    println!("✅ AI-optimized HTTP client created successfully");
    
    // Test 4: Streaming-optimized configuration (predefined)  
    let streaming_config = HttpConfig::streaming_optimized();
    let streaming_client = HttpClient::with_config(streaming_config)?;
    println!("✅ Streaming-optimized HTTP client created successfully");
    
    // Test 5: HTTP/3 disabled explicitly
    let http2_config = HttpConfig::default()
        .with_http3(false)
        .with_timeout(Duration::from_secs(60));
    
    let http2_client = HttpClient::with_config(http2_config)?;
    println!("✅ HTTP/2 client created successfully");
    
    // Test 6: Configuration with reqwest_unstable features
    #[cfg(feature = "reqwest_unstable")]
    {
        let advanced_config = HttpConfig::default()
            .with_http3(true)
            .with_quic_max_idle_timeout(Duration::from_secs(30));
            
        let advanced_client = HttpClient::with_config(advanced_config)?;
        println!("✅ Advanced HTTP/3 client with QUIC features created successfully");
        println!("   Advanced QUIC features: ENABLED ⚡");
    }
    
    #[cfg(not(feature = "reqwest_unstable"))]
    {
        println!("   Advanced QUIC features: BASIC (reqwest_unstable not enabled)");
    }
    
    // Test 7: Client statistics
    let stats = default_client.stats_snapshot();
    println!("✅ Client statistics retrieved: {} requests", stats.request_count);
    
    println!("\n🎯 All basic HTTP client tests passed!");
    println!("   reqwest_unstable feature: {}", if cfg!(feature = "reqwest_unstable") { "ENABLED" } else { "DISABLED" });
    
    Ok(())
}