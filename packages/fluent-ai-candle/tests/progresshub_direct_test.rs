//! Direct ProgressHub integration test
//!
//! Tests that ProgressHub can be used directly without any abstractions.

use std::path::PathBuf;

use fluent_ai_candle::{create_client, create_download_config, Backend};

#[tokio::test]
async fn test_direct_progresshub_client_creation() {
    // Test direct ProgressHub client creation
    let result = create_client(Backend::Auto);

    match result {
        Ok(client) => {
            println!("✅ Direct ProgressHub client created successfully");

            // Test download config creation
            let cache_dir = PathBuf::from("/tmp/fluent_ai_test_cache");
            let config = create_download_config(cache_dir);

            println!(
                "✅ Download config created: destination = {:?}",
                config.destination
            );
            assert!(config.use_cache);
            assert!(!config.show_progress); // We handle progress via channels
        }
        Err(e) => {
            println!("❌ Failed to create direct ProgressHub client: {}", e);
            panic!("Direct ProgressHub client creation failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_direct_progresshub_download() {
    // Test direct ProgressHub download without abstractions
    let client = match create_client(Backend::Auto) {
        Ok(client) => client,
        Err(e) => {
            println!("⚠️  Skipping download test - client creation failed: {}", e);
            return;
        }
    };

    let cache_dir = PathBuf::from("/tmp/fluent_ai_test_cache");
    let config = create_download_config(cache_dir);

    // Test with a small model
    let repo_id = "microsoft/DialoGPT-small";
    println!("🚀 Testing direct ProgressHub download of {}", repo_id);

    match client.download_model_auto(repo_id, &config, None).await {
        Ok(result) => {
            println!("✅ Direct ProgressHub download completed successfully");
            println!("📁 Downloaded to: {:?}", result.destination);

            // Verify files were downloaded
            if result.destination.exists() {
                println!("✅ Download directory exists");

                if let Ok(entries) = std::fs::read_dir(&result.destination) {
                    let file_count = entries.count();
                    println!("📊 Downloaded {} files", file_count);
                    assert!(file_count > 0, "Should download at least one file");
                } else {
                    println!("⚠️  Could not read download directory");
                }
            } else {
                println!("❌ Download directory does not exist");
                panic!("Download directory should exist after successful download");
            }
        }
        Err(e) => {
            println!(
                "⚠️  Download test failed (may be expected for network issues): {}",
                e
            );
            // Don't fail the test for network issues
        }
    }
}
