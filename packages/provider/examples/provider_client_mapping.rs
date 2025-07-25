//! Example demonstrating provider-to-client mapping
//!
//! This example shows how to use the zero-allocation factory pattern to map
//! provider enum variants to their corresponding client implementations.

use std::sync::Arc;

use fluent_ai_provider::{ClientConfig, ClientFactoryResult, Providers, UnifiedClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Provider-to-Client Mapping Example");
    println!("=====================================\n");

    // Example 1: Direct provider-to-client mapping
    println!("📋 Example 1: Direct Provider-to-Client Mapping");
    println!("-----------------------------------------------");

    // Create OpenAI client
    let openai_config = ClientConfig {
        api_key: Some("sk-test-key".to_string()),
        base_url: Some("https://api.openai.com/v1".to_string()),
        ..Default::default()
    };

    match Providers::Openai.create_client(openai_config).await {
        Ok(client) => {
            println!("✅ OpenAI client created successfully");
            println!("   Provider: {}", client.provider_name());

            // Test connection (will fail with fake key but demonstrates the pattern)
            match client.test_connection().await {
                Ok(()) => println!("✅ Connection test passed"),
                Err(e) => println!("❌ Connection test failed (expected with fake key): {}", e)}
        }
        Err(e) => println!("❌ Failed to create OpenAI client: {}", e)}

    // Create Anthropic client
    let anthropic_config = ClientConfig {
        api_key: Some("sk-ant-test-key".to_string()),
        base_url: Some("https://api.anthropic.com".to_string()),
        ..Default::default()
    };

    match Providers::Claude.create_client(anthropic_config).await {
        Ok(client) => {
            println!("✅ Anthropic client created successfully");
            println!("   Provider: {}", client.provider_name());

            // Test connection (will fail with fake key but demonstrates the pattern)
            match client.test_connection().await {
                Ok(()) => println!("✅ Connection test passed"),
                Err(e) => println!("❌ Connection test failed (expected with fake key): {}", e)}
        }
        Err(e) => println!("❌ Failed to create Anthropic client: {}", e)}

    println!("\n📋 Example 2: Environment Variable Configuration");
    println!("-----------------------------------------------");

    // Example 2: Environment variable configuration
    for provider in [Providers::Openai, Providers::Claude] {
        let env_vars = provider.required_env_vars();
        println!("🔧 {} requires: {:?}", provider.name(), env_vars);

        // Try to create client from environment
        match provider.create_client_from_env().await {
            Ok(client) => {
                println!("✅ {} client created from environment", provider.name());
                println!("   Provider: {}", client.provider_name());
            }
            Err(e) => {
                println!("❌ {} client creation failed: {}", provider.name(), e);
            }
        }
    }

    println!("\n📋 Example 3: Provider Name Mapping");
    println!("-----------------------------------");

    // Example 3: Provider name mapping
    let provider_names = vec![
        "openai",
        "anthropic",
        "claude",
        "gemini",
        "mistral",
        "groq",
        "perplexity",
        "xai",
    ];

    for name in provider_names {
        match Providers::from_name(name) {
            Some(provider) => {
                println!("✅ {} -> {:?}", name, provider);
                println!("   Canonical name: {}", provider.name());
                println!("   Supported: {}", provider.is_supported());
            }
            None => {
                println!("❌ Unknown provider: {}", name);
            }
        }
    }

    println!("\n📋 Example 4: Convenience Functions");
    println!("----------------------------------");

    // Example 4: Convenience functions

    // Direct OpenAI client creation
    match Providers::openai_client("sk-test-key".to_string()).await {
        Ok(client) => {
            println!("✅ OpenAI client created with convenience function");
            println!("   Provider: {}", client.provider_name());
            println!(
                "   Available models: {:?}",
                client.get_models().await.unwrap_or_default()
            );
        }
        Err(e) => println!("❌ OpenAI client creation failed: {}", e)}

    // Direct Anthropic client creation
    match Providers::anthropic_client("sk-ant-test-key".to_string()).await {
        Ok(client) => {
            println!("✅ Anthropic client created with convenience function");
            println!("   Provider: {}", client.provider_name());
            println!(
                "   Available models: {:?}",
                client.get_models().await.unwrap_or_default()
            );
        }
        Err(e) => println!("❌ Anthropic client creation failed: {}", e)}

    // Client creation from name with config
    let config = ClientConfig {
        api_key: Some("test-key".to_string()),
        ..Default::default()
    };

    match Providers::from_name_with_config("openai", config).await {
        Ok(client) => {
            println!("✅ Client created from name 'openai'");
            println!("   Provider: {}", client.provider_name());
        }
        Err(e) => println!("❌ Client creation from name failed: {}", e)}

    println!("\n📋 Example 5: Unified Client Interface");
    println!("-------------------------------------");

    // Example 5: Unified client interface
    let clients: Vec<Arc<dyn UnifiedClient>> = vec![
        // These would work with real API keys
        // Providers::openai_client("real-openai-key".to_string()).await?,
        // Providers::anthropic_client("real-anthropic-key".to_string()).await?,
    ];

    for client in clients {
        println!(
            "🔧 Testing unified interface for {}",
            client.provider_name()
        );

        match client.get_models().await {
            Ok(models) => {
                println!("   Available models: {:?}", models);
            }
            Err(e) => {
                println!("   Failed to get models: {}", e);
            }
        }

        // Example completion request
        let completion_request = serde_json::json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ],
            "max_tokens": 100
        });

        match client.send_completion(&completion_request).await {
            Ok(response) => {
                println!("   ✅ Completion successful: {:?}", response);
            }
            Err(e) => {
                println!("   ❌ Completion failed: {}", e);
            }
        }
    }

    println!("\n📋 Example 6: Provider Feature Matrix");
    println!("-------------------------------------");

    // Example 6: Provider feature matrix
    let all_providers = vec![
        Providers::Openai,
        Providers::Claude,
        Providers::Gemini,
        Providers::Mistral,
        Providers::Groq,
        Providers::Perplexity,
        Providers::Xai,
    ];

    println!("Provider Feature Matrix:");
    println!("========================");
    println!("| Provider    | Supported | Env Vars Required  |");
    println!("| ----------- | --------- | ------------------ |");

    for provider in all_providers {
        let supported = if provider.is_supported() {
            "✅"
        } else {
            "❌"
        };
        let env_vars = provider.required_env_vars().join(", ");
        println!(
            "| {:10} | {:8} | {:18} |",
            provider.name(),
            supported,
            env_vars
        );
    }

    println!("\n🎯 Summary");
    println!("==========");
    println!("The provider-to-client mapping system provides:");
    println!("1. ✅ Zero-allocation factory pattern");
    println!("2. ✅ Type-safe client instantiation");
    println!("3. ✅ Unified async interface");
    println!("4. ✅ QUIC/HTTP3 prioritization");
    println!("5. ✅ Comprehensive error handling");
    println!("6. ✅ Environment variable configuration");
    println!("7. ✅ Provider name resolution");
    println!("8. ✅ Feature detection and capability queries");

    Ok(())
}

/// Helper function to demonstrate streaming completion
async fn _demonstrate_streaming(client: Arc<dyn UnifiedClient>) -> ClientFactoryResult<()> {
    let streaming_request = serde_json::json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Count from 1 to 10"}
        ],
        "max_tokens": 100,
        "stream": true
    });

    let mut stream = client.send_streaming_completion(&streaming_request).await?;

    println!("📡 Streaming completion:");
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                println!("   📦 Chunk: {:?}", chunk);
            }
            Err(e) => {
                println!("   ❌ Stream error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

/// Helper function to demonstrate embedding generation
async fn _demonstrate_embedding(client: Arc<dyn UnifiedClient>) -> ClientFactoryResult<()> {
    let embedding_request = serde_json::json!({
        "model": "text-embedding-3-small",
        "input": "This is a test sentence for embedding generation.",
        "encoding_format": "float"
    });

    match client.send_embedding(&embedding_request).await {
        Ok(response) => {
            println!("📊 Embedding generated: {:?}", response);
        }
        Err(e) => {
            println!("❌ Embedding failed: {}", e);
        }
    }

    Ok(())
}
