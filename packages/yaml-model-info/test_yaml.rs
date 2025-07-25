use yaml_model_info::models::YamlProvider;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ProviderInfo {
    provider: String,
    models: Vec<ModelConfig>}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelConfig {
    name: String,
    #[serde(default)]
    max_input_tokens: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    input_price: Option<f64>,
    #[serde(default)]
    output_price: Option<f64>,
    #[serde(default)]
    supports_vision: Option<bool>,
    #[serde(default)]
    supports_function_calling: Option<bool>,
    #[serde(default)]
    require_max_tokens: Option<bool>}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let yaml_content = std::fs::read_to_string(".yaml-cache/models.yaml")?;
    println!("Testing yyaml parsing with actual downloaded file...");
    println!("File size: {} bytes", yaml_content.len());
    
    // Test basic yyaml parsing as raw value
    match yyaml::from_str::<serde_json::Value>(&yaml_content) {
        Ok(value) => {
            println!("✅ Raw yyaml parse successful");
            if let Some(array) = value.as_array() {
                println!("   Root is array with {} items", array.len());
            } else if let Some(object) = value.as_object() {
                println!("   Root is object with keys: {:?}", object.keys().take(10).collect::<Vec<_>>());
            } else {
                println!("   Root is neither array nor object: {:?}", value);
            }
        },
        Err(e) => {
            println!("❌ Raw yyaml parse failed: {:?}", e);
            return Ok(());
        }
    }
    
    // Test with exact structs from working yyaml test
    match yyaml::from_str::<Vec<ProviderInfo>>(&yaml_content) {
        Ok(providers) => {
            println!("✅ SUCCESS with ProviderInfo: Parsed {} providers", providers.len());
            for provider in providers.iter().take(3) {
                println!("  Provider: {}, Models: {}", provider.provider, provider.models.len());
            }
        },
        Err(e) => {
            println!("❌ ERROR with ProviderInfo: {:?}", e);
        }
    }
    
    // Test with my structs
    match yyaml::from_str::<Vec<YamlProvider>>(&yaml_content) {
        Ok(providers) => {
            println!("✅ SUCCESS with YamlProvider: Parsed {} providers", providers.len());
            for provider in providers.iter().take(3) {
                println!("  Provider: {}, Models: {}", provider.provider, provider.models.len());
            }
        },
        Err(e) => {
            println!("❌ ERROR with YamlProvider: {:?}", e);
        }
    }
    
    Ok(())
}