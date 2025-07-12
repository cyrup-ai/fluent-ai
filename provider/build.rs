// Removed proc_macro2 and quote - using direct string generation for better control
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Duration;
use reqwest::Client;
use tokio;

/// The models.yaml is a top-level array of providers, not a struct with providers field
type ModelYaml = Vec<ProviderInfo>;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ProviderInfo {
    provider: String,
    models: Vec<ModelConfig>,
}

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
    require_max_tokens: Option<bool>,
}

/// Surgical file operations with zero-allocation design
struct SurgicalFileManager {
    auto_gen_start: &'static str,
    auto_gen_end: &'static str,
}

impl SurgicalFileManager {
    #[inline(always)]
    const fn new() -> Self {
        Self {
            auto_gen_start: "// AUTO-GENERATED START",
            auto_gen_end: "// AUTO-GENERATED END",
        }
    }

    // Removed unused extract_existing_variants function

    /// Surgically update file preserving hand-written sections
    fn update_file_surgical(
        &self,
        file_path: &Path,
        new_content: &str,
        preserve_hand_written: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let updated_content = if preserve_hand_written && file_path.exists() {
            let existing = fs::read_to_string(file_path)?;
            self.replace_auto_generated_section(&existing, new_content)?
        } else {
            new_content.to_string()
        };

        let mut file = fs::File::create(file_path)?;
        file.write_all(updated_content.as_bytes())?;
        file.flush()?;

        println!("cargo:rerun-if-changed={}", file_path.display());
        Ok(())
    }

    fn replace_auto_generated_section(
        &self,
        existing: &str,
        new_content: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        if let Some(start_pos) = existing.find(self.auto_gen_start) {
            if let Some(end_pos) = existing.find(self.auto_gen_end) {
                let before = &existing[..start_pos];
                let after = &existing[end_pos + self.auto_gen_end.len()..];
                return Ok(format!("{}{}{}", before, new_content, after));
            }
        }

        // If no auto-generated section found, append the new content
        Ok(format!("{}\n{}", existing, new_content))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=models.yaml");
    println!("cargo:rerun-if-changed=build.rs");
    
    let _rt = tokio::runtime::Handle::current();
    
    // Download latest models.yaml
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    
    let response = client
        .get("https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json")
        .send()
        .await?;
    
    if !response.status().is_success() {
        eprintln!("Warning: Failed to download models file, using existing local copy if available");
        if !Path::new("models.yaml").exists() {
            return Err("No models.yaml found and download failed".into());
        }
    } else {
        let content = response.text().await?;
        
        // Parse JSON and convert to our YAML format
        let json_data: serde_json::Value = serde_json::from_str(&content)?;
        let mut providers: Vec<ProviderInfo> = Vec::new();
        
        if let Some(obj) = json_data.as_object() {
            for (model_name, model_data) in obj {
                if let Some(model_obj) = model_data.as_object() {
                    // Extract provider from model name (e.g., "gpt-4" -> "openai")
                    let provider = if model_name.starts_with("gpt-") || model_name.starts_with("text-") {
                        "openai"
                    } else if model_name.starts_with("claude-") {
                        "anthropic"
                    } else if model_name.contains("gemini") || model_name.contains("palm") {
                        "google"
                    } else if model_name.contains("llama") {
                        "meta"
                    } else {
                        "unknown"
                    };
                    
                    let model_config = ModelConfig {
                        name: model_name.clone(),
                        max_input_tokens: model_obj.get("max_input_tokens").and_then(|v| v.as_u64()),
                        max_output_tokens: model_obj.get("max_output_tokens").and_then(|v| v.as_u64()),
                        input_price: model_obj.get("input_cost_per_token").and_then(|v| v.as_f64()),
                        output_price: model_obj.get("output_cost_per_token").and_then(|v| v.as_f64()),
                        supports_vision: model_obj.get("supports_vision").and_then(|v| v.as_bool()),
                        supports_function_calling: model_obj.get("supports_function_calling").and_then(|v| v.as_bool()),
                        require_max_tokens: None,
                    };
                    
                    // Find or create provider
                    if let Some(provider_info) = providers.iter_mut().find(|p| p.provider == provider) {
                        provider_info.models.push(model_config);
                    } else {
                        providers.push(ProviderInfo {
                            provider: provider.to_string(),
                            models: vec![model_config],
                        });
                    }
                }
            }
        }
        
        // Write to models.yaml
        let yaml_content = serde_yaml::to_string(&providers)?;
        fs::write("models.yaml", yaml_content)?;
    }
    
    // Load providers from YAML
    let yaml_content = fs::read_to_string("models.yaml")?;
    let providers: ModelYaml = serde_yaml::from_str(&yaml_content)?;
    
    let file_manager = SurgicalFileManager::new();
    
    // Generate all files
    generate_all_files(&providers, &file_manager)?;
    
    println!("✅ Code generation completed successfully!");
    Ok(())
}

/// Generate all files with surgical precision and zero-allocation optimizations
fn generate_all_files(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    generate_models_enum_file(providers, file_manager)?;
    generate_providers_enum_file(providers, file_manager)?;
    generate_model_implementations(providers, file_manager)?;
    generate_provider_implementations(providers, file_manager)?;
    
    println!("✅ All files generated with surgical precision");
    Ok(())
}

/// Generate models.rs with pure Models enum (zero-allocation, blazing-fast)
fn generate_models_enum_file(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let variants = generate_model_variants_optimized(providers);
    
    let content = format!(
        r#"// This file is auto-generated. Do not edit manually.
use serde::{{{}, {}}};

// AUTO-GENERATED START
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Models {{
{}
}}
// AUTO-GENERATED END
"#,
        "Serialize", "Deserialize",
        variants.iter().map(|v| format!("    {},", v)).collect::<Vec<_>>().join("\n")
    );
    
    file_manager.update_file_surgical(Path::new("src/models.rs"), &content, false)?;
    Ok(())
}

/// Generate providers.rs with pure Providers enum (zero-allocation, blazing-fast)
fn generate_providers_enum_file(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let variants: Vec<String> = providers
        .iter()
        .map(|p| to_pascal_case_optimized(&p.provider))
        .collect();
    
    let content = format!(
        r#"// This file is auto-generated. Do not edit manually.
use serde::{{{}, {}}};

// AUTO-GENERATED START
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Providers {{
{}
}}
// AUTO-GENERATED END
"#,
        "Serialize", "Deserialize",
        variants.iter().map(|v| format!("    {},", v)).collect::<Vec<_>>().join("\n")
    );
    
    file_manager.update_file_surgical(Path::new("src/providers.rs"), &content, false)?;
    Ok(())
}

/// Generate Model trait implementations with surgical precision
fn generate_model_implementations(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let impl_code = generate_model_info_implementation_optimized(providers);
    
    let content = format!(
        "// AUTO-GENERATED START\nuse crate::models::Models;\n\n{}\n// AUTO-GENERATED END\n",
        impl_code
    );
    
    file_manager.update_file_surgical(Path::new("src/model.rs"), &content, true)?;
    Ok(())
}

/// Generate Provider trait implementations with surgical precision
fn generate_provider_implementations(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let (models_match_arms, name_match_arms) = generate_provider_match_arms_optimized(providers);
    
    let impl_code = format!(
        "use crate::ZeroOneOrMany;\nuse crate::models::Models;\n\n\
        #[inline(always)]\n\
        impl crate::Provider for Providers {{\n\
            fn models(&self) -> ZeroOneOrMany<Models> {{\n\
                match self {{\n\
{}\n\
                }}\n\
            }}\n\n\
            #[inline(always)]\n\
            fn name(&self) -> &'static str {{\n\
                match self {{\n\
{}\n\
                }}\n\
            }}\n\
        }}",
        models_match_arms.join(",\n"),
        name_match_arms.join(",\n")
    );
    
    let content = format!(
        "// AUTO-GENERATED START\nuse crate::providers::Providers;\n\n{}\n// AUTO-GENERATED END\n",
        impl_code
    );
    
    file_manager.update_file_surgical(Path::new("src/provider.rs"), &content, true)?;
    Ok(())
}

/// Generate model variants with zero-allocation optimization using BTreeSet
fn generate_model_variants_optimized(providers: &[ProviderInfo]) -> BTreeSet<String> {
    let mut variants = BTreeSet::new();
    
    for provider in providers {
        for model in &provider.models {
            let variant_name = to_pascal_case_optimized(&model.name);
            variants.insert(variant_name);
        }
    }
    
    variants
}

/// Optimized PascalCase conversion with minimal allocations
fn to_pascal_case_optimized(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut capitalize_next = true;
    
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    
    while i < chars.len() {
        let ch = chars[i];
        
        match ch {
            '-' | '_' | '.' | ' ' => {
                capitalize_next = true;
            }
            c if c.is_alphabetic() => {
                if capitalize_next {
                    result.push(c.to_ascii_uppercase());
                    capitalize_next = false;
                } else {
                    result.push(c.to_ascii_lowercase());
                }
            }
            c if c.is_ascii_digit() => {
                result.push(c);
            }
            _ => {} // Skip other characters
        }
        i += 1;
    }
    
    // Ensure it starts with a letter for valid Rust identifier
    if result.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        result = format!("Model{}", result);
    }
    
    result
}

/// Generate optimized Model info implementation with proper formatting
fn generate_model_info_implementation_optimized(providers: &[ProviderInfo]) -> String {
    let mut match_arms = Vec::new();
    
    for provider in providers {
        for model in &provider.models {
            let variant_name = to_pascal_case_optimized(&model.name);
            
            // Generate properly formatted field values with correct Option handling
            let max_input_tokens = match model.max_input_tokens {
                Some(v) => format!("Some({}u64)", v),
                None => "None".to_string(),
            };
            let max_output_tokens = match model.max_output_tokens {
                Some(v) => format!("Some({}u64)", v),
                None => "None".to_string(),
            };
            let input_price = match model.input_price {
                Some(v) => format!("Some({}f64)", v),
                None => "None".to_string(),
            };
            let output_price = match model.output_price {
                Some(v) => format!("Some({}f64)", v),
                None => "None".to_string(),
            };
            let supports_vision = match model.supports_vision {
                Some(v) => format!("Some({})", v),
                None => "None".to_string(),
            };
            let supports_function_calling = match model.supports_function_calling {
                Some(v) => format!("Some({})", v),
                None => "None".to_string(),
            };
            let require_max_tokens = match model.require_max_tokens {
                Some(v) => format!("Some({})", v),
                None => "None".to_string(),
            };
            
            let match_arm = format!(
                "            Models::{} => crate::ModelInfoData {{\n\
                 provider_name: \"{}\".to_string(),\n\
                 name: \"{}\".to_string(),\n\
                 max_input_tokens: {},\n\
                 max_output_tokens: {},\n\
                 input_price: {},\n\
                 output_price: {},\n\
                 supports_vision: {},\n\
                 supports_function_calling: {},\n\
                 require_max_tokens: {},\n\
            }}",
                variant_name,
                provider.provider,
                model.name,
                max_input_tokens,
                max_output_tokens,
                input_price,
                output_price,
                supports_vision,
                supports_function_calling,
                require_max_tokens
            );
            match_arms.push(match_arm);
        }
    }
    
    format!(
        "#[inline(always)]\n\
        impl crate::Model for Models {{\n\
            fn info(&self) -> crate::ModelInfoData {{\n\
                match self {{\n\
{}\n\
                }}\n\
            }}\n\
        }}",
        match_arms.join(",\n")
    )
}

/// Generate optimized Provider match arms with proper string formatting
fn generate_provider_match_arms_optimized(
    providers: &[ProviderInfo],
) -> (Vec<String>, Vec<String>) {
    let mut models_arms = Vec::new();
    let mut name_arms = Vec::new();
    
    for provider in providers {
        let provider_variant = to_pascal_case_optimized(&provider.provider);
        let provider_name = &provider.provider;
        
        let model_variants: Vec<_> = provider.models
            .iter()
            .map(|m| to_pascal_case_optimized(&m.name))
            .collect();
        
        let models_arm = if model_variants.len() == 1 {
            let model = &model_variants[0];
            format!(
                "            Providers::{} => ZeroOneOrMany::One(Models::{})",
                provider_variant, model
            )
        } else if model_variants.is_empty() {
            format!(
                "            Providers::{} => ZeroOneOrMany::Zero",
                provider_variant
            )
        } else {
            let model_list = model_variants
                .iter()
                .map(|m| format!("Models::{}", m))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "            Providers::{} => ZeroOneOrMany::Many(vec![{}])",
                provider_variant, model_list
            )
        };
        
        let name_arm = format!(
            "            Providers::{} => \"{}\"",
            provider_variant, provider_name
        );
        
        models_arms.push(models_arm);
        name_arms.push(name_arm);
    }
    
    (models_arms, name_arms)
}
