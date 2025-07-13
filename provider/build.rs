use serde::{Deserialize, Serialize};
use syn::{parse_quote, ItemImpl, Arm};
use quote::{quote, format_ident};
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;

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
    
    // Download models.yaml from sigoden/aichat
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    
    let response = client
        .get("https://raw.githubusercontent.com/sigoden/aichat/refs/heads/main/models.yaml")
        .send()
        .await?;
    
    if !response.status().is_success() {
        eprintln!("Warning: Failed to download models.yaml, using existing local copy if available");
        if !Path::new("models.yaml").exists() {
            return Err("No models.yaml found and download failed".into());
        }
    } else {
        let content = response.text().await?;
        fs::write("models.yaml", content)?;
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
    generate_model_info_data(providers, file_manager)?;
    
    // NOTE: Trait implementations are now permanent code, not auto-generated
    // generate_model_implementations(providers, file_manager)?; // DISABLED
    // generate_provider_implementations(providers, file_manager)?; // DISABLED
    
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

/// Generate Model info data in model_info.rs (without trait implementations)
fn generate_model_info_data(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let info_data = generate_model_info_data_only(providers);
    
    let content = format!(
        "// AUTO-GENERATED START\n{}\n// AUTO-GENERATED END\n",
        info_data
    );
    
    file_manager.update_file_surgical(Path::new("src/model_info.rs"), &content, true)?;
    Ok(())
}

// NOTE: This function is no longer used as Provider implementations are now permanent code
#[allow(dead_code)]
fn generate_provider_implementations(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let (models_match_arms, name_match_arms) = generate_provider_match_arms_ast(providers);
    
    // Generate the Provider trait implementation using syn/quote
    let provider_impl: ItemImpl = parse_quote! {
        impl crate::Provider for Providers {
            #[inline(always)]
            fn models(&self) -> ZeroOneOrMany<Models> {
                match self {
                    #(#models_match_arms)*
                }
            }

            #[inline(always)]
            fn name(&self) -> &'static str {
                match self {
                    #(#name_match_arms)*
                }
            }
        }
    };
    
    // Generate the complete file content with imports
    let tokens = quote! {
        // AUTO-GENERATED START
        use crate::providers::Providers;
        use crate::models::Models;
        
        #provider_impl
        // AUTO-GENERATED END
    };
    
    let content = tokens.to_string();
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

/// Generate model info data only (no trait implementations) 
fn generate_model_info_data_only(providers: &[ProviderInfo]) -> String {
    let mut model_data = Vec::new();
    let mut seen_models = std::collections::HashSet::new();
    
    for provider in providers {
        for model in &provider.models {
            let variant_name = to_pascal_case_optimized(&model.name);
            let function_name = format!("get_{}_info", variant_name.to_lowercase());
            
            // Skip if we've already generated this model function
            if seen_models.contains(&function_name) {
                continue;
            }
            seen_models.insert(function_name.clone());
            
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
            
            let model_info = format!(
                "/// Get model info for {}\npub fn get_{}_info() -> ModelInfoData {{\n\
                    ModelInfoData {{\n\
                        provider_name: \"{}\".to_string(),\n\
                        name: \"{}\".to_string(),\n\
                        max_input_tokens: {},\n\
                        max_output_tokens: {},\n\
                        input_price: {},\n\
                        output_price: {},\n\
                        supports_vision: {},\n\
                        supports_function_calling: {},\n\
                        require_max_tokens: {},\n\
                    }}\n\
                }}",
                model.name,
                variant_name.to_lowercase(),
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
            model_data.push(model_info);
        }
    }
    
    model_data.join("\n\n")
}

// NOTE: This function is no longer used as Provider implementations are now permanent code
#[allow(dead_code)]
fn generate_provider_match_arms_ast(
    providers: &[ProviderInfo],
) -> (Vec<Arm>, Vec<Arm>) {
    let mut models_arms = Vec::new();
    let mut name_arms = Vec::new();
    
    for provider in providers {
        let provider_variant = format_ident!("{}", to_pascal_case_optimized(&provider.provider));
        let provider_name = &provider.provider;
        
        let model_variants: Vec<_> = provider.models
            .iter()
            .map(|m| format_ident!("{}", to_pascal_case_optimized(&m.name)))
            .collect();
        
        // Generate models() match arm
        let models_arm = if model_variants.len() == 1 {
            let model = &model_variants[0];
            parse_quote! {
                Providers::#provider_variant => ZeroOneOrMany::One(Models::#model),
            }
        } else if model_variants.is_empty() {
            parse_quote! {
                Providers::#provider_variant => ZeroOneOrMany::Zero,
            }
        } else {
            parse_quote! {
                Providers::#provider_variant => ZeroOneOrMany::Many(vec![#(Models::#model_variants),*]),
            }
        };
        
        // Generate name() match arm
        let name_arm: Arm = parse_quote! {
            Providers::#provider_variant => #provider_name,
        };
        
        models_arms.push(models_arm);
        name_arms.push(name_arm);
    }
    
    (models_arms, name_arms)
}
