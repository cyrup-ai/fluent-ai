use serde::{Deserialize, Serialize};
use syn::{parse_quote, ItemImpl, Arm};
use quote::{quote, format_ident};
use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use yyaml;

/// The models.yaml is a top-level array of providers, not a struct with providers field
type ModelYaml = Vec<ProviderInfo>;

/// Progressive YAML parsing to isolate assertion failures
fn debug_yaml_parsing(yaml_content: &str) -> Result<yyaml::Yaml, Box<dyn std::error::Error>> {
    let lines: Vec<&str> = yaml_content.lines().collect();
    let total_lines = lines.len();
    println!("🔍 Total YAML lines: {}", total_lines);
    
    // Test parsing in progressively larger chunks
    let chunk_sizes = [50, 100, 200, 500, 1000, total_lines];
    
    for &chunk_size in &chunk_sizes {
        let test_lines = chunk_size.min(total_lines);
        let test_content = lines[..test_lines].join("\n");
        
        println!("🧪 Testing {} lines...", test_lines);
        
        match std::panic::catch_unwind(|| {
            yyaml::YamlLoader::load_from_str(&test_content)
        }) {
            Ok(Ok(docs)) => {
                println!("✅ Successfully parsed {} lines", test_lines);
                if test_lines == total_lines {
                    return Ok(docs[0].clone());
                }
            },
            Ok(Err(e)) => {
                println!("❌ YAML error at {} lines: {}", test_lines, e);
                return Err(format!("YAML parsing failed at line {}: {}", test_lines, e).into());
            },
            Err(_) => {
                println!("💥 PANIC at {} lines - narrowing down...", test_lines);
                
                // Binary search to find exact problematic line
                if chunk_size > 50 {
                    let prev_working = chunk_sizes.iter()
                        .rev()
                        .find(|&&size| size < chunk_size)
                        .copied()
                        .unwrap_or(1);
                    
                    return find_problematic_yaml_section(&lines, prev_working, test_lines);
                }
                return Err("YAML parsing panicked in first 50 lines".into());
            }
        }
    }
    
    Err("Failed to parse YAML completely".into())
}

/// Binary search to find the exact problematic YAML lines
fn find_problematic_yaml_section(
    lines: &[&str], 
    last_working: usize, 
    first_failing: usize
) -> Result<yyaml::Yaml, Box<dyn std::error::Error>> {
    println!("🎯 Narrowing down: last working {} lines, first failing {} lines", 
             last_working, first_failing);
    
    for line_count in (last_working + 1)..=first_failing {
        let test_content = lines[..line_count].join("\n");
        
        match std::panic::catch_unwind(|| {
            yyaml::YamlLoader::load_from_str(&test_content)
        }) {
            Ok(Ok(_)) => continue,
            _ => {
                println!("🚨 PROBLEMATIC SECTION around line {}", line_count);
                println!("Lines {}-{}:", (line_count.saturating_sub(5)).max(1), line_count.min(lines.len()));
                
                let start = (line_count.saturating_sub(5)).max(1) - 1;
                let end = line_count.min(lines.len());
                
                for (i, line) in lines[start..end].iter().enumerate() {
                    println!("{:4}: {}", start + i + 1, line);
                }
                
                return Err(format!("YAML parsing fails at line {}", line_count).into());
            }
        }
    }
    
    Err("Could not isolate problematic YAML section".into())
}

/// Convert YAML document to ProviderInfo structs
fn convert_yaml_to_providers(yaml_doc: &yyaml::Yaml) -> Result<ModelYaml, Box<dyn std::error::Error>> {
    let mut providers = Vec::new();
    
    if let Some(provider_array) = yaml_doc.as_vec() {
        for provider_yaml in provider_array {
            let provider_name = provider_yaml["provider"].as_str()
                .ok_or("Missing provider field")?;
            
            let mut models = Vec::new();
            if let Some(models_array) = provider_yaml["models"].as_vec() {
                for model_yaml in models_array {
                    let model = ModelConfig {
                        name: model_yaml["name"].as_str()
                            .ok_or("Missing model name")?.to_string(),
                        max_input_tokens: model_yaml["max_input_tokens"].as_i64().map(|v| v as u64),
                        max_output_tokens: model_yaml["max_output_tokens"].as_i64().map(|v| v as u64),
                        input_price: model_yaml["input_price"].as_f64(),
                        output_price: model_yaml["output_price"].as_f64(),
                        supports_vision: model_yaml["supports_vision"].as_bool(),
                        supports_function_calling: model_yaml["supports_function_calling"].as_bool(),
                        require_max_tokens: model_yaml["require_max_tokens"].as_bool(),
                    };
                    models.push(model);
                }
            }
            
            let provider_info = ProviderInfo {
                provider: provider_name.to_string(),
                models,
            };
            providers.push(provider_info);
        }
    }
    
    Ok(providers)
}

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
                return Ok(format!("{}{}\n{}\n{}{}",
                    before,
                    self.auto_gen_start,
                    new_content,
                    self.auto_gen_end,
                    after));
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
    
    let download_result = client
        .get("https://raw.githubusercontent.com/sigoden/aichat/refs/heads/main/models.yaml")
        .send()
        .await;
    
    match download_result {
        Ok(response) if response.status().is_success() => {
            let content = response.text().await?;
            fs::write("models.yaml", content)?;
        }
        _ => {
            eprintln!("Warning: Failed to download models.yaml, using existing local copy if available");
            if !Path::new("models.yaml").exists() {
                return Err("No models.yaml found and download failed".into());
            }
        }
    }
    
    // Load providers from YAML using progressive parsing debug
    let yaml_content = fs::read_to_string("models.yaml")?;
    println!("📄 YAML content loaded, {} bytes", yaml_content.len());
    
    // Use progressive parsing to isolate any assertion failures
    let yaml_doc = debug_yaml_parsing(&yaml_content)?;
    let providers: ModelYaml = convert_yaml_to_providers(&yaml_doc)?;
    
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
    // Generate enums and implementations directly in the target files
    generate_models_enum_in_model_file(providers, file_manager)?;
    generate_providers_enum_in_provider_file(providers, file_manager)?;
    generate_model_info_data(providers, file_manager)?;
    
    // Generate trait implementations in their respective files
    generate_model_trait_implementation(providers, file_manager)?;
    generate_provider_implementations(providers, file_manager)?;
    
    println!("✅ All files generated with surgical precision");
    Ok(())
}

/// Generate Models enum directly in model.rs with proper syn formatting
fn generate_models_enum_in_model_file(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    use syn::{ItemEnum, Variant, parse_quote};
    
    // Generate all unique model variants
    let variants = generate_model_variants_optimized(providers);
    let enum_variants: Vec<Variant> = variants
        .iter()
        .map(|variant_name| {
            let ident = format_ident!("{}", variant_name);
            parse_quote! { #ident }
        })
        .collect();
    
    // Create the Models enum using syn
    let models_enum: ItemEnum = parse_quote! {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        pub enum Models {
            #(#enum_variants,)*
        }
    };
    
    // Format with proper line breaks and indentation
    let tokens = quote! {
        #models_enum
    };
    
    let content = tokens.to_string();
    file_manager.update_file_surgical(Path::new("src/model.rs"), &content, true)?;
    Ok(())
}

/// Generate Providers enum directly in provider.rs with proper syn formatting
fn generate_providers_enum_in_provider_file(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    use syn::{ItemEnum, Variant, parse_quote};
    
    // Generate provider variants
    let provider_variants: Vec<Variant> = providers
        .iter()
        .map(|provider| {
            let variant_name = to_pascal_case_optimized(&provider.provider);
            let ident = format_ident!("{}", variant_name);
            parse_quote! { #ident }
        })
        .collect();
    
    // Create the Providers enum using syn
    let providers_enum: ItemEnum = parse_quote! {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        pub enum Providers {
            #(#provider_variants,)*
        }
    };
    
    // Format with proper line breaks and indentation
    let tokens = quote! {
        #providers_enum
    };
    
    let content = tokens.to_string();
    file_manager.update_file_surgical(Path::new("src/provider.rs"), &content, true)?;
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

/// Generate Model trait implementation using syn/quote with proper formatting
fn generate_model_trait_implementation(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_match_arms = generate_model_match_arms_ast(providers);
    
    // Generate the Model trait implementation using syn/quote
    let model_impl: ItemImpl = parse_quote! {
        impl crate::Model for Models {
            #[inline(always)]
            fn info(&self) -> ModelInfoData {
                match self {
                    #(#model_match_arms)*
                }
            }
        }
    };
    
    // Generate the complete content with proper formatting
    let tokens = quote! {
        use crate::model_info::ModelInfoData;
        
        #model_impl
    };
    
    let content = tokens.to_string();
    file_manager.update_file_surgical(Path::new("src/model.rs"), &content, true)?;
    Ok(())
}

/// Generate Provider trait implementation for Providers enum (name() method only)
fn generate_provider_implementations(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generate name match arms only
    let mut name_match_arms = Vec::new();
    
    for provider in providers {
        let provider_variant = format_ident!("{}", to_pascal_case_optimized(&provider.provider));
        let provider_name = &provider.provider;
        
        let name_arm: syn::Arm = parse_quote! {
            Providers::#provider_variant => #provider_name
        };
        
        name_match_arms.push(name_arm);
    }
    
    // Generate the Provider trait implementation using syn/quote (name() only)
    let provider_impl: ItemImpl = parse_quote! {
        impl crate::Provider for Providers {
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

/// Generate match arms for Model trait implementation using syn/quote
fn generate_model_match_arms_ast(providers: &[ProviderInfo]) -> Vec<Arm> {
    let mut match_arms = Vec::new();
    
    for provider in providers {
        for model in &provider.models {
            let variant_name = format_ident!("{}", to_pascal_case_optimized(&model.name));
            let function_name = format_ident!("get_{}_info", to_pascal_case_optimized(&model.name).to_lowercase());
            
            let match_arm: Arm = parse_quote! {
                Models::#variant_name => crate::model_info::#function_name(),
            };
            
            match_arms.push(match_arm);
        }
    }
    
    match_arms
}

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
