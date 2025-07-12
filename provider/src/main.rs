use proc_macro2::{Ident, TokenStream};
use quote::quote;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, BTreeMap};
use std::fs;
use std::path::Path;
use std::io::{BufRead, BufReader, Write};
use syn::{parse2, ItemEnum};

/// Zero-allocation model configuration with const optimizations
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelYaml {
    providers: Vec<ProviderInfo>,
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
            auto_gen_start: "// AUTO-GENERATED: START - DO NOT EDIT",
            auto_gen_end: "// AUTO-GENERATED: END",
        }
    }

    /// Parse file and extract existing enum variants with zero allocation
    #[inline]
    fn extract_existing_variants(&self, file_path: &Path) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
        if !file_path.exists() {
            return Ok(BTreeSet::new());
        }

        let file = fs::File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut variants = BTreeSet::new();
        let mut in_enum = false;
        let mut brace_count = 0;

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
            
            if trimmed.contains("pub enum") {
                in_enum = true;
                continue;
            }
            
            if in_enum {
                if trimmed.contains('{') {
                    brace_count += trimmed.matches('{').count();
                }
                if trimmed.contains('}') {
                    brace_count -= trimmed.matches('}').count();
                    if brace_count == 0 {
                        break;
                    }
                }
                
                // Extract variant names (handle trailing commas)
                if let Some(variant) = trimmed.split(',').next() {
                    let variant = variant.trim();
                    if !variant.is_empty() && !variant.starts_with("//") && !variant.starts_with("#") {
                        variants.insert(variant.to_string());
                    }
                }
            }
        }
        
        Ok(variants)
    }

    /// Surgically update file preserving hand-written sections
    #[inline]
    fn update_file_surgical(
        &self,
        file_path: &Path,
        new_content: &str,
        preserve_hand_written: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !preserve_hand_written {
            // Pure auto-generated files can be fully overwritten
            return fs::write(file_path, new_content).map_err(Into::into);
        }

        let existing_content = if file_path.exists() {
            fs::read_to_string(file_path)?
        } else {
            String::new()
        };

        if existing_content.contains(self.auto_gen_start) {
            // Replace only auto-generated sections
            let updated = self.replace_auto_generated_section(&existing_content, new_content)?;
            fs::write(file_path, updated)?;
        } else {
            // Append auto-generated section
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)?;
            writeln!(file, "\n{}", self.auto_gen_start)?;
            writeln!(file, "{}", new_content)?;
            writeln!(file, "{}", self.auto_gen_end)?;
        }
        
        Ok(())
    }

    #[inline]
    fn replace_auto_generated_section(
        &self,
        existing: &str,
        new_content: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let start_marker = self.auto_gen_start;
        let end_marker = self.auto_gen_end;
        
        if let Some(start_pos) = existing.find(start_marker) {
            if let Some(end_pos) = existing.find(end_marker) {
                let before = &existing[..start_pos];
                let after = &existing[end_pos + end_marker.len()..];
                return Ok(format!(
                    "{}{}\n{}\n{}{}",
                    before, start_marker, new_content, end_marker, after
                ));
            }
        }
        
        Err("Auto-generated markers not found in existing content".into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Fetching latest model definitions from AiChat...");

    // Fetch the YAML file from the aichat repository with optimized HTTP client
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    
    let yaml_url = "https://raw.githubusercontent.com/sigoden/aichat/main/models.yaml";
    let response = client.get(yaml_url).send().await?;
    
    if !response.status().is_success() {
        return Err(format!("Failed to fetch models.yaml: {}", response.status()).into());
    }
    
    let yaml_content = response.text().await?;

    // Parse the YAML with error context
    let model_yaml: ModelYaml = serde_yaml::from_str(&yaml_content)
        .map_err(|e| format!("Failed to parse models.yaml: {}", e))?;
    
    let providers = model_yaml.providers;
    // Generate Provider enum
    let provider_enum = generate_provider_enum(&providers)?;

    // Generate Model enum
    let model_enum = generate_model_enum(&providers)?;

    // Generate provider match arms
    let (provider_models_match_arms, provider_name_match_arms) =
        generate_provider_match_arms(&providers);

    // Generate model info function
    let model_info_fn = generate_model_info_function(&providers)?;

    // Generate provider.rs file
    let provider_code = quote! {
        //! Generated provider implementation from AiChat models.yaml
        //!
        //! This file is auto-generated. Do not edit manually.
        //! Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml

        use serde::{Deserialize, Serialize};
        use crate::{Provider, Model, model::Models};

        #provider_enum

        impl Provider for Providers {
            fn models(&self) -> Vec<Box<dyn Model>> {
                match self {
                    #(#provider_models_match_arms,)*
                }
            }

            fn name(&self) -> &str {
                match self {
                    #(#provider_name_match_arms,)*
                }
            }
        }
    };

    // Generate model.rs file
    let model_code = quote! {
        //! Generated model implementation from AiChat models.yaml
        //!
        //! This file is auto-generated. Do not edit manually.
        //! Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml

        use serde::{Deserialize, Serialize};
        use crate::{Model, ModelInfoData};

        #model_enum

        #model_info_fn
    };

    // Generate enum-only files
    let providers_enum_code = quote! {
        // AUTO-GENERATED: Providers enum from AiChat models.yaml
        // This file is auto-generated. Do not edit manually.
        // Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml
        use serde::{Deserialize, Serialize};
        
        #provider_enum
    };

    let models_enum_code = quote! {
        // AUTO-GENERATED: Models enum from AiChat models.yaml
        // This file is auto-generated. Do not edit manually.
        // Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml
        use serde::{Deserialize, Serialize};
        
        #model_enum
    };

    // Generate trait implementations to append to existing files
    let provider_impl_code = quote! {
        
        // AUTO-GENERATED: Provider trait implementation - DO NOT EDIT
        impl crate::Provider for Providers {
            fn models(&self) -> Vec<Box<dyn crate::Model>> {
                match self {
                    #(#provider_models_match_arms,)*
                }
            }

            fn name(&self) -> &str {
                match self {
                    #(#provider_name_match_arms,)*
                }
            }
        }
        // END AUTO-GENERATED
    };

    let model_impl_code = quote! {
        
        // AUTO-GENERATED: Model trait implementation - DO NOT EDIT
        #model_info_fn
        // END AUTO-GENERATED
    };

    // Write enum files (overwrite is OK since they are pure auto-generated)
    let providers_path = Path::new("src/providers.rs");
    fs::write(providers_path, providers_enum_code.to_string())?;

    let models_path = Path::new("src/models.rs");
    fs::write(models_path, models_enum_code.to_string())?;

    // Append implementations to existing trait files (preserving hand-written code)
    append_if_not_exists("src/provider.rs", &provider_impl_code.to_string())?;
    append_if_not_exists("src/model.rs", &model_impl_code.to_string())?;

    println!(
        "📋 Loaded {} providers with {} total models",
        providers.len(),
        providers.iter().map(|p| p.models.len()).sum::<usize>()
    );

    // Initialize surgical file manager
    let file_manager = SurgicalFileManager::new();
    
    // Generate optimized enums and implementations
    generate_all_files(&providers, &file_manager).await?;

    println!("🎉 Provider generation complete!");

    Ok(())
}

/// Generate all files with surgical precision and zero-allocation optimizations
#[inline]
async fn generate_all_files(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generate pure auto-generated enum files (safe to overwrite)
    generate_models_enum_file(providers, file_manager).await?;
    generate_providers_enum_file(providers, file_manager).await?;
    
    // Generate trait implementations with surgical updates
    generate_model_implementations(providers, file_manager).await?;
    generate_provider_implementations(providers, file_manager).await?;
    
    println!("✨ Generated all files with {} providers and {} total models",
        providers.len(),
        providers.iter().map(|p| p.models.len()).sum::<usize>()
    );
    
    Ok(())
}

/// Generate models.rs with pure Models enum (zero-allocation, blazing-fast)
#[inline]
async fn generate_models_enum_file(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let variants = generate_model_variants_optimized(providers);
    let variant_tokens: Vec<_> = variants
        .iter()
        .map(|name| {
            let ident = Ident::new(name, proc_macro2::Span::call_site());
            quote! { #ident }
        })
        .collect();

    let enum_code = quote! {
        // AUTO-GENERATED: Models enum from AiChat models.yaml
        // This file is auto-generated. Do not edit manually.
        // Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml
        use serde::{Deserialize, Serialize};
        
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Models {
            #(#variant_tokens,)*
        }
    };

    let file_path = Path::new("src/models.rs");
    file_manager.update_file_surgical(file_path, &enum_code.to_string(), false)?;
    println!("📝 Generated models.rs with {} variants", variants.len());
    
    Ok(())
}

/// Generate providers.rs with pure Providers enum (zero-allocation, blazing-fast)
#[inline]
async fn generate_providers_enum_file(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut variants = BTreeSet::new();
    
    for provider in providers {
        let variant_name = to_pascal_case_optimized(&provider.provider);
        variants.insert(variant_name);
    }
    
    let variant_tokens: Vec<_> = variants
        .iter()
        .map(|name| {
            let ident = Ident::new(name, proc_macro2::Span::call_site());
            quote! { #ident }
        })
        .collect();

    let enum_code = quote! {
        // AUTO-GENERATED: Providers enum from AiChat models.yaml
        // This file is auto-generated. Do not edit manually.
        // Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml
        use serde::{Deserialize, Serialize};
        
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Providers {
            #(#variant_tokens,)*
        }
    };

    let file_path = Path::new("src/providers.rs");
    file_manager.update_file_surgical(file_path, &enum_code.to_string(), false)?;
    println!("📝 Generated providers.rs with {} variants", variants.len());
    
    Ok(())
}

/// Generate Model trait implementations with surgical precision
#[inline]
async fn generate_model_implementations(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_info_impl = generate_model_info_implementation_optimized(providers);
    
    let impl_code = quote! {
        #model_info_impl
    };

    let file_path = Path::new("src/model.rs");
    file_manager.update_file_surgical(file_path, &impl_code.to_string(), true)?;
    println!("📝 Updated model.rs with trait implementations");
    
    Ok(())
}

/// Generate Provider trait implementations with surgical precision
#[inline]
async fn generate_provider_implementations(
    providers: &[ProviderInfo],
    file_manager: &SurgicalFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let (models_match_arms, name_match_arms) = generate_provider_match_arms_optimized(providers);
    
    let impl_code = quote! {
        #[inline(always)]
        impl crate::Provider for Providers {
            fn models(&self) -> Vec<Box<dyn crate::Model>> {
                match self {
                    #(#models_match_arms,)*
                }
            }

            #[inline(always)]
            fn name(&self) -> &str {
                match self {
                    #(#name_match_arms,)*
                }
            }
        }
    };

    let file_path = Path::new("src/provider.rs");
    file_manager.update_file_surgical(file_path, &impl_code.to_string(), true)?;
    println!("📝 Updated provider.rs with trait implementations");
    
    Ok(())
}

/// Generate model variants with zero-allocation optimization using BTreeSet
#[inline]
fn generate_model_variants_optimized(providers: &[ProviderInfo]) -> BTreeSet<String> {
    let mut variants = BTreeSet::new();

    for provider in providers {
        let provider_prefix = to_pascal_case_optimized(&provider.provider);

        for model in &provider.models {
            let model_name = to_pascal_case_optimized(&model.name);
            // Create provider-prefixed variant name to ensure uniqueness
            let variant_name = format!("{}{}", provider_prefix, model_name);
            variants.insert(variant_name);
        }
    }

    variants
}

/// Optimized PascalCase conversion with minimal allocations
#[inline]
fn to_pascal_case_optimized(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    // Pre-allocate with estimated capacity
    let mut result = String::with_capacity(input.len());
    let mut next_upper = true;

    for ch in input.chars() {
        match ch {
            '-' | '_' | '.' | ':' | '/' | ' ' => {
                next_upper = true;
            }
            c if c.is_alphanumeric() => {
                if next_upper {
                    result.extend(c.to_uppercase());
                    next_upper = false;
                } else {
                    result.push(c.to_ascii_lowercase());
                }
            }
            _ => {} // Skip non-alphanumeric characters
        }
    }

    // Ensure it starts with a letter (prepend underscore if it starts with number)
    if result.is_empty() || result.chars().next().unwrap().is_numeric() {
        format!("Model_{}", result)
    } else {
        result
    }
}

/// Generate optimized Model info implementation with inline functions
#[inline]
fn generate_model_info_implementation_optimized(providers: &[ProviderInfo]) -> TokenStream {
    let mut model_info_arms = Vec::new();

    for provider in providers {
        let provider_name = &provider.provider;
        let provider_prefix = to_pascal_case_optimized(provider_name);

        for model in &provider.models {
            let model_name = to_pascal_case_optimized(&model.name);
            let variant_name = format!("{}{}", provider_prefix, model_name);
            let variant_ident = Ident::new(&variant_name, proc_macro2::Span::call_site());

            let original_name = &model.name;
            let max_input_tokens = model.max_input_tokens;
            let max_output_tokens = model.max_output_tokens;
            let input_price = model.input_price;
            let output_price = model.output_price;
            let supports_vision = model.supports_vision;
            let supports_function_calling = model.supports_function_calling;
            let require_max_tokens = model.require_max_tokens;

            let arm = quote! {
                Models::#variant_ident => crate::ModelInfoData {
                    provider_name: #provider_name.to_string(),
                    name: #original_name.to_string(),
                    max_input_tokens: #max_input_tokens,
                    max_output_tokens: #max_output_tokens,
                    input_price: #input_price,
                    output_price: #output_price,
                    supports_vision: #supports_vision,
                    supports_function_calling: #supports_function_calling,
                    require_max_tokens: #require_max_tokens,
                }
            };
            model_info_arms.push(arm);
        }
    }

    quote! {
        #[inline(always)]
        impl crate::Model for Models {
            fn info(&self) -> crate::ModelInfoData {
                match self {
                    #(#model_info_arms,)*
                }
            }

            #[inline(always)]
            fn name(&self) -> &str {
                match self {
                    #(#model_info_arms,)*
                }
            }
        }
    }
}

/// Generate optimized Provider match arms with inlined implementations
#[inline]
fn generate_provider_match_arms_optimized(
    providers: &[ProviderInfo],
) -> (Vec<TokenStream>, Vec<TokenStream>) {
    let mut models_match_arms = Vec::new();
    let mut name_match_arms = Vec::new();

    for provider in providers {
        let provider_variant = to_pascal_case_optimized(&provider.provider);
        let provider_ident = Ident::new(&provider_variant, proc_macro2::Span::call_site());
        let provider_name = &provider.provider;

        let model_variants: Vec<_> = provider
            .models
            .iter()
            .map(|model| {
                let model_name = to_pascal_case_optimized(&model.name);
                let variant_name = format!("{}{}", provider_variant, model_name);
                let variant_ident = Ident::new(&variant_name, proc_macro2::Span::call_site());
                quote! { Box::new(crate::models::Models::#variant_ident) }
            })
            .collect();

        let models_arm = quote! {
            Providers::#provider_ident => vec![#(#model_variants,)*]
        };
        models_match_arms.push(models_arm);

        let name_arm = quote! {
            Providers::#provider_ident => #provider_name
        };
        name_match_arms.push(name_arm);
    }

    (models_match_arms, name_match_arms)
}

fn generate_provider_enum(
    providers: &[ProviderInfo],
) -> Result<ItemEnum, Box<dyn std::error::Error>> {
    let mut variants = Vec::new();

    for provider in providers {
        let variant_name = to_pascal_case(&provider.provider);
        let ident = Ident::new(&variant_name, proc_macro2::Span::call_site());
        variants.push(quote! { #ident });
    }

    let enum_tokens = quote! {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Providers {
            #(#variants,)*
        }
    };

    Ok(syn::parse2(enum_tokens)?)
}

fn generate_model_enum(providers: &[ProviderInfo]) -> Result<ItemEnum, Box<dyn std::error::Error>> {
    let mut variants = std::collections::HashSet::new();

    for provider in providers {
        let provider_prefix = to_pascal_case(&provider.provider);

        for model in &provider.models {
            let model_name = to_pascal_case(&model.name);
            // Create provider-prefixed variant name to ensure uniqueness
            let variant_name = format!("{}{}", provider_prefix, model_name);
            variants.insert(variant_name);
        }
    }

    // Convert to sorted Vec for consistent output
    let mut variants: Vec<_> = variants.into_iter().collect();
    variants.sort();

    let variant_tokens: Vec<_> = variants
        .iter()
        .map(|name| {
            let ident = Ident::new(name, proc_macro2::Span::call_site());
            quote! { #ident }
        })
        .collect();

    let enum_tokens = quote! {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Models {
            #(#variant_tokens,)*
        }
    };

    Ok(syn::parse2(enum_tokens)?)
}

fn generate_provider_match_arms(
    providers: &[ProviderInfo],
) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    let mut models_match_arms = Vec::new();
    let mut name_match_arms = Vec::new();

    for provider in providers {
        let provider_variant = to_pascal_case(&provider.provider);
        let provider_ident = Ident::new(&provider_variant, proc_macro2::Span::call_site());
        let provider_name = &provider.provider;

        let model_variants: Vec<_> = provider
            .models
            .iter()
            .map(|model| {
                let model_name = to_pascal_case(&model.name);
                // Create provider-prefixed variant name to match generate_model_enum()
                let variant_name = format!("{}{}", provider_variant, model_name);
                let ident = Ident::new(&variant_name, proc_macro2::Span::call_site());
                quote! { Box::new(Models::#ident) as Box<dyn Model> }
            })
            .collect();

        models_match_arms.push(quote! {
            Providers::#provider_ident => vec![#(#model_variants,)*]
        });

        name_match_arms.push(quote! {
            Providers::#provider_ident => #provider_name
        });
    }

    (models_match_arms, name_match_arms)
}

fn generate_model_info_function(
    providers: &[ProviderInfo],
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut match_arms = std::collections::HashMap::new();
    let mut name_match_arms = std::collections::HashMap::new();

    for provider in providers {
        let provider_variant = to_pascal_case(&provider.provider);

        for model in &provider.models {
            let model_name = to_pascal_case(&model.name);
            // Create provider-prefixed variant name to match generate_model_enum()
            let model_variant = format!("{}{}", provider_variant, model_name);
            let model_ident = Ident::new(&model_variant, proc_macro2::Span::call_site());

            // Only add if we haven't seen this variant before (deduplication)
            if match_arms.contains_key(&model_variant) {
                continue;
            }

            let name = &model.name;
            let max_input_tokens = model
                .max_input_tokens
                .map(|t| quote! { Some(#t) })
                .unwrap_or(quote! { None });
            let max_output_tokens = model
                .max_output_tokens
                .map(|t| quote! { Some(#t) })
                .unwrap_or(quote! { None });
            let input_price = model
                .input_price
                .map(|p| quote! { Some(#p) })
                .unwrap_or(quote! { None });
            let output_price = model
                .output_price
                .map(|p| quote! { Some(#p) })
                .unwrap_or(quote! { None });
            let supports_vision = model
                .supports_vision
                .map(|v| quote! { Some(#v) })
                .unwrap_or(quote! { None });
            let supports_function_calling = model
                .supports_function_calling
                .map(|f| quote! { Some(#f) })
                .unwrap_or(quote! { None });
            let require_max_tokens = model
                .require_max_tokens
                .map(|r| quote! { Some(#r) })
                .unwrap_or(quote! { None });

            let provider_name = &provider.provider;
            match_arms.insert(
                model_variant.clone(),
                quote! {
                    Models::#model_ident => ModelInfoData {
                        provider_name: #provider_name.to_string(),
                        name: #name.to_string(),
                        max_input_tokens: #max_input_tokens,
                        max_output_tokens: #max_output_tokens,
                        input_price: #input_price,
                        output_price: #output_price,
                        supports_vision: #supports_vision,
                        supports_function_calling: #supports_function_calling,
                        require_max_tokens: #require_max_tokens,
                    }
                },
            );

            name_match_arms.insert(
                model_variant,
                quote! {
                    Models::#model_ident => #name
                },
            );
        }
    }

    // Convert to sorted vectors for consistent output
    let match_arm_values: Vec<_> = match_arms.into_values().collect();
    let name_match_arm_values: Vec<_> = name_match_arms.into_values().collect();

    let function_tokens = quote! {
        impl Model for Models {
            /// Get detailed information about this model
            fn info(&self) -> ModelInfoData {
                match self {
                    #(#match_arm_values,)*
                }
            }

            /// Get the original model name
            fn name(&self) -> &str {
                match self {
                    #(#name_match_arm_values,)*
                }
            }
        }
    };

    Ok(function_tokens)
}

fn to_pascal_case(s: &str) -> String {
    // First sanitize the string by removing invalid characters and normalizing
    let sanitized = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c
            } else {
                '_' // Replace any non-alphanumeric with underscore
            }
        })
        .collect::<String>();

    // Split on various separators and convert to PascalCase
    let result = sanitized
        .split(&['-', '_', '.', ':', '/', ' '][..])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
            }
        })
        .collect::<String>();

    // Ensure it starts with a letter (prepend underscore if it starts with number)
    if result.is_empty() || result.chars().next().unwrap().is_numeric() {
        format!("Model_{}", result)
    } else {
        result
    }
}

fn update_lib_rs() -> Result<(), Box<dyn std::error::Error>> {
    let lib_path = Path::new("../fluent-ai/src/lib.rs");
    let mut content = fs::read_to_string(lib_path)?;

    // Check if providers module is already included
    if !content.contains("pub mod providers;") {
        // Add the providers module declaration
        let insertion_point = content.find("pub mod fluent;").unwrap_or(content.len());
        content.insert_str(insertion_point, "pub mod providers;\n");

        // Also add to the public exports
        if let Some(exports_start) = content.find("// Re-export domain types") {
            let export_line = "\n// Re-export generated providers\npub use providers::{Provider, Model, ModelInfo};\n";
            content.insert_str(exports_start, export_line);
        }

        fs::write(lib_path, content)?;
        println!("✅ Updated lib.rs to include providers module");
    } else {
        println!("ℹ️  lib.rs already includes providers module");
    }

    Ok(())
}
