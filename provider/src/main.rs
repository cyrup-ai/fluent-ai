use proc_macro2::TokenStream;
use quote::quote;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use syn::{Ident, ItemEnum};

#[derive(Debug, Deserialize, Serialize)]
struct ModelInfo {
    name: String,
    max_input_tokens: Option<u64>,
    max_output_tokens: Option<u64>,
    input_price: Option<f64>,
    output_price: Option<f64>,
    supports_vision: Option<bool>,
    supports_function_calling: Option<bool>,
    require_max_tokens: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProviderInfo {
    provider: String,
    models: Vec<ModelInfo>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Fluent AI Provider Generator");
    println!("Downloading models.yaml from AiChat repository...");
    
    // Download models.yaml from AiChat repository
    let response = reqwest::get("https://raw.githubusercontent.com/sigoden/aichat/main/models.yaml")
        .await?
        .text()
        .await?;
    
    println!("✅ Downloaded models.yaml ({} bytes)", response.len());
    
    // Parse YAML
    let providers: Vec<ProviderInfo> = serde_yaml::from_str(&response)?;
    
    println!("📊 Found {} providers", providers.len());
    for provider in &providers {
        println!("  - {}: {} models", provider.provider, provider.models.len());
    }
    
    // Generate Provider enum
    let provider_enum = generate_provider_enum(&providers)?;
    
    // Generate Model enum
    let model_enum = generate_model_enum(&providers)?;
    
    // Generate provider match arms
    let (provider_models_match_arms, provider_name_match_arms) = generate_provider_match_arms(&providers);
    
    // Generate model info function
    let model_info_fn = generate_model_info_function(&providers)?;
    
    // Generate provider.rs file
    let provider_code = quote! {
        //! Generated provider implementation from AiChat models.yaml
        //! 
        //! This file is auto-generated. Do not edit manually.
        //! Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml
        
        use serde::{Deserialize, Serialize};
        use crate::{Provider, Model, model::ModelImpl};
        
        #provider_enum
        
        impl Provider for ProviderImpl {
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
    
    // Write provider.rs
    let provider_path = Path::new("src/provider.rs");
    fs::write(provider_path, provider_code.to_string())?;
    
    // Write model.rs
    let model_path = Path::new("src/model.rs");
    fs::write(model_path, model_code.to_string())?;
    
    println!("✨ Generated provider.rs and model.rs with {} providers and {} total models", 
             providers.len(), 
             providers.iter().map(|p| p.models.len()).sum::<usize>());
    
    // Update lib.rs to include the module
    update_lib_rs()?;
    
    println!("🎉 Provider generation complete!");
    
    Ok(())
}

fn generate_provider_enum(providers: &[ProviderInfo]) -> Result<ItemEnum, Box<dyn std::error::Error>> {
    let mut variants = Vec::new();
    
    for provider in providers {
        let variant_name = to_pascal_case(&provider.provider);
        let ident = Ident::new(&variant_name, proc_macro2::Span::call_site());
        variants.push(quote! { #ident });
    }
    
    let enum_tokens = quote! {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum ProviderImpl {
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
    
    let variant_tokens: Vec<_> = variants.iter().map(|name| {
        let ident = Ident::new(name, proc_macro2::Span::call_site());
        quote! { #ident }
    }).collect();
    
    let enum_tokens = quote! {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum ModelImpl {
            #(#variant_tokens,)*
        }
    };
    
    Ok(syn::parse2(enum_tokens)?)
}



fn generate_provider_match_arms(providers: &[ProviderInfo]) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    let mut models_match_arms = Vec::new();
    let mut name_match_arms = Vec::new();
    
    for provider in providers {
        let provider_variant = to_pascal_case(&provider.provider);
        let provider_ident = Ident::new(&provider_variant, proc_macro2::Span::call_site());
        let provider_name = &provider.provider;
        
        let model_variants: Vec<_> = provider.models.iter()
            .map(|model| {
                let model_name = to_pascal_case(&model.name);
                // Create provider-prefixed variant name to match generate_model_enum()
                let variant_name = format!("{}{}", provider_variant, model_name);
                let ident = Ident::new(&variant_name, proc_macro2::Span::call_site());
                quote! { Box::new(ModelImpl::#ident) as Box<dyn Model> }
            })
            .collect();
        
        models_match_arms.push(quote! {
            ProviderImpl::#provider_ident => vec![#(#model_variants,)*]
        });
        
        name_match_arms.push(quote! {
            ProviderImpl::#provider_ident => #provider_name
        });
    }
    
    (models_match_arms, name_match_arms)
}

fn generate_model_info_function(providers: &[ProviderInfo]) -> Result<TokenStream, Box<dyn std::error::Error>> {
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
            let max_input_tokens = model.max_input_tokens.map(|t| quote! { Some(#t) }).unwrap_or(quote! { None });
            let max_output_tokens = model.max_output_tokens.map(|t| quote! { Some(#t) }).unwrap_or(quote! { None });
            let input_price = model.input_price.map(|p| quote! { Some(#p) }).unwrap_or(quote! { None });
            let output_price = model.output_price.map(|p| quote! { Some(#p) }).unwrap_or(quote! { None });
            let supports_vision = model.supports_vision.map(|v| quote! { Some(#v) }).unwrap_or(quote! { None });
            let supports_function_calling = model.supports_function_calling.map(|f| quote! { Some(#f) }).unwrap_or(quote! { None });
            let require_max_tokens = model.require_max_tokens.map(|r| quote! { Some(#r) }).unwrap_or(quote! { None });
            
            let provider_name = &provider.provider;
            match_arms.insert(model_variant.clone(), quote! {
                ModelImpl::#model_ident => ModelInfoData {
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
            });
            
            name_match_arms.insert(model_variant, quote! {
                ModelImpl::#model_ident => #name
            });
        }
    }
    
    // Convert to sorted vectors for consistent output
    let match_arm_values: Vec<_> = match_arms.into_values().collect();
    let name_match_arm_values: Vec<_> = name_match_arms.into_values().collect();
    
    let function_tokens = quote! {
        impl Model for ModelImpl {
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
                Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
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