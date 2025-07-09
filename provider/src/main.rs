use proc_macro2::TokenStream;
use quote::quote;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use syn::{Ident, ItemEnum, ItemStruct};

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
    
    // Generate ModelInfo struct
    let model_info_struct = generate_model_info_struct();
    
    // Generate provider model mappings
    let provider_models_fn = generate_provider_models_function(&providers)?;
    
    // Generate model info function
    let model_info_fn = generate_model_info_function(&providers)?;
    
    // Create the complete module
    let generated_code = quote! {
        //! Generated provider and model definitions from AiChat models.yaml
        //! 
        //! This file is auto-generated. Do not edit manually.
        //! Generated from: https://github.com/sigoden/aichat/blob/main/models.yaml
        
        use serde::{Deserialize, Serialize};
        use std::collections::HashMap;
        
        #model_info_struct
        
        #provider_enum
        
        #model_enum
        
        #provider_models_fn
        
        #model_info_fn
    };
    
    // Write to output file
    let output_path = Path::new("../fluent-ai/src/providers.rs");
    fs::write(output_path, generated_code.to_string())?;
    
    println!("✨ Generated providers.rs with {} providers and {} total models", 
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
        pub enum Provider {
            #(#variants,)*
        }
    };
    
    Ok(syn::parse2(enum_tokens)?)
}

fn generate_model_enum(providers: &[ProviderInfo]) -> Result<ItemEnum, Box<dyn std::error::Error>> {
    let mut variants = std::collections::HashSet::new();
    
    for provider in providers {
        for model in &provider.models {
            let variant_name = to_pascal_case(&model.name);
            variants.insert(variant_name);
        }
    }
    
    let mut variant_list: Vec<_> = variants.into_iter().collect();
    variant_list.sort();
    
    let variant_tokens: Vec<_> = variant_list.iter().map(|name| {
        let ident = Ident::new(name, proc_macro2::Span::call_site());
        quote! { #ident }
    }).collect();
    
    let enum_tokens = quote! {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Model {
            #(#variant_tokens,)*
        }
    };
    
    Ok(syn::parse2(enum_tokens)?)
}

fn generate_model_info_struct() -> ItemStruct {
    let struct_tokens = quote! {
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct ModelInfo {
            pub provider: Provider,
            pub name: String,
            pub max_input_tokens: Option<u64>,
            pub max_output_tokens: Option<u64>,
            pub input_price: Option<f64>,
            pub output_price: Option<f64>,
            pub supports_vision: Option<bool>,
            pub supports_function_calling: Option<bool>,
            pub require_max_tokens: Option<bool>,
        }
    };
    
    syn::parse2(struct_tokens).unwrap()
}

fn generate_provider_models_function(providers: &[ProviderInfo]) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut match_arms = Vec::new();
    
    for provider in providers {
        let provider_variant = to_pascal_case(&provider.provider);
        let provider_ident = Ident::new(&provider_variant, proc_macro2::Span::call_site());
        
        let model_variants: Vec<_> = provider.models.iter()
            .map(|model| {
                let variant_name = to_pascal_case(&model.name);
                let ident = Ident::new(&variant_name, proc_macro2::Span::call_site());
                quote! { Model::#ident }
            })
            .collect();
        
        match_arms.push(quote! {
            Provider::#provider_ident => vec![#(#model_variants,)*]
        });
    }
    
    let function_tokens = quote! {
        impl Provider {
            /// Get all models supported by this provider
            pub fn models(&self) -> Vec<Model> {
                match self {
                    #(#match_arms,)*
                }
            }
        }
    };
    
    Ok(function_tokens)
}

fn generate_model_info_function(providers: &[ProviderInfo]) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut match_arms = Vec::new();
    let mut name_match_arms = Vec::new();
    
    for provider in providers {
        let provider_variant = to_pascal_case(&provider.provider);
        let provider_ident = Ident::new(&provider_variant, proc_macro2::Span::call_site());
        
        for model in &provider.models {
            let model_variant = to_pascal_case(&model.name);
            let model_ident = Ident::new(&model_variant, proc_macro2::Span::call_site());
            
            let name = &model.name;
            let max_input_tokens = model.max_input_tokens.map(|t| quote! { Some(#t) }).unwrap_or(quote! { None });
            let max_output_tokens = model.max_output_tokens.map(|t| quote! { Some(#t) }).unwrap_or(quote! { None });
            let input_price = model.input_price.map(|p| quote! { Some(#p) }).unwrap_or(quote! { None });
            let output_price = model.output_price.map(|p| quote! { Some(#p) }).unwrap_or(quote! { None });
            let supports_vision = model.supports_vision.map(|v| quote! { Some(#v) }).unwrap_or(quote! { None });
            let supports_function_calling = model.supports_function_calling.map(|f| quote! { Some(#f) }).unwrap_or(quote! { None });
            let require_max_tokens = model.require_max_tokens.map(|r| quote! { Some(#r) }).unwrap_or(quote! { None });
            
            match_arms.push(quote! {
                Model::#model_ident => ModelInfo {
                    provider: Provider::#provider_ident,
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
            
            name_match_arms.push(quote! {
                Model::#model_ident => #name
            });
        }
    }
    
    let function_tokens = quote! {
        impl Model {
            /// Get detailed information about this model
            pub fn info(&self) -> ModelInfo {
                match self {
                    #(#match_arms,)*
                }
            }
            
            /// Get the provider for this model
            pub fn provider(&self) -> Provider {
                self.info().provider
            }
            
            /// Get the original model name
            pub fn name(&self) -> &str {
                match self {
                    #(#name_match_arms,)*
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