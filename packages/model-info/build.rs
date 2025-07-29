use anyhow::{Context, Result};
use fluent_ai_http3::{Http3, HttpStreamExt};
use serde::{Deserialize};
use serde_json::Value;
use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use syn::Ident;
use proc_macro2::Span;

// Utility function to sanitize model IDs into valid Rust identifiers
fn sanitize_ident(id: &str) -> Ident {
    let words: Vec<&str> = id.split(|c: char| !c.is_alphanumeric()).collect();
    let mut pascal_case = String::new();
    
    for word in words {
        if !word.is_empty() {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                pascal_case.push(first.to_ascii_uppercase());
                for ch in chars {
                    pascal_case.push(ch.to_ascii_lowercase());
                }
            }
        }
    }
    
    // Handle leading digits and empty strings
    let final_name = if pascal_case.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        format!("Model{}", pascal_case)
    } else if pascal_case.is_empty() {
        "Unknown".to_string()
    } else {
        pascal_case
    };
    
    Ident::new(&final_name, Span::call_site())
}

// Parse existing enum variants from generated_models.rs
fn parse_existing_enum_variants(path: &PathBuf, enum_name: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read generated_models.rs")?;
    
    let file = syn::parse_file(&content)
        .context("Failed to parse generated_models.rs as Rust code")?;
    
    for item in file.items {
        if let syn::Item::Enum(enum_item) = item {
            if enum_item.ident.to_string() == enum_name {
                let variants: Vec<String> = enum_item.variants
                    .iter()
                    .map(|variant| variant.ident.to_string())
                    .collect();
                return Ok(variants);
            }
        }
    }
    
    // Enum not found in file
    Ok(vec![])
}

// Append new model variants to existing enum in generated_models.rs
fn append_new_models_to_enum(
    path: &PathBuf, 
    enum_name: &str, 
    new_models: &[(String, u64, f64, f64, bool, Option<f64>)]
) -> Result<()> {
    if !path.exists() {
        // File doesn't exist, create it with complete enum
        create_complete_enum(path, enum_name, new_models)?;
        return Ok(());
    }
    
    // Parse existing file
    let content = std::fs::read_to_string(path)
        .context("Failed to read generated_models.rs")?;
    
    let mut file = syn::parse_file(&content)
        .context("Failed to parse generated_models.rs as Rust code")?;
    
    // Find and update the enum
    let mut enum_found = false;
    for item in &mut file.items {
        if let syn::Item::Enum(enum_item) = item {
            if enum_item.ident.to_string() == enum_name {
                // Add new variants to existing enum
                for (id, ..) in new_models {
                    let variant_ident = sanitize_ident(id);
                    let variant = syn::Variant {
                        attrs: vec![],
                        ident: variant_ident,
                        fields: syn::Fields::Unit,
                        discriminant: None,
                    };
                    enum_item.variants.push(variant);
                }
                enum_found = true;
                break;
            }
        }
    }
    
    if !enum_found {
        // Enum doesn't exist in file, add it
        let new_enum_item = create_enum_item(enum_name, new_models)?;
        file.items.push(syn::Item::Enum(new_enum_item));
        
        // Also add the impl block
        let impl_item = create_impl_item(enum_name, new_models)?;
        file.items.push(syn::Item::Impl(impl_item));
    } else {
        // Update existing impl blocks with new match arms
        update_impl_blocks(&mut file, enum_name, new_models)?;
    }
    
    // Write updated file back
    let updated_content = prettyplease::unparse(&file);
    std::fs::write(path, updated_content)
        .context("Failed to write updated generated_models.rs")?;
    
    Ok(())
}

// Create complete enum when file doesn't exist
fn create_complete_enum(
    path: &PathBuf,
    enum_name: &str, 
    models: &[(String, u64, f64, f64, bool, Option<f64>)]
) -> Result<()> {
    let enum_ident = syn::Ident::new(enum_name, Span::call_site());
    
    // Create enum variants
    let variants: Vec<_> = models.iter().map(|(id, ..)| {
        let variant_ident = sanitize_ident(id);
        syn::Variant {
            attrs: vec![],
            ident: variant_ident,
            fields: syn::Fields::Unit,
            discriminant: None,
        }
    }).collect();
    
    let enum_item = syn::ItemEnum {
        attrs: vec![syn::parse_quote! { #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] }],
        vis: syn::parse_quote! { pub },
        enum_token: Default::default(),
        ident: enum_ident.clone(),
        generics: Default::default(),
        brace_token: Default::default(),
        variants: variants.into_iter().collect(),
    };
    
    // Create impl block
    let impl_item = create_impl_item(enum_name, models)?;
    
    // Create file content
    let file = syn::File {
        shebang: None,
        attrs: vec![],
        items: vec![
            syn::Item::Enum(enum_item),
            syn::Item::Impl(impl_item),
        ],
    };
    
    let content = prettyplease::unparse(&file);
    std::fs::write(path, content)
        .context("Failed to write new generated_models.rs")?;
    
    Ok(())
}

// Helper function to create impl item (simplified for now)
fn create_impl_item(enum_name: &str, _models: &[(String, u64, f64, f64, bool, Option<f64>)]) -> Result<syn::ItemImpl> {
    // This is a simplified version - in reality we'd need to create all the match arms
    // For now, just create an empty impl block
    let enum_ident = syn::Ident::new(enum_name, Span::call_site());
    
    Ok(syn::parse_quote! {
        impl crate::common::Model for #enum_ident {
            fn name(&self) -> &'static str {
                match self {
                    // TODO: Add match arms for new variants
                }
            }
            
            fn provider_name(&self) -> &'static str {
                // TODO: Set appropriate provider name
                "unknown"
            }
            
            fn max_input_tokens(&self) -> Option<u32> {
                // TODO: Add logic to split context into input/output
                Some(4096)
            }
            
            fn max_output_tokens(&self) -> Option<u32> {
                // TODO: Add logic to split context into input/output  
                Some(2048)
            }
            
            fn pricing_input(&self) -> Option<f64> { 
                // TODO: Make pricing optional
                Some(0.0) 
            }
            
            fn pricing_output(&self) -> Option<f64> { 
                // TODO: Make pricing optional
                Some(0.0) 
            }
            
            fn supports_vision(&self) -> bool { false }
            fn supports_function_calling(&self) -> bool { true }
            fn supports_streaming(&self) -> bool { true }
            fn supports_embeddings(&self) -> bool { false }
            fn requires_max_tokens(&self) -> bool { true }
            fn supports_thinking(&self) -> bool { false }
            fn required_temperature(&self) -> Option<f64> { None }
        }
    })
}

// Helper function to create enum item
fn create_enum_item(enum_name: &str, models: &[(String, u64, f64, f64, bool, Option<f64>)]) -> Result<syn::ItemEnum> {
    let enum_ident = syn::Ident::new(enum_name, Span::call_site());
    
    let variants: Vec<_> = models.iter().map(|(id, ..)| {
        let variant_ident = sanitize_ident(id);
        syn::Variant {
            attrs: vec![],
            ident: variant_ident,
            fields: syn::Fields::Unit,
            discriminant: None,
        }
    }).collect();
    
    Ok(syn::ItemEnum {
        attrs: vec![syn::parse_quote! { #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)] }],
        vis: syn::parse_quote! { pub },
        enum_token: Default::default(),
        ident: enum_ident,
        generics: Default::default(),
        brace_token: Default::default(),
        variants: variants.into_iter().collect(),
    })
}

// Update existing impl blocks with new match arms (simplified)
fn update_impl_blocks(
    _file: &mut syn::File,
    enum_name: &str, 
    _new_models: &[(String, u64, f64, f64, bool, Option<f64>)]
) -> Result<()> {
    // This is complex to implement properly with syn
    // For now, we'll just print a warning and let the user handle it
    println!("cargo:warning=New variants added to {}. You may need to update match arms manually.", enum_name);
    Ok(())
}


#[derive(Deserialize, Default)]
struct OpenAiModelsResponse {
    _object: String,
    data: Vec<Value>,
}

#[derive(Deserialize, Default)]
struct MistralModelsResponse {
    _object: String,
    data: Vec<Value>,
}

#[derive(Deserialize, Default)]
struct XaiModelsResponse {
    _object: String,
    data: Vec<Value>,
}

#[derive(Deserialize, Default)]
struct TogetherModel {
    id: String,
    context_length: u64,
    pricing: TogetherPricing,
    #[serde(default)]
    description: String,
}

#[derive(Deserialize, Default)]
struct TogetherPricing {
    input: f64,
    output: f64,
}

#[derive(Deserialize, Default)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Deserialize, Default)]
struct OpenRouterModel {
    id: String,
    context_length: u64,
    pricing: OpenRouterPricing,
    description: String,
}

#[derive(Deserialize, Default)]
struct OpenRouterPricing {
    prompt: String,
    completion: String,
}


fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    println!("cargo:warning=Starting additive model generation system");
    
    let src_dir = Path::new("src");
    
    // Target generated_models.rs for additive updates
    let generated_models_path = src_dir.join("generated_models.rs");
    
    // Create supporting files
    let provider_path = src_dir.join("provider.rs");
    let model_info_path = src_dir.join("model_info.rs");
    
    let _provider_file = File::create(&provider_path).context("Failed to create provider.rs")?;
    let _model_info_file = File::create(&model_info_path).context("Failed to create model_info.rs")?;

    // Function to additively generate enum and impl for a provider
    fn generate_enum_code(
        provider_name: &str,
        models: &[(String, u64, f64, f64, bool, Option<f64>)],
        generated_models_path: &PathBuf,
        should_regenerate: bool,
    ) -> Result<()> {
        if !should_regenerate {
            println!("cargo:warning=Skipping code generation for {} (no changes detected)", provider_name);
            return Ok(());
        }
        
        // Parse existing generated_models.rs to extract current variants
        let existing_variants = if generated_models_path.exists() {
            parse_existing_enum_variants(generated_models_path, provider_name)?
        } else {
            vec![]
        };
        
        // Filter models to only include new ones not in existing variants
        let new_models: Vec<_> = models.iter()
            .filter(|(id, ..)| {
                let variant_name = sanitize_ident(id).to_string();
                !existing_variants.contains(&variant_name)
            })
            .cloned()
            .collect();
        
        if new_models.is_empty() {
            println!("cargo:warning=No new models found for {} - skipping generation", provider_name);
            return Ok(());
        }
        
        // Append new models to existing enum by modifying the file
        append_new_models_to_enum(generated_models_path, provider_name, &new_models)?;
        Ok(())
    }

    // OpenAI
    let mut openai_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    let mut api_response_available = false;
    
    if let Ok(key) = env::var("OPENAI_API_KEY") {
        let stream = Http3::json()
            .api_key(&key)
            .max_age(86400)
            .get("https://api.openai.com/v1/models");
        let responses: Vec<OpenAiModelsResponse> = HttpStreamExt::collect(stream);
        if let Some(response) = responses.into_iter().next() {
            api_response_available = true;
            
            for m in response.data {
                let id = m["id"].as_str().unwrap_or("").to_string();
                if id.is_empty() {
                    continue;
                }
                let (context, input, output, thinking, temp) = match id.as_str() {
                    "gpt-4.1" => (128000, 2.0, 8.0, false, None),
                    "gpt-4.1-mini" => (128000, 0.4, 1.6, false, None),
                    "o3" => (200000, 3.0, 12.0, true, Some(1.0)),
                    "o4-mini" => (128000, 1.1, 4.4, true, Some(1.0)),
                    "gpt-4o" => (128000, 5.0, 15.0, false, None),
                    "gpt-4o-mini" => (128000, 0.15, 0.6, false, None),
                    _ => (8192, 0.0, 0.0, false, None),
                };
                openai_models.push((id, context, input, output, thinking, temp));
            }
        }
    }
    
    if !api_response_available {
        // Use fallback static data when API is unavailable
        openai_models = vec![
            ("gpt-4.1".to_string(), 128000, 2.0, 8.0, false, None),
            ("gpt-4.1-mini".to_string(), 128000, 0.4, 1.6, false, None),
            ("o3".to_string(), 200000, 3.0, 12.0, true, Some(1.0)),
            ("o4-mini".to_string(), 128000, 1.1, 4.4, true, Some(1.0)),
            ("gpt-4o".to_string(), 128000, 5.0, 15.0, false, None),
            ("gpt-4o-mini".to_string(), 128000, 0.15, 0.6, false, None),
        ];
    }

    // Mistral
    let mut mistral_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    if let Ok(key) = env::var("MISTRAL_API_KEY") {
        let stream = Http3::json()
            .api_key(&key)
            .max_age(86400)
            .get("https://api.mistral.ai/v1/models");
        let responses: Vec<MistralModelsResponse> = HttpStreamExt::collect(stream);
        let response = responses.into_iter().next().unwrap_or_default();
        
        for m in response.data {
            let id = m["id"].as_str().unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }
            let (context, input, output, thinking, temp) = match id.as_str() {
                "mistral-large-2407" => (128000, 8.0, 24.0, false, None),
                "mistral-large-2312" => (32000, 3.0, 9.0, false, None),
                "mistral-small-2409" => (128000, 0.2, 0.6, false, None),
                "mistral-nemo-2407" => (128000, 0.2, 0.6, false, None),
                "open-mistral-nemo" => (128000, 0.3, 1.0, false, None),
                "codestral-2405" => (32000, 0.8, 2.5, false, None),
                "mistral-embed" => (8192, 0.1, 0.1, false, None),
                "mistral-tiny" => (32000, 0.25, 0.75, false, None),
                "mistral-small" => (32000, 2.0, 6.0, false, None),
                "mistral-medium" => (32000, 8.0, 24.0, false, None),
                "mistral-large" => (32000, 20.0, 60.0, false, None),
                _ => (8192, 0.0, 0.0, false, None),
            };
            mistral_models.push((id, context, input, output, thinking, temp));
        }
    }
    if mistral_models.is_empty() {
        mistral_models = vec![
            ("mistral-large-2407".to_string(), 128000, 8.0, 24.0, false, None),
            ("mistral-large-2312".to_string(), 32000, 3.0, 9.0, false, None),
            ("mistral-small-2409".to_string(), 128000, 0.2, 0.6, false, None),
            ("mistral-nemo-2407".to_string(), 128000, 0.2, 0.6, false, None),
            ("open-mistral-nemo".to_string(), 128000, 0.3, 1.0, false, None),
            ("codestral-2405".to_string(), 32000, 0.8, 2.5, false, None),
            ("mistral-embed".to_string(), 8192, 0.1, 0.1, false, None),
            ("mistral-tiny".to_string(), 32000, 0.25, 0.75, false, None),
            ("mistral-small".to_string(), 32000, 2.0, 6.0, false, None),
            ("mistral-medium".to_string(), 32000, 8.0, 24.0, false, None),
            ("mistral-large".to_string(), 32000, 20.0, 60.0, false, None),
        ];
    }

    // Anthropic - no dynamic list endpoint, hardcoded
    let anthropic_models = vec![
        ("claude-3-5-sonnet-20240620".to_string(), 200000, 3.0, 15.0, false, None),
        ("claude-3-haiku-20240307".to_string(), 200000, 0.25, 1.25, false, None),
        ("claude-3-opus-20240229".to_string(), 200000, 15.0, 75.0, false, None),
    ];

    // Together
    let mut together_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    let together_result: Result<Vec<TogetherModel>, anyhow::Error> = (|| {
        let mut builder = Http3::json().max_age(86400);
        if let Ok(key) = env::var("TOGETHER_API_KEY") {
            builder = builder.api_key(&key);
        }
        let stream = builder.get("https://api.together.ai/v1/models");
        Ok(HttpStreamExt::collect(stream))
    })();
    
    if let Ok(response) = together_result {
        for m in response.into_iter().take(20) {
            let is_thinking = m.description.to_lowercase().contains("reasoning") || m.id.contains("reasoning");
            let temp = if is_thinking { Some(1.0) } else { None };
            together_models.push((m.id, m.context_length, m.pricing.input, m.pricing.output, is_thinking, temp));
        }
    }
    if together_models.is_empty() {
        together_models = vec![
            ("meta-llama/Llama-3-8b-chat-hf".to_string(), 8192, 0.2, 0.2, false, None),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(), 32768, 0.27, 0.27, false, None),
            ("togethercomputer/CodeLlama-34b-Instruct".to_string(), 16384, 0.5, 0.5, false, None),
        ];
    }

    // OpenRouter
    let mut openrouter_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    let openrouter_result: Result<OpenRouterModelsResponse, anyhow::Error> = (|| {
        let stream = Http3::json()
            .max_age(86400)
            .get("https://openrouter.ai/api/v1/models");
        let responses: Vec<OpenRouterModelsResponse> = HttpStreamExt::collect(stream);
        Ok(responses.into_iter().next().unwrap_or_default())
    })();
    
    if let Ok(response) = openrouter_result {
        for m in response.data.into_iter().take(20) {
            let input = m.pricing.prompt.parse::<f64>().unwrap_or(0.0) * 1_000_000.0;
            let output = m.pricing.completion.parse::<f64>().unwrap_or(0.0) * 1_000_000.0;
            let is_thinking = m.description.to_lowercase().contains("reasoning") || m.id.contains("o1") || m.id.contains("o3");
            let temp = if is_thinking { Some(1.0) } else { None };
            openrouter_models.push((m.id, m.context_length, input, output, is_thinking, temp));
        }
    }
    if openrouter_models.is_empty() {
        openrouter_models = vec![
            ("openai/gpt-4o".to_string(), 128000, 5.0, 15.0, false, None),
            ("anthropic/claude-3.5-sonnet".to_string(), 200000, 3.0, 15.0, false, None),
            ("google/gemini-pro-1.5".to_string(), 1000000, 0.5, 1.5, false, None),
        ];
    }

    // HuggingFace
    let mut huggingface_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    let hf_result: Result<Vec<Value>, anyhow::Error> = (|| {
        let mut builder = Http3::json().max_age(86400);
        if let Ok(key) = env::var("HF_API_KEY") {
            builder = builder.api_key(&key);
        }
        let stream = builder.get("https://api.huggingface.co/models?sort=downloads&direction=-1&limit=20");
        Ok(HttpStreamExt::collect(stream))
    })();
    
    if let Ok(response) = hf_result {
        for m in response {
            let id = m["id"].as_str().unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }
            let context = m["config"]["max_position_embeddings"].as_u64().unwrap_or(m["max_length"].as_u64().unwrap_or(8192));
            let input = 0.0;
            let output = 0.0;
            let is_thinking = m["tags"].as_array().map(|t| t.iter().any(|tag| tag.as_str().unwrap_or("").contains("reasoning"))) .unwrap_or(false);
            let temp = if is_thinking { Some(1.0) } else { None };
            huggingface_models.push((id, context, input, output, is_thinking, temp));
        }
    }
    if huggingface_models.is_empty() {
        huggingface_models = vec![
            ("meta-llama/Meta-Llama-3-8B-Instruct".to_string(), 8192, 0.0, 0.0, false, None),
            ("mistralai/Mistral-7B-Instruct-v0.3".to_string(), 32768, 0.0, 0.0, false, None),
            ("google/gemma-2-9b-it".to_string(), 8192, 0.0, 0.0, false, None),
        ];
    }

    // xAI
    let mut xai_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    if let Ok(key) = env::var("XAI_API_KEY") {
        let stream = Http3::json()
            .api_key(&key)
            .max_age(86400)
            .get("https://api.x.ai/v1/models");
        let responses: Vec<XaiModelsResponse> = HttpStreamExt::collect(stream);
        let response = responses.into_iter().next().unwrap_or_default();
        
        for m in response.data {
            let id = m["id"].as_str().unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }
            let (context, input, output, thinking, temp) = match id.as_str() {
                "grok-4" => (256000, 3.0, 15.0, true, Some(1.0)),
                "grok-3" => (131072, 3.0, 15.0, true, Some(1.0)),
                "grok-3-mini" => (131072, 0.3, 0.5, true, None),
                _ => (8192, 0.0, 0.0, true, None),
            };
            xai_models.push((id, context, input, output, thinking, temp));
        }
    }
    if xai_models.is_empty() {
        xai_models = vec![
            ("grok-4".to_string(), 256000, 3.0, 15.0, true, Some(1.0)),
            ("grok-3".to_string(), 131072, 3.0, 15.0, true, Some(1.0)),
            ("grok-3-mini".to_string(), 131072, 0.3, 0.5, true, None),
        ];
    }
    // Perform additive enum generation with new models only
    println!("cargo:warning=Performing additive model generation");
    
    // TODO: Implement additive generation functions
    // For now, generate all enums (will be replaced with additive logic)
    generate_enum_code("OpenAiModel", &openai_models, &generated_models_path, true)?;
    generate_enum_code("MistralModel", &mistral_models, &generated_models_path, true)?;
    generate_enum_code("AnthropicModel", &anthropic_models, &generated_models_path, true)?;
    generate_enum_code("TogetherModel", &together_models, &generated_models_path, true)?;
    generate_enum_code("OpenRouterModel", &openrouter_models, &generated_models_path, true)?;
    generate_enum_code("HuggingFaceModel", &huggingface_models, &generated_models_path, true)?;
    generate_enum_code("XaiModel", &xai_models, &generated_models_path, true)?;
    
    println!("cargo:warning=Additive model generation completed");

    Ok(())
}