use anyhow::{anyhow, Context, Result};
use quote::quote;
use fluent_ai_http3::{Http3, HttpStreamExt};
use serde::Deserialize;
use serde_json::Value;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use syn::Ident;
use proc_macro2::Span;

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
    let out_dir = env::var_os("OUT_DIR").ok_or(anyhow!("OUT_DIR not set"))?;
    let dest_path = Path::new(&out_dir).join("generated_models.rs");
    let mut f = File::create(&dest_path).context("Failed to create generated file")?;

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

    // Function to generate enum and impl for a provider
    fn generate_enum_code(
        provider_name: &str,
        models: &[(String, u64, f64, f64, bool, Option<f64>)],
        f: &mut File,
    ) -> Result<()> {
        let enum_name = syn::Ident::new(&format!("{}Model", provider_name), Span::call_site());
        let variants = models.iter().map(|(id, ..)| sanitize_ident(id)).collect::<Vec<_>>();
        let name_matches = models.iter().map(|(id, ..)| {
            let variant = sanitize_ident(id);
            let name = id.as_str();
            quote! { #enum_name::#variant => #name }
        }).collect::<Vec<_>>();
        let context_matches = models.iter().map(|(id, context, ..)| {
            let variant = sanitize_ident(id);
            quote! { #enum_name::#variant => #context }
        }).collect::<Vec<_>>();
        let input_matches = models.iter().map(|(id, _, input, ..)| {
            let variant = sanitize_ident(id);
            quote! { #enum_name::#variant => #input }
        }).collect::<Vec<_>>();
        let output_matches = models.iter().map(|(id, _, _, output, ..)| {
            let variant = sanitize_ident(id);
            quote! { #enum_name::#variant => #output }
        }).collect::<Vec<_>>();
        let thinking_matches = models.iter().map(|(id, _, _, _, thinking, ..)| {
            let variant = sanitize_ident(id);
            quote! { #enum_name::#variant => #thinking }
        }).collect::<Vec<_>>();
        let temp_matches = models.iter().map(|(id, _, _, _, _, temp)| {
            let variant = sanitize_ident(id);
            let t = match temp {
                Some(v) => quote! { Some(#v) },
                None => quote! { None },
            };
            quote! { #enum_name::#variant => #t }
        }).collect::<Vec<_>>();

        let code = quote! {
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            pub enum #enum_name {
                #(#variants,)*
            }

            impl crate::common::Model for #enum_name {
                fn name(&self) -> &'static str {
                    match self {
                        #(#name_matches,)*
                    }
                }
                fn max_context_length(&self) -> u64 {
                    match self {
                        #(#context_matches,)*
                    }
                }
                fn pricing_input(&self) -> f64 {
                    match self {
                        #(#input_matches,)*
                    }
                }
                fn pricing_output(&self) -> f64 {
                    match self {
                        #(#output_matches,)*
                    }
                }
                fn is_thinking(&self) -> bool {
                    match self {
                        #(#thinking_matches,)*
                    }
                }
                fn required_temperature(&self) -> Option<f64> {
                    match self {
                        #(#temp_matches,)*
                    }
                }
            }

        };
        f.write_all(code.to_string().as_bytes()).context("Failed to write generated code")?;
        Ok(())
    }

    // OpenAI
    let mut openai_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    if let Ok(key) = env::var("OPENAI_API_KEY") {
        let stream = Http3::json()
            .api_key(&key)
            .get("https://api.openai.com/v1/models");
        let response: OpenAiModelsResponse = HttpStreamExt::collect(stream);
        
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
    if openai_models.is_empty() {
        openai_models = vec![
            ("gpt-4.1".to_string(), 128000, 2.0, 8.0, false, None),
            ("gpt-4.1-mini".to_string(), 128000, 0.4, 1.6, false, None),
            ("o3".to_string(), 200000, 3.0, 12.0, true, Some(1.0)),
            ("o4-mini".to_string(), 128000, 1.1, 4.4, true, Some(1.0)),
            ("gpt-4o".to_string(), 128000, 5.0, 15.0, false, None),
            ("gpt-4o-mini".to_string(), 128000, 0.15, 0.6, false, None),
        ];
    }
    generate_enum_code("OpenAi", &openai_models, &mut f)?;

    // Mistral
    let mut mistral_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    if let Ok(key) = env::var("MISTRAL_API_KEY") {
        let stream = Http3::json()
            .api_key(&key)
            .get("https://api.mistral.ai/v1/models");
        let response: MistralModelsResponse = HttpStreamExt::collect(stream);
        
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
    generate_enum_code("Mistral", &mistral_models, &mut f)?;

    // Anthropic - no dynamic list endpoint, hardcoded
    let anthropic_models = vec![
        ("claude-3-5-sonnet-20240620".to_string(), 200000, 3.0, 15.0, false, None),
        ("claude-3-haiku-20240307".to_string(), 200000, 0.25, 1.25, false, None),
        ("claude-3-opus-20240229".to_string(), 200000, 15.0, 75.0, false, None),
    ];
    generate_enum_code("Anthropic", &anthropic_models, &mut f)?;

    // Together
    let mut together_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    let together_result: Result<Vec<TogetherModel>, anyhow::Error> = (|| {
        let mut builder = Http3::json();
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
    generate_enum_code("Together", &together_models, &mut f)?;

    // OpenRouter
    let mut openrouter_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    let openrouter_result: Result<OpenRouterModelsResponse, anyhow::Error> = (|| {
        let stream = Http3::json()
            .get("https://openrouter.ai/api/v1/models");
        Ok(HttpStreamExt::collect(stream))
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
    generate_enum_code("OpenRouter", &openrouter_models, &mut f)?;

    // HuggingFace
    let mut huggingface_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    let hf_result: Result<Vec<Value>, anyhow::Error> = (|| {
        let mut builder = Http3::json();
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
    generate_enum_code("HuggingFace", &huggingface_models, &mut f)?;

    // xAI
    let mut xai_models: Vec<(String, u64, f64, f64, bool, Option<f64>)> = vec![];
    if let Ok(key) = env::var("XAI_API_KEY") {
        let stream = Http3::json()
            .api_key(&key)
            .get("https://api.x.ai/v1/models");
        let response: XaiModelsResponse = HttpStreamExt::collect(stream);
        
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
    generate_enum_code("Xai", &xai_models, &mut f)?;

    Ok(())
}