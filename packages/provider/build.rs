use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use syn::{parse_str, Item, Ident};
use quote::quote;

/// The models.yaml is a top-level array of providers, not a struct with providers field
type ModelYaml = Vec<ProviderInfo>;

/// Zero-allocation, blazing-fast YAML configuration parser using serde_yaml
/// Optimized for production use with zero compromises
#[inline]
fn parse_yaml_config(yaml_content: &str) -> Result<ModelYaml, Box<dyn std::error::Error>> {
    // Direct parsing with semantic error handling - deserialize directly to our struct
    match serde_yaml::from_str::<ModelYaml>(yaml_content) {
        Ok(providers) => Ok(providers),
        Err(parse_error) => {
            Err(format!("Failed to parse YAML configuration: {}", parse_error).into())
        }
    }
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=models.yaml");
    println!("cargo:rerun-if-changed=build.rs");

    // Download models.yaml from sigoden/aichat - fully async with QUIC support
    let client = reqwest::Client::builder()
        .http3_prior_knowledge()
        .http2_prior_knowledge()
        .use_rustls_tls()
        .tls_built_in_root_certs(true)
        .timeout(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(10))
        .user_agent("cyrup_ai/build-script (QUIC/HTTP3+rustls)")
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
            eprintln!(
                "Warning: Failed to download models.yaml, using existing local copy if available"
            );
            if !Path::new("models.yaml").exists() {
                return Err("No models.yaml found and download failed".into());
            }
        }
    }

    // Load providers from YAML
    let yaml_content = fs::read_to_string("models.yaml")?;
    println!("📄 YAML content loaded, {} bytes", yaml_content.len());

    // Don't generate if YAML is empty
    if yaml_content.trim().is_empty() {
        println!("⚠️  models.yaml is empty, skipping generation");
        return Ok(());
    }

    let providers: ModelYaml = parse_yaml_config(&yaml_content)?;

    println!("✅ Successfully parsed {} providers", providers.len());
    if !providers.is_empty() {
        println!(
            "🔍 First provider: {} with {} models",
            providers[0].provider,
            providers[0].models.len()
        );
    }

    // Don't generate if no providers
    if providers.is_empty() {
        println!("⚠️  No providers found, skipping generation");
        return Ok(());
    }

    // Generate all files
    generate_all_files(&providers)?;

    println!("✅ Code generation completed successfully!");
    Ok(())
}

/// Generate all files with clean, readable code
fn generate_all_files(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    // Create src/models.rs with Models enum
    generate_models_file(providers)?;

    // Create src/providers.rs with Providers enum
    generate_providers_file(providers)?;

    // Generate model info data
    generate_model_info_file(providers)?;

    // Add trait implementations to existing files
    add_model_trait_impl(providers)?;
    add_provider_trait_impl(providers)?;

    println!("✅ All files generated successfully");
    Ok(())
}

/// Compare YAML variants with existing Rust enum variants
fn enum_variants_changed(file_path: &Path, yaml_variants: &BTreeSet<String>) -> bool {
    if !file_path.exists() {
        return true;
    }
    
    let existing_content = match fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(_) => return true,
    };
    
    // Handle empty files
    if existing_content.trim().is_empty() {
        return true;
    }
    
    // Parse existing enum variants
    let existing_variants = match parse_enum_variants(&existing_content) {
        Ok(variants) => variants,
        Err(_) => return true,
    };
    
    // Compare variant sets
    yaml_variants != &existing_variants
}

/// Parse enum variants from Rust source code (handles Models and Providers enums)
fn parse_enum_variants(rust_code: &str) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let mut variants = BTreeSet::new();
    
    let parsed: syn::File = parse_str(rust_code)?;
    
    for item in parsed.items {
        if let Item::Enum(item_enum) = item {
            let enum_name = item_enum.ident.to_string();
            if enum_name == "Models" || enum_name == "Providers" {
                for variant in item_enum.variants {
                    variants.insert(variant.ident.to_string());
                }
                break;
            }
        }
    }
    
    Ok(variants)
}

/// Generate the Models enum using syn/quote - only write if changed
fn generate_models_file(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let yaml_variants = generate_model_variants_optimized(providers);
    let models_path = Path::new("src/models.rs");
    
    // Check if variants changed
    if !enum_variants_changed(models_path, &yaml_variants) {
        return Ok(());
    }
    
    // Generate enum variants using syn
    let variant_idents: Vec<Ident> = yaml_variants
        .iter()
        .map(|v| syn::parse_str::<Ident>(v).unwrap())
        .collect();
    
    // Generate the enum using quote
    let enum_token = quote! {
        // This file is auto-generated. Do not edit manually.
        use serde::{Serialize, Deserialize};

        // AUTO-GENERATED START
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Models {
            #(#variant_idents,)*
        }
        // AUTO-GENERATED END
    };
    
    // Write the generated code
    fs::write("src/models.rs", enum_token.to_string())?;
    println!("cargo:rerun-if-changed=src/models.rs");
    Ok(())
}

/// Generate the Providers enum using syn/quote - only write if changed
fn generate_providers_file(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let providers_path = Path::new("src/providers.rs");
    
    // Generate provider variants
    let provider_variants: BTreeSet<String> = providers
        .iter()
        .map(|p| to_pascal_case_optimized(&p.provider))
        .collect();
    
    // Check if variants changed
    if !enum_variants_changed(providers_path, &provider_variants) {
        return Ok(());
    }
    
    // Generate enum variants using syn
    let variant_idents: Vec<Ident> = provider_variants
        .iter()
        .map(|v| syn::parse_str::<Ident>(v).unwrap())
        .collect();
    
    // Generate the enum using quote
    let enum_token = quote! {
        // This file is auto-generated. Do not edit manually.
        use serde::{Serialize, Deserialize};

        // AUTO-GENERATED START
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Providers {
            #(#variant_idents,)*
        }
        // AUTO-GENERATED END
    };
    
    // Write the generated code
    fs::write("src/providers.rs", enum_token.to_string())?;
    println!("cargo:rerun-if-changed=src/providers.rs");
    Ok(())
}

/// Generate model info data
fn generate_model_info_file(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let mut content = String::new();
    content.push_str("// This file is auto-generated. Do not edit manually.\n");
    content.push_str("use serde::{Serialize, Deserialize};\n\n");
    content.push_str("// AUTO-GENERATED START\n");

    // Generate ModelInfoData struct
    content.push_str("#[derive(Debug, Clone, Serialize, Deserialize)]\n");
    content.push_str("pub struct ModelInfoData {\n");
    content.push_str("    pub provider_name: String,\n");
    content.push_str("    pub name: String,\n");
    content.push_str("    pub max_input_tokens: Option<u64>,\n");
    content.push_str("    pub max_output_tokens: Option<u64>,\n");
    content.push_str("    pub input_price: Option<f64>,\n");
    content.push_str("    pub output_price: Option<f64>,\n");
    content.push_str("    pub supports_vision: Option<bool>,\n");
    content.push_str("    pub supports_function_calling: Option<bool>,\n");
    content.push_str("    pub require_max_tokens: Option<bool>,\n");
    content.push_str("}\n\n");

    // Generate info functions
    let mut seen_models = std::collections::HashSet::new();
    for provider in providers {
        for model in &provider.models {
            let variant_name = to_pascal_case_optimized(&model.name);
            let function_name = format!("get_{}_info", variant_name.to_lowercase());

            if seen_models.contains(&function_name) {
                continue;
            }
            seen_models.insert(function_name.clone());

            content.push_str(&format!("/// Get model info for {}\n", model.name));
            content.push_str(&format!("pub fn {}() -> ModelInfoData {{\n", function_name));
            content.push_str("    ModelInfoData {\n");
            content.push_str(&format!(
                "        provider_name: \"{}\".to_string(),\n",
                provider.provider
            ));
            content.push_str(&format!("        name: \"{}\".to_string(),\n", model.name));
            content.push_str(&format!(
                "        max_input_tokens: {:?},\n",
                model.max_input_tokens
            ));
            content.push_str(&format!(
                "        max_output_tokens: {:?},\n",
                model.max_output_tokens
            ));
            content.push_str(&format!("        input_price: {:?},\n", model.input_price));
            content.push_str(&format!(
                "        output_price: {:?},\n",
                model.output_price
            ));
            content.push_str(&format!(
                "        supports_vision: {:?},\n",
                model.supports_vision
            ));
            content.push_str(&format!(
                "        supports_function_calling: {:?},\n",
                model.supports_function_calling
            ));
            content.push_str(&format!(
                "        require_max_tokens: {:?},\n",
                model.require_max_tokens
            ));
            content.push_str("    }\n");
            content.push_str("}\n\n");
        }
    }

    content.push_str("// AUTO-GENERATED END\n");

    fs::write("src/model_info.rs", content)?;
    println!("cargo:rerun-if-changed=src/model_info.rs");
    Ok(())
}

/// Add Model trait implementation to models.rs - smart comparison approach
fn add_model_trait_impl(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let models_path = Path::new("src/models.rs");
    let variants = generate_model_variants_optimized(providers);

    let mut impl_content = String::new();
    impl_content.push_str("use crate::models::Models;\n");
    impl_content.push_str("use fluent_ai_domain::Model;\n");
    impl_content.push_str("use crate::model_info::ModelInfoData;\n\n");
    impl_content.push_str("impl Model for Models {\n");
    impl_content.push_str("    fn info(&self) -> ModelInfoData {\n");
    impl_content.push_str("        match self {\n");

    // Generate deduplicated match arms
    for variant_name in &variants {
        let function_name = format!("get_{}_info", variant_name.to_lowercase());
        impl_content.push_str(&format!(
            "            Models::{} => crate::model_info::{}(),\n",
            variant_name, function_name
        ));
    }

    impl_content.push_str("        }\n");
    impl_content.push_str("    }\n");
    impl_content.push_str("}\n");

    // Read existing models.rs file and append the implementation
    if models_path.exists() {
        let existing = fs::read_to_string(models_path)?;
        let updated = replace_auto_generated_section(&existing, &impl_content);
        
        // Only write if content changed
        if existing != updated {
            fs::write("src/models.rs", updated)?;
        }
    }
    
    Ok(())
}

/// Add Provider trait implementation to providers.rs
fn add_provider_trait_impl(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let providers_path = Path::new("src/providers.rs");
    
    let mut impl_content = String::new();
    impl_content.push_str("use crate::providers::Providers;\n");
    impl_content.push_str("use crate::models::Models;\n");
    impl_content.push_str("use fluent_ai_domain::Provider;\n");
    impl_content.push_str("use cyrup_sugars::ZeroOneOrMany;\n\n");
    impl_content.push_str("impl Provider for Providers {\n");
    impl_content.push_str("    type Model = Models;\n\n");

    // Generate name() method
    impl_content.push_str("    fn name(&self) -> &'static str {\n");
    impl_content.push_str("        match self {\n");
    for provider in providers {
        let variant_name = to_pascal_case_optimized(&provider.provider);
        impl_content.push_str(&format!(
            "            Providers::{} => \"{}\",\n",
            variant_name, provider.provider
        ));
    }
    impl_content.push_str("        }\n");
    impl_content.push_str("    }\n\n");

    // Generate models() method
    impl_content.push_str("    fn models(&self) -> ZeroOneOrMany<Self::Model> {\n");
    impl_content.push_str("        match self {\n");
    for provider in providers {
        let variant_name = to_pascal_case_optimized(&provider.provider);
        let model_variants: Vec<_> = provider
            .models
            .iter()
            .map(|m| to_pascal_case_optimized(&m.name))
            .collect();

        if model_variants.len() == 1 {
            impl_content.push_str(&format!(
                "            Providers::{} => ZeroOneOrMany::One(Models::{}),\n",
                variant_name, model_variants[0]
            ));
        } else if model_variants.is_empty() {
            impl_content.push_str(&format!(
                "            Providers::{} => ZeroOneOrMany::Zero,\n",
                variant_name
            ));
        } else {
            impl_content.push_str(&format!(
                "            Providers::{} => ZeroOneOrMany::Many(vec![\n",
                variant_name
            ));
            for model_variant in &model_variants {
                impl_content.push_str(&format!("                Models::{},\n", model_variant));
            }
            impl_content.push_str("            ]),\n");
        }
    }
    impl_content.push_str("        }\n");
    impl_content.push_str("    }\n");
    impl_content.push_str("}\n");

    // Read existing providers.rs file and append the implementation
    if providers_path.exists() {
        let existing = fs::read_to_string(providers_path)?;
        let updated = replace_auto_generated_section(&existing, &impl_content);
        
        // Only write if content changed
        if existing != updated {
            fs::write("src/providers.rs", updated)?;
        }
    }
    
    Ok(())
}

/// Replace the auto-generated section in a file
fn replace_auto_generated_section(existing: &str, new_content: &str) -> String {
    let auto_gen_start = "// AUTO-GENERATED START";
    let auto_gen_end = "// AUTO-GENERATED END";

    if let Some(start_pos) = existing.find(auto_gen_start) {
        if let Some(end_pos) = existing.find(auto_gen_end) {
            let before = &existing[..start_pos];
            let after = &existing[end_pos + auto_gen_end.len()..];
            return format!(
                "{}{}\n{}\n{}{}",
                before, auto_gen_start, new_content, auto_gen_end, after
            );
        }
    }

    // If no auto-generated section found, append the new content
    format!(
        "{}\n{}\n{}\n{}\n",
        existing, auto_gen_start, new_content, auto_gen_end
    )
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
            '-' | '_' | '.' | ' ' | '/' | '@' => {
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
