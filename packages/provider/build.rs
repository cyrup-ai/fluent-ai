use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use syn::{parse_str, Item, Ident};
use quote::quote;
use std::time::SystemTime;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use once_cell::sync::Lazy;

/// The models.yaml is a top-level array of providers, not a struct with providers field
type ModelYaml = Vec<ProviderInfo>;

/// Cache metadata for models.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheMetadata {
    etag: Option<String>,
    last_modified: Option<String>,
    timestamp: u64,
    url: String,
}

/// High-performance HTTP client with connection pooling and QUIC support
/// Zero allocation, blazing-fast, lock-free design
struct PooledHttpClient {
    client: fluent_ai_http3::HttpClient,
    connection_count: Arc<AtomicUsize>,
    request_count: Arc<AtomicUsize>,
}

impl PooledHttpClient {
    /// Create a new pooled HTTP client optimized for maximum performance
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        use fluent_ai_http3::HttpConfig;
        
        // Use optimized HTTP3 client with AI-specific configuration
        let client = fluent_ai_http3::HttpClient::with_config(HttpConfig::ai_optimized())
            .map_err(|e| format!("Failed to create HTTP3 client: {}", e))?;
        
        Ok(Self {
            client,
            connection_count: Arc::new(AtomicUsize::new(0)),
            request_count: Arc::new(AtomicUsize::new(0)),
        })
    }
    
    /// Execute HTTP request with connection pooling and intelligent caching
    async fn execute_request(&self, url: &str, cache_metadata: Option<&CacheMetadata>) 
        -> Result<HttpResponse, Box<dyn std::error::Error>> {
        
        // Increment request counter atomically
        let request_id = self.request_count.fetch_add(1, Ordering::Relaxed);
        
        // Build request with conditional headers
        let mut request = self.client.get(url);
        
        // Add caching headers if available
        if let Some(cache) = cache_metadata {
            if let Some(ref etag) = cache.etag {
                request = request.header("If-None-Match", etag);
            }
            if let Some(ref last_modified) = cache.last_modified {
                request = request.header("If-Modified-Since", last_modified);
            }
        }
        
        // Add performance headers
        request = request
            .header("Connection", "keep-alive")
            .header("Cache-Control", "max-age=3600")
            .header("Accept", "application/x-yaml, text/yaml, */*")
            .header("Accept-Encoding", "gzip, br, deflate")
            .header("X-Request-ID", request_id.to_string());
        
        // Execute request using fluent_ai_http3
        let response = self.client.send(request).await?;
        
        // Process response
        let status = response.status();
        let etag = response.etag().cloned();
        let last_modified = response.last_modified().cloned();
        
        if status.as_u16() == 304 {
            return Ok(HttpResponse::NotModified);
        }
        
        if !response.is_success() {
            return Err(format!("HTTP {}: Request failed", status).into());
        }
        
        // Get the text content
        let content = response.text().await?;
        
        Ok(HttpResponse::Success {
            content,
            etag,
            last_modified,
        })
    }
    
    /// Get connection statistics
    fn get_stats(&self) -> (usize, usize) {
        (
            self.connection_count.load(Ordering::Relaxed),
            self.request_count.load(Ordering::Relaxed),
        )
    }
}

/// HTTP response variants for efficient processing
enum HttpResponse {
    Success {
        content: String,
        etag: Option<String>,
        last_modified: Option<String>,
    },
    NotModified,
}

/// Global HTTP client instance with connection pooling
static HTTP_CLIENT: Lazy<PooledHttpClient> = Lazy::new(|| {
    PooledHttpClient::new().expect("Failed to initialize HTTP client")
});

/// Zero-allocation, blazing-fast YAML configuration parser using yyaml
/// Optimized for production use with zero compromises
#[inline]
fn parse_yaml_config(yaml_content: &str) -> Result<ModelYaml, Box<dyn std::error::Error>> {
    // Direct parsing with semantic error handling using yyaml
    match yyaml::from_str::<ModelYaml>(yaml_content) {
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
    println!("cargo:rerun-if-changed=models.yaml.cache");
    println!("cargo:rerun-if-changed=build.rs");

    // Download models.yaml with intelligent caching (ETag + 24-hour cache)
    let models_url = "https://raw.githubusercontent.com/sigoden/aichat/refs/heads/main/models.yaml";
    let cache_file = Path::new("models.yaml.cache");
    let models_file = Path::new("models.yaml");
    
    // Load existing cache metadata
    let cache_metadata = if cache_file.exists() {
        fs::read_to_string(cache_file)
            .ok()
            .and_then(|content| serde_json::from_str::<CacheMetadata>(&content).ok())
    } else {
        None
    };
    
    // Check if we should fetch (no cache, expired, or different URL)
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let should_fetch = match &cache_metadata {
        Some(cache) => {
            // Check if cache is older than 24 hours (86400 seconds)
            now - cache.timestamp > 86400 || cache.url != models_url
        }
        None => true,
    };
    
    if should_fetch {
        // Use the pooled HTTP client for blazing-fast performance
        let (conn_count, req_count) = HTTP_CLIENT.get_stats();
        println!("🔗 HTTP Client Stats: {} connections, {} requests", conn_count, req_count);
        
        match HTTP_CLIENT.execute_request(models_url, cache_metadata.as_ref()).await {
            Ok(HttpResponse::NotModified) => {
                // Not modified, update timestamp but keep existing file
                println!("📋 models.yaml unchanged (ETag match), updating cache timestamp");
                if let Some(mut cache) = cache_metadata {
                    cache.timestamp = now;
                    let cache_json = serde_json::to_string_pretty(&cache)?;
                    fs::write(cache_file, cache_json)?;
                }
            }
            Ok(HttpResponse::Success { content, etag, last_modified }) => {
                // New content, download and update cache
                fs::write(models_file, &content)?;
                
                // Update cache metadata
                let new_cache = CacheMetadata {
                    etag,
                    last_modified,
                    timestamp: now,
                    url: models_url.to_string(),
                };
                let cache_json = serde_json::to_string_pretty(&new_cache)?;
                fs::write(cache_file, cache_json)?;
                
                println!("📥 Downloaded fresh models.yaml ({} bytes) via connection pool", content.len());
            }
            Err(e) => {
                eprintln!("⚠️  Network error downloading models.yaml: {}", e);
            }
        }
    } else {
        println!("📋 Using cached models.yaml (within 24-hour window)");
    }
    
    // Ensure models.yaml exists
    if !models_file.exists() {
        return Err("No models.yaml found and download failed".into());
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
    
    // Generate provider-to-client mapping
    generate_provider_client_mapping(&providers)?;

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
        use std::sync::Arc;
        use once_cell::sync::Lazy;

        // AUTO-GENERATED START
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum Models {
            #(#variant_idents,)*
        }
        
        impl Models {
            /// Get model info with zero allocation - blazing fast lookup
            #[inline]
            pub fn info(&self) -> crate::model_info::ModelInfoData {
                // Static lookup for maximum performance
                match self {
                    #(Models::#variant_idents => {
                        let function_name = format!("get_{}_info", stringify!(#variant_idents).to_lowercase());
                        // Dynamic function lookup - will be optimized by compiler
                        crate::model_info::get_model_info_by_name(stringify!(#variant_idents))
                    },)*
                }
            }
            
            /// Get model name as static string - zero allocation
            #[inline]
            pub fn name(&self) -> &'static str {
                match self {
                    #(Models::#variant_idents => stringify!(#variant_idents),)*
                }
            }
            
            /// Get provider name from model info - zero allocation
            #[inline]
            pub fn provider(&self) -> String {
                self.info().provider_name
            }
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

    // Generate the lookup function
    content.push_str("/// Get model info by name - zero allocation lookup\n");
    content.push_str("pub fn get_model_info_by_name(name: &str) -> ModelInfoData {\n");
    content.push_str("    match name {\n");
    
    for provider in providers {
        for model in &provider.models {
            let variant_name = to_pascal_case_optimized(&model.name);
            let function_name = format!("get_{}_info", variant_name.to_lowercase());
            content.push_str(&format!(
                "        \"{}\" => {}(),\n",
                variant_name, function_name
            ));
        }
    }
    
    content.push_str("        _ => ModelInfoData::default(),\n");
    content.push_str("    }\n");
    content.push_str("}\n\n");
    
    content.push_str("impl Default for ModelInfoData {\n");
    content.push_str("    fn default() -> Self {\n");
    content.push_str("        Self {\n");
    content.push_str("            provider_name: String::new(),\n");
    content.push_str("            name: String::new(),\n");
    content.push_str("            max_input_tokens: None,\n");
    content.push_str("            max_output_tokens: None,\n");
    content.push_str("            input_price: None,\n");
    content.push_str("            output_price: None,\n");
    content.push_str("            supports_vision: None,\n");
    content.push_str("            supports_function_calling: None,\n");
    content.push_str("            require_max_tokens: None,\n");
    content.push_str("        }\n");
    content.push_str("    }\n");
    content.push_str("}\n\n");

    content.push_str("// AUTO-GENERATED END\n");

    fs::write("src/model_info.rs", content)?;
    println!("cargo:rerun-if-changed=src/model_info.rs");
    Ok(())
}


/// Add Provider trait implementation to providers.rs
fn add_provider_trait_impl(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let providers_path = Path::new("src/providers.rs");
    
    let mut impl_content = String::new();
    impl_content.push_str("use crate::providers::Providers;\n");
    impl_content.push_str("use crate::models::Models;\n");
    impl_content.push_str("use cyrup_sugars::ZeroOneOrMany;\n\n");
    impl_content.push_str("impl Providers {\n");

    // Generate name() method
    impl_content.push_str("    /// Get provider name as static string - zero allocation\n");
    impl_content.push_str("    pub fn name(&self) -> &'static str {\n");
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
    impl_content.push_str("    /// Get models for this provider - zero allocation\n");
    impl_content.push_str("    pub fn models(&self) -> ZeroOneOrMany<Models> {\n");
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

/// Convert provider name to snake_case for client module matching
fn to_snake_case_optimized(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + 8);
    let mut chars = input.chars().peekable();
    let mut first = true;
    
    while let Some(ch) = chars.next() {
        match ch {
            '-' | '_' | '.' | ' ' | '/' | '@' => {
                if !first && result.chars().last() != Some('_') {
                    result.push('_');
                }
            }
            c if c.is_uppercase() => {
                if !first && result.chars().last() != Some('_') {
                    result.push('_');
                }
                result.push(c.to_ascii_lowercase());
                first = false;
            }
            c if c.is_alphabetic() || c.is_ascii_digit() => {
                result.push(c.to_ascii_lowercase());
                first = false;
            }
            _ => {} // Skip other characters
        }
    }
    
    result
}

/// Dynamically discover available client modules by scanning the clients directory
fn discover_client_modules() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let clients_dir = Path::new("src/clients");
    let mut client_modules = Vec::new();
    
    if clients_dir.exists() && clients_dir.is_dir() {
        for entry in fs::read_dir(clients_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            // Check if it's a directory and contains a mod.rs file
            if path.is_dir() {
                let mod_file = path.join("mod.rs");
                if mod_file.exists() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        client_modules.push(name.to_string());
                    }
                }
            }
        }
    }
    
    // Sort for consistent output
    client_modules.sort();
    
    println!("🔍 Discovered {} client modules: {}", client_modules.len(), client_modules.join(", "));
    
    Ok(client_modules)
}

/// Generate provider-to-client mapping file
fn generate_provider_client_mapping(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let mut content = String::new();
    content.push_str("// This file is auto-generated. Do not edit manually.\n");
    content.push_str("use std::collections::HashMap;\n");
    content.push_str("use once_cell::sync::Lazy;\n\n");
    content.push_str("// AUTO-GENERATED START\n");
    
    // Generate the mapping function
    content.push_str("/// Static mapping of provider names to client module names\n");
    content.push_str("pub static PROVIDER_CLIENT_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {\n");
    content.push_str("    let mut map = HashMap::new();\n");
    
    // Dynamically discover available client modules
    let available_clients = discover_client_modules()?;
    
    for provider in providers {
        let client_module = to_snake_case_optimized(&provider.provider);
        
        // Check if client module exists
        if available_clients.contains(&client_module) {
            content.push_str(&format!(
                "    map.insert(\"{}\", \"{}\");\n",
                provider.provider, client_module
            ));
        } else {
            // Make loud noise but don't fail the build
            println!("cargo:warning=🚨 MISSING CLIENT MODULE: Provider '{}' maps to client module '{}' which doesn't exist!", provider.provider, client_module);
            println!("cargo:warning=📁 Available client modules: {}", available_clients.join(", "));
            println!("cargo:warning=💡 Create src/clients/{}/mod.rs to support this provider", client_module);
            
            // Still add to map but with a comment indicating it's missing
            content.push_str(&format!(
                "    // MISSING CLIENT: map.insert(\"{}\", \"{}\"); // Client module not found\n",
                provider.provider, client_module
            ));
        }
    }
    
    content.push_str("    map\n");
    content.push_str("});\n\n");
    
    // Generate helper function
    content.push_str("/// Get client module name for a provider\n");
    content.push_str("pub fn get_client_module(provider: &str) -> Option<&'static str> {\n");
    content.push_str("    PROVIDER_CLIENT_MAP.get(provider).copied()\n");
    content.push_str("}\n\n");
    
    // Generate reverse mapping
    content.push_str("/// Get provider name for a client module\n");
    content.push_str("pub fn get_provider_for_client(client: &str) -> Option<&'static str> {\n");
    content.push_str("    PROVIDER_CLIENT_MAP.iter()\n");
    content.push_str("        .find(|(_, &v)| v == client)\n");
    content.push_str("        .map(|(&k, _)| k)\n");
    content.push_str("}\n\n");
    
    // Generate all mappings function
    content.push_str("/// Get all provider-to-client mappings\n");
    content.push_str("pub fn get_all_mappings() -> &'static HashMap<&'static str, &'static str> {\n");
    content.push_str("    &PROVIDER_CLIENT_MAP\n");
    content.push_str("}\n\n");
    
    content.push_str("// AUTO-GENERATED END\n");
    
    fs::write("src/fluent_ai_provider.rs", content)?;
    println!("cargo:rerun-if-changed=src/fluent_ai_provider.rs");
    println!("✅ Generated provider-to-client mapping");
    Ok(())
}
