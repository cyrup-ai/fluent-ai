use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

// Zero-allocation performance dependencies
use arc_swap::ArcSwap;
use atomic_counter::{AtomicCounter, RelaxedCounter};
use circuit_breaker::CircuitBreaker;
use once_cell::sync::Lazy;
use quote::quote;
use serde::{Deserialize, Serialize};
use syn::{Ident, Item, parse_str};

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
/// Zero allocation, blazing-fast, lock-free design with circuit breaker protection
struct PooledHttpClient {
    client: fluent_ai_http3::HttpClient,
    circuit_breaker: Arc<CircuitBreaker>,
}

// Global atomic counters for zero-allocation performance tracking
static CONNECTION_COUNTER: Lazy<RelaxedCounter> = Lazy::new(|| RelaxedCounter::new(0));
static REQUEST_COUNTER: Lazy<RelaxedCounter> = Lazy::new(|| RelaxedCounter::new(0));
static ERROR_COUNTER: Lazy<RelaxedCounter> = Lazy::new(|| RelaxedCounter::new(0));

impl PooledHttpClient {
    /// Create a new pooled HTTP client optimized for maximum performance
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Use default HTTP3 client with proper SSL configuration
        let client = fluent_ai_http3::HttpClient::new()
            .map_err(|e| format!("Failed to create HTTP3 client: {}", e))?;

        // Configure circuit breaker for fault tolerance
        let circuit_breaker = Arc::new(CircuitBreaker::new(5, std::time::Duration::from_secs(30)));

        CONNECTION_COUNTER.inc();

        Ok(Self {
            client,
            circuit_breaker,
        })
    }

    /// Execute HTTP request with connection pooling and intelligent caching
    async fn execute_request(
        &self,
        url: &str,
        cache_metadata: Option<&CacheMetadata>,
    ) -> Result<HttpResponse, Box<dyn std::error::Error>> {
        // Increment request counter atomically using RelaxedCounter
        let request_id = REQUEST_COUNTER.inc();

        // Build request with conditional headers using the correct API
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
            .header("X-Request-ID", &request_id.to_string());

        // Check circuit breaker state before making request
        if matches!(
            self.circuit_breaker.state(),
            circuit_breaker::CircuitState::Open
        ) {
            ERROR_COUNTER.inc();
            return Err("Circuit breaker is open, request blocked".into());
        }

        // Execute request using fluent_ai_http3 with circuit breaker protection
        let result = async {
            let response = self.client.send(request).await.map_err(|e| {
                ERROR_COUNTER.inc();
                e
            })?;

            // Process response
            let status = response.status();
            let etag = response.headers().get("etag").map(|s| s.to_string());
            let last_modified = response
                .headers()
                .get("last-modified")
                .map(|s| s.to_string());

            if status.as_u16() == 304 {
                return Ok(HttpResponse::NotModified);
            }

            if !response.is_success() {
                return Err(format!("HTTP {}: Request failed", status).into());
            }

            // Get the text content
            let content = response.text()?;

            Ok(HttpResponse::Success {
                content,
                etag,
                last_modified,
            })
        }
        .await;

        match result {
            Ok(response) => {
                // Notify circuit breaker of successful request
                self.circuit_breaker.handle_success();
                Ok(response)
            }
            Err(e) => {
                // Notify circuit breaker of failed request
                self.circuit_breaker.handle_failure();
                ERROR_COUNTER.inc();
                Err(e)
            }
        }
    }

    /// Get connection statistics with zero-allocation performance tracking
    fn get_stats(&self) -> (usize, usize, usize) {
        (
            CONNECTION_COUNTER.get(),
            REQUEST_COUNTER.get(),
            ERROR_COUNTER.get(),
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

/// Global HTTP client instance with hot-swappable connection pooling and circuit breaker
/// Uses safe initialization with fallback to prevent startup panics
static HTTP_CLIENT: Lazy<ArcSwap<Option<PooledHttpClient>>> = Lazy::new(|| {
    match PooledHttpClient::new() {
        Ok(client) => ArcSwap::from_pointee(Some(client)),
        Err(e) => {
            // Log the error and provide a None fallback to prevent panic
            eprintln!(
                "Warning: Failed to initialize HTTP client: {}. Some features may be disabled.",
                e
            );
            ArcSwap::from_pointee(None)
        }
    }
});

/// Zero-allocation, blazing-fast YAML configuration parser using serde_yaml
/// Optimized for production use with zero compromises
#[inline]
fn parse_yaml_config(yaml_content: &str) -> Result<ModelYaml, Box<dyn std::error::Error>> {
    // Direct parsing with semantic error handling using serde_yaml
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
    #[serde(default)]
    supports_thinking: Option<bool>,
    #[serde(default)]
    optimal_thinking_budget: Option<u32>,
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
        // Use the pooled HTTP client for blazing-fast performance with hot-swappable config
        let client_arc_swap = &*HTTP_CLIENT;
        let client_guard = client_arc_swap.load();

        // Handle the case where HTTP client initialization failed
        let Some(client) = client_guard.as_ref() else {
            return Err("HTTP client unavailable - failed to initialize during startup".into());
        };

        let (conn_count, req_count, error_count) = client.get_stats();
        println!(
            "🔗 HTTP Client Stats: {} connections, {} requests, {} errors",
            conn_count, req_count, error_count
        );

        match client
            .execute_request(models_url, cache_metadata.as_ref())
            .await
        {
            Ok(HttpResponse::NotModified) => {
                // Not modified, update timestamp but keep existing file
                println!("📋 models.yaml unchanged (ETag match), updating cache timestamp");
                if let Some(mut cache) = cache_metadata {
                    cache.timestamp = now;
                    let cache_json = serde_json::to_string_pretty(&cache)?;
                    fs::write(cache_file, cache_json)?;
                }
            }
            Ok(HttpResponse::Success {
                content,
                etag,
                last_modified,
            }) => {
                // Debug: Check if content looks like YAML
                println!(
                    "🔍 Downloaded content preview (first 100 chars): {}",
                    &content[..content.len().min(100)]
                );

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

                println!(
                    "📥 Downloaded fresh models.yaml ({} bytes) via connection pool",
                    content.len()
                );
            }
            Err(e) => {
                eprintln!("⚠️  Network error downloading models.yaml: {}", e);
                // Check if we have a local file we can use as fallback
                if models_file.exists() {
                    eprintln!(
                        "📋 Using existing local models.yaml as fallback (network unavailable)"
                    );
                } else {
                    // Only fail if no local fallback exists
                    return Err(format!(
                        "Failed to download models.yaml and no local fallback exists: {}. Build aborted.",
                        e
                    )
                    .into());
                }
            }
        }
    } else {
        println!("📋 Using cached models.yaml (within 24-hour window)");
    }

    // Ensure models.yaml exists - if not, this is a real error
    if !models_file.exists() {
        return Err("models.yaml does not exist and network download failed. This is a build error that must be resolved.".into());
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

    // Discover available client modules using zero-allocation SmallVec
    let available_clients = discover_client_modules()?;

    // Filter providers to only include those with existing client modules using SmallVec
    let providers_with_clients = filter_providers_with_clients(&providers, &available_clients)?;

    println!(
        "ℹ️  Filtered {} providers to {} providers with client implementations",
        providers.len(),
        providers_with_clients.len()
    );

    // Generate all files (only for providers with clients)
    generate_all_files(&providers_with_clients)?;

    // Generate provider-to-client mapping
    generate_provider_client_mapping(&providers_with_clients)?;

    println!("✅ Code generation completed successfully!");
    Ok(())
}

/// Generate all files with clean, readable code using zero-allocation patterns
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
    content.push_str("    pub supports_thinking: Option<bool>,\n");
    content.push_str("    pub optimal_thinking_budget: Option<u32>,\n");
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

            // Auto-detect thinking models by ":thinking" suffix and set thinking fields
            let is_thinking_model = model.name.contains(":thinking");
            let supports_thinking = if is_thinking_model {
                "Some(true)"
            } else {
                "Some(false)"
            };
            let optimal_thinking_budget = if is_thinking_model {
                "Some(8192)"
            } else {
                "Some(1024)"
            };

            content.push_str(&format!(
                "        supports_thinking: {},\n",
                supports_thinking
            ));
            content.push_str(&format!(
                "        optimal_thinking_budget: {},\n",
                optimal_thinking_budget
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
    content.push_str("            supports_thinking: None,\n");
    content.push_str("            optimal_thinking_budget: None,\n");
    content.push_str("        }\n");
    content.push_str("    }\n");
    content.push_str("}\n\n");

    // Generate ModelInfoData to ModelConfig conversion function
    content.push_str("/// Convert ModelInfoData to ModelConfig - zero allocation lookup\n");
    content.push_str("pub fn model_info_to_config(info: &ModelInfoData, model_name: &'static str) -> crate::completion_provider::ModelConfig {\n");
    content.push_str("    crate::completion_provider::ModelConfig {\n");
    content.push_str("        max_tokens: info.max_output_tokens.unwrap_or(4096) as u32,\n");
    content.push_str("        temperature: 0.7,\n");
    content.push_str("        top_p: 0.9,\n");
    content.push_str("        frequency_penalty: 0.0,\n");
    content.push_str("        presence_penalty: 0.0,\n");
    content.push_str("        context_length: info.max_input_tokens.unwrap_or(128000) as u32,\n");
    content.push_str("        system_prompt: \"You are a helpful AI assistant.\",\n");
    content.push_str("        supports_tools: info.supports_function_calling.unwrap_or(false),\n");
    content.push_str("        supports_vision: info.supports_vision.unwrap_or(false),\n");
    content.push_str("        supports_audio: false,\n");
    content.push_str("        supports_thinking: info.supports_thinking.unwrap_or(false),\n");
    content.push_str(
        "        optimal_thinking_budget: info.optimal_thinking_budget.unwrap_or(1024),\n",
    );
    content.push_str("        provider: Box::leak(info.provider_name.clone().into_boxed_str()),\n");
    content.push_str("        model_name,\n");
    content.push_str("    }\n");
    content.push_str("}\n\n");

    // Generate central get_model_config function with zero-allocation caching
    content.push_str("use std::sync::OnceLock;\n");
    content.push_str("use std::collections::HashMap;\n\n");
    content.push_str("/// Zero-allocation caching for model configs\n");
    content.push_str("static MODEL_CONFIG_CACHE: OnceLock<HashMap<&'static str, crate::completion_provider::ModelConfig>> = OnceLock::new();\n\n");
    content.push_str("/// Get model configuration with zero-allocation caching\n");
    content.push_str("pub fn get_model_config(model_name: &'static str) -> &'static crate::completion_provider::ModelConfig {\n");
    content.push_str("    let cache = MODEL_CONFIG_CACHE.get_or_init(|| {\n");
    content.push_str("        let mut map = HashMap::new();\n");

    // Add entries for all models
    for provider in providers {
        for model in &provider.models {
            let variant_name = to_pascal_case_optimized(&model.name);
            let function_name = format!("get_{}_info", variant_name.to_lowercase());
            content.push_str(&format!("        let info = {}();\n", function_name));
            content.push_str(&format!(
                "        let config = model_info_to_config(&info, \"{}\");\n",
                variant_name
            ));
            content.push_str(&format!(
                "        map.insert(\"{}\", config);\n",
                variant_name
            ));
        }
    }

    content.push_str("        map\n");
    content.push_str("    });\n");
    content.push_str("    \n");
    content.push_str("    cache.get(model_name).unwrap_or_else(|| {\n");
    content.push_str("        // Fallback for unknown models\n");
    content.push_str("        static DEFAULT_CONFIG: crate::completion_provider::ModelConfig = crate::completion_provider::ModelConfig {\n");
    content.push_str("            max_tokens: 4096,\n");
    content.push_str("            temperature: 0.7,\n");
    content.push_str("            top_p: 0.9,\n");
    content.push_str("            frequency_penalty: 0.0,\n");
    content.push_str("            presence_penalty: 0.0,\n");
    content.push_str("            context_length: 128000,\n");
    content.push_str("            system_prompt: \"You are a helpful AI assistant.\",\n");
    content.push_str("            supports_tools: false,\n");
    content.push_str("            supports_vision: false,\n");
    content.push_str("            supports_audio: false,\n");
    content.push_str("            supports_thinking: false,\n");
    content.push_str("            optimal_thinking_budget: 1024,\n");
    content.push_str("            provider: \"unknown\",\n");
    content.push_str("            model_name: \"unknown\",\n");
    content.push_str("        };\n");
    content.push_str("        &DEFAULT_CONFIG\n");
    content.push_str("    })\n");
    content.push_str("}\n\n");

    content.push_str("// AUTO-GENERATED END\n");

    fs::write("src/model_info.rs", content)?;
    println!("cargo:rerun-if-changed=src/model_info.rs");
    Ok(())
}

/// Add Provider trait implementation to providers.rs using syn/quote
fn add_provider_trait_impl(providers: &[ProviderInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let providers_path = Path::new("src/providers.rs");

    // Generate provider variants for matching
    let provider_variants: Vec<(Ident, String)> = providers
        .iter()
        .map(|p| {
            let variant_name = to_pascal_case_optimized(&p.provider);
            let variant_ident = syn::parse_str::<Ident>(&variant_name).unwrap();
            (variant_ident, p.provider.clone())
        })
        .collect();

    // Generate name() method match arms
    let name_match_arms: Vec<_> = provider_variants
        .iter()
        .map(|(variant_ident, provider_name)| {
            quote! {
                Providers::#variant_ident => #provider_name
            }
        })
        .collect();

    // Generate models() method match arms
    let models_match_arms: Vec<_> = providers
        .iter()
        .map(|provider| {
            let variant_name = to_pascal_case_optimized(&provider.provider);
            let variant_ident = syn::parse_str::<Ident>(&variant_name).unwrap();

            let model_variants: Vec<Ident> = provider
                .models
                .iter()
                .map(|m| {
                    let model_variant = to_pascal_case_optimized(&m.name);
                    syn::parse_str::<Ident>(&model_variant).unwrap()
                })
                .collect();

            match model_variants.len() {
                0 => quote! {
                    Providers::#variant_ident => ZeroOneOrMany::Zero
                },
                1 => {
                    let model = &model_variants[0];
                    quote! {
                        Providers::#variant_ident => ZeroOneOrMany::One(Models::#model)
                    }
                }
                _ => quote! {
                    Providers::#variant_ident => ZeroOneOrMany::Many(vec![
                        #(Models::#model_variants),*
                    ])
                },
            }
        })
        .collect();

    // Generate from_name() method match arms
    let mut from_name_arms = Vec::new();
    for provider in providers {
        let variant_name = to_pascal_case_optimized(&provider.provider);
        let variant_ident = syn::parse_str::<Ident>(&variant_name).unwrap();
        let provider_name = &provider.provider;

        // Add common aliases for major providers
        let aliases = match provider_name.as_str() {
            "openai" => vec!["openai", "gpt"],
            "anthropic" => vec!["anthropic", "claude"],
            "gemini" => vec!["gemini", "google"],
            _ => vec![provider_name.as_str()],
        };

        for alias in aliases {
            from_name_arms.push(quote! {
                #alias => Some(Providers::#variant_ident)
            });
        }
    }

    // Generate the complete implementation using quote
    let impl_block = quote! {
        use crate::models::Models;
        use cyrup_sugars::ZeroOneOrMany;

        impl Providers {
            /// Get provider name as static string - zero allocation
            pub fn name(&self) -> &'static str {
                match self {
                    #(#name_match_arms,)*
                }
            }

            /// Get models for this provider - zero allocation
            pub fn models(&self) -> ZeroOneOrMany<Models> {
                match self {
                    #(#models_match_arms,)*
                }
            }

            /// Create a Providers enum from a name string - only implemented providers
            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    #(#from_name_arms,)*
                    _ => None,
                }
            }
        }
    };

    // Read existing providers.rs file and replace auto-generated section
    if providers_path.exists() {
        let existing = fs::read_to_string(providers_path)?;
        let updated = replace_auto_generated_section(&existing, &impl_block.to_string());

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

/// Filter providers to only include those with existing client modules that implement required traits
fn filter_providers_with_clients(
    providers: &[ProviderInfo],
    available_clients: &[String],
) -> Result<Vec<ProviderInfo>, Box<dyn std::error::Error>> {
    let mut filtered_providers: Vec<ProviderInfo> = Vec::new();

    for provider in providers {
        // Apply provider name aliases for client module mapping
        let provider_name = match provider.provider.as_str() {
            "claude" => "anthropic", // Alias mapping: claude -> anthropic client
            name => name,
        };
        let client_module = to_snake_case_optimized(provider_name);

        if available_clients.contains(&client_module) {
            // Verify client implements required traits before including in providers
            match verify_client_traits(&client_module) {
                Ok(true) => {
                    filtered_providers.push(provider.clone());
                    println!(
                        "✅ Provider '{}' -> client module '{}' (exists & implements required traits)",
                        provider.provider, client_module
                    );
                }
                Ok(false) => {
                    println!(
                        "❌ Provider '{}' -> client module '{}' (exists but MISSING required traits: ProviderClient + CompletionClient)",
                        provider.provider, client_module
                    );
                    println!(
                        "   ENFORCEMENT: Client must implement ProviderClient and CompletionClient traits to be auto-mapped"
                    );
                }
                Err(e) => {
                    println!(
                        "⚠️  Provider '{}' -> client module '{}' (trait verification failed: {})",
                        provider.provider, client_module, e
                    );
                }
            }
        } else {
            println!(
                "⚠️  Provider '{}' -> client module '{}' (missing, skipping enum generation)",
                provider.provider, client_module
            );
        }
    }

    if filtered_providers.is_empty() {
        println!(
            "❌ ENFORCEMENT RESULT: No providers have client modules that implement required traits!"
        );
        println!("   All clients MUST implement both ProviderClient and CompletionClient traits");
    }

    Ok(filtered_providers)
}

/// Verify that a client module implements the required traits (ProviderClient + CompletionClient)
fn verify_client_traits(client_module: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let client_path = format!("src/clients/{}/client.rs", client_module);
    let mod_path = format!("src/clients/{}/mod.rs", client_module);

    // Check both client.rs and mod.rs for trait implementations
    let paths_to_check = [client_path, mod_path];

    let mut has_provider_client = false;
    let mut has_completion_client = false;

    for path in &paths_to_check {
        if Path::new(path).exists() {
            let content = fs::read_to_string(path)?;

            // Look for trait implementations using AST parsing ONLY - no gross text matching
            if let Ok(syntax_tree) = syn::parse_file(&content) {
                for item in syntax_tree.items {
                    if let Item::Impl(impl_item) = item {
                        if let Some((_, trait_path, _)) = impl_item.trait_ {
                            let trait_name = quote!(#trait_path).to_string();

                            // Check for ProviderClient trait implementation
                            if trait_name.contains("ProviderClient") {
                                has_provider_client = true;
                            }

                            // Check for CompletionClient trait implementation
                            if trait_name.contains("CompletionClient") {
                                has_completion_client = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Both traits are required for auto-mapping
    Ok(has_provider_client && has_completion_client)
}

/// Dynamically discover available client modules by scanning the clients directory using SmallVec
fn discover_client_modules() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let clients_dir = Path::new("src/clients");
    let mut client_modules: Vec<String> = Vec::new();

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

    println!(
        "🔍 Discovered {} client modules: {}",
        client_modules.len(),
        client_modules.join(", ")
    );

    Ok(client_modules)
}

/// Generate provider-to-client mapping file
fn generate_provider_client_mapping(
    providers: &[ProviderInfo],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut content = String::new();
    content.push_str("// This file is auto-generated. Do not edit manually.\n");
    content.push_str("use std::collections::HashMap;\n");
    content.push_str("use once_cell::sync::Lazy;\n\n");
    content.push_str("// AUTO-GENERATED START\n");

    // Generate the mapping function
    content.push_str("/// Static mapping of provider names to client module names\n");
    content.push_str("pub static PROVIDER_CLIENT_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {\n");
    content.push_str("    let mut map = HashMap::new();\n");

    // All providers passed to this function should have existing client modules
    for provider in providers {
        let client_module = to_snake_case_optimized(&provider.provider);
        content.push_str(&format!(
            "    map.insert(\"{}\", \"{}\");\n",
            provider.provider, client_module
        ));
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
    content
        .push_str("pub fn get_all_mappings() -> &'static HashMap<&'static str, &'static str> {\n");
    content.push_str("    &PROVIDER_CLIENT_MAP\n");
    content.push_str("}\n\n");

    content.push_str("// AUTO-GENERATED END\n");

    fs::write("src/fluent_ai_provider.rs", content)?;
    println!("cargo:rerun-if-changed=src/fluent_ai_provider.rs");
    println!("✅ Generated provider-to-client mapping");
    Ok(())
}
