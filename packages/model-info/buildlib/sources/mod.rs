use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod openrouter;
pub mod litellm;
pub mod scraper;

/// Comprehensive model specification with all possible attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub provider: String,
    pub max_input_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub pricing_input: Option<f64>,  // per 1M tokens
    pub pricing_output: Option<f64>, // per 1M tokens
    pub supports_vision: bool,
    pub supports_function_calling: bool,
    pub supports_embeddings: bool,
    pub supports_thinking: bool,
    pub required_temperature: Option<f64>,
    pub optimal_thinking_budget: Option<u32>,
    pub context_window: Option<u32>,
    pub description: Option<String>,
    pub last_updated: Option<String>,
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            id: String::new(),
            provider: String::new(),
            max_input_tokens: None,
            max_output_tokens: None,
            pricing_input: None,
            pricing_output: None,
            supports_vision: false,
            supports_function_calling: false,
            supports_embeddings: false,
            supports_thinking: false,
            required_temperature: None,
            optimal_thinking_budget: None,
            context_window: None,
            description: None,
            last_updated: None,
        }
    }
}

/// Multi-source specification fetcher with fallback chain
pub struct SpecificationFetcher {
    // TODO: Update to use Http3 when needed
    // http_client: Http3,
    cache: HashMap<String, ModelSpec>,
}

impl SpecificationFetcher {
    pub fn new() -> Self {
        Self {
            // http_client: Http3::default(),
            cache: HashMap::new(),
        }
    }
    
    /// Fetch model specifications using multi-source strategy
    /// Tries sources in order: OpenRouter.ai -> LiteLLM -> Web Scraping -> Conservative defaults
    pub async fn fetch_specification(&mut self, model_id: &str, provider: &str) -> Result<ModelSpec> {
        // Check cache first
        let cache_key = format!("{}:{}", provider, model_id);
        if let Some(cached_spec) = self.cache.get(&cache_key) {
            return Ok(cached_spec.clone());
        }
        
        // Try OpenRouter.ai first (most comprehensive)
        if let Ok(spec) = self.try_openrouter(model_id, provider).await {
            self.cache.insert(cache_key.clone(), spec.clone());
            return Ok(spec);
        }
        
        // Try LiteLLM registry
        if let Ok(spec) = self.try_litellm(model_id, provider).await {
            self.cache.insert(cache_key.clone(), spec.clone());
            return Ok(spec);
        }
        
        // Try web scraping as last resort
        if let Ok(spec) = self.try_scraping(model_id, provider).await {
            self.cache.insert(cache_key.clone(), spec.clone());
            return Ok(spec);
        }
        
        // Fall back to conservative defaults based on model patterns
        let spec = self.generate_conservative_defaults(model_id, provider);
        self.cache.insert(cache_key, spec.clone());
        Ok(spec)
    }
    
    async fn try_openrouter(&self, model_id: &str, provider: &str) -> Result<ModelSpec> {
        // TODO: Update to use Http3 when needed
        // openrouter::fetch_model_spec(&Http3::default(), model_id, provider).await
        Err(anyhow::anyhow!("OpenRouter fetching temporarily disabled"))
    }
    
    async fn try_litellm(&self, model_id: &str, provider: &str) -> Result<ModelSpec> {
        // TODO: Update to use Http3 when needed
        // litellm::fetch_model_spec(&Http3::default(), model_id, provider).await
        Err(anyhow::anyhow!("LiteLLM fetching temporarily disabled"))
    }
    
    async fn try_scraping(&self, model_id: &str, provider: &str) -> Result<ModelSpec> {
        // TODO: Update to use Http3 when needed
        // scraper::fetch_model_spec(&Http3::default(), model_id, provider).await
        Err(anyhow::anyhow!("Scraping temporarily disabled"))
    }
    
    fn generate_conservative_defaults(&self, model_id: &str, provider: &str) -> ModelSpec {
        let mut spec = ModelSpec {
            id: model_id.to_string(),
            provider: provider.to_string(),
            ..Default::default()
        };
        
        // Apply intelligent defaults based on model name patterns
        match provider {
            "openai" => {
                if model_id.starts_with("gpt-4") {
                    spec.max_input_tokens = Some(128000);
                    spec.max_output_tokens = Some(32000);
                    spec.pricing_input = Some(10.0);
                    spec.pricing_output = Some(30.0);
                    spec.supports_vision = model_id.contains("gpt-4o");
                    spec.supports_function_calling = true;
                } else if model_id.starts_with("o") {
                    spec.max_input_tokens = Some(128000);
                    spec.max_output_tokens = Some(50000);
                    spec.pricing_input = Some(15.0);
                    spec.pricing_output = Some(60.0);
                    spec.supports_thinking = true;
                    spec.required_temperature = Some(1.0);
                    spec.optimal_thinking_budget = Some(100000);
                }
            }
            "anthropic" => {
                spec.max_input_tokens = Some(200000);
                spec.max_output_tokens = Some(50000);
                spec.supports_vision = true;
                spec.supports_function_calling = true;
                if model_id.contains("opus") {
                    spec.pricing_input = Some(15.0);
                    spec.pricing_output = Some(75.0);
                } else if model_id.contains("sonnet") {
                    spec.pricing_input = Some(3.0);
                    spec.pricing_output = Some(15.0);
                } else if model_id.contains("haiku") {
                    spec.pricing_input = Some(0.25);
                    spec.pricing_output = Some(1.25);
                }
            }
            "mistral" => {
                spec.supports_function_calling = true;
                if model_id.contains("large") {
                    spec.max_input_tokens = Some(128000);
                    spec.max_output_tokens = Some(32000);
                    spec.pricing_input = Some(2.0);
                    spec.pricing_output = Some(6.0);
                } else {
                    spec.max_input_tokens = Some(32000);
                    spec.max_output_tokens = Some(8000);
                    spec.pricing_input = Some(1.0);
                    spec.pricing_output = Some(3.0);
                }
            }
            "xai" => {
                spec.max_input_tokens = Some(131072);
                spec.max_output_tokens = Some(30000);
                spec.supports_thinking = true;
                spec.supports_function_calling = true;
                if model_id.contains("mini") {
                    spec.pricing_input = Some(0.3);
                    spec.pricing_output = Some(0.5);
                } else {
                    spec.pricing_input = Some(3.0);
                    spec.pricing_output = Some(15.0);
                    spec.required_temperature = Some(1.0);
                    spec.optimal_thinking_budget = Some(100000);
                }
            }
            _ => {
                // Very conservative defaults for unknown providers
                spec.max_input_tokens = Some(16384);
                spec.max_output_tokens = Some(4096);
                spec.pricing_input = Some(2.0);
                spec.pricing_output = Some(6.0);
            }
        }
        
        spec
    }
}

impl Default for SpecificationFetcher {
    fn default() -> Self {
        Self::new()
    }
}