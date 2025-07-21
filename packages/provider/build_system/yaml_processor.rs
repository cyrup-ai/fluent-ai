//! YAML processing for provider and model definitions
//!
//! This module provides zero-allocation YAML parsing and validation for
//! provider and model definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::errors::{BuildError, BuildResult, YamlError};
use super::string_utils::sanitize_identifier;
use fluent_ai_domain::model::ModelInfo;

/// Information about a model provider
#[derive(Debug, Clone, Serialize)]
pub struct ProviderInfo {
    /// Provider ID (e.g., "openai", "anthropic")
    pub id: String,
    /// Display name for the provider
    pub name: String,
    /// Base URL for the provider's API
    pub base_url: String,
    /// List of models provided by this provider
    pub models: Vec<ModelInfo>,
    /// Authentication requirements
    pub auth: AuthConfig,
    /// Rate limiting configuration
    pub rate_limit: Option<RateLimitConfig>,
    /// Provider-specific features
    pub features: Vec<String>,
}

/// Authentication configuration for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type (e.g., "api_key", "bearer")
    pub r#type: String,
    /// Environment variable containing the API key
    pub env_var: String,
    /// Header name for the API key (if applicable)
    pub header_name: Option<String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per minute
    pub requests_per_min: u32,
    /// Tokens per minute
    pub tokens_per_min: u32,
}

/// YAML processor for provider and model definitions
#[derive(Debug, Default, Clone)]
pub struct YamlProcessor;

impl YamlProcessor {
    /// Create a new YAML processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse provider definitions from a YAML string
    pub fn parse_providers(&self, yaml_content: &str) -> BuildResult<Vec<ProviderInfo>> {
        // Parse with yyaml using serde_json::Value as intermediate
        let yaml_value: serde_json::Value = yyaml::from_str(yaml_content).map_err(|e| {
            let msg = e.to_string();
            let err = YamlError::new(msg);
            BuildError::YamlError(err)
        })?;

        // Convert sigoden YAML format to our ProviderInfo structure
        let mut providers = Vec::new();

        if let Some(array) = yaml_value.as_array() {
            for item in array {
                if let Some(obj) = item.as_object() {
                    if let Some(provider_name) = obj.get("provider").and_then(|v| v.as_str()) {
                        let models = self.parse_sigoden_models(provider_name, obj.get("models"))?;
                        
                        let provider = ProviderInfo {
                            id: provider_name.to_string(),
                            name: provider_name.replace('-', " ").replace('_', " "),
                            base_url: self.get_provider_base_url(provider_name),
                            models,
                            auth: AuthConfig {
                                r#type: "api_key".to_string(),
                                env_var: format!("{}_API_KEY", provider_name.to_uppercase().replace('-', "_")),
                                header_name: Some("Authorization".to_string()),
                            },
                            rate_limit: None,
                            features: vec!["chat".to_string()],
                        };

                        self.validate_provider(&provider)?;
                        providers.push(provider);
                    }
                }
            }
        }

        Ok(providers)
    }

    /// Parse models from sigoden YAML format
    fn parse_sigoden_models(&self, provider_name: &str, models_value: Option<&serde_json::Value>) -> BuildResult<Vec<ModelInfo>> {
        let mut models = Vec::new();
        
        if let Some(models_array) = models_value.and_then(|v| v.as_array()) {
            for model_value in models_array {
                if let Some(model_obj) = model_value.as_object() {
                    let name = model_obj.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    
                    let max_input_tokens = model_obj.get("max_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(4096) as u32;
                    
                    let max_output_tokens = model_obj.get("max_output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1024) as u32;
                    
                    let supports_vision = model_obj.get("supports_vision")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    
                    let supports_function_calling = model_obj.get("supports_function_calling")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    
                    // Create capabilities based on features
                    let mut capabilities = vec!["chat".to_string()];
                    if supports_vision {
                        capabilities.push("vision".to_string());
                    }
                    if supports_function_calling {
                        capabilities.push("function_calling".to_string());
                    }
                    
                    // Create parameters map
                    let mut parameters = HashMap::new();
                    if let Some(input_price) = model_obj.get("input_price").and_then(|v| v.as_f64()) {
                        parameters.insert("input_price".to_string(), yyaml::Value::Number(yyaml::Number::from(input_price)));
                    }
                    if let Some(output_price) = model_obj.get("output_price").and_then(|v| v.as_f64()) {
                        parameters.insert("output_price".to_string(), yyaml::Value::Number(yyaml::Number::from(output_price)));
                    }
                    
                    // Convert String to &'static str for build script context
                    let static_name: &'static str = Box::leak(name.to_string().into_boxed_str());
                    let static_provider: &'static str = Box::leak(provider_name.to_string().into_boxed_str());
                    
                    let mut builder = ModelInfo::builder()
                        .provider_name(static_provider)
                        .name(static_name)
                        .max_input_tokens(max_input_tokens as u32)
                        .max_output_tokens(max_output_tokens as u32)
                        .with_vision(capabilities.iter().any(|cap| cap.to_lowercase().contains("vision")))
                        .with_function_calling(capabilities.iter().any(|cap| cap.to_lowercase().contains("function")));

                    // Add pricing if available
                    if let (Some(input_price), Some(output_price)) = (
                        model_obj.get("input_price").and_then(|v| v.as_f64()),
                        model_obj.get("output_price").and_then(|v| v.as_f64())
                    ) {
                        builder = builder.pricing(input_price, output_price);
                    }

                    let model = builder.build()?;
                    
                    self.validate_model(&model)?;
                    models.push(model);
                }
            }
        }
        
        Ok(models)
    }

    /// Get the base URL for a provider
    fn get_provider_base_url(&self, provider_name: &str) -> String {
        match provider_name {
            "openai" => "https://api.openai.com/v1".to_string(),
            "anthropic" => "https://api.anthropic.com".to_string(),
            "google" => "https://generativelanguage.googleapis.com/v1beta".to_string(),
            "mistral" => "https://api.mistral.ai/v1".to_string(),
            "cohere" => "https://api.cohere.ai/v1".to_string(),
            "perplexity" => "https://api.perplexity.ai".to_string(),
            "groq" => "https://api.groq.com/openai/v1".to_string(),
            "together" => "https://api.together.xyz/v1".to_string(),
            "deepseek" => "https://api.deepseek.com/v1".to_string(),
            "ollama" => "http://localhost:11434/v1".to_string(),
            _ => format!("https://api.{}.com/v1", provider_name),
        }
    }

    /// Validate a provider definition
    fn validate_provider(&self, provider: &ProviderInfo) -> BuildResult<()> {
        if provider.id.is_empty() {
            return Err(BuildError::YamlError(YamlError::new(
                "Provider ID cannot be empty",
            )));
        }

        if sanitize_identifier(&provider.id) != provider.id {
            return Err(BuildError::YamlError(YamlError::new(format!(
                "Provider ID '{}' contains invalid characters. Use alphanumeric and underscores only.",
                provider.id
            ))));
        }

        if provider.name.is_empty() {
            return Err(BuildError::YamlError(YamlError::new(
                "Provider name cannot be empty",
            )));
        }

        if provider.base_url.is_empty() {
            return Err(BuildError::YamlError(YamlError::new(
                "Provider base URL cannot be empty",
            )));
        }

        Ok(())
    }

    /// Validate a model definition
    fn validate_model(&self, model: &ModelInfo) -> BuildResult<()> {
        if model.id().is_empty() {
            return Err(BuildError::YamlError(YamlError::new(
                "Model ID cannot be empty",
            )));
        }

        if sanitize_identifier(model.id()) != model.id() {
            return Err(BuildError::YamlError(YamlError::new(format!(
                "Model ID '{}' contains invalid characters. Use alphanumeric and underscores only.",
                model.id()
            ))));
        }

        if model.max_input_tokens.map_or(true, |tokens| tokens.get() == 0) {
            return Err(BuildError::YamlError(YamlError::new(
                "Max input tokens must be greater than 0",
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_YAML: &str = r#"
- id: openai
  name: OpenAI
  base_url: https://api.openai.com/v1
  auth:
    type: api_key
    env_var: OPENAI_API_KEY
    header_name: Authorization
  models:
    - id: gpt-4
      name: GPT-4
      description: Most capable model, great for complex tasks
      maxTokens: 8192
      supportsStreaming: true
      capabilities: ["chat", "completion"]
      parameters:
        temperature: 0.7
        topP: 1.0
    - id: gpt-3.5-turbo
      name: GPT-3.5 Turbo
      description: Fast and capable model, great for most tasks
      maxTokens: 4096
      supportsStreaming: true
      capabilities: ["chat", "completion"]
      parameters:
        temperature: 0.7
        topP: 1.0
"#;

    #[test]
    fn test_parse_providers() {
        let processor = YamlProcessor::new();
        let providers = processor.parse_providers(TEST_YAML).unwrap();

        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].id, "openai");
        assert_eq!(providers[0].models.len(), 2);
        assert_eq!(providers[0].models[0].id, "gpt-4");
        assert_eq!(providers[0].models[1].id, "gpt-3.5-turbo");
    }

    #[test]
    fn test_validate_invalid_provider() {
        let processor = YamlProcessor::new();
        let invalid_yaml = r#"- id: ""
  name: ""
  base_url: ""
  auth: { type: "", env_var: "" }
  models: []
"#;

        let result = processor.parse_providers(invalid_yaml);
        assert!(result.is_err());
    }
}
