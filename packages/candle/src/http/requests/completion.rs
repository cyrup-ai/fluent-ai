//! Completion Request Models for AI Provider Integration
//!
//! This module provides unified completion request structures that work across all 17 AI providers
//! while maintaining type safety, zero-allocation patterns, and provider-specific optimizations.
//!
//! # Supported Providers
//!
//! - **OpenAI**: GPT models with function calling and vision
//! - **Anthropic**: Claude models with cache control and tool use
//! - **Google**: Vertex AI and Gemini with safety settings
//! - **AWS Bedrock**: Multi-model support with converse API

#![allow(missing_docs)]
//! - **Cohere**: Command models with chat history
//! - **Azure OpenAI**: All OpenAI models via Azure deployment
//! - **Local/OSS**: Ollama, HuggingFace, Together, etc.
//! - **Commercial**: AI21, Groq, Mistral, Perplexity, OpenRouter, xAI, DeepSeek
//!
//! # Architecture
//!
//! Uses a trait-based design with a base [`CompletionRequest`] struct and provider-specific
//! extensions. Zero-allocation patterns with `ArrayVec` for bounded collections and stack
//! allocation for common use cases.

// # Usage Examples
//
// ```rust
// use fluent_ai_domain::http::requests::completion::{CompletionRequest, ProviderExtensions};
// use fluent_ai_domain::http::common::{BaseMessage, ModelParameters};
//
// // Basic completion request
// let request = CompletionRequest::new("gpt-4")
//     .with_message(BaseMessage::user("Hello, world!"))
//     .with_temperature(0.7)?
//     .with_max_tokens(1000)?
//     .with_streaming(true);
//
// // Provider-specific extensions
// let anthropic_request = request
//     .with_anthropic_cache_control(true)
//     .with_anthropic_system_message("You are helpful");
//
// let openai_request = request
//     .with_openai_response_format("json_object")
//     .with_openai_seed(42);
// ```
#[forbid(unsafe_code)]
#[deny(clippy::all)]
#[deny(clippy::pedantic)]
#[allow(clippy::module_name_repetitions)]
// Removed duplicate import
use std::fmt;

use arrayvec::{ArrayString, ArrayVec};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use std::collections::HashMap;
use crate::domain::http::Provider;
use crate::http::common::{
    BaseMessage, MAX_IDENTIFIER_LEN, MAX_MESSAGES, MAX_STOP_SEQUENCE_LEN, MAX_STOP_SEQUENCES,
    MAX_TOOLS, ModelParameters, ValidationError,
};

/// Maximum number of provider-specific parameters
pub const MAX_PROVIDER_PARAMS: usize = 16;

/// Maximum length for provider-specific string parameters
pub const MAX_PARAM_VALUE_LEN: usize = 256;

/// Maximum number of safety settings (for Google providers)
pub const MAX_SAFETY_SETTINGS: usize = 8;

/// Maximum number of cache control entries (for Anthropic)
pub const MAX_CACHE_ENTRIES: usize = 4;

/// Universal completion request supporting all AI providers
///
/// This struct provides a unified interface for completion requests across all providers
/// while allowing provider-specific extensions through the `extensions` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier (required)
    pub model: ArrayString<MAX_IDENTIFIER_LEN>,

    /// Conversation messages with bounded capacity for zero allocation
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub messages: ArrayVec<BaseMessage, MAX_MESSAGES>,

    /// System message/prompt (separate from messages for some providers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Temperature for randomness (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Top-p nucleus sampling (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling (1 or higher)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    /// Stop sequences for completion termination
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub stop: ArrayVec<ArrayString<MAX_STOP_SEQUENCE_LEN>, MAX_STOP_SEQUENCES>,

    /// Enable streaming response
    pub stream: bool,

    /// Tools/functions available for the model to call
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub tools: ArrayVec<ToolDefinition, MAX_TOOLS>,

    /// Tool choice strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Provider-specific extensions
    #[serde(skip_serializing_if = "ProviderExtensions::is_empty")]
    pub extensions: ProviderExtensions,

    /// User identifier for tracking and rate limiting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<ArrayString<MAX_IDENTIFIER_LEN>>,
}

impl CompletionRequest {
    /// Create a new completion request with the specified model
    #[inline]
    pub fn new(model: &str) -> Result<Self, CompletionRequestError> {
        let model_name = ArrayString::from(model)
            .map_err(|_| CompletionRequestError::ModelNameTooLong(model.len()))?;

        if model.is_empty() {
            return Err(CompletionRequestError::EmptyModelName);
        }

        Ok(Self {
            model: model_name,
            messages: ArrayVec::new(),
            system: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: ArrayVec::new(),
            stream: false,
            tools: ArrayVec::new(),
            tool_choice: None,
            extensions: ProviderExtensions::new(),
            user: None,
        })
    }

    /// Add a message to the conversation
    #[inline]
    pub fn with_message(mut self, message: BaseMessage) -> Result<Self, CompletionRequestError> {
        self.messages
            .try_push(message)
            .map_err(|_| CompletionRequestError::TooManyMessages)?;
        Ok(self)
    }

    /// Add multiple messages to the conversation
    #[inline]
    pub fn with_messages<I>(mut self, messages: I) -> Result<Self, CompletionRequestError>
    where
        I: IntoIterator<Item = BaseMessage>,
    {
        for message in messages {
            self.messages
                .try_push(message)
                .map_err(|_| CompletionRequestError::TooManyMessages)?;
        }
        Ok(self)
    }

    /// Set system message/prompt
    #[inline]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set temperature with validation
    #[inline]
    pub fn with_temperature(mut self, temperature: f64) -> Result<Self, CompletionRequestError> {
        if !(0.0..=2.0).contains(&temperature) {
            return Err(CompletionRequestError::TemperatureOutOfRange(temperature));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    /// Set max tokens with validation
    #[inline]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Result<Self, CompletionRequestError> {
        if max_tokens == 0 {
            return Err(CompletionRequestError::MaxTokensZero);
        }
        if max_tokens > 2_000_000 {
            return Err(CompletionRequestError::MaxTokensTooLarge(max_tokens));
        }
        self.max_tokens = Some(max_tokens);
        Ok(self)
    }

    /// Set top-p with validation
    #[inline]
    pub fn with_top_p(mut self, top_p: f64) -> Result<Self, CompletionRequestError> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(CompletionRequestError::TopPOutOfRange(top_p));
        }
        self.top_p = Some(top_p);
        Ok(self)
    }

    /// Set top-k with validation
    #[inline]
    pub fn with_top_k(mut self, top_k: u32) -> Result<Self, CompletionRequestError> {
        if top_k == 0 {
            return Err(CompletionRequestError::TopKZero);
        }
        self.top_k = Some(top_k);
        Ok(self)
    }

    /// Set frequency penalty with validation
    #[inline]
    pub fn with_frequency_penalty(mut self, penalty: f64) -> Result<Self, CompletionRequestError> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(CompletionRequestError::FrequencyPenaltyOutOfRange(penalty));
        }
        self.frequency_penalty = Some(penalty);
        Ok(self)
    }

    /// Set presence penalty with validation
    #[inline]
    pub fn with_presence_penalty(mut self, penalty: f64) -> Result<Self, CompletionRequestError> {
        if !(-2.0..=2.0).contains(&penalty) {
            return Err(CompletionRequestError::PresencePenaltyOutOfRange(penalty));
        }
        self.presence_penalty = Some(penalty);
        Ok(self)
    }

    /// Add a stop sequence
    #[inline]
    pub fn with_stop_sequence(mut self, sequence: &str) -> Result<Self, CompletionRequestError> {
        if sequence.is_empty() {
            return Err(CompletionRequestError::EmptyStopSequence);
        }

        let stop_seq = ArrayString::from(sequence)
            .map_err(|_| CompletionRequestError::StopSequenceTooLong(sequence.len()))?;

        self.stop
            .try_push(stop_seq)
            .map_err(|_| CompletionRequestError::TooManyStopSequences)?;

        Ok(self)
    }

    /// Enable or disable streaming
    #[inline]
    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Add a tool definition
    #[inline]
    pub fn with_tool(mut self, tool: ToolDefinition) -> Result<Self, CompletionRequestError> {
        self.tools
            .try_push(tool)
            .map_err(|_| CompletionRequestError::TooManyTools)?;
        Ok(self)
    }

    /// Set tool choice strategy
    #[inline]
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set user identifier
    #[inline]
    pub fn with_user(mut self, user: &str) -> Result<Self, CompletionRequestError> {
        let user_id = ArrayString::from(user)
            .map_err(|_| CompletionRequestError::UserIdTooLong(user.len()))?;
        self.user = Some(user_id);
        Ok(self)
    }

    /// Apply model parameters from ModelParameters struct
    #[inline]
    pub fn with_model_parameters(
        mut self,
        params: &ModelParameters,
    ) -> Result<Self, CompletionRequestError> {
        if let Some(temp) = params.temperature {
            self = self.with_temperature(temp)?;
        }
        if let Some(max_tokens) = params.max_tokens {
            self = self.with_max_tokens(max_tokens)?;
        }
        if let Some(top_p) = params.top_p {
            self = self.with_top_p(top_p)?;
        }
        if let Some(top_k) = params.top_k {
            self = self.with_top_k(top_k)?;
        }
        if let Some(freq_penalty) = params.frequency_penalty {
            self = self.with_frequency_penalty(freq_penalty)?;
        }
        if let Some(pres_penalty) = params.presence_penalty {
            self = self.with_presence_penalty(pres_penalty)?;
        }

        // Add stop sequences
        for stop_seq in &params.stop_sequences {
            self = self.with_stop_sequence(stop_seq.as_str())?;
        }

        self.stream = params.stream;

        Ok(self)
    }

    /// Validate the entire request for consistency and provider requirements
    #[inline]
    pub fn validate(&self) -> Result<(), CompletionRequestError> {
        // Check that we have at least one message
        if self.messages.is_empty() && self.system.is_none() {
            return Err(CompletionRequestError::NoMessages);
        }

        // Validate parameter combinations
        if self.temperature.is_some() && self.top_p.is_some() {
            if let (Some(temp), Some(top_p)) = (self.temperature, self.top_p) {
                // Some providers don't like both high temperature and low top_p
                if temp > 1.5 && top_p < 0.3 {
                    return Err(CompletionRequestError::ConflictingParameters);
                }
            }
        }

        // Validate tools and tool choice consistency
        if self.tool_choice.is_some() && self.tools.is_empty() {
            return Err(CompletionRequestError::ToolChoiceWithoutTools);
        }

        // Validate provider-specific constraints
        self.extensions.validate()?;

        Ok(())
    }

    /// Convert to provider-specific format for the given provider
    #[inline]
    pub fn to_provider_format(&self, provider: Provider) -> Result<Value, CompletionRequestError> {
        match provider {
            Provider::OpenAI | Provider::Azure => self.to_openai_format(),
            Provider::Anthropic => self.to_anthropic_format(),
            Provider::VertexAI | Provider::Gemini => self.to_google_format(),
            Provider::Bedrock => self.to_bedrock_format(),
            Provider::Cohere => self.to_cohere_format(),
            Provider::Ollama => self.to_ollama_format(),
            Provider::Groq | Provider::OpenRouter | Provider::Together => {
                self.to_openai_compatible_format()
            }
            Provider::AI21 => self.to_ai21_format(),
            Provider::Mistral => self.to_mistral_format(),
            Provider::HuggingFace => self.to_huggingface_format(),
            Provider::Perplexity => self.to_perplexity_format(),
            Provider::XAI => self.to_xai_format(),
            Provider::DeepSeek => self.to_deepseek_format(),
        }
    }

    /// Convert to OpenAI API format
    #[inline]
    fn to_openai_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "model": self.model.as_str(),
            "messages": self.messages,
            "stream": self.stream
        });

        if let Some(temp) = self.temperature {
            request["temperature"] = temp.into();
        }
        if let Some(max_tokens) = self.max_tokens {
            request["max_tokens"] = max_tokens.into();
        }
        if let Some(top_p) = self.top_p {
            request["top_p"] = top_p.into();
        }
        if let Some(freq_penalty) = self.frequency_penalty {
            request["frequency_penalty"] = freq_penalty.into();
        }
        if let Some(pres_penalty) = self.presence_penalty {
            request["presence_penalty"] = pres_penalty.into();
        }
        if !self.stop.is_empty() {
            request["stop"] = self
                .stop
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .into();
        }
        if !self.tools.is_empty() {
            request["tools"] = serde_json::to_value(&self.tools)
                .map_err(|_| CompletionRequestError::SerializationError)?;
        }
        if let Some(ref tool_choice) = self.tool_choice {
            request["tool_choice"] = serde_json::to_value(tool_choice)
                .map_err(|_| CompletionRequestError::SerializationError)?;
        }
        if let Some(ref user) = self.user {
            request["user"] = user.as_str().into();
        }

        // Add OpenAI-specific extensions
        if let Some(ref openai_ext) = self.extensions.openai {
            if let Some(ref response_format) = openai_ext.response_format {
                request["response_format"] = serde_json::json!({"type": response_format});
            }
            if let Some(seed) = openai_ext.seed {
                request["seed"] = seed.into();
            }
            if let Some(logit_bias) = &openai_ext.logit_bias {
                request["logit_bias"] = logit_bias.clone();
            }
        }

        Ok(request)
    }

    /// Convert to Anthropic API format
    #[inline]
    fn to_anthropic_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "model": self.model.as_str(),
            "messages": self.messages,
            "max_tokens": self.max_tokens.unwrap_or(1024),
            "stream": self.stream
        });

        if let Some(ref system) = self.system {
            request["system"] = system.as_str().into();
        }
        if let Some(temp) = self.temperature {
            request["temperature"] = temp.into();
        }
        if let Some(top_p) = self.top_p {
            request["top_p"] = top_p.into();
        }
        if let Some(top_k) = self.top_k {
            request["top_k"] = top_k.into();
        }
        if !self.stop.is_empty() {
            request["stop_sequences"] = self
                .stop
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .into();
        }
        if !self.tools.is_empty() {
            request["tools"] = serde_json::to_value(&self.tools)
                .map_err(|_| CompletionRequestError::SerializationError)?;
        }

        // Add Anthropic-specific extensions
        if let Some(ref anthropic_ext) = self.extensions.anthropic {
            if anthropic_ext.cache_control {
                // Anthropic cache control would be added here
            }
        }

        Ok(request)
    }

    /// Convert to Google (Vertex AI/Gemini) format
    #[inline]
    fn to_google_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "model": self.model.as_str(),
            "contents": self.messages
        });

        let mut generation_config = serde_json::Map::new();
        if let Some(temp) = self.temperature {
            generation_config.insert("temperature".to_string(), temp.into());
        }
        if let Some(max_tokens) = self.max_tokens {
            generation_config.insert("maxOutputTokens".to_string(), max_tokens.into());
        }
        if let Some(top_p) = self.top_p {
            generation_config.insert("topP".to_string(), top_p.into());
        }
        if let Some(top_k) = self.top_k {
            generation_config.insert("topK".to_string(), top_k.into());
        }
        if !self.stop.is_empty() {
            generation_config.insert(
                "stopSequences".to_string(),
                self.stop
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .into(),
            );
        }

        if !generation_config.is_empty() {
            request["generationConfig"] = generation_config.into();
        }

        // Add Google-specific extensions
        if let Some(ref google_ext) = self.extensions.google {
            if !google_ext.safety_settings.is_empty() {
                // Convert ArrayVec to Vec, then to Value
                let safety_vec: Vec<Value> = google_ext.safety_settings.iter().cloned().collect();
                request["safetySettings"] = Value::Array(safety_vec);
            }
        }

        Ok(request)
    }

    /// Convert to AWS Bedrock format
    #[inline]
    fn to_bedrock_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "modelId": self.model.as_str(),
            "messages": self.messages
        });

        let mut inference_config = serde_json::Map::new();
        if let Some(temp) = self.temperature {
            inference_config.insert("temperature".to_string(), temp.into());
        }
        if let Some(max_tokens) = self.max_tokens {
            inference_config.insert("maxTokens".to_string(), max_tokens.into());
        }
        if let Some(top_p) = self.top_p {
            inference_config.insert("topP".to_string(), top_p.into());
        }
        if !self.stop.is_empty() {
            inference_config.insert(
                "stopSequences".to_string(),
                self.stop
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .into(),
            );
        }

        if !inference_config.is_empty() {
            request["inferenceConfig"] = inference_config.into();
        }

        if let Some(ref system) = self.system {
            request["system"] = serde_json::json!([{"text": system}]);
        }

        Ok(request)
    }

    /// Convert to Cohere format
    #[inline]
    fn to_cohere_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "model": self.model.as_str(),
            "stream": self.stream
        });

        // Cohere uses chat_history instead of messages
        if !self.messages.is_empty() {
            if let Some(last_message) = self.messages.last() {
                request["message"] = last_message.content.clone().into();
                if self.messages.len() > 1 {
                    request["chat_history"] = serde_json::to_value(
                        &self.messages[..self.messages.len() - 1]
                    ).unwrap_or_default();
                }
            }
        }

        if let Some(temp) = self.temperature {
            request["temperature"] = temp.into();
        }
        if let Some(max_tokens) = self.max_tokens {
            request["max_tokens"] = max_tokens.into();
        }
        if let Some(top_p) = self.top_p {
            request["p"] = top_p.into();
        }
        if let Some(top_k) = self.top_k {
            request["k"] = top_k.into();
        }

        Ok(request)
    }

    /// Convert to Ollama format
    #[inline]
    fn to_ollama_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "model": self.model.as_str(),
            "messages": self.messages,
            "stream": self.stream
        });

        if let Some(ref system) = self.system {
            request["system"] = system.as_str().into();
        }

        let mut options = serde_json::Map::new();
        if let Some(temp) = self.temperature {
            options.insert("temperature".to_string(), temp.into());
        }
        if let Some(top_p) = self.top_p {
            options.insert("top_p".to_string(), top_p.into());
        }
        if let Some(top_k) = self.top_k {
            options.insert("top_k".to_string(), top_k.into());
        }

        if !options.is_empty() {
            request["options"] = options.into();
        }

        Ok(request)
    }

    /// Convert to OpenAI-compatible format (for Groq, OpenRouter, Together)
    #[inline]
    fn to_openai_compatible_format(&self) -> Result<Value, CompletionRequestError> {
        // Most providers are OpenAI-compatible
        self.to_openai_format()
    }

    /// Convert to AI21 format
    #[inline]
    fn to_ai21_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "model": self.model.as_str(),
            "messages": self.messages
        });

        if let Some(temp) = self.temperature {
            request["temperature"] = temp.into();
        }
        if let Some(max_tokens) = self.max_tokens {
            request["maxTokens"] = max_tokens.into();
        }
        if let Some(top_p) = self.top_p {
            request["topP"] = top_p.into();
        }

        Ok(request)
    }

    /// Convert to Mistral format
    #[inline]
    fn to_mistral_format(&self) -> Result<Value, CompletionRequestError> {
        // Mistral is mostly OpenAI-compatible
        self.to_openai_format()
    }

    /// Convert to HuggingFace format
    #[inline]
    fn to_huggingface_format(&self) -> Result<Value, CompletionRequestError> {
        let mut request = serde_json::json!({
            "model": self.model.as_str(),
            "messages": self.messages,
            "stream": self.stream
        });

        let mut parameters = serde_json::Map::new();
        if let Some(temp) = self.temperature {
            parameters.insert("temperature".to_string(), temp.into());
        }
        if let Some(max_tokens) = self.max_tokens {
            parameters.insert("max_new_tokens".to_string(), max_tokens.into());
        }
        if let Some(top_p) = self.top_p {
            parameters.insert("top_p".to_string(), top_p.into());
        }

        if !parameters.is_empty() {
            request["parameters"] = parameters.into();
        }

        Ok(request)
    }

    /// Convert to Perplexity format
    #[inline]
    fn to_perplexity_format(&self) -> Result<Value, CompletionRequestError> {
        // Perplexity is OpenAI-compatible
        self.to_openai_format()
    }

    /// Convert to xAI format
    #[inline]
    fn to_xai_format(&self) -> Result<Value, CompletionRequestError> {
        // xAI is OpenAI-compatible
        self.to_openai_format()
    }

    /// Convert to DeepSeek format
    #[inline]
    fn to_deepseek_format(&self) -> Result<Value, CompletionRequestError> {
        // DeepSeek is OpenAI-compatible
        self.to_openai_format()
    }
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool type (always "function" for now)
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    /// Function definition
    pub function: FunctionDefinition,
}

impl ToolDefinition {
    /// Create a new function tool
    #[inline]
    pub fn function(
        name: &str,
        description: &str,
        parameters: Value,
    ) -> Result<Self, CompletionRequestError> {
        let func_def = FunctionDefinition::new(name, description, parameters)?;
        Ok(Self {
            tool_type: ToolType::Function,
            function: func_def,
        })
    }
}

/// Tool type enum
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    /// Function tool
    Function,
}

/// Function definition for tool calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: ArrayString<MAX_IDENTIFIER_LEN>,
    /// Function description
    pub description: String,
    /// JSON schema for function parameters
    pub parameters: Value,
}

impl FunctionDefinition {
    /// Create a new function definition
    #[inline]
    pub fn new(
        name: &str,
        description: &str,
        parameters: Value,
    ) -> Result<Self, CompletionRequestError> {
        let func_name = ArrayString::from(name)
            .map_err(|_| CompletionRequestError::FunctionNameTooLong(name.len()))?;

        if name.is_empty() {
            return Err(CompletionRequestError::EmptyFunctionName);
        }

        Ok(Self {
            name: func_name,
            description: description.to_string(),
            parameters,
        })
    }
}

/// Tool choice strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Automatic tool choice (model decides)
    Auto,
    /// No tools should be called
    None,
    /// Force a specific tool to be called
    Required { function: FunctionChoice },
    /// Force any tool to be called
    RequiredAny,
}

/// Specific function choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChoice {
    /// Function name to call
    pub name: ArrayString<MAX_IDENTIFIER_LEN>,
}

/// Provider-specific extensions for completion requests
///
/// This struct contains provider-specific parameters that don't fit into
/// the universal completion request structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderExtensions {
    /// OpenAI-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAIExtensions>,

    /// Anthropic-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic: Option<AnthropicExtensions>,

    /// Google-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google: Option<GoogleExtensions>,

    /// AWS Bedrock-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bedrock: Option<BedrockExtensions>,

    /// Cohere-specific extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohere: Option<CohereExtensions>,

    /// Generic provider parameters for other providers
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub custom: HashMap<String, Value>,
}

impl ProviderExtensions {
    /// Create new empty provider extensions
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if extensions are empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.openai.is_none()
            && self.anthropic.is_none()
            && self.google.is_none()
            && self.bedrock.is_none()
            && self.cohere.is_none()
            && self.custom.is_empty()
    }

    /// Validate all extensions
    #[inline]
    pub fn validate(&self) -> Result<(), CompletionRequestError> {
        if let Some(ref openai) = self.openai {
            openai.validate()?;
        }
        if let Some(ref anthropic) = self.anthropic {
            anthropic.validate()?;
        }
        if let Some(ref google) = self.google {
            google.validate()?;
        }
        if let Some(ref bedrock) = self.bedrock {
            bedrock.validate()?;
        }
        if let Some(ref cohere) = self.cohere {
            cohere.validate()?;
        }
        Ok(())
    }
}

/// OpenAI-specific extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIExtensions {
    /// Response format (e.g., "json_object")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,

    /// Seed for deterministic generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Logit bias for token probability modification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,

    /// Number of completions to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to return log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Number of most likely tokens to return log probabilities for
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

impl OpenAIExtensions {
    /// Validate OpenAI extensions
    #[inline]
    pub fn validate(&self) -> Result<(), CompletionRequestError> {
        if let Some(n) = self.n {
            if n == 0 || n > 128 {
                return Err(CompletionRequestError::InvalidParameterValue(
                    "n must be between 1 and 128",
                ));
            }
        }
        if let Some(top_logprobs) = self.top_logprobs {
            if top_logprobs > 20 {
                return Err(CompletionRequestError::InvalidParameterValue(
                    "top_logprobs must be <= 20",
                ));
            }
        }
        Ok(())
    }
}

/// Anthropic-specific extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicExtensions {
    /// Enable cache control
    pub cache_control: bool,

    /// Anthropic beta features
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub anthropic_beta: ArrayVec<ArrayString<32>, 4>,
}

impl AnthropicExtensions {
    /// Validate Anthropic extensions
    #[inline]
    pub fn validate(&self) -> Result<(), CompletionRequestError> {
        // Currently no validation needed
        Ok(())
    }
}

/// Google-specific extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleExtensions {
    /// Safety settings for content filtering
    #[serde(skip_serializing_if = "ArrayVec::is_empty")]
    pub safety_settings: ArrayVec<Value, MAX_SAFETY_SETTINGS>,

    /// Candidate count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<u32>,
}

impl GoogleExtensions {
    /// Validate Google extensions
    #[inline]
    pub fn validate(&self) -> Result<(), CompletionRequestError> {
        if let Some(count) = self.candidate_count {
            if count == 0 || count > 8 {
                return Err(CompletionRequestError::InvalidParameterValue(
                    "candidate_count must be between 1 and 8",
                ));
            }
        }
        Ok(())
    }
}

/// AWS Bedrock-specific extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockExtensions {
    /// Additional model parameters
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub additional_model_request_fields: HashMap<String, Value>,

    /// Guardrail configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrail_config: Option<Value>,
}

impl BedrockExtensions {
    /// Validate Bedrock extensions
    #[inline]
    pub fn validate(&self) -> Result<(), CompletionRequestError> {
        // Currently no validation needed
        Ok(())
    }
}

/// Cohere-specific extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereExtensions {
    /// Preamble override
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preamble_override: Option<String>,

    /// Conversation ID for chat continuity
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,

    /// Prompt truncation setting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_truncation: Option<String>,
}

impl CohereExtensions {
    /// Validate Cohere extensions
    #[inline]
    pub fn validate(&self) -> Result<(), CompletionRequestError> {
        // Currently no validation needed
        Ok(())
    }
}

/// Completion request errors
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionRequestError {
    /// Model name is empty
    EmptyModelName,
    /// Model name is too long
    ModelNameTooLong(usize),
    /// Too many messages in request
    TooManyMessages,
    /// No messages provided
    NoMessages,
    /// Temperature out of valid range
    TemperatureOutOfRange(f64),
    /// Max tokens is zero
    MaxTokensZero,
    /// Max tokens too large
    MaxTokensTooLarge(u32),
    /// Top-p out of valid range
    TopPOutOfRange(f64),
    /// Top-k is zero
    TopKZero,
    /// Frequency penalty out of range
    FrequencyPenaltyOutOfRange(f64),
    /// Presence penalty out of range
    PresencePenaltyOutOfRange(f64),
    /// Stop sequence is empty
    EmptyStopSequence,
    /// Stop sequence too long
    StopSequenceTooLong(usize),
    /// Too many stop sequences
    TooManyStopSequences,
    /// Too many tools
    TooManyTools,
    /// Tool choice specified without tools
    ToolChoiceWithoutTools,
    /// Function name is empty
    EmptyFunctionName,
    /// Function name too long
    FunctionNameTooLong(usize),
    /// User ID too long
    UserIdTooLong(usize),
    /// Conflicting parameters
    ConflictingParameters,
    /// Invalid parameter value
    InvalidParameterValue(&'static str),
    /// Serialization error
    SerializationError,
    /// Validation error from common module
    ValidationError(ValidationError),
}

impl fmt::Display for CompletionRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompletionRequestError::EmptyModelName => write!(f, "Model name cannot be empty"),
            CompletionRequestError::ModelNameTooLong(len) => write!(
                f,
                "Model name too long: {len} characters (max {MAX_IDENTIFIER_LEN})"
            ),
            CompletionRequestError::TooManyMessages => {
                write!(f, "Too many messages (max {MAX_MESSAGES})")
            }
            CompletionRequestError::NoMessages => {
                write!(f, "No messages or system prompt provided")
            }
            CompletionRequestError::TemperatureOutOfRange(temp) => {
                write!(f, "Temperature {temp} out of range (0.0 to 2.0)")
            }
            CompletionRequestError::MaxTokensZero => write!(f, "Max tokens cannot be zero"),
            CompletionRequestError::MaxTokensTooLarge(tokens) => {
                write!(f, "Max tokens {tokens} too large (max 2,000,000)")
            }
            CompletionRequestError::TopPOutOfRange(top_p) => {
                write!(f, "Top-p {top_p} out of range (0.0 to 1.0)")
            }
            CompletionRequestError::TopKZero => write!(f, "Top-k cannot be zero"),
            CompletionRequestError::FrequencyPenaltyOutOfRange(penalty) => {
                write!(f, "Frequency penalty {penalty} out of range (-2.0 to 2.0)")
            }
            CompletionRequestError::PresencePenaltyOutOfRange(penalty) => {
                write!(f, "Presence penalty {penalty} out of range (-2.0 to 2.0)")
            }
            CompletionRequestError::EmptyStopSequence => write!(f, "Stop sequence cannot be empty"),
            CompletionRequestError::StopSequenceTooLong(len) => write!(
                f,
                "Stop sequence too long: {len} characters (max {MAX_STOP_SEQUENCE_LEN})"
            ),
            CompletionRequestError::TooManyStopSequences => {
                write!(f, "Too many stop sequences (max {MAX_STOP_SEQUENCES})")
            }
            CompletionRequestError::TooManyTools => write!(f, "Too many tools (max {MAX_TOOLS})"),
            CompletionRequestError::ToolChoiceWithoutTools => {
                write!(f, "Tool choice specified but no tools provided")
            }
            CompletionRequestError::EmptyFunctionName => write!(f, "Function name cannot be empty"),
            CompletionRequestError::FunctionNameTooLong(len) => write!(
                f,
                "Function name too long: {len} characters (max {MAX_IDENTIFIER_LEN})"
            ),
            CompletionRequestError::UserIdTooLong(len) => write!(
                f,
                "User ID too long: {len} characters (max {MAX_IDENTIFIER_LEN})"
            ),
            CompletionRequestError::ConflictingParameters => {
                write!(f, "Conflicting parameters specified")
            }
            CompletionRequestError::InvalidParameterValue(msg) => {
                write!(f, "Invalid parameter value: {msg}")
            }
            CompletionRequestError::SerializationError => write!(f, "Failed to serialize request"),
            CompletionRequestError::ValidationError(err) => write!(f, "Validation error: {err}"),
        }
    }
}

impl std::error::Error for CompletionRequestError {}

impl From<ValidationError> for CompletionRequestError {
    fn from(err: ValidationError) -> Self {
        CompletionRequestError::ValidationError(err)
    }
}

impl From<serde_json::Error> for CompletionRequestError {
    fn from(_err: serde_json::Error) -> Self {
        CompletionRequestError::SerializationError
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::common::BaseMessage;

    #[test]
    fn test_completion_request_creation() {
        let request = CompletionRequest::new("gpt-4").expect("Valid model name");
        assert_eq!(request.model.as_str(), "gpt-4");
        assert!(request.messages.is_empty());
        assert!(!request.stream);
    }

    #[test]
    fn test_request_with_messages() {
        let request = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_message(BaseMessage::user("Hello"))
            .expect("Valid message")
            .with_message(BaseMessage::assistant("Hi there!"))
            .expect("Valid message");

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].content, "Hello");
        assert_eq!(request.messages[1].content, "Hi there!");
    }

    #[test]
    fn test_parameter_validation() {
        let request = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_temperature(0.7)
            .expect("Valid temperature")
            .with_max_tokens(1000)
            .expect("Valid max tokens")
            .with_top_p(0.9)
            .expect("Valid top-p");

        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_tokens, Some(1000));
        assert_eq!(request.top_p, Some(0.9));
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid temperature
        let invalid_temp = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_temperature(3.0);
        assert!(invalid_temp.is_err());

        // Invalid max tokens
        let invalid_tokens = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_max_tokens(0);
        assert!(invalid_tokens.is_err());

        // Invalid top-p
        let invalid_top_p = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_top_p(1.5);
        assert!(invalid_top_p.is_err());
    }

    #[test]
    fn test_tool_definitions() {
        let tool = ToolDefinition::function(
            "get_weather",
            "Get current weather for a location",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        )
        .expect("Valid tool definition");

        assert_eq!(tool.function.name.as_str(), "get_weather");
        assert_eq!(
            tool.function.description,
            "Get current weather for a location"
        );
    }

    #[test]
    fn test_provider_format_conversion() {
        let request = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_message(BaseMessage::user("Hello"))
            .expect("Valid message")
            .with_temperature(0.7)
            .expect("Valid temperature")
            .with_streaming(true);

        // Test OpenAI format
        let openai_format = request
            .to_provider_format(Provider::OpenAI)
            .expect("Valid conversion");
        assert!(openai_format.get("model").is_some());
        assert!(openai_format.get("messages").is_some());
        assert!(openai_format.get("temperature").is_some());
        assert_eq!(
            openai_format.get("stream"),
            Some(&serde_json::Value::Bool(true))
        );

        // Test Anthropic format
        let anthropic_format = request
            .to_provider_format(Provider::Anthropic)
            .expect("Valid conversion");
        assert!(anthropic_format.get("model").is_some());
        assert!(anthropic_format.get("messages").is_some());
        assert!(anthropic_format.get("max_tokens").is_some());
    }

    #[test]
    fn test_request_validation() {
        // Valid request
        let valid_request = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_message(BaseMessage::user("Hello"))
            .expect("Valid message");
        assert!(valid_request.validate().is_ok());

        // Invalid request - no messages
        let invalid_request = CompletionRequest::new("gpt-4").expect("Valid model");
        assert!(invalid_request.validate().is_err());

        // Invalid request - tool choice without tools
        let mut invalid_tool_request = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_message(BaseMessage::user("Hello"))
            .expect("Valid message");
        invalid_tool_request.tool_choice = Some(ToolChoice::Auto);
        assert!(invalid_tool_request.validate().is_err());
    }

    #[test]
    fn test_provider_extensions() {
        let mut request = CompletionRequest::new("gpt-4").expect("Valid model");

        // Add OpenAI extensions
        request.extensions.openai = Some(OpenAIExtensions {
            response_format: Some("json_object".to_string()),
            seed: Some(42),
            logit_bias: None,
            n: Some(1),
            logprobs: Some(true),
            top_logprobs: Some(5),
        });

        assert!(!request.extensions.is_empty());
        assert!(request.extensions.validate().is_ok());

        // Test invalid extensions
        request.extensions.openai.as_mut().unwrap().n = Some(0);
        assert!(request.extensions.validate().is_err());
    }

    #[test]
    fn test_stop_sequences() {
        let request = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_stop_sequence("\\n")
            .expect("Valid stop sequence")
            .with_stop_sequence("END")
            .expect("Valid stop sequence");

        assert_eq!(request.stop.len(), 2);
        assert_eq!(request.stop[0].as_str(), "\\n");
        assert_eq!(request.stop[1].as_str(), "END");
    }

    #[test]
    fn test_streaming_mode() {
        let request = CompletionRequest::new("gpt-4")
            .expect("Valid model")
            .with_streaming(true);

        assert!(request.stream);

        let non_streaming = request.with_streaming(false);
        assert!(!non_streaming.stream);
    }
}
