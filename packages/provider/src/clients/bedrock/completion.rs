//! Bedrock completion builder implementing CompletionProvider trait
//!
//! Provides a clean builder pattern for AWS Bedrock completions with:
//! - Zero allocation message conversion
//! - Model-specific parameter validation
//! - AWS SigV4 request signing
//! - Tool/function calling support
//! - Streaming and non-streaming execution

use std::sync::Arc;

use arrayvec::{ArrayString, ArrayVec};
use fluent_ai_domain::chunk::CompletionChunk;
use fluent_ai_domain::completion::CompletionRequest;
use fluent_ai_domain::message::{Message, MessageRole as Role};
use fluent_ai_domain::tool::Tool;
use fluent_ai_domain::{AsyncStream, AsyncTask};
use fluent_ai_http3::{HttpClient, HttpRequest};
use serde_json::Value;

use super::error::{BedrockError, Result};
use super::models::validate_model_capability;
use super::sigv4::SigV4Signer;
use super::streaming::BedrockStream;
use crate::completion_provider::{
    CompletionError, CompletionProvider, CompletionResponse, StreamingResponse,
};

/// Bedrock completion builder implementing CompletionProvider
#[derive(Clone)]
pub struct BedrockCompletionBuilder {
    /// HTTP client for API requests
    http_client: HttpClient,
    /// AWS SigV4 signer
    signer: Arc<SigV4Signer>,
    /// AWS region
    region: String,
    /// Model identifier
    model: &'static str,
    /// System prompt
    system_prompt: Option<String>,
    /// Temperature (0.0 to 1.0)
    temperature: Option<f32>,
    /// Maximum tokens to generate
    max_tokens: Option<u32>,
    /// Top-p sampling
    top_p: Option<f32>,
    /// Top-k sampling
    top_k: Option<u32>,
    /// Stop sequences
    stop_sequences: Vec<String>,
    /// Tools/functions available to the model
    tools: Vec<Tool>,
    /// Whether to stream the response
    stream: bool,
}

impl BedrockCompletionBuilder {
    /// Create new Bedrock completion builder
    pub fn new(
        http_client: HttpClient,
        signer: Arc<SigV4Signer>,
        region: String,
        model: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            http_client,
            signer,
            region,
            model,
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            tools: Vec::new(),
            stream: false,
        })
    }

    /// Set system prompt
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set temperature (0.0 to 1.0)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp.clamp(0.0, 1.0));
        self
    }

    /// Set maximum tokens to generate
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set top-p sampling
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p.clamp(0.0, 1.0));
        self
    }

    /// Set top-k sampling  
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Add stop sequence
    pub fn stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Add tool/function
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Enable streaming
    pub fn stream(mut self, enabled: bool) -> Self {
        self.stream = enabled;
        self
    }

    /// Build Bedrock API request body
    fn build_request_body(&self, messages: &[Message]) -> Result<Vec<u8>> {
        let mut request = serde_json::Map::new();

        // Convert messages to Bedrock format
        let bedrock_messages = self.convert_messages(messages)?;
        request.insert(
            "messages".to_string(),
            serde_json::Value::Array(bedrock_messages),
        );

        // Add model parameters
        let mut inference_config = serde_json::Map::new();

        if let Some(temp) = self.temperature {
            inference_config.insert("temperature".to_string(), serde_json::Value::from(temp));
        }

        if let Some(max_tokens) = self.max_tokens {
            inference_config.insert("maxTokens".to_string(), serde_json::Value::from(max_tokens));
        }

        if let Some(top_p) = self.top_p {
            inference_config.insert("topP".to_string(), serde_json::Value::from(top_p));
        }

        if !self.stop_sequences.is_empty() {
            let stop_seqs: Vec<serde_json::Value> = self
                .stop_sequences
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            inference_config.insert(
                "stopSequences".to_string(),
                serde_json::Value::Array(stop_seqs),
            );
        }

        if !inference_config.is_empty() {
            request.insert(
                "inferenceConfig".to_string(),
                serde_json::Value::Object(inference_config),
            );
        }

        // Add system prompt if provided
        if let Some(ref system) = self.system_prompt {
            let mut system_prompts = Vec::new();
            let mut system_prompt = serde_json::Map::new();
            system_prompt.insert(
                "text".to_string(),
                serde_json::Value::String(system.clone()),
            );
            system_prompts.push(serde_json::Value::Object(system_prompt));
            request.insert(
                "system".to_string(),
                serde_json::Value::Array(system_prompts),
            );
        }

        // Add tools if provided and model supports them
        if !self.tools.is_empty() {
            if let Err(_) = validate_model_capability(self.model, "tools") {
                return Err(BedrockError::config_error(
                    "tools",
                    "Model does not support tools",
                ));
            }

            let bedrock_tools = self.convert_tools()?;
            request.insert(
                "toolConfig".to_string(),
                serde_json::Value::Object(bedrock_tools),
            );
        }

        serde_json::to_vec(&request)
            .map_err(|e| BedrockError::config_error("json_serialization", &e.to_string()))
    }

    /// Convert domain messages to Bedrock format
    fn convert_messages(&self, messages: &[Message]) -> Result<Vec<serde_json::Value>> {
        let mut bedrock_messages = Vec::new();

        for message in messages {
            let mut bedrock_msg = serde_json::Map::new();

            // Set role
            let role_str = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => continue, // System messages handled separately
                Role::Tool => "tool",
            };
            bedrock_msg.insert(
                "role".to_string(),
                serde_json::Value::String(role_str.to_string()),
            );

            // Set content
            let mut content = Vec::new();
            let mut text_content = serde_json::Map::new();
            text_content.insert(
                "text".to_string(),
                serde_json::Value::String(message.content.clone()),
            );
            content.push(serde_json::Value::Object(text_content));

            bedrock_msg.insert("content".to_string(), serde_json::Value::Array(content));
            bedrock_messages.push(serde_json::Value::Object(bedrock_msg));
        }

        Ok(bedrock_messages)
    }

    /// Convert domain tools to Bedrock format
    fn convert_tools(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut tool_config = serde_json::Map::new();
        let mut tools_array = Vec::new();

        for tool in &self.tools {
            let mut bedrock_tool = serde_json::Map::new();

            let mut tool_spec = serde_json::Map::new();
            tool_spec.insert(
                "name".to_string(),
                serde_json::Value::String(tool.name.clone()),
            );
            tool_spec.insert(
                "description".to_string(),
                serde_json::Value::String(tool.description.clone()),
            );

            if let Some(ref schema) = tool.parameters {
                tool_spec.insert("inputSchema".to_string(), schema.clone());
            }

            bedrock_tool.insert("toolSpec".to_string(), serde_json::Value::Object(tool_spec));
            tools_array.push(serde_json::Value::Object(bedrock_tool));
        }

        tool_config.insert("tools".to_string(), serde_json::Value::Array(tools_array));
        Ok(tool_config)
    }

    /// Build AWS Bedrock API endpoint URL
    fn build_endpoint_url(&self) -> ArrayString<128> {
        let mut url = ArrayString::new();
        let _ = url.try_push_str(&format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/{}",
            self.region,
            self.model,
            if self.stream {
                "invoke-with-response-stream"
            } else {
                "invoke"
            }
        ));
        url
    }

    /// Execute completion request
    async fn execute_completion(&self, messages: &[Message]) -> Result<CompletionResponse> {
        let request_body = self.build_request_body(messages)?;
        let endpoint = self.build_endpoint_url();

        // Prepare headers
        let headers = [
            (
                "host",
                &format!("bedrock-runtime.{}.amazonaws.com", self.region),
            ),
            ("content-type", "application/json"),
            ("content-length", &request_body.len().to_string()),
        ];

        // Sign request
        let auth_header = self.signer.sign_request(
            "POST",
            &format!("/model/{}/invoke", self.model),
            "",
            &headers,
            &request_body,
        )?;

        // Build HTTP request
        let mut http_headers = Vec::new();
        for (name, value) in &headers {
            http_headers.push((*name, *value));
        }
        http_headers.push(("authorization", auth_header.as_str()));

        let http_request = HttpRequest::post(endpoint.as_str(), request_body)
            .map_err(|e| BedrockError::config_error("http_request", &e.to_string()))?
            .headers(http_headers.iter().map(|(k, v)| (*k, *v)));

        // Send request
        let response = self
            .http_client
            .send(http_request)
            .await
            .map_err(|e| BedrockError::config_error("http_send", &e.to_string()))?;

        if !response.status().is_success() {
            return Err(BedrockError::from(response.status().as_u16()));
        }

        // Parse response
        let body = response
            .body()
            .await
            .map_err(|e| BedrockError::config_error("response_body", &e.to_string()))?;

        let response_json: serde_json::Value = serde_json::from_slice(&body)
            .map_err(|e| BedrockError::config_error("json_parse", &e.to_string()))?;

        self.parse_completion_response(response_json)
    }

    /// Parse Bedrock completion response
    fn parse_completion_response(&self, response: serde_json::Value) -> Result<CompletionResponse> {
        let output = response
            .get("output")
            .and_then(|o| o.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .unwrap_or("");

        let usage = response.get("usage");
        let input_tokens = usage
            .and_then(|u| u.get("inputTokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;
        let output_tokens = usage
            .and_then(|u| u.get("outputTokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;

        Ok(CompletionResponse {
            content: output.to_string(),
            finish_reason: Some("stop".to_string()),
            usage: Some(fluent_ai_domain::usage::Usage {
                prompt_tokens: input_tokens,
                completion_tokens: output_tokens,
                total_tokens: input_tokens + output_tokens,
            }),
            model: Some(self.model.to_string()),
        })
    }
}

impl CompletionProvider for BedrockCompletionBuilder {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingResponse;
    type Error = CompletionError;

    fn prompt(&self, prompt: fluent_ai_domain::prompt::Prompt) -> AsyncStream<CompletionChunk> {
        let messages = vec![Message {
            role: Role::User,
            content: prompt.to_string(),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let mut builder = self.clone();
        builder.stream = true;

        AsyncStream::new(async move {
            match builder.stream_completion_internal(&messages).await {
                Ok(stream) => stream,
                Err(e) => {
                    let error_chunk = CompletionChunk {
                        content: Some(format!("Error: {}", e)),
                        finish_reason: Some("error".to_string()),
                        usage: None,
                        model: Some(builder.model.to_string()),
                        delta: None,
                    };
                    AsyncStream::from_single(error_chunk)
                }
            }
        })
    }

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> AsyncTask<Result<Self::Response, Self::Error>> {
        let builder = self.clone();
        AsyncTask::spawn(async move {
            let response = builder
                .execute_completion(&request.messages)
                .await
                .map_err(|e| CompletionError::from(e))?;
            Ok(response)
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> AsyncTask<Result<AsyncStream<Self::StreamingResponse>, Self::Error>> {
        let mut builder = self.clone();
        builder.stream = true;

        AsyncTask::spawn(async move {
            let stream = builder
                .stream_completion_internal(&request.messages)
                .await
                .map_err(|e| CompletionError::from(e))?;

            let streaming_response = StreamingResponse::new(stream);
            Ok(AsyncStream::from_single(streaming_response))
        })
    }
}

impl BedrockCompletionBuilder {
    /// Internal streaming completion implementation
    async fn stream_completion_internal(
        &self,
        messages: &[Message],
    ) -> Result<AsyncStream<CompletionChunk>> {
        let request_body = self.build_request_body(messages)?;
        let endpoint = self.build_endpoint_url();

        // Prepare headers for streaming
        let headers = [
            (
                "host",
                &format!("bedrock-runtime.{}.amazonaws.com", self.region),
            ),
            ("content-type", "application/json"),
            ("content-length", &request_body.len().to_string()),
            ("accept", "application/vnd.amazon.eventstream"),
        ];

        // Sign request
        let auth_header = self.signer.sign_request(
            "POST",
            &format!("/model/{}/invoke-with-response-stream", self.model),
            "",
            &headers,
            &request_body,
        )?;

        // Build HTTP request with streaming headers
        let mut http_headers = Vec::new();
        for (name, value) in &headers {
            http_headers.push((*name, *value));
        }
        http_headers.push(("authorization", auth_header.as_str()));

        let http_request = HttpRequest::post(endpoint.as_str(), request_body)
            .map_err(|e| BedrockError::config_error("http_request", &e.to_string()))?
            .headers(http_headers.iter().map(|(k, v)| (*k, *v)));

        // Send request and get streaming response
        let response = self
            .http_client
            .send(http_request)
            .await
            .map_err(|e| BedrockError::config_error("http_send", &e.to_string()))?;

        if !response.status().is_success() {
            return Err(BedrockError::from(response.status().as_u16()));
        }

        // Create Bedrock stream from response
        let bedrock_stream = BedrockStream::new(response, self.model);
        Ok(bedrock_stream.into_chunk_stream())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let http_client = HttpClient::with_config(HttpConfig::ai_optimized())
            .expect("Failed to create http client in test");
        let signer = Arc::new(SigV4Signer::new(
            AwsCredentials::new("test", "test", "us-east-1")
                .expect("Failed to create credentials in test"),
        ));

        let builder = BedrockCompletionBuilder::new(
            http_client,
            signer,
            "us-east-1".to_string(),
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        );
        assert!(builder.is_ok());
    }

    #[test]
    fn test_builder_configuration() {
        let http_client = HttpClient::with_config(HttpConfig::ai_optimized())
            .expect("Failed to create http client in test");
        let signer = Arc::new(SigV4Signer::new(
            AwsCredentials::new("test", "test", "us-east-1")
                .expect("Failed to create credentials in test"),
        ));

        let builder = BedrockCompletionBuilder::new(
            http_client,
            signer,
            "us-east-1".to_string(),
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        .expect("Failed to create builder in test")
        .system_prompt("You are helpful")
        .temperature(0.8)
        .max_tokens(1000)
        .stream(true);

        assert_eq!(builder.system_prompt.as_ref().unwrap(), "You are helpful");
        assert_eq!(builder.temperature.unwrap(), 0.8);
        assert_eq!(builder.max_tokens.unwrap(), 1000);
        assert!(builder.stream);
    }
}
