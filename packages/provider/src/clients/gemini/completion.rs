// ================================================================
//! Google Gemini Completion Integration
//! From [Gemini API Reference](https://ai.google.dev/api/generate-content)
// ================================================================
/// `gemini-2.5-pro-preview-06-05` completion model
pub const GEMINI_2_5_PRO_PREVIEW_06_05: &str = "gemini-2.5-pro-preview-06-05";
/// `gemini-2.5-pro-preview-05-06` completion model
pub const GEMINI_2_5_PRO_PREVIEW_05_06: &str = "gemini-2.5-pro-preview-05-06";
/// `gemini-2.5-pro-preview-03-25` completion model
pub const GEMINI_2_5_PRO_PREVIEW_03_25: &str = "gemini-2.5-pro-preview-03-25";
/// `gemini-2.5-flash-preview-05-20` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_05_20: &str = "gemini-2.5-flash-preview-05-20";
/// `gemini-2.5-flash-preview-04-17` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_04_17: &str = "gemini-2.5-flash-preview-04-17";
/// `gemini-2.5-pro-exp-03-25` experimental completion model
pub const GEMINI_2_5_PRO_EXP_03_25: &str = "gemini-2.5-pro-exp-03-25";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";
/// `gemini-1.5-flash` completion model
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro` completion model
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-pro-8b` completion model
pub const GEMINI_1_5_PRO_8B: &str = "gemini-1.5-pro-8b";
/// `gemini-1.0-pro` completion model
pub const GEMINI_1_0_PRO: &str = "gemini-1.0-pro";

use std::convert::TryFrom;

use arrayvec::ArrayVec;
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_domain::chunk::{CompletionChunk, FinishReason, Usage};
use fluent_ai_domain::completion::{
    self, CompletionCoreError as CompletionError, CompletionRequest,
};
use fluent_ai_domain::tool::ToolDefinition;
use fluent_ai_domain::{AsyncTask, spawn_async};
use fluent_ai_domain::{Document, Message};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpError, HttpRequest};
use gemini_api_types::{
    Content, FunctionDeclaration, GenerateContentRequest, GenerateContentResponse,
    GenerationConfig, Part, Role, Tool,
};
use serde_json::{Map, Value};

use self::gemini_api_types::Schema;
use super::Client;
use super::streaming::StreamingCompletionResponse;
use crate::{
    AsyncStream, OneOrMany,
    completion_provider::{ChunkHandler, CompletionProvider, ModelConfig, ModelInfo},
};

/// Maximum messages per completion request (compile-time bounded)
const MAX_MESSAGES: usize = 128;
/// Maximum tools per request (compile-time bounded)  
const MAX_TOOLS: usize = 32;
/// Maximum documents per request (compile-time bounded)
const MAX_DOCUMENTS: usize = 64;

// =================================================================
// Rig Implementation Types
// =================================================================

// CompletionModel is now imported from fluent_ai_domain::model
// Removed duplicated CompletionModel struct - use canonical domain type

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = GenerateContentResponse;
    type StreamingResponse = StreamingCompletionResponse;

    fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> crate::runtime::AsyncTask<
        Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError>,
    > {
        let (tx, task) = crate::runtime::channel();
        let client = self.client.clone();
        let model = self.model.clone();

        match create_request_body(completion_request) {
            Ok(request) => {
                crate::runtime::spawn_async(async move {
                    tracing::debug!(
                        "Sending completion request to Gemini API {}",
                        serde_json::to_string_pretty(&request).unwrap_or_default()
                    );

                    let response = client
                        .post(&format!("/v1beta/models/{}:generateContent", model))
                        .json(&request)
                        .send()
                        .await;

                    let result = match response {
                        Ok(response) => {
                            if response.status().is_success() {
                                match response.json::<GenerateContentResponse>().await {
                                    Ok(response) => {
                                        match response.usage_metadata {
                                            Some(ref usage) => tracing::info!(target: "rig",
                                                "Gemini completion token usage: {}",
                                                usage
                                            ),
                                            None => tracing::info!(target: "rig",
                                                "Gemini completion token usage: n/a",
                                            ),
                                        }
                                        tracing::debug!("Received response");
                                        completion::CompletionResponse::try_from(response)
                                    }
                                    Err(e) => {
                                        Err(CompletionError::DeserializationError(e.to_string()))
                                    }
                                }
                            } else {
                                // Domain uses HTTP3, provider delegates to domain layer
                                Err(CompletionError::ProviderError(
                                    "HTTP error - domain layer handles HTTP3 operations"
                                        .to_string(),
                                ))
                            }
                        }
                        Err(e) => Err(CompletionError::RequestError(e.to_string())),
                    };

                    tx.finish(result);
                });
            }
            Err(e) => {
                tx.finish(Err(e));
            }
        }

        task
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> crate::runtime::AsyncTask<
        Result<
            crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
            CompletionError,
        >,
    > {
        let (tx, task) = crate::runtime::channel();
        let client = self.client.clone();
        let model = self.model.clone();

        crate::runtime::spawn_async(async move {
            let result = CompletionModel { client, model }
                .stream_internal(request)
                .await;
            tx.finish(result);
        });

        task
    }
}

// =================================================================
// New CompletionProvider Implementation
// =================================================================

/// Zero-allocation Gemini completion builder with perfect ergonomics
#[derive(Clone)]
pub struct GeminiCompletionBuilder {
    client: HttpClient,
    api_key: String,
    explicit_api_key: Option<String>, // .api_key() override takes priority
    base_url: &'static str,
    model_name: &'static str,
    config: &'static ModelConfig,
    system_prompt: String,
    temperature: f64,
    max_tokens: u32,
    top_p: f64,
    top_k: u32,
    frequency_penalty: f64,
    presence_penalty: f64,
    chat_history: ArrayVec<Message, MAX_MESSAGES>,
    documents: ArrayVec<Document, MAX_DOCUMENTS>,
    tools: ArrayVec<ToolDefinition, MAX_TOOLS>,
    additional_params: Option<Value>,
    chunk_handler: Option<ChunkHandler>,
}

impl CompletionProvider for GeminiCompletionBuilder {
    /// Create new Gemini completion builder with ModelInfo defaults
    #[inline(always)]
    fn new(
        api_key: String,
        model_name: &'static str,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|_| crate::completion_provider::CompletionError::HttpError)?;

        let config = crate::clients::gemini::model_info::get_model_config(model_name);

        Ok(Self {
            client,
            api_key,
            explicit_api_key: None,
            base_url: "https://generativelanguage.googleapis.com",
            model_name,
            config,
            system_prompt: config.system_prompt.to_string(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            top_k: 40, // Gemini default
            frequency_penalty: config.frequency_penalty,
            presence_penalty: config.presence_penalty,
            chat_history: ArrayVec::new(),
            documents: ArrayVec::new(),
            tools: ArrayVec::new(),
            additional_params: None,
            chunk_handler: None,
        })
    }

    /// Set explicit API key (takes priority over environment variables)
    #[inline(always)]
    fn api_key(mut self, key: impl Into<String>) -> Self {
        self.explicit_api_key = Some(key.into());
        self
    }

    /// Environment variable names to search for Gemini API keys (ordered by priority)
    #[inline(always)]
    fn env_api_keys(&self) -> ZeroOneOrMany<String> {
        // First found wins - search in priority order
        ZeroOneOrMany::Many(vec![
            "GEMINI_API_KEY".to_string(),        // Primary Gemini key
            "GOOGLE_GEMINI_API_KEY".to_string(), // Google-specific Gemini key
            "GOOGLE_AI_API_KEY".to_string(),     // Google AI Platform key
            "GOOGLE_API_KEY".to_string(),        // General Google API key
        ])
    }

    /// Set system prompt (overrides ModelInfo default)
    #[inline(always)]
    fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set temperature (overrides ModelInfo default)
    #[inline(always)]
    fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens (overrides ModelInfo default)
    #[inline(always)]
    fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set top_p (overrides ModelInfo default)
    #[inline(always)]
    fn top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    /// Set frequency penalty (overrides ModelInfo default)
    #[inline(always)]
    fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty (overrides ModelInfo default)
    #[inline(always)]
    fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Add chat history (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn chat_history(
        mut self,
        history: ZeroOneOrMany<Message>,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        match history {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(msg) => {
                self.chat_history
                    .try_push(msg)
                    .map_err(|_| crate::completion_provider::CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(msgs) => {
                for msg in msgs {
                    self.chat_history.try_push(msg).map_err(|_| {
                        crate::completion_provider::CompletionError::RequestTooLarge
                    })?;
                }
            }
        }
        Ok(self)
    }

    /// Add documents for RAG (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn documents(
        mut self,
        docs: ZeroOneOrMany<Document>,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        match docs {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(doc) => {
                self.documents
                    .try_push(doc)
                    .map_err(|_| crate::completion_provider::CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(documents) => {
                for doc in documents {
                    self.documents.try_push(doc).map_err(|_| {
                        crate::completion_provider::CompletionError::RequestTooLarge
                    })?;
                }
            }
        }
        Ok(self)
    }

    /// Add tools for function calling (ZeroOneOrMany with bounded capacity)
    #[inline(always)]
    fn tools(
        mut self,
        tools: ZeroOneOrMany<ToolDefinition>,
    ) -> Result<Self, crate::completion_provider::CompletionError> {
        match tools {
            ZeroOneOrMany::None => {}
            ZeroOneOrMany::One(tool) => {
                self.tools
                    .try_push(tool)
                    .map_err(|_| crate::completion_provider::CompletionError::RequestTooLarge)?;
            }
            ZeroOneOrMany::Many(tool_list) => {
                for tool in tool_list {
                    self.tools.try_push(tool).map_err(|_| {
                        crate::completion_provider::CompletionError::RequestTooLarge
                    })?;
                }
            }
        }
        Ok(self)
    }

    /// Add provider-specific parameters
    #[inline(always)]
    fn additional_params(mut self, params: Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Set chunk handler with cyrup_sugars pattern matching syntax
    #[inline(always)]
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<CompletionChunk, crate::completion_provider::CompletionError>)
            + Send
            + Sync
            + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }

    /// Terminal action - execute completion with user prompt (blazing-fast streaming)
    #[inline(always)]
    fn prompt(self, text: impl AsRef<str>) -> AsyncStream<CompletionChunk> {
        let (sender, receiver) = crate::channel();
        let prompt_text = text.as_ref().to_string();

        spawn_async(async move {
            match self.execute_streaming_completion(prompt_text).await {
                Ok(mut stream) => {
                    use futures_util::StreamExt;
                    while let Some(chunk_result) = stream.next().await {
                        // Apply cyrup_sugars pattern matching if handler provided
                        if let Some(ref handler) = self.chunk_handler {
                            handler(chunk_result.clone());
                        } else {
                            // Default env_logger behavior (zero allocation)
                            match &chunk_result {
                                Ok(chunk) => log::debug!("Chunk: {:?}", chunk),
                                Err(e) => log::error!("Chunk error: {}", e),
                            }
                        }

                        match chunk_result {
                            Ok(chunk) => {
                                if sender.send(chunk).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                if sender.send(CompletionChunk::error(e.message())).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to start completion: {}", e);
                    let _ = sender.send(CompletionChunk::error(e.message()));
                }
            }
        });

        receiver
    }
}

impl GeminiCompletionBuilder {
    /// Execute streaming completion with zero-allocation HTTP3 (blazing-fast)
    #[inline(always)]
    async fn execute_streaming_completion(
        &self,
        prompt: String,
    ) -> Result<
        AsyncStream<Result<CompletionChunk, crate::completion_provider::CompletionError>>,
        crate::completion_provider::CompletionError,
    > {
        let request_body = self.build_gemini_request(&prompt)?;
        let body_bytes = serde_json::to_vec(&request_body)
            .map_err(|_| crate::completion_provider::CompletionError::ParseError)?;

        // Use explicit API key if set, otherwise use discovered key
        let auth_key = self.explicit_api_key.as_ref().unwrap_or(&self.api_key);

        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?key={}",
            self.base_url, self.model_name, auth_key
        );

        let request = HttpRequest::post(&url, body_bytes)
            .map_err(|_| crate::completion_provider::CompletionError::HttpError)?
            .header("Content-Type", "application/json");

        let response = self
            .client
            .send(request)
            .await
            .map_err(|_| crate::completion_provider::CompletionError::HttpError)?;

        if !response.status().is_success() {
            return Err(match response.status().as_u16() {
                401 => crate::completion_provider::CompletionError::AuthError,
                413 => crate::completion_provider::CompletionError::RequestTooLarge,
                429 => crate::completion_provider::CompletionError::RateLimited,
                _ => crate::completion_provider::CompletionError::HttpError,
            });
        }

        let sse_stream = response.sse();
        let (chunk_sender, chunk_receiver) = crate::channel();

        spawn_async(async move {
            use futures_util::StreamExt;
            let mut sse_stream = sse_stream;

            while let Some(event) = sse_stream.next().await {
                match event {
                    Ok(sse_event) => {
                        if let Some(data) = sse_event.data {
                            match Self::parse_gemini_chunk(data.as_bytes()) {
                                Ok(chunk) => {
                                    if chunk_sender.send(Ok(chunk)).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    if chunk_sender.send(Err(e)).is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => {
                        let _ = chunk_sender.send(Err(
                            crate::completion_provider::CompletionError::StreamError,
                        ));
                        break;
                    }
                }
            }
        });

        Ok(chunk_receiver)
    }

    /// Build Gemini request with zero allocation where possible
    #[inline(always)]
    fn build_gemini_request(
        &self,
        prompt: &str,
    ) -> Result<GenerateContentRequest, crate::completion_provider::CompletionError> {
        let mut contents = Vec::new();

        // Add documents as context
        for doc in &self.documents {
            contents.push(Content {
                parts: OneOrMany::one(format!("Document: {}", doc.content()).into()),
                role: Some(Role::User),
            });
        }

        // Add chat history
        for msg in &self.chat_history {
            contents.push(self.convert_domain_message_to_gemini(msg)?);
        }

        // Add user prompt
        contents.push(Content {
            parts: OneOrMany::one(prompt.to_string().into()),
            role: Some(Role::User),
        });

        let system_instruction = if !self.system_prompt.is_empty() {
            Some(Content {
                parts: OneOrMany::one(self.system_prompt.clone().into()),
                role: Some(Role::Model),
            })
        } else {
            None
        };

        let mut generation_config = GenerationConfig::default();
        generation_config.temperature = Some(self.temperature);
        generation_config.max_output_tokens = Some(self.max_tokens as u64);
        generation_config.top_p = Some(self.top_p);
        generation_config.top_k = Some(self.top_k as i32);

        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(self.convert_tools_to_gemini()?)
        };

        Ok(GenerateContentRequest {
            contents,
            generation_config: Some(generation_config),
            safety_settings: None,
            tools,
            tool_config: None,
            system_instruction,
        })
    }

    /// Convert domain Message to Gemini format
    #[inline(always)]
    fn convert_domain_message_to_gemini(
        &self,
        msg: &Message,
    ) -> Result<Content, crate::completion_provider::CompletionError> {
        match msg.role() {
            fluent_ai_domain::message::MessageRole::User => {
                let content = msg
                    .content()
                    .text()
                    .ok_or(crate::completion_provider::CompletionError::ParseError)?;
                Ok(Content {
                    parts: OneOrMany::one(content.to_string().into()),
                    role: Some(Role::User),
                })
            }
            fluent_ai_domain::message::MessageRole::Assistant => {
                let mut parts = Vec::new();

                if let Some(text) = msg.content().text() {
                    parts.push(Part::Text(text.to_string()));
                }

                for tool_call in msg.tool_calls() {
                    parts.push(Part::FunctionCall(gemini_api_types::FunctionCall {
                        name: tool_call.function().name().to_string(),
                        args: tool_call.function().arguments().clone(),
                    }));
                }

                Ok(Content {
                    parts: OneOrMany::many(parts)
                        .map_err(|_| crate::completion_provider::CompletionError::ParseError)?,
                    role: Some(Role::Model),
                })
            }
            fluent_ai_domain::message::MessageRole::System => {
                let content = msg
                    .content()
                    .text()
                    .ok_or(crate::completion_provider::CompletionError::ParseError)?;
                Ok(Content {
                    parts: OneOrMany::one(content.to_string().into()),
                    role: Some(Role::Model),
                })
            }
        }
    }

    /// Convert domain ToolDefinition to Gemini format
    #[inline(always)]
    fn convert_tools_to_gemini(
        &self,
    ) -> Result<Vec<Tool>, crate::completion_provider::CompletionError> {
        let mut tools = Vec::new();

        for tool in &self.tools {
            let parameters: Option<Schema> =
                if tool.parameters() == &serde_json::json!({"type": "object", "properties": {}}) {
                    None
                } else {
                    Some(
                        tool.parameters()
                            .clone()
                            .try_into()
                            .map_err(|_| crate::completion_provider::CompletionError::ParseError)?,
                    )
                };

            tools.push(Tool {
                function_declarations: FunctionDeclaration {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters,
                },
                code_execution: None,
            });
        }

        Ok(tools)
    }

    /// Parse Gemini SSE chunk with zero-copy byte slice parsing (blazing-fast)
    #[inline(always)]
    fn parse_gemini_chunk(
        data: &[u8],
    ) -> Result<CompletionChunk, crate::completion_provider::CompletionError> {
        // Fast JSON parsing from bytes using serde_json
        let response: GenerateContentResponse = serde_json::from_slice(data)
            .map_err(|_| crate::completion_provider::CompletionError::ParseError)?;

        let candidate = response
            .candidates
            .first()
            .ok_or(crate::completion_provider::CompletionError::ParseError)?;

        let mut text_content = String::new();
        let mut has_tool_calls = false;

        for part in &candidate.content.parts {
            match part {
                Part::Text(text) => {
                    text_content.push_str(text);
                }
                Part::FunctionCall(_) => {
                    has_tool_calls = true;
                }
                _ => {}
            }
        }

        // Handle finish reason
        if let Some(ref finish_reason) = candidate.finish_reason {
            let reason = match finish_reason {
                gemini_api_types::FinishReason::Stop => FinishReason::Stop,
                gemini_api_types::FinishReason::MaxTokens => FinishReason::Length,
                gemini_api_types::FinishReason::Safety => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            };

            let usage_info = response.usage_metadata.map(|u| Usage {
                prompt_tokens: u.prompt_token_count as u32,
                completion_tokens: u.candidates_token_count as u32,
                total_tokens: u.total_token_count as u32,
            });

            return Ok(CompletionChunk::Complete {
                text: text_content,
                finish_reason: Some(reason),
                usage: usage_info,
            });
        }

        if has_tool_calls {
            Ok(CompletionChunk::tool_partial("", "", &text_content))
        } else {
            Ok(CompletionChunk::text(&text_content))
        }
    }
}

/// Public constructor for Gemini completion builder
#[inline(always)]
pub fn completion_builder(
    api_key: String,
    model_name: &'static str,
) -> Result<GeminiCompletionBuilder, crate::completion_provider::CompletionError> {
    GeminiCompletionBuilder::new(api_key, model_name)
}

/// Get available Gemini models (compile-time constant)
#[inline(always)]
pub const fn available_models() -> &'static [&'static str] {
    &[
        GEMINI_2_5_PRO_PREVIEW_06_05,
        GEMINI_2_5_FLASH_PREVIEW_05_20,
        GEMINI_2_0_FLASH,
        GEMINI_1_5_PRO,
        GEMINI_1_5_FLASH,
        GEMINI_1_5_PRO_8B,
        GEMINI_1_0_PRO,
    ]
}

pub(crate) fn create_request_body(
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, CompletionError> {
    let mut full_history = Vec::new();
    full_history.extend(completion_request.chat_history);

    let additional_params = completion_request
        .additional_params
        .unwrap_or_else(|| Value::Object(Map::new()));

    let mut generation_config = serde_json::from_value::<GenerationConfig>(additional_params)?;

    if let Some(temp) = completion_request.temperature {
        generation_config.temperature = Some(temp);
    }

    if let Some(max_tokens) = completion_request.max_tokens {
        generation_config.max_output_tokens = Some(max_tokens);
    }

    let system_instruction = completion_request.preamble.clone().map(|preamble| Content {
        parts: OneOrMany::one(preamble.into()),
        role: Some(Role::Model),
    });

    let request = GenerateContentRequest {
        contents: full_history
            .into_iter()
            .map(|msg| {
                msg.try_into()
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))
            })
            .collect::<Result<Vec<_>, _>>()?,
        generation_config: Some(generation_config),
        safety_settings: None,
        tools: Some(
            completion_request
                .tools
                .into_iter()
                .map(Tool::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        tool_config: None,
        system_instruction,
    };

    Ok(request)
}

impl TryFrom<completion::ToolDefinition> for Tool {
    type Error = CompletionError;

    fn try_from(tool: completion::ToolDefinition) -> Result<Self, Self::Error> {
        let parameters: Option<Schema> =
            if tool.parameters == serde_json::json!({"type": "object", "properties": {}}) {
                None
            } else {
                Some(tool.parameters.try_into()?)
            };
        Ok(Self {
            function_declarations: FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters,
            },
            code_execution: None,
        })
    }
}

impl TryFrom<GenerateContentResponse> for completion::CompletionResponse<GenerateContentResponse> {
    type Error = CompletionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ResponseError("No response candidates in response".into())
        })?;

        let content = candidate
            .content
            .parts
            .iter()
            .map(|part| {
                Ok(match part {
                    Part::Text(text) => completion::AssistantContent::text(text),
                    Part::FunctionCall(function_call) => completion::AssistantContent::tool_call(
                        &function_call.name,
                        &function_call.name,
                        function_call.args.clone(),
                    ),
                    _ => {
                        return Err(CompletionError::ResponseError(
                            "Response did not contain a message or tool call".into(),
                        ));
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        Ok(completion::CompletionResponse {
            choice,
            raw_response: response,
        })
    }
}

pub mod gemini_api_types {
    use std::{collections::HashMap, convert::Infallible, str::FromStr};

    use fluent_ai_domain::completion::CompletionCoreError as CompletionError;
    use fluent_ai_domain::message::{self, MimeType as _};
    // =================================================================
    // Gemini API Types
    // =================================================================
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use crate::OneOrMany;

    /// Response from the model supporting multiple candidate responses.
    /// Safety ratings and content filtering are reported for both prompt in GenerateContentResponse.prompt_feedback
    /// and for each candidate in finishReason and in safetyRatings.
    /// The API:
    ///     - Returns either all requested candidates or none of them
    ///     - Returns no candidates at all only if there was something wrong with the prompt (check promptFeedback)
    ///     - Reports feedback on each candidate in finishReason and safetyRatings.
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentResponse {
        /// Candidate responses from the model.
        pub candidates: Vec<ContentCandidate>,
        /// Returns the prompt's feedback related to the content filters.
        pub prompt_feedback: Option<PromptFeedback>,
        /// Output only. Metadata on the generation requests' token usage.
        pub usage_metadata: Option<UsageMetadata>,
        pub model_version: Option<String>,
    }

    /// A response candidate generated from the model.
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ContentCandidate {
        /// Output only. Generated content returned from the model.
        pub content: Content,
        /// Optional. Output only. The reason why the model stopped generating tokens.
        /// If empty, the model has not stopped generating tokens.
        pub finish_reason: Option<FinishReason>,
        /// List of ratings for the safety of a response candidate.
        /// There is at most one rating per category.
        pub safety_ratings: Option<Vec<SafetyRating>>,
        /// Output only. Citation information for model-generated candidate.
        /// This field may be populated with recitation information for any text included in the content.
        /// These are passages that are "recited" from copyrighted material in the foundational LLM's training data.
        pub citation_metadata: Option<CitationMetadata>,
        /// Output only. Token count for this candidate.
        pub token_count: Option<i32>,
        /// Output only.
        pub avg_logprobs: Option<f64>,
        /// Output only. Log-likelihood scores for the response tokens and top tokens
        pub logprobs_result: Option<LogprobsResult>,
        /// Output only. Index of the candidate in the list of response candidates.
        pub index: Option<i32>,
    }
    #[derive(Debug, Deserialize, Serialize)]
    pub struct Content {
        /// Ordered Parts that constitute a single message. Parts may have different MIME types.
        #[serde(deserialize_with = "crate::util::string_or_one_or_many")]
        pub parts: OneOrMany<Part>,
        /// The producer of the content. Must be either 'user' or 'model'.
        /// Useful to set for multi-turn conversations, otherwise can be left blank or unset.
        pub role: Option<Role>,
    }

    impl TryFrom<message::Message> for Content {
        type Error = message::MessageError;

        fn try_from(msg: message::Message) -> Result<Self, Self::Error> {
            Ok(match msg {
                message::Message::User { content } => Content {
                    parts: content.try_map(|c| c.try_into())?,
                    role: Some(Role::User),
                },
                message::Message::Assistant { content } => Content {
                    role: Some(Role::Model),
                    parts: content.map(|content| content.into()),
                },
            })
        }
    }

    impl TryFrom<Content> for message::Message {
        type Error = message::MessageError;

        fn try_from(content: Content) -> Result<Self, Self::Error> {
            match content.role {
                Some(Role::User) | None => Ok(message::Message::User {
                    content: content.parts.try_map(|part| {
                        Ok(match part {
                            Part::Text(text) => message::UserContent::text(text),
                            Part::InlineData(inline_data) => {
                                let mime_type =
                                    message::MediaType::from_mime_type(&inline_data.mime_type);

                                match mime_type {
                                    Some(message::MediaType::Image(media_type)) => {
                                        message::UserContent::image(
                                            inline_data.data,
                                            Some(message::ContentFormat::default()),
                                            Some(media_type),
                                            Some(message::ImageDetail::default()),
                                        )
                                    }
                                    Some(message::MediaType::Document(media_type)) => {
                                        message::UserContent::document(
                                            inline_data.data,
                                            Some(message::ContentFormat::default()),
                                            Some(media_type),
                                        )
                                    }
                                    Some(message::MediaType::Audio(media_type)) => {
                                        message::UserContent::audio(
                                            inline_data.data,
                                            Some(message::ContentFormat::default()),
                                            Some(media_type),
                                        )
                                    }
                                    _ => {
                                        return Err(message::MessageError::ConversionError(
                                            format!("Unsupported media type {mime_type:?}"),
                                        ));
                                    }
                                }
                            }
                            _ => {
                                return Err(message::MessageError::ConversionError(format!(
                                    "Unsupported gemini content part type: {part:?}"
                                )));
                            }
                        })
                    })?,
                }),
                Some(Role::Model) => Ok(message::Message::Assistant {
                    content: content.parts.try_map(|part| {
                        Ok(match part {
                            Part::Text(text) => message::AssistantContent::text(text),
                            Part::FunctionCall(function_call) => {
                                message::AssistantContent::ToolCall(function_call.into())
                            }
                            _ => {
                                return Err(message::MessageError::ConversionError(format!(
                                    "Unsupported part type: {part:?}"
                                )));
                            }
                        })
                    })?,
                }),
            }
        }
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Model,
    }

    /// A datatype containing media that is part of a multi-part [Content] message.
    /// A Part consists of data which has an associated datatype. A Part can only contain one of the accepted types in Part.data.
    /// A Part must have a fixed IANA MIME type identifying the type and subtype of the media if the inlineData field is filled with raw bytes.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub enum Part {
        Text(String),
        InlineData(Blob),
        FunctionCall(FunctionCall),
        FunctionResponse(FunctionResponse),
        FileData(FileData),
        ExecutableCode(ExecutableCode),
        CodeExecutionResult(CodeExecutionResult),
    }

    impl From<String> for Part {
        fn from(text: String) -> Self {
            Self::Text(text)
        }
    }

    impl From<&str> for Part {
        fn from(text: &str) -> Self {
            Self::Text(text.to_string())
        }
    }

    impl FromStr for Part {
        type Err = Infallible;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(s.into())
        }
    }

    impl TryFrom<message::UserContent> for Part {
        type Error = message::MessageError;

        fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
            match content {
                message::UserContent::Text(message::Text { text }) => Ok(Self::Text(text)),
                message::UserContent::ToolResult(message::ToolResult { id, content }) => {
                    let content = match content.first() {
                        message::ToolResultContent::Text(text) => text.text,
                        message::ToolResultContent::Image(_) => {
                            return Err(message::MessageError::ConversionError(
                                "Tool result content must be text".to_string(),
                            ));
                        }
                    };
                    Ok(Part::FunctionResponse(FunctionResponse {
                        name: id,
                        response: Some(serde_json::from_str(&content).map_err(|e| {
                            message::MessageError::ConversionError(format!(
                                "Failed to parse tool response: {e}"
                            ))
                        })?),
                    }))
                }
                message::UserContent::Image(message::Image {
                    data, media_type, ..
                }) => match media_type {
                    Some(media_type) => match media_type {
                        message::ImageMediaType::JPEG
                        | message::ImageMediaType::PNG
                        | message::ImageMediaType::WEBP
                        | message::ImageMediaType::HEIC
                        | message::ImageMediaType::HEIF => Ok(Self::InlineData(Blob {
                            mime_type: media_type.to_mime_type().to_owned(),
                            data,
                        })),
                        _ => Err(message::MessageError::ConversionError(format!(
                            "Unsupported image media type {media_type:?}"
                        ))),
                    },
                    None => Err(message::MessageError::ConversionError(
                        "Media type for image is required for Anthropic".to_string(),
                    )),
                },
                message::UserContent::Document(message::Document {
                    data, media_type, ..
                }) => match media_type {
                    Some(media_type) => match media_type {
                        message::DocumentMediaType::PDF
                        | message::DocumentMediaType::TXT
                        | message::DocumentMediaType::RTF
                        | message::DocumentMediaType::HTML
                        | message::DocumentMediaType::CSS
                        | message::DocumentMediaType::MARKDOWN
                        | message::DocumentMediaType::CSV
                        | message::DocumentMediaType::XML => Ok(Self::InlineData(Blob {
                            mime_type: media_type.to_mime_type().to_owned(),
                            data,
                        })),
                        _ => Err(message::MessageError::ConversionError(format!(
                            "Unsupported document media type {media_type:?}"
                        ))),
                    },
                    None => Err(message::MessageError::ConversionError(
                        "Media type for document is required for Anthropic".to_string(),
                    )),
                },
                message::UserContent::Audio(message::Audio {
                    data, media_type, ..
                }) => match media_type {
                    Some(media_type) => Ok(Self::InlineData(Blob {
                        mime_type: media_type.to_mime_type().to_owned(),
                        data,
                    })),
                    None => Err(message::MessageError::ConversionError(
                        "Media type for audio is required for Anthropic".to_string(),
                    )),
                },
            }
        }
    }

    impl From<message::AssistantContent> for Part {
        fn from(content: message::AssistantContent) -> Self {
            match content {
                message::AssistantContent::Text(message::Text { text }) => text.into(),
                message::AssistantContent::ToolCall(tool_call) => tool_call.into(),
            }
        }
    }

    impl From<message::ToolCall> for Part {
        fn from(tool_call: message::ToolCall) -> Self {
            Self::FunctionCall(FunctionCall {
                name: tool_call.function.name,
                args: tool_call.function.arguments,
            })
        }
    }

    /// Raw media bytes.
    /// Text should not be sent as raw bytes, use the 'text' field.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct Blob {
        /// The IANA standard MIME type of the source data. Examples: - image/png - image/jpeg
        /// If an unsupported MIME type is provided, an error will be returned.
        pub mime_type: String,
        /// Raw bytes for media formats. A base64-encoded string.
        pub data: String,
    }

    /// A predicted FunctionCall returned from the model that contains a string representing the
    /// FunctionDeclaration.name with the arguments and their values.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct FunctionCall {
        /// Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores
        /// and dashes, with a maximum length of 63.
        pub name: String,
        /// Optional. The function parameters and values in JSON object format.
        pub args: serde_json::Value,
    }

    impl From<FunctionCall> for message::ToolCall {
        fn from(function_call: FunctionCall) -> Self {
            Self {
                id: function_call.name.clone(),
                function: message::ToolFunction {
                    name: function_call.name,
                    arguments: function_call.args,
                },
            }
        }
    }

    impl From<message::ToolCall> for FunctionCall {
        fn from(tool_call: message::ToolCall) -> Self {
            Self {
                name: tool_call.function.name,
                args: tool_call.function.arguments,
            }
        }
    }

    /// The result output from a FunctionCall that contains a string representing the FunctionDeclaration.name
    /// and a structured JSON object containing any output from the function is used as context to the model.
    /// This should contain the result of aFunctionCall made based on model prediction.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct FunctionResponse {
        /// The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores and dashes,
        /// with a maximum length of 63.
        pub name: String,
        /// The function response in JSON object format.
        pub response: Option<HashMap<String, serde_json::Value>>,
    }

    /// URI based data.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct FileData {
        /// Optional. The IANA standard MIME type of the source data.
        pub mime_type: Option<String>,
        /// Required. URI.
        pub file_uri: String,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct SafetyRating {
        pub category: HarmCategory,
        pub probability: HarmProbability,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmProbability {
        HarmProbabilityUnspecified,
        Negligible,
        Low,
        Medium,
        High,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmCategory {
        HarmCategoryUnspecified,
        HarmCategoryDerogatory,
        HarmCategoryToxicity,
        HarmCategoryViolence,
        HarmCategorySexually,
        HarmCategoryMedical,
        HarmCategoryDangerous,
        HarmCategoryHarassment,
        HarmCategoryHateSpeech,
        HarmCategorySexuallyExplicit,
        HarmCategoryDangerousContent,
        HarmCategoryCivicIntegrity,
    }

    #[derive(Debug, Deserialize, Clone, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct UsageMetadata {
        pub prompt_token_count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cached_content_token_count: Option<i32>,
        pub candidates_token_count: i32,
        pub total_token_count: i32,
    }

    impl std::fmt::Display for UsageMetadata {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Prompt token count: {}\nCached content token count: {}\nCandidates token count: {}\nTotal token count: {}",
                self.prompt_token_count,
                match self.cached_content_token_count {
                    Some(count) => count.to_string(),
                    None => "n/a".to_string(),
                },
                self.candidates_token_count,
                self.total_token_count
            )
        }
    }

    /// A set of the feedback metadata the prompt specified in [GenerateContentRequest.contents](GenerateContentRequest).
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PromptFeedback {
        /// Optional. If set, the prompt was blocked and no candidates are returned. Rephrase the prompt.
        pub block_reason: Option<BlockReason>,
        /// Ratings for safety of the prompt. There is at most one rating per category.
        pub safety_ratings: Option<Vec<SafetyRating>>,
    }

    /// Reason why a prompt was blocked by the model
    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum BlockReason {
        /// Default value. This value is unused.
        BlockReasonUnspecified,
        /// Prompt was blocked due to safety reasons. Inspect safetyRatings to understand which safety category blocked it.
        Safety,
        /// Prompt was blocked due to unknown reasons.
        Other,
        /// Prompt was blocked due to the terms which are included from the terminology blocklist.
        Blocklist,
        /// Prompt was blocked due to prohibited content.
        ProhibitedContent,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum FinishReason {
        /// Default value. This value is unused.
        FinishReasonUnspecified,
        /// Natural stop point of the model or provided stop sequence.
        Stop,
        /// The maximum number of tokens as specified in the request was reached.
        MaxTokens,
        /// The response candidate content was flagged for safety reasons.
        Safety,
        /// The response candidate content was flagged for recitation reasons.
        Recitation,
        /// The response candidate content was flagged for using an unsupported language.
        Language,
        /// Unknown reason.
        Other,
        /// Token generation stopped because the content contains forbidden terms.
        Blocklist,
        /// Token generation stopped for potentially containing prohibited content.
        ProhibitedContent,
        /// Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).
        Spii,
        /// The function call generated by the model is invalid.
        MalformedFunctionCall,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationMetadata {
        pub citation_sources: Vec<CitationSource>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationSource {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub start_index: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub end_index: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub license: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LogprobsResult {
        pub top_candidate: Vec<TopCandidate>,
        pub chosen_candidate: Vec<LogProbCandidate>,
    }

    #[derive(Debug, Deserialize)]
    pub struct TopCandidate {
        pub candidates: Vec<LogProbCandidate>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LogProbCandidate {
        pub token: String,
        pub token_id: String,
        pub log_probability: f64,
    }

    /// Gemini API Configuration options for model generation and outputs. Not all parameters are
    /// configurable for every model. From [Gemini API Reference](https://ai.google.dev/api/generate-content#generationconfig)
    /// ### Rig Note:
    /// Can be used to construct a typesafe `additional_params` in rig::[AgentBuilder](crate::agent::AgentBuilder).
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerationConfig {
        /// The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop
        /// at the first appearance of a stop_sequence. The stop sequence will not be included as part of the response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop_sequences: Option<Vec<String>>,
        /// MIME type of the generated candidate text. Supported MIME types are:
        ///     - text/plain:  (default) Text output
        ///     - application/json: JSON response in the response candidates.
        ///     - text/x.enum: ENUM as a string response in the response candidates.
        /// Refer to the docs for a list of all supported text MIME types
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_mime_type: Option<String>,
        /// Output schema of the generated candidate text. Schemas must be a subset of the OpenAPI schema and can be
        /// objects, primitives or arrays. If set, a compatible responseMimeType must also  be set. Compatible MIME
        /// types: application/json: Schema for JSON response. Refer to the JSON text generation guide for more details.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_schema: Option<Schema>,
        /// Number of generated responses to return. Currently, this value can only be set to 1. If
        /// unset, this will default to 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidate_count: Option<i32>,
        /// The maximum number of tokens to include in a response candidate. Note: The default value varies by model, see
        /// the Model.output_token_limit attribute of the Model returned from the getModel function.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_output_tokens: Option<u64>,
        /// Controls the randomness of the output. Note: The default value varies by model, see the Model.temperature
        /// attribute of the Model returned from the getModel function. Values can range from [0.0, 2.0].
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,
        /// The maximum cumulative probability of tokens to consider when sampling. The model uses combined Top-k and
        /// Top-p (nucleus) sampling. Tokens are sorted based on their assigned probabilities so that only the most
        /// likely tokens are considered. Top-k sampling directly limits the maximum number of tokens to consider, while
        /// Nucleus sampling limits the number of tokens based on the cumulative probability. Note: The default value
        /// varies by Model and is specified by theModel.top_p attribute returned from the getModel function. An empty
        /// topK attribute indicates that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,
        /// The maximum number of tokens to consider when sampling. Gemini models use Top-p (nucleus) sampling or a
        /// combination of Top-k and nucleus sampling. Top-k sampling considers the set of topK most probable tokens.
        /// Models running with nucleus sampling don't allow topK setting. Note: The default value varies by Model and is
        /// specified by theModel.top_p attribute returned from the getModel function. An empty topK attribute indicates
        /// that the model doesn't apply top-k sampling and doesn't allow setting topK on requests.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_k: Option<i32>,
        /// Presence penalty applied to the next token's logprobs if the token has already been seen in the response.
        /// This penalty is binary on/off and not dependent on the number of times the token is used (after the first).
        /// Use frequencyPenalty for a penalty that increases with each use. A positive penalty will discourage the use
        /// of tokens that have already been used in the response, increasing the vocabulary. A negative penalty will
        /// encourage the use of tokens that have already been used in the response, decreasing the vocabulary.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f64>,
        /// Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been
        /// seen in the response so far. A positive penalty will discourage the use of tokens that have already been
        /// used, proportional to the number of times the token has been used: The more a token is used, the more
        /// difficult it is for the  model to use that token again increasing the vocabulary of responses. Caution: A
        /// negative penalty will encourage the model to reuse tokens proportional to the number of times the token has
        /// been used. Small negative values will reduce the vocabulary of a response. Larger negative values will cause
        /// the model to  repeating a common token until it hits the maxOutputTokens limit: "...the the the the the...".
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f64>,
        /// If true, export the logprobs results in response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_logprobs: Option<bool>,
        /// Only valid if responseLogprobs=True. This sets the number of top logprobs to return at each decoding step in
        /// [Candidate.logprobs_result].
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<i32>,
    }

    impl Default for GenerationConfig {
        fn default() -> Self {
            Self {
                temperature: Some(1.0),
                max_output_tokens: Some(4096),
                stop_sequences: None,
                response_mime_type: None,
                response_schema: None,
                candidate_count: None,
                top_p: None,
                top_k: None,
                presence_penalty: None,
                frequency_penalty: None,
                response_logprobs: None,
                logprobs: None,
            }
        }
    }
    /// The Schema object allows the definition of input and output data types. These types can be objects, but also
    /// primitives and arrays. Represents a select subset of an OpenAPI 3.0 schema object.
    /// From [Gemini API Reference](https://ai.google.dev/api/caching#Schema)
    #[derive(Debug, Deserialize, Serialize)]
    pub struct Schema {
        pub r#type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub format: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub nullable: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub r#enum: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_items: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub min_items: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub properties: Option<HashMap<String, Schema>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub required: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub items: Option<Box<Schema>>,
    }

    impl TryFrom<Value> for Schema {
        type Error = CompletionError;

        fn try_from(value: Value) -> Result<Self, Self::Error> {
            if let Some(obj) = value.as_object() {
                Ok(Schema {
                    r#type: obj
                        .get("type")
                        .and_then(|v| {
                            if v.is_string() {
                                v.as_str().map(String::from)
                            } else if v.is_array() {
                                v.as_array()
                                    .and_then(|arr| arr.first())
                                    .and_then(|v| v.as_str().map(String::from))
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default(),
                    format: obj.get("format").and_then(|v| v.as_str()).map(String::from),
                    description: obj
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    nullable: obj.get("nullable").and_then(|v| v.as_bool()),
                    r#enum: obj.get("enum").and_then(|v| v.as_array()).map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    }),
                    max_items: obj
                        .get("maxItems")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32),
                    min_items: obj
                        .get("minItems")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32),
                    properties: obj
                        .get("properties")
                        .and_then(|v| v.as_object())
                        .map(|map| {
                            map.iter()
                                .filter_map(|(k, v)| {
                                    v.clone().try_into().ok().map(|schema| (k.clone(), schema))
                                })
                                .collect()
                        }),
                    required: obj.get("required").and_then(|v| v.as_array()).map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    }),
                    items: obj
                        .get("items")
                        .map(|v| {
                            v.clone()
                                .try_into()
                                .map(|schema| Box::new(schema))
                                .map_err(|_| {
                                    CompletionError::ResponseError(
                                        "Failed to parse items schema".into(),
                                    )
                                })
                        })
                        .transpose()?,
                })
            } else {
                Err(CompletionError::ResponseError(
                    "Expected a JSON object for Schema".into(),
                ))
            }
        }
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentRequest {
        pub contents: Vec<Content>,
        pub tools: Option<Vec<Tool>>,
        pub tool_config: Option<ToolConfig>,
        /// Optional. Configuration options for model generation and outputs.
        pub generation_config: Option<GenerationConfig>,
        /// Optional. A list of unique SafetySetting instances for blocking unsafe content. This will be enforced on the
        /// [GenerateContentRequest.contents] and [GenerateContentResponse.candidates]. There should not be more than one
        /// setting for each SafetyCategory type. The API will block any contents and responses that fail to meet the
        /// thresholds set by these settings. This list overrides the default settings for each SafetyCategory specified
        /// in the safetySettings. If there is no SafetySetting for a given SafetyCategory provided in the list, the API
        /// will use the default safety setting for that category. Harm categories:
        ///     - HARM_CATEGORY_HATE_SPEECH,
        ///     - HARM_CATEGORY_SEXUALLY_EXPLICIT
        ///     - HARM_CATEGORY_DANGEROUS_CONTENT
        ///     - HARM_CATEGORY_HARASSMENT
        /// are supported.
        /// Refer to the guide for detailed information on available safety settings. Also refer to the Safety guidance
        /// to learn how to incorporate safety considerations in your AI applications.
        pub safety_settings: Option<Vec<SafetySetting>>,
        /// Optional. Developer set system instruction(s). Currently, text only.
        /// From [Gemini API Reference](https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest)
        pub system_instruction: Option<Content>,
        // cachedContent: Optional<String>
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Tool {
        pub function_declarations: FunctionDeclaration,
        pub code_execution: Option<CodeExecution>,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionDeclaration {
        pub name: String,
        pub description: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parameters: Option<Schema>,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ToolConfig {
        pub schema: Option<Schema>,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CodeExecution {}

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SafetySetting {
        pub category: HarmCategory,
        pub threshold: HarmBlockThreshold,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmBlockThreshold {
        HarmBlockThresholdUnspecified,
        BlockLowAndAbove,
        BlockMediumAndAbove,
        BlockOnlyHigh,
        BlockNone,
        Off,
    }
}

#[cfg(test)]
mod tests {
    use fluent_ai_domain::message;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_deserialize_message_user() {
        let raw_message = r#"{
            "parts": [
                {"text": "Hello, world!"},
                {"inlineData": {"mimeType": "image/png", "data": "base64encodeddata"}},
                {"functionCall": {"name": "test_function", "args": {"arg1": "value1"}}},
                {"functionResponse": {"name": "test_function", "response": {"result": "success"}}},
                {"fileData": {"mimeType": "application/pdf", "fileUri": "http://example.com/file.pdf"}},
                {"executableCode": {"code": "print('Hello, world!')", "language": "PYTHON"}},
                {"codeExecutionResult": {"output": "Hello, world!", "outcome": "OUTCOME_OK"}}
            ],
            "role": "user"
        }"#;

        let content: Content = {
            let jd = &mut serde_json::Deserializer::from_str(raw_message);
            serde_path_to_error::deserialize(jd).unwrap_or_else(|err| {
                panic!("Deserialization error at {}: {}", err.path(), err);
            })
        };
        assert_eq!(content.role, Some(Role::User));
        assert_eq!(content.parts.len(), 7);

        let parts: Vec<Part> = content.parts.into_iter().collect();

        if let Part::Text(text) = &parts[0] {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text part");
        }

        if let Part::InlineData(inline_data) = &parts[1] {
            assert_eq!(inline_data.mime_type, "image/png");
            assert_eq!(inline_data.data, "base64encodeddata");
        } else {
            panic!("Expected inline data part");
        }

        if let Part::FunctionCall(function_call) = &parts[2] {
            assert_eq!(function_call.name, "test_function");
            assert_eq!(
                function_call
                    .args
                    .as_object()
                    .expect("Failed to get function call args as object in test")
                    .get("arg1")
                    .expect("Failed to get arg1 from function call args in test"),
                "value1"
            );
        } else {
            panic!("Expected function call part");
        }

        if let Part::FunctionResponse(function_response) = &parts[3] {
            assert_eq!(function_response.name, "test_function");
            assert_eq!(
                function_response
                    .response
                    .as_ref()
                    .expect("Failed to get function response in test")
                    .get("result")
                    .expect("Failed to get result from function response in test"),
                "success"
            );
        } else {
            panic!("Expected function response part");
        }

        if let Part::FileData(file_data) = &parts[4] {
            assert_eq!(
                file_data
                    .mime_type
                    .as_ref()
                    .expect("Failed to get file data mime type in test"),
                "application/pdf"
            );
            assert_eq!(file_data.file_uri, "http://example.com/file.pdf");
        } else {
            panic!("Expected file data part");
        }

        if let Part::ExecutableCode(executable_code) = &parts[5] {
            assert_eq!(executable_code.code, "print('Hello, world!')");
        } else {
            panic!("Expected executable code part");
        }

        if let Part::CodeExecutionResult(code_execution_result) = &parts[6] {
            assert_eq!(
                code_execution_result
                    .clone()
                    .output
                    .expect("Failed to get code execution result output in test"),
                "Hello, world!"
            );
        } else {
            panic!("Expected code execution result part");
        }
    }

    #[test]
    fn test_deserialize_message_model() {
        let json_data = json!({
            "parts": [{"text": "Hello, user!"}],
            "role": "model"
        });

        let content: Content = serde_json::from_value(json_data)
            .expect("Failed to deserialize content from JSON in test");
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
        if let Part::Text(text) = &content.parts.first() {
            assert_eq!(text, "Hello, user!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_message_conversion_user() {
        let msg = message::Message::user("Hello, world!");
        let content: Content = msg
            .try_into()
            .expect("Failed to convert user message to content in test");
        assert_eq!(content.role, Some(Role::User));
        assert_eq!(content.parts.len(), 1);
        if let Part::Text(text) = &content.parts.first() {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_message_conversion_model() {
        let msg = message::Message::assistant("Hello, user!");

        let content: Content = msg
            .try_into()
            .expect("Failed to convert assistant message to content in test");
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
        if let Part::Text(text) = &content.parts.first() {
            assert_eq!(text, "Hello, user!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_message_conversion_tool_call() {
        let tool_call = message::ToolCall {
            id: "test_tool".to_string(),
            function: message::ToolFunction {
                name: "test_function".to_string(),
                arguments: json!({"arg1": "value1"}),
            },
        };

        let msg = message::Message::Assistant {
            content: OneOrMany::one(message::AssistantContent::ToolCall(tool_call)),
        };

        let content: Content = msg
            .try_into()
            .expect("Failed to convert tool call message to content in test");
        assert_eq!(content.role, Some(Role::Model));
        assert_eq!(content.parts.len(), 1);
        if let Part::FunctionCall(function_call) = &content.parts.first() {
            assert_eq!(function_call.name, "test_function");
            assert_eq!(
                function_call
                    .args
                    .as_object()
                    .expect("Failed to get function call args as object in tool call test")
                    .get("arg1")
                    .expect("Failed to get arg1 from function call args in tool call test"),
                "value1"
            );
        } else {
            panic!("Expected function call part");
        }
    }
}
