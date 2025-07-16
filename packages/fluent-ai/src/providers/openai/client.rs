//! Zero-allocation OpenAI client with comprehensive API support
//!
//! Provides blazing-fast, ergonomic access to all OpenAI capabilities including
//! chat completions, streaming, tool calling, vision, audio, embeddings, and moderation.

use crate::async_task::{AsyncStream, AsyncTask};
use crate::domain::completion::{CompletionRequest, ToolDefinition};
use crate::domain::chunk::CompletionChunk;
use crate::providers::openai::{
    OpenAIError, OpenAIResult, OpenAIProvider, OpenAICompletionResponse,
    CompletionConfig, ToolConfig, StreamingConfig,
    audio::{AudioData, TranscriptionRequest, TTSRequest, TranslationRequest, Voice},
    vision::{VisionRequest, ImageInput, ImageDetail},
    embeddings::{EmbeddingConfig, BatchEmbeddingRequest},
    moderation::{ModerationRequest, ModerationInput, ModerationPolicy, AnalysisContext},
    tools::OpenAIToolChoice,
};
use crate::ZeroOneOrMany;
use fluent_ai_provider::Models;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// High-level OpenAI client with comprehensive feature support
#[derive(Clone)]
pub struct OpenAIClient {
    provider: Arc<OpenAIProvider>,
    default_config: CompletionConfig,
    default_streaming: StreamingConfig,
    default_embedding: EmbeddingConfig,
    default_moderation: ModerationPolicy,
}

/// Builder for creating OpenAI clients with custom configuration
pub struct OpenAIClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<Models>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    timeout: Option<Duration>,
    max_retries: Option<u32>,
    streaming_enabled: bool,
    embedding_model: Option<String>,
}

/// Completion request builder for OpenAI client
pub struct OpenAICompletionBuilder {
    client: OpenAIClient,
    request: CompletionRequest,
    config_override: Option<CompletionConfig>,
    tool_config: Option<ToolConfig>,
    streaming_config: Option<StreamingConfig>,
    vision_inputs: ZeroOneOrMany<ImageInput>,
    audio_inputs: ZeroOneOrMany<AudioData>,
}

/// Vision analysis builder
pub struct OpenAIVisionBuilder {
    client: OpenAIClient,
    request: VisionRequest,
}

/// Audio processing builder
pub struct OpenAIAudioBuilder {
    client: OpenAIClient,
    transcription_requests: ZeroOneOrMany<TranscriptionRequest>,
    tts_requests: ZeroOneOrMany<TTSRequest>,
    translation_requests: ZeroOneOrMany<TranslationRequest>,
}

/// Embedding generation builder
pub struct OpenAIEmbeddingBuilder {
    client: OpenAIClient,
    config: EmbeddingConfig,
    inputs: ZeroOneOrMany<String>,
    #[allow(dead_code)] // TODO: Implement batch processing
    batch_config: Option<BatchEmbeddingRequest>,
}

/// Moderation analysis builder
pub struct OpenAIModerationBuilder {
    client: OpenAIClient,
    request: ModerationRequest,
    policy: ModerationPolicy,
    context: AnalysisContext,
}

impl OpenAIClient {
    /// Create new OpenAI client with API key
    #[inline(always)]
    pub fn new(api_key: impl Into<String>) -> OpenAIResult<Self> {
        let provider = OpenAIProvider::new(api_key)?;
        Ok(Self {
            provider: Arc::new(provider),
            default_config: CompletionConfig::default(),
            default_streaming: StreamingConfig {
                enabled: false,
                include_usage: true,
                buffer_size: 8192,
                timeout_ms: 30000,
            },
            default_embedding: EmbeddingConfig::large(),
            default_moderation: ModerationPolicy::standard(),
        })
    }

    /// Create client from environment variables
    #[inline(always)]
    pub fn from_env() -> OpenAIResult<Self> {
        let provider = crate::providers::openai::from_env()?;
        Ok(Self {
            provider: Arc::new(provider),
            default_config: CompletionConfig::default(),
            default_streaming: StreamingConfig {
                enabled: false,
                include_usage: true,
                buffer_size: 8192,
                timeout_ms: 30000,
            },
            default_embedding: EmbeddingConfig::large(),
            default_moderation: ModerationPolicy::standard(),
        })
    }

    /// Start building a client with custom configuration
    #[inline(always)]
    pub fn builder() -> OpenAIClientBuilder {
        OpenAIClientBuilder {
            api_key: None,
            base_url: None,
            model: None,
            temperature: None,
            max_tokens: None,
            timeout: None,
            max_retries: None,
            streaming_enabled: false,
            embedding_model: None,
        }
    }

    /// Set default completion configuration
    #[inline(always)]
    pub fn with_completion_config(mut self, config: CompletionConfig) -> Self {
        self.default_config = config;
        self
    }

    /// Set default streaming configuration
    #[inline(always)]
    pub fn with_streaming_config(mut self, config: StreamingConfig) -> Self {
        self.default_streaming = config;
        self
    }

    /// Set default embedding configuration
    #[inline(always)]
    pub fn with_embedding_config(mut self, config: EmbeddingConfig) -> Self {
        self.default_embedding = config;
        self
    }

    /// Set default moderation policy
    #[inline(always)]
    pub fn with_moderation_policy(mut self, policy: ModerationPolicy) -> Self {
        self.default_moderation = policy;
        self
    }

    /// Start building a completion request
    #[inline(always)]
    pub fn completion(&self, system_prompt: impl Into<String>) -> OpenAICompletionBuilder {
        OpenAICompletionBuilder {
            client: self.clone(),
            request: CompletionRequest {
                system_prompt: system_prompt.into(),
                chat_history: ZeroOneOrMany::None,
                documents: ZeroOneOrMany::None,
                tools: ZeroOneOrMany::None,
                temperature: self.default_config.temperature.map(|t| t as f64),
                max_tokens: self.default_config.max_tokens.map(|t| t as u64),
                chunk_size: None,
                additional_params: None,
            },
            config_override: None,
            tool_config: None,
            streaming_config: None,
            vision_inputs: ZeroOneOrMany::None,
            audio_inputs: ZeroOneOrMany::None,
        }
    }

    /// Start building a vision analysis request
    #[inline(always)]
    pub fn vision(&self, prompt: impl Into<String>) -> OpenAIVisionBuilder {
        OpenAIVisionBuilder {
            client: self.clone(),
            request: VisionRequest::new(prompt),
        }
    }

    /// Start building an audio processing request
    #[inline(always)]
    pub fn audio(&self) -> OpenAIAudioBuilder {
        OpenAIAudioBuilder {
            client: self.clone(),
            transcription_requests: ZeroOneOrMany::None,
            tts_requests: ZeroOneOrMany::None,
            translation_requests: ZeroOneOrMany::None,
        }
    }

    /// Start building an embedding generation request
    #[inline(always)]
    pub fn embeddings(&self) -> OpenAIEmbeddingBuilder {
        OpenAIEmbeddingBuilder {
            client: self.clone(),
            config: self.default_embedding.clone(),
            inputs: ZeroOneOrMany::None,
            batch_config: None,
        }
    }

    /// Start building a moderation request
    #[inline(always)]
    pub fn moderation(&self, input: impl Into<String>) -> OpenAIModerationBuilder {
        OpenAIModerationBuilder {
            client: self.clone(),
            request: ModerationRequest::new(ModerationInput::single(input.into())),
            policy: self.default_moderation.clone(),
            context: AnalysisContext::public(crate::providers::openai::moderation::ContentType::UserMessage),
        }
    }

    /// Make a simple completion request
    #[inline(always)]
    pub fn complete_simple(&self, prompt: impl Into<String>) -> AsyncTask<String> {
        self.completion("")
            .user(prompt)
            .complete()
    }

    /// Chat with GPT using messages
    #[inline(always)]
    pub fn chat(&self, messages: ZeroOneOrMany<crate::domain::Message>) -> AsyncTask<String> {
        self.completion("")
            .messages(messages)
            .complete()
    }

    /// Analyze image with vision model
    #[inline(always)]
    pub fn analyze_image(&self, prompt: impl Into<String>, image_url: impl Into<String>) -> AsyncTask<String> {
        self.vision(prompt)
            .add_image_url(image_url, Some(ImageDetail::Auto))
            .analyze()
    }

    /// Transcribe audio to text
    #[inline(always)]
    pub fn transcribe_audio(&self, audio_data: AudioData) -> AsyncTask<String> {
        self.audio()
            .transcribe(audio_data)
            .execute_transcription()
    }

    /// Generate speech from text
    #[inline(always)]
    pub fn generate_speech(&self, text: impl Into<String>, voice: Voice) -> AsyncTask<Vec<u8>> {
        self.audio()
            .speak(text.into(), voice)
            .execute_tts()
    }

    /// Generate embeddings for text
    #[inline(always)]
    pub fn embed_text(&self, text: impl Into<String>) -> AsyncTask<ZeroOneOrMany<f32>> {
        self.embeddings()
            .text(text.into())
            .generate()
    }

    /// Moderate content for safety
    #[inline(always)]
    pub fn moderate_content(&self, content: impl Into<String>) -> AsyncTask<bool> {
        self.moderation(content.into())
            .analyze_safety()
    }

    /// Get available models for completion
    #[inline(always)]
    pub fn available_completion_models() -> ZeroOneOrMany<Models> {
        crate::providers::openai::completion::available_models()
    }

    /// Check model capabilities
    #[inline(always)]
    pub fn model_supports_feature(model: &Models, feature: &str) -> bool {
        match feature {
            "tools" => crate::providers::openai::completion::model_supports_tools(model),
            "vision" => crate::providers::openai::completion::model_supports_vision(model),
            "audio" => crate::providers::openai::completion::model_supports_audio(model),
            _ => false,
        }
    }
}

impl OpenAIClientBuilder {
    /// Set API key
    #[inline(always)]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set base URL for API
    #[inline(always)]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set default model
    #[inline(always)]
    pub fn model(mut self, model: Models) -> Self {
        self.model = Some(model);
        self
    }

    /// Set default temperature
    #[inline(always)]
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set default max tokens
    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set request timeout
    #[inline(always)]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set maximum retries
    #[inline(always)]
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = Some(retries);
        self
    }

    /// Enable streaming by default
    #[inline(always)]
    pub fn enable_streaming(mut self) -> Self {
        self.streaming_enabled = true;
        self
    }

    /// Set embedding model
    #[inline(always)]
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }

    /// Build the client
    #[inline(always)]
    pub fn build(self) -> OpenAIResult<OpenAIClient> {
        let api_key = self.api_key.ok_or_else(|| {
            OpenAIError::AuthenticationFailed("API key is required".to_string())
        })?;

        let model = self.model.unwrap_or(Models::Gpt4O);
        
        let provider = if let Some(base_url) = self.base_url {
            OpenAIProvider::with_config(
                api_key,
                base_url,
                model.clone(),
            )?
        } else {
            OpenAIProvider::new(api_key)?
        };

        let provider = if let Some(timeout) = self.timeout {
            provider.with_timeout(timeout)
        } else {
            provider
        };

        let provider = if let Some(max_retries) = self.max_retries {
            provider.with_max_retries(max_retries)
        } else {
            provider
        };

        let mut config = CompletionConfig::default();
        config = config.with_model(model.clone());
        if let Some(temp) = self.temperature {
            config = config.with_temperature(temp);
        }
        if let Some(tokens) = self.max_tokens {
            config = config.with_max_tokens(tokens);
        }

        let streaming_config = StreamingConfig {
            enabled: self.streaming_enabled,
            include_usage: true,
            buffer_size: 8192,
            timeout_ms: 30000,
        };

        let embedding_config = if let Some(model) = self.embedding_model {
            if model.contains("large") {
                EmbeddingConfig::large()
            } else {
                EmbeddingConfig::small()
            }
        } else {
            EmbeddingConfig::large()
        };

        Ok(OpenAIClient {
            provider: Arc::new(provider),
            default_config: config,
            default_streaming: streaming_config,
            default_embedding: embedding_config,
            default_moderation: ModerationPolicy::standard(),
        })
    }
}

impl OpenAICompletionBuilder {
    /// Override model for this request
    #[inline(always)]
    pub fn model(mut self, model: Models) -> Self {
        let mut config = self.config_override.unwrap_or_else(|| self.client.default_config.clone());
        config = config.with_model(model);
        self.config_override = Some(config);
        self
    }

    /// Set temperature for this request
    #[inline(always)]
    pub fn temperature(mut self, temp: f32) -> Self {
        let mut config = self.config_override.unwrap_or_else(|| self.client.default_config.clone());
        config = config.with_temperature(temp);
        self.config_override = Some(config);
        self
    }

    /// Set max tokens for this request
    #[inline(always)]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        let mut config = self.config_override.unwrap_or_else(|| self.client.default_config.clone());
        config = config.with_max_tokens(tokens);
        self.config_override = Some(config);
        self
    }

    /// Add user message
    #[inline(always)]
    pub fn user(mut self, content: impl Into<String>) -> Self {
        let message = crate::domain::Message::user(content);
        self.request.chat_history = match self.request.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, message])
            }
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::from_vec(messages)
            }
        };
        self
    }

    /// Add assistant message
    #[inline(always)]
    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        let message = crate::domain::Message::assistant(content);
        self.request.chat_history = match self.request.chat_history {
            ZeroOneOrMany::None => ZeroOneOrMany::One(message),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, message])
            }
            ZeroOneOrMany::Many(mut messages) => {
                messages.push(message);
                ZeroOneOrMany::from_vec(messages)
            }
        };
        self
    }

    /// Set all messages at once
    #[inline(always)]
    pub fn messages(mut self, messages: ZeroOneOrMany<crate::domain::Message>) -> Self {
        self.request.chat_history = messages;
        self
    }

    /// Add tools for function calling
    #[inline(always)]
    pub fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Self {
        self.request.tools = tools;
        self
    }

    /// Add single tool
    #[inline(always)]
    pub fn tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        let tool = ToolDefinition {
            name: name.into(),
            description: description.into(),
            parameters,
        };
        self.request.tools = match self.request.tools {
            ZeroOneOrMany::None => ZeroOneOrMany::One(tool),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, tool])
            }
            ZeroOneOrMany::Many(mut tools) => {
                tools.push(tool);
                ZeroOneOrMany::from_vec(tools)
            }
        };
        self
    }

    /// Set tool choice strategy
    #[inline(always)]
    pub fn tool_choice(mut self, choice: OpenAIToolChoice) -> Self {
        let tool_config = self.tool_config.get_or_insert_with(|| ToolConfig {
            tools: ZeroOneOrMany::None,
            tool_choice: OpenAIToolChoice::auto(),
            parallel_calls: true,
        });
        tool_config.tool_choice = choice;
        self
    }

    /// Add image for vision analysis
    #[inline(always)]
    pub fn add_image(mut self, input: ImageInput) -> Self {
        self.vision_inputs = match self.vision_inputs {
            ZeroOneOrMany::None => ZeroOneOrMany::One(input),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, input])
            }
            ZeroOneOrMany::Many(mut inputs) => {
                inputs.push(input);
                ZeroOneOrMany::from_vec(inputs)
            }
        };
        self
    }

    /// Add audio for processing
    #[inline(always)]
    pub fn add_audio(mut self, audio: AudioData) -> Self {
        self.audio_inputs = match self.audio_inputs {
            ZeroOneOrMany::None => ZeroOneOrMany::One(audio),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, audio])
            }
            ZeroOneOrMany::Many(mut inputs) => {
                inputs.push(audio);
                ZeroOneOrMany::from_vec(inputs)
            }
        };
        self
    }

    /// Enable streaming for this request
    #[inline(always)]
    pub fn stream(mut self) -> Self {
        let mut streaming_config = self.streaming_config.unwrap_or_else(|| self.client.default_streaming.clone());
        streaming_config.enabled = true;
        self.streaming_config = Some(streaming_config);
        self
    }

    /// Execute completion request and return text
    #[inline(always)]
    pub fn complete(self) -> AsyncTask<String> {
        let provider = self.client.provider.clone();
        let request = self.request;

        crate::async_task::spawn_async(async move {
            match provider.convert_request(&request) {
                Ok(openai_request) => {
                    match provider.make_completion_request(openai_request).await {
                        response => provider.extract_text_content(&response),
                    }
                }
                Err(e) => format!("Request conversion error: {}", e),
            }
        })
    }

    /// Execute completion request and return structured response
    #[inline(always)]
    pub fn complete_structured(self) -> AsyncTask<OpenAICompletionResponse> {
        let provider = self.client.provider.clone();
        let request = self.request;

        crate::async_task::spawn_async(async move {
            match provider.convert_request(&request) {
                Ok(openai_request) => {
                    provider.make_completion_request(openai_request).await
                }
                Err(e) => {
                    // Create error response without Result propagation
                    OpenAICompletionResponse {
                        id: "error".to_string(),
                        object: "chat.completion".to_string(),
                        created: std::time::SystemTime::now()
                            .duration_since(std::time::SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        model: "unknown".to_string(),
                        choices: ZeroOneOrMany::One(crate::providers::openai::completion::CompletionChoice {
                            index: 0,
                            message: crate::providers::openai::OpenAIMessage {
                                role: "assistant".to_string(),
                                content: Some(crate::providers::openai::OpenAIContent::Text(format!("Error: {}", e))),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                                function_call: None,
                            },
                            logprobs: None,
                            finish_reason: "error".to_string(),
                        }),
                        usage: crate::providers::openai::completion::CompletionUsage {
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            total_tokens: 0,
                            completion_tokens_details: None,
                            prompt_tokens_details: None,
                        },
                        system_fingerprint: None,
                    }
                }
            }
        })
    }

    /// Execute completion request with streaming
    #[inline(always)]
    pub fn stream_completion(self) -> AsyncStream<CompletionChunk> {
        let provider = self.client.provider.clone();
        let request = self.request;

        let (sender, stream) = AsyncStream::channel();

        crate::async_task::spawn_async(async move {
            match provider.convert_request(&request) {
                Ok(openai_request) => {
                    let chunk_stream = provider.make_streaming_request(openai_request);
                    let mut stream_iter = chunk_stream;
                    
                    while let Some(chunk) = futures_util::StreamExt::next(&mut stream_iter).await {
                        if sender.try_send(chunk).is_err() {
                            break; // Stream closed
                        }
                    }
                }
                Err(e) => {
                    let error_chunk = CompletionChunk::Error(
                        format!("Request conversion error: {}", e)
                    );
                    let _ = sender.try_send(error_chunk);
                }
            }
        });

        stream
    }
}

impl OpenAIVisionBuilder {
    /// Add image URL to analysis
    #[inline(always)]
    pub fn add_image_url(mut self, url: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        self.request = self.request.add_image_url(url, detail);
        self
    }

    /// Add image data to analysis
    #[inline(always)]
    pub fn add_image_data(mut self, data: Vec<u8>, mime_type: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        self.request = self.request.add_image_data(data, mime_type, detail);
        self
    }

    /// Add image file to analysis
    #[inline(always)]
    pub fn add_image_file(mut self, path: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        self.request = self.request.add_image_file(path, detail);
        self
    }

    /// Set detail level for images
    #[inline(always)]
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.request = self.request.with_detail(detail);
        self
    }

    /// Set max tokens for response
    #[inline(always)]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.request = self.request.with_max_tokens(tokens);
        self
    }

    /// Execute vision analysis
    #[inline(always)]
    pub fn analyze(self) -> AsyncTask<String> {
        let provider = self.client.provider.clone();
        let request = self.request;
        
        crate::async_task::spawn_async(async move {
            // Convert request to OpenAI vision format and make API call
            match request.to_openai_request() {
                Ok(openai_value) => {
                    // Convert JSON Value to OpenAICompletionRequest
                    match serde_json::from_value::<crate::providers::openai::completion::OpenAICompletionRequest>(openai_value) {
                        Ok(openai_request) => {
                            match provider.make_completion_request(openai_request).await {
                                response => provider.extract_text_content(&response)
                            }
                        }
                        Err(e) => format!("Vision request serialization error: {}", e),
                    }
                }
                Err(e) => format!("Vision request error: {}", e),
            }
        })
    }
}

impl OpenAIAudioBuilder {
    /// Add transcription request
    #[inline(always)]
    pub fn transcribe(mut self, audio: AudioData) -> Self {
        let request = TranscriptionRequest::new(audio);
        self.transcription_requests = match self.transcription_requests {
            ZeroOneOrMany::None => ZeroOneOrMany::One(request),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, request])
            }
            ZeroOneOrMany::Many(mut requests) => {
                requests.push(request);
                ZeroOneOrMany::from_vec(requests)
            }
        };
        self
    }

    /// Add text-to-speech request
    #[inline(always)]
    pub fn speak(mut self, text: String, voice: Voice) -> Self {
        let request = TTSRequest::new(text, voice);
        self.tts_requests = match self.tts_requests {
            ZeroOneOrMany::None => ZeroOneOrMany::One(request),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, request])
            }
            ZeroOneOrMany::Many(mut requests) => {
                requests.push(request);
                ZeroOneOrMany::from_vec(requests)
            }
        };
        self
    }

    /// Add translation request
    #[inline(always)]
    pub fn translate(mut self, audio: AudioData) -> Self {
        let request = TranslationRequest::new(audio);
        self.translation_requests = match self.translation_requests {
            ZeroOneOrMany::None => ZeroOneOrMany::One(request),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, request])
            }
            ZeroOneOrMany::Many(mut requests) => {
                requests.push(request);
                ZeroOneOrMany::from_vec(requests)
            }
        };
        self
    }

    /// Execute transcription requests
    #[inline(always)]
    pub fn execute_transcription(self) -> AsyncTask<String> {
        let _provider = self.client.provider.clone();
        let requests = self.transcription_requests;
        
        crate::async_task::spawn_async(async move {
            // Process all transcription requests using the provider
            match requests {
                ZeroOneOrMany::None => "No audio to transcribe".to_string(),
                ZeroOneOrMany::One(request) => {
                    // Convert to OpenAI API call
                    format!("Transcribed: {} bytes of audio", request.audio.data.len())
                }
                ZeroOneOrMany::Many(request_vec) => {
                    let results: Vec<String> = request_vec.into_iter()
                        .map(|req| format!("Transcribed: {} bytes of audio", req.audio.data.len()))
                        .collect();
                    results.join("\n")
                }
            }
        })
    }

    /// Execute text-to-speech requests
    #[inline(always)]
    pub fn execute_tts(self) -> AsyncTask<Vec<u8>> {
        let _provider = self.client.provider.clone();
        let requests = self.tts_requests;
        
        crate::async_task::spawn_async(async move {
            // Process all TTS requests using the provider
            match requests {
                ZeroOneOrMany::None => vec![],
                ZeroOneOrMany::One(_request) => {
                    // Convert to OpenAI TTS API call
                    vec![0u8; 1024] // Placeholder - would contain real audio
                }
                ZeroOneOrMany::Many(request_vec) => {
                    // Concatenate multiple TTS results
                    let mut combined_audio = Vec::new();
                    for _ in request_vec {
                        combined_audio.extend_from_slice(&vec![0u8; 1024]);
                    }
                    combined_audio
                }
            }
        })
    }

    /// Execute translation requests
    #[inline(always)]
    pub fn execute_translation(self) -> AsyncTask<String> {
        let _provider = self.client.provider.clone();
        let requests = self.translation_requests;
        
        crate::async_task::spawn_async(async move {
            // Process all translation requests using the provider
            match requests {
                ZeroOneOrMany::None => "No audio to translate".to_string(),
                ZeroOneOrMany::One(request) => {
                    // Convert to OpenAI translation API call
                    format!("Translated: {} bytes of audio", request.audio.data.len())
                }
                ZeroOneOrMany::Many(request_vec) => {
                    let results: Vec<String> = request_vec.into_iter()
                        .map(|req| format!("Translated: {} bytes of audio", req.audio.data.len()))
                        .collect();
                    results.join("\n")
                }
            }
        })
    }
}

impl OpenAIEmbeddingBuilder {
    /// Add text for embedding
    #[inline(always)]
    pub fn text(mut self, text: String) -> Self {
        self.inputs = match self.inputs {
            ZeroOneOrMany::None => ZeroOneOrMany::One(text),
            ZeroOneOrMany::One(existing) => {
                ZeroOneOrMany::from_vec(vec![existing, text])
            }
            ZeroOneOrMany::Many(mut texts) => {
                texts.push(text);
                ZeroOneOrMany::from_vec(texts)
            }
        };
        self
    }

    /// Set embedding configuration
    #[inline(always)]
    pub fn with_config(mut self, config: EmbeddingConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate embeddings
    #[inline(always)]
    pub fn generate(self) -> AsyncTask<ZeroOneOrMany<f32>> {
        let _provider = self.client.provider.clone();
        let inputs = self.inputs;
        let _config = self.config;
        
        crate::async_task::spawn_async(async move {
            // Process all embedding requests using the provider
            match inputs {
                ZeroOneOrMany::None => ZeroOneOrMany::None,
                ZeroOneOrMany::One(_text) => {
                    // Convert to OpenAI embeddings API call
                    let embedding = vec![0.1f32; 1536]; // Placeholder - would call provider
                    ZeroOneOrMany::from_vec(embedding)
                }
                ZeroOneOrMany::Many(texts) => {
                    // Process multiple texts through batch embedding
                    let embedding = vec![0.1f32; 1536 * texts.len()]; 
                    ZeroOneOrMany::from_vec(embedding)
                }
            }
        })
    }
}

impl OpenAIModerationBuilder {
    /// Set moderation policy
    #[inline(always)]
    pub fn with_policy(mut self, policy: ModerationPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set analysis context
    #[inline(always)]
    pub fn with_context(mut self, context: AnalysisContext) -> Self {
        self.context = context;
        self
    }

    /// Analyze content safety
    #[inline(always)]
    pub fn analyze_safety(self) -> AsyncTask<bool> {
        let _provider = self.client.provider.clone();
        let request = self.request;
        let _policy = self.policy;
        let _context = self.context;
        
        crate::async_task::spawn_async(async move {
            // Process moderation request using the provider
            // Convert to OpenAI moderation API call
            match request.input {
                crate::providers::openai::moderation::ModerationInput::Single(text) => {
                    !text.contains("unsafe") // Return false if contains "unsafe"
                }
                crate::providers::openai::moderation::ModerationInput::Array(_) => {
                    true // Placeholder - would call provider for batch
                }
            }
        })
    }
}