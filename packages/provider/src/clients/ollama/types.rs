//! Ollama API request and response structures.
//! 
//! This module provides zero-allocation structures for interacting with Ollama,
//! a local LLM inference server. All collections use ArrayVec for bounded, 
//! stack-allocated storage.

use serde::{Deserialize, Serialize};
use arrayvec::ArrayVec;
use crate::{MAX_MESSAGES, MAX_TOOLS, MAX_IMAGES};

// ============================================================================
// Generate/Chat Completion API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaGenerateRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<ArrayVec<&'a str, MAX_IMAGES>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ArrayVec<i64, 2048>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub messages: ArrayVec<OllamaMessage<'a>, MAX_MESSAGES>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ArrayVec<OllamaTool<'a>, MAX_TOOLS>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage<'a> {
    #[serde(borrow)]
    pub role: &'a str,
    #[serde(borrow)]
    pub content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<ArrayVec<&'a str, MAX_IMAGES>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<OllamaToolCall<'a>, MAX_TOOLS>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall<'a> {
    #[serde(borrow)]
    pub function: OllamaFunction<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool<'a> {
    #[serde(rename = "type")]
    #[serde(borrow)]
    pub tool_type: &'a str,
    pub function: OllamaToolFunction<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolFunction<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub description: &'a str,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_vram: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub f16_kv: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits_all: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mlock: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalize_newline: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ArrayVec<String, 4>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numa: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tfs_z: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ArrayVec<i64, 2048>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaResponseMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponseMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ArrayVec<OllamaResponseToolCall, MAX_TOOLS>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponseToolCall {
    pub function: OllamaResponseFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponseFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ============================================================================
// Embeddings API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingRequest<'a> {
    #[serde(borrow)]
    pub model: &'a str,
    #[serde(borrow)]
    pub prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingResponse {
    pub embedding: ArrayVec<f32, 4096>,
}

// ============================================================================
// Model Management API
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaCreateRequest<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(borrow)]
    pub modelfile: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaCreateResponse {
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPullRequest<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub insecure: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPullResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPushRequest<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub insecure: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPushResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaListResponse {
    pub models: ArrayVec<OllamaModelInfo, 256>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    pub parent_model: String,
    pub format: String,
    pub family: String,
    pub families: ArrayVec<String, 8>,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaShowRequest<'a> {
    #[serde(borrow)]
    pub name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbose: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: OllamaModelDetails,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaCopyRequest<'a> {
    #[serde(borrow)]
    pub source: &'a str,
    #[serde(borrow)]
    pub destination: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaDeleteRequest<'a> {
    #[serde(borrow)]
    pub name: &'a str,
}

// ============================================================================
// Streaming Support
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaStreamingChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<OllamaResponseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ArrayVec<i64, 2048>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaErrorResponse {
    pub error: String,
}

// ============================================================================
// Builder Patterns for Http3 Integration
// ============================================================================

impl<'a> OllamaGenerateRequest<'a> {
    pub fn new(model: &'a str, prompt: &'a str) -> Self {
        Self {
            model,
            prompt,
            images: None,
            format: None,
            options: None,
            system: None,
            template: None,
            context: None,
            stream: None,
            raw: None,
            keep_alive: None,
        }
    }

    pub fn with_system(mut self, system: &'a str) -> Self {
        self.system = Some(system);
        self
    }

    pub fn with_options(mut self, options: OllamaOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn stream(mut self, streaming: bool) -> Self {
        self.stream = Some(streaming);
        self
    }

    pub fn with_images(mut self, images: ArrayVec<&'a str, MAX_IMAGES>) -> Self {
        self.images = Some(images);
        self
    }

    pub fn with_format(mut self, format: &'a str) -> Self {
        self.format = Some(format);
        self
    }

    pub fn keep_alive(mut self, keep_alive: &'a str) -> Self {
        self.keep_alive = Some(keep_alive);
        self
    }
}

impl<'a> OllamaChatRequest<'a> {
    pub fn new(model: &'a str) -> Self {
        Self {
            model,
            messages: ArrayVec::new(),
            tools: None,
            format: None,
            options: None,
            stream: None,
            keep_alive: None,
        }
    }

    pub fn add_message(mut self, role: &'a str, content: &'a str) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(OllamaMessage {
                role,
                content,
                images: None,
                tool_calls: None,
            });
        }
        self
    }

    pub fn add_message_with_images(mut self, role: &'a str, content: &'a str, images: ArrayVec<&'a str, MAX_IMAGES>) -> Self {
        if self.messages.len() < MAX_MESSAGES {
            let _ = self.messages.try_push(OllamaMessage {
                role,
                content,
                images: Some(images),
                tool_calls: None,
            });
        }
        self
    }

    pub fn with_tools(mut self, tools: ArrayVec<OllamaTool<'a>, MAX_TOOLS>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_options(mut self, options: OllamaOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn stream(mut self, streaming: bool) -> Self {
        self.stream = Some(streaming);
        self
    }

    pub fn with_format(mut self, format: &'a str) -> Self {
        self.format = Some(format);
        self
    }
}

impl OllamaOptions {
    pub fn new() -> Self {
        Self {
            temperature: None,
            top_k: None,
            top_p: None,
            num_predict: None,
            num_ctx: None,
            num_batch: None,
            num_gpu: None,
            main_gpu: None,
            low_vram: None,
            f16_kv: None,
            logits_all: None,
            vocab_only: None,
            use_mmap: None,
            use_mlock: None,
            num_thread: None,
            repeat_last_n: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            penalize_newline: None,
            stop: None,
            numa: None,
            seed: None,
            tfs_z: None,
            typical_p: None,
        }
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn num_predict(mut self, num: i32) -> Self {
        self.num_predict = Some(num);
        self
    }

    pub fn num_ctx(mut self, ctx: u32) -> Self {
        self.num_ctx = Some(ctx);
        self
    }

    pub fn stop_sequences(mut self, stops: ArrayVec<String, 4>) -> Self {
        self.stop = Some(stops);
        self
    }

    pub fn seed(mut self, seed: i32) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl<'a> OllamaEmbeddingRequest<'a> {
    pub fn new(model: &'a str, prompt: &'a str) -> Self {
        Self {
            model,
            prompt,
            options: None,
            keep_alive: None,
        }
    }

    pub fn with_options(mut self, options: OllamaOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn keep_alive(mut self, keep_alive: &'a str) -> Self {
        self.keep_alive = Some(keep_alive);
        self
    }
}

impl<'a> OllamaCreateRequest<'a> {
    pub fn new(name: &'a str, modelfile: &'a str) -> Self {
        Self {
            name,
            modelfile,
            stream: None,
            path: None,
        }
    }

    pub fn stream(mut self, streaming: bool) -> Self {
        self.stream = Some(streaming);
        self
    }

    pub fn with_path(mut self, path: &'a str) -> Self {
        self.path = Some(path);
        self
    }
}

impl<'a> OllamaPullRequest<'a> {
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            insecure: None,
            stream: None,
        }
    }

    pub fn insecure(mut self, insecure: bool) -> Self {
        self.insecure = Some(insecure);
        self
    }

    pub fn stream(mut self, streaming: bool) -> Self {
        self.stream = Some(streaming);
        self
    }
}

impl<'a> OllamaShowRequest<'a> {
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            verbose: None,
        }
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }
}