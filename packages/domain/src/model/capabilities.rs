//! Model capability utilities for filtering and querying
//!
//! This module provides utility types for working with model capabilities.
//! ModelCapabilities is imported from model-info (the single source of truth).

use serde::{Deserialize, Serialize};

// RE-EXPORT ModelCapabilities from model-info (single source of truth)
pub use model_info::common::ModelCapabilities;

/// Specific capabilities that models can support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capability {
    /// Supports vision/image understanding
    Vision,
    /// Supports function/tool calling
    FunctionCalling,
    /// Supports streaming responses
    Streaming,
    /// Supports fine-tuning
    FineTuning,
    /// Supports batch processing
    BatchProcessing,
    /// Supports real-time processing
    Realtime,
    /// Supports multimodal inputs (text + images, etc.)
    Multimodal,
    /// Supports thinking/reasoning modes
    Thinking,
    /// Supports embedding generation
    Embedding,
    /// Supports code completion
    CodeCompletion,
    /// Supports chat/conversation
    Chat,
    /// Supports instruction following
    InstructionFollowing,
    /// Supports few-shot learning
    FewShotLearning,
    /// Supports zero-shot learning
    ZeroShotLearning,
    /// Supports long context windows
    LongContext,
    /// Supports low-latency inference
    LowLatency,
    /// Supports high-throughput inference
    HighThroughput,
    /// Supports model quantization
    Quantization,
    /// Supports model distillation
    Distillation,
    /// Supports model pruning
    Pruning}

/// Domain-specific model capability flags for filtering and selection
///
/// This is a utility struct derived from ModelInfo for capability-based filtering.
/// ModelInfo (which deserializes from the external models.yaml) is the single source of truth.
/// Use ModelInfo::to_capabilities() to create this struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DomainModelCapabilities {
    /// Supports vision/image understanding
    pub vision: bool,
    /// Supports function/tool calling  
    pub function_calling: bool,
    /// Supports streaming responses
    pub streaming: bool,
    /// Supports embeddings generation
    pub embeddings: bool,
    /// Supports thinking/reasoning
    pub thinking: bool,
}

/// Extension trait to add utility methods to ModelCapabilities from model-info
pub trait ModelCapabilitiesExt {
    /// Enable a specific capability
    fn with_capability(self, capability: Capability) -> Self;
    /// Disable a specific capability  
    fn without_capability(self, capability: Capability) -> Self;
    /// Set a specific capability
    fn set_capability(&mut self, capability: Capability, enabled: bool);
    /// Check if a specific capability is enabled
    fn has_capability(&self, capability: Capability) -> bool;
    /// Check if all specified capabilities are enabled
    fn has_all_capabilities(&self, capabilities: &[Capability]) -> bool;
    /// Check if any of the specified capabilities are enabled
    fn has_any_capability(&self, capabilities: &[Capability]) -> bool;
    /// Get an iterator over all enabled capabilities
    fn enabled_capabilities(&self) -> impl Iterator<Item = Capability> + '_;
    /// Get all enabled capabilities as a vector
    fn to_vec(&self) -> Vec<Capability>;
}

impl ModelCapabilitiesExt for ModelCapabilities {
    fn with_capability(mut self, capability: Capability) -> Self {
        self.set_capability(capability, true);
        self
    }

    fn without_capability(mut self, capability: Capability) -> Self {
        self.set_capability(capability, false);
        self
    }

    fn set_capability(&mut self, capability: Capability, enabled: bool) {
        match capability {
            Capability::Vision => self.supports_vision = enabled,
            Capability::FunctionCalling => self.supports_function_calling = enabled,
            Capability::Streaming => self.supports_streaming = enabled,
            Capability::FineTuning => self.supports_fine_tuning = enabled,
            Capability::BatchProcessing => self.supports_batch_processing = enabled,
            Capability::Realtime => self.supports_realtime = enabled,
            Capability::Multimodal => self.supports_multimodal = enabled,
            Capability::Thinking => self.supports_thinking = enabled,
            Capability::Embedding => self.supports_embedding = enabled,
            Capability::CodeCompletion => self.supports_code_completion = enabled,
            Capability::Chat => self.supports_chat = enabled,
            Capability::InstructionFollowing => self.supports_instruction_following = enabled,
            Capability::FewShotLearning => self.supports_few_shot_learning = enabled,
            Capability::ZeroShotLearning => self.supports_zero_shot_learning = enabled,
            Capability::LongContext => self.has_long_context = enabled,
            Capability::LowLatency => self.is_low_latency = enabled,
            Capability::HighThroughput => self.is_high_throughput = enabled,
            Capability::Quantization => self.supports_quantization = enabled,
            Capability::Distillation => self.supports_distillation = enabled,
            Capability::Pruning => self.supports_pruning = enabled,
        }
    }

    fn has_capability(&self, capability: Capability) -> bool {
        match capability {
            Capability::Vision => self.supports_vision,
            Capability::FunctionCalling => self.supports_function_calling,
            Capability::Streaming => self.supports_streaming,
            Capability::FineTuning => self.supports_fine_tuning,
            Capability::BatchProcessing => self.supports_batch_processing,
            Capability::Realtime => self.supports_realtime,
            Capability::Multimodal => self.supports_multimodal,
            Capability::Thinking => self.supports_thinking,
            Capability::Embedding => self.supports_embedding,
            Capability::CodeCompletion => self.supports_code_completion,
            Capability::Chat => self.supports_chat,
            Capability::InstructionFollowing => self.supports_instruction_following,
            Capability::FewShotLearning => self.supports_few_shot_learning,
            Capability::ZeroShotLearning => self.supports_zero_shot_learning,
            Capability::LongContext => self.has_long_context,
            Capability::LowLatency => self.is_low_latency,
            Capability::HighThroughput => self.is_high_throughput,
            Capability::Quantization => self.supports_quantization,
            Capability::Distillation => self.supports_distillation,
            Capability::Pruning => self.supports_pruning,
        }
    }

    fn has_all_capabilities(&self, capabilities: &[Capability]) -> bool {
        capabilities.iter().all(|&cap| self.has_capability(cap))
    }

    fn has_any_capability(&self, capabilities: &[Capability]) -> bool {
        capabilities.iter().any(|&cap| self.has_capability(cap))
    }

    fn enabled_capabilities(&self) -> impl Iterator<Item = Capability> + '_ {
        use Capability::*;
        [
            Vision,
            FunctionCalling,
            Streaming,
            FineTuning,
            BatchProcessing,
            Realtime,
            Multimodal,
            Thinking,
            Embedding,
            CodeCompletion,
            Chat,
            InstructionFollowing,
            FewShotLearning,
            ZeroShotLearning,
            LongContext,
            LowLatency,
            HighThroughput,
            Quantization,
            Distillation,
            Pruning,
        ]
        .iter()
        .filter(move |&&capability| self.has_capability(capability))
        .copied()
    }

    fn to_vec(&self) -> Vec<Capability> {
        self.enabled_capabilities().collect()
    }
}

/// Model performance characteristics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Tokens per second for input processing
    pub input_tokens_per_second: f32,
    /// Tokens per second for output generation
    pub output_tokens_per_second: f32,
    /// Latency in milliseconds for the first token
    pub first_token_latency_ms: f32,
    /// Latency in milliseconds per token
    pub per_token_latency_ms: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f32,
    /// GPU memory usage in MB (if applicable)
    pub gpu_memory_usage_mb: Option<f32>,
    /// Number of parameters in billions
    pub parameter_count_billions: f32,
    /// Floating-point operations per token
    pub flops_per_token: Option<u64>}

impl Default for ModelPerformance {
    fn default() -> Self {
        Self {
            input_tokens_per_second: 0.0,
            output_tokens_per_second: 0.0,
            first_token_latency_ms: 0.0,
            per_token_latency_ms: 0.0,
            memory_usage_mb: 0.0,
            gpu_memory_usage_mb: None,
            parameter_count_billions: 0.0,
            flops_per_token: None}
    }
}

/// Common use cases for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UseCase {
    /// General chat/conversation
    Chat,
    /// Code generation and completion
    CodeGeneration,
    /// Text summarization
    Summarization,
    /// Text classification
    Classification,
    /// Named entity recognition
    NamedEntityRecognition,
    /// Question answering
    QuestionAnswering,
    /// Text embedding generation
    Embedding,
    /// Text generation
    TextGeneration,
    /// Translation between languages
    Translation,
    /// Sentiment analysis
    SentimentAnalysis,
    /// Text-to-Speech
    TextToSpeech,
    /// Speech-to-Text
    SpeechToText,
    /// Image generation
    ImageGeneration,
    /// Image classification
    ImageClassification,
    /// Object detection
    ObjectDetection,
    /// Video understanding
    VideoUnderstanding,
    /// Audio processing
    AudioProcessing,
    /// Multimodal tasks
    Multimodal,
    /// Reasoning/thinking tasks
    Reasoning,
    /// Few-shot learning
    FewShotLearning,
    /// Zero-shot learning
    ZeroShotLearning}
