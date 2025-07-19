//! Model capability definitions and capability-based filtering
//! 
//! This module defines the core model capabilities structure and provides
//! functionality for filtering and querying models based on their capabilities.

use serde::{Serialize, Deserialize};

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
    Pruning,
}

/// Model capability flags for filtering and selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Whether the model supports vision/image inputs
    pub supports_vision: bool,
    /// Whether the model supports function/tool calling
    pub supports_function_calling: bool,
    /// Whether the model supports streaming responses
    pub supports_streaming: bool,
    /// Whether the model supports fine-tuning
    pub supports_fine_tuning: bool,
    /// Whether the model supports batch processing
    pub supports_batch_processing: bool,
    /// Whether the model supports real-time processing
    pub supports_realtime: bool,
    /// Whether the model supports multimodal inputs
    pub supports_multimodal: bool,
    /// Whether the model supports thinking/reasoning modes
    pub supports_thinking: bool,
    /// Whether the model supports embedding generation
    pub supports_embedding: bool,
    /// Whether the model supports code completion
    pub supports_code_completion: bool,
    /// Whether the model supports chat/conversation
    pub supports_chat: bool,
    /// Whether the model supports instruction following
    pub supports_instruction_following: bool,
    /// Whether the model supports few-shot learning
    pub supports_few_shot_learning: bool,
    /// Whether the model supports zero-shot learning
    pub supports_zero_shot_learning: bool,
    /// Whether the model has a long context window
    pub has_long_context: bool,
    /// Whether the model is optimized for low-latency inference
    pub is_low_latency: bool,
    /// Whether the model is optimized for high-throughput inference
    pub is_high_throughput: bool,
    /// Whether the model supports quantization
    pub supports_quantization: bool,
    /// Whether the model supports distillation
    pub supports_distillation: bool,
    /// Whether the model supports pruning
    pub supports_pruning: bool,
}

impl ModelCapabilities {
    /// Create a new ModelCapabilities with all capabilities disabled
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable a specific capability
    pub fn with_capability(mut self, capability: Capability) -> Self {
        self.set_capability(capability, true);
        self
    }

    /// Disable a specific capability
    pub fn without_capability(mut self, capability: Capability) -> Self {
        self.set_capability(capability, false);
        self
    }

    /// Set a specific capability
    pub fn set_capability(&mut self, capability: Capability, enabled: bool) {
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

    /// Check if a specific capability is enabled
    pub fn has_capability(&self, capability: Capability) -> bool {
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

    /// Check if all specified capabilities are enabled
    pub fn has_all_capabilities(&self, capabilities: &[Capability]) -> bool {
        capabilities.iter().all(|&cap| self.has_capability(cap))
    }

    /// Check if any of the specified capabilities are enabled
    pub fn has_any_capability(&self, capabilities: &[Capability]) -> bool {
        capabilities.iter().any(|&cap| self.has_capability(cap))
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
    pub flops_per_token: Option<u64>,
}

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
            flops_per_token: None,
        }
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
    ZeroShotLearning,
}


