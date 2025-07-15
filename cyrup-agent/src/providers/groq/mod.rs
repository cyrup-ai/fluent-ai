// ============================================================================
// File: src/providers/groq/mod.rs
// ----------------------------------------------------------------------------
// Groq provider implementation following Anthropic template pattern
// ============================================================================

mod client;
mod completion;
mod streaming;

pub use client::{Client, GroqCompletionBuilder};
pub use completion::CompletionModel;

// Re-export model constants
pub use completion::{
    DEEPSEEK_R1_DISTILL_LLAMA_70B, GEMMA2_9B_IT, LLAMA_3_1_8B_INSTANT,
    LLAMA_3_2_11B_VISION_PREVIEW, LLAMA_3_2_1B_PREVIEW, LLAMA_3_2_3B_PREVIEW,
    LLAMA_3_2_70B_SPECDEC, LLAMA_3_2_70B_VERSATILE, LLAMA_3_2_90B_VISION_PREVIEW, LLAMA_3_70B_8192,
    LLAMA_3_8B_8192, LLAMA_GUARD_3_8B, MIXTRAL_8X7B_32768,
};

// Transcription models
pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
pub const DISTIL_WHISPER_LARGE_V3: &str = "distil-whisper-large-v3-en";
