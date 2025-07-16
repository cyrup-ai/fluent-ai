// ============================================================================
// File: src/providers/ollama/mod.rs
// ----------------------------------------------------------------------------
// Ollama provider implementation with typestate builder pattern
// ============================================================================

mod client;
mod completion;
mod streaming;

pub use client::{Client, OllamaCompletionBuilder};
pub use completion::{CompletionModel, EmbeddingModel};

// Re-export model constants
pub use completion::{LLAMA3_2, LLAVA, MISTRAL, MISTRAL_MAGISTRAR_SMALL};

// Re-export embedding constants
pub use completion::{ALL_MINILM, NOMIC_EMBED_TEXT};
