pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_info;

pub use client::{Client, MistralCompletionBuilder};
pub use completion::{
    CompletionModel, MistralCompletionBuilder as NewMistralCompletionBuilder,
    mistral_completion_builder,
};
pub use fluent_ai_domain::context::provider::EmbeddingModel;
// Model constants removed - use model-info package exclusively
