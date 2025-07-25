pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_info;

pub use client::{Client, MistralCompletionBuilder};
pub use completion::{
    CODESTRAL, CODESTRAL_MAMBA, CompletionModel, MINISTRAL_3B, MINISTRAL_8B, MISTRAL_LARGE,
    MISTRAL_NEMO, MISTRAL_SABA, MISTRAL_SMALL,
    MistralCompletionBuilder as NewMistralCompletionBuilder, PIXTRAL_LARGE, PIXTRAL_SMALL,
    available_mistral_models, mistral_completion_builder};
pub use embedding::{EmbeddingModel, MISTRAL_EMBED};
pub use model_info::{
    Codestral, CodestralMamba, Ministral3B, Ministral8B, MistralLarge, MistralNemo, MistralSaba,
    MistralSmall, PixtralLarge, PixtralSmall};
