pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_info;

pub use client::{Client, MistralCompletionBuilder};
pub use completion::{
    CompletionModel, CODESTRAL, CODESTRAL_MAMBA, MINISTRAL_3B, MINISTRAL_8B, MISTRAL_LARGE,
    MISTRAL_NEMO, MISTRAL_SABA, MISTRAL_SMALL, PIXTRAL_LARGE, PIXTRAL_SMALL,
    MistralCompletionBuilder as NewMistralCompletionBuilder, mistral_completion_builder,
    available_mistral_models,
};
pub use embedding::{EmbeddingModel, MISTRAL_EMBED};
pub use model_info::{
    MistralLarge, Codestral, PixtralLarge, MistralSaba,
    Ministral3B, Ministral8B, MistralSmall, PixtralSmall,
    MistralNemo, CodestralMamba,
};
