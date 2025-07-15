pub mod client;
pub mod completion;
pub mod embedding;

pub use client::{Client, MistralCompletionBuilder};
pub use completion::{
    CompletionModel, CODESTRAL, CODESTRAL_MAMBA, MINISTRAL_3B, MINISTRAL_8B, MISTRAL_LARGE,
    MISTRAL_NEMO, MISTRAL_SABA, MISTRAL_SMALL, PIXTRAL_LARGE, PIXTRAL_SMALL,
};
pub use embedding::{EmbeddingModel, MISTRAL_EMBED};
