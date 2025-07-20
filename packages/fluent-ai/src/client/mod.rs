pub mod builder;
pub use builder::*;

// Conversion trait implementations moved to proper trait impls
// No exposed macros

/// Base trait for provider clients
pub trait ProviderClient: Send + Sync {
    /// Get the provider name
    fn provider_name(&self) -> &'static str;

    /// Create client from environment variables
    fn from_env() -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        Self: Sized;
}

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "audio")]
pub use audio::{AsAudioGeneration, *};

pub mod completion;
pub use completion::{AsCompletion, *};

pub mod embeddings;
pub use embeddings::{AsEmbeddings, *};

#[cfg(feature = "image")]
pub mod images;
#[cfg(feature = "image")]
pub use images::{AsImageGeneration, *};

pub mod transcription;
pub use transcription::{AsTranscription, *};
