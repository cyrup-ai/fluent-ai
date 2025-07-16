// ============================================================================
// File: src/embeddings/embed.rs
// ----------------------------------------------------------------------------
// Zero-overhead primitives for turning arbitrary data into text snippets that
// an embedding model can consume.
// ============================================================================

use serde_json::Value as Json;

// ---------------------------------------------------------------------------
// 0. Error type
// ---------------------------------------------------------------------------

/// Error surfaced by any `Embed::embed` implementation.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl EmbedError {
    #[inline(always)]
    pub fn new<E: std::error::Error + Send + Sync + 'static>(err: E) -> Self {
        Self(Box::new(err))
    }
}

// ---------------------------------------------------------------------------
// 1. Core trait
// ---------------------------------------------------------------------------

/// Convert `Self` into one or more strings ready for vectorisation.
///
/// Implementations must _only_ push to the provided `TextEmbedder`; doing any
/// heavy-weight work (e.g. network, file-IO) here would block the fast path.
pub trait Embed {
    fn embed(&self, dst: &mut TextEmbedder) -> Result<(), EmbedError>;
}

// ---------------------------------------------------------------------------
// 2. Accumulator
// ---------------------------------------------------------------------------

/// Collects raw strings that an `EmbeddingModel` should process.
///
/// Internally just a `Vec<String>` – no extra indirections, no locking.
#[derive(Default)]
pub struct TextEmbedder {
    pub(crate) texts: Vec<String>,
}

impl TextEmbedder {
    #[inline(always)]
    pub fn embed(&mut self, text: impl Into<String>) {
        self.texts.push(text.into());
    }
}

/// Utility: eagerly extract the string list from any `Embed`.
#[inline]
pub fn to_texts(item: impl Embed) -> Result<Vec<String>, EmbedError> {
    let mut acc = TextEmbedder::default();
    item.embed(&mut acc)?;
    Ok(acc.texts)
}

// ---------------------------------------------------------------------------
// 3. Blanket helpers & primitive implementations
// ---------------------------------------------------------------------------

macro_rules! impl_embed_to_string {
    ($($t:ty),+ $(,)?) => {$(
        impl Embed for $t {
            #[inline(always)]
            fn embed(&self, dst: &mut TextEmbedder) -> Result<(), EmbedError> {
                dst.embed(self.to_string());
                Ok(())
            }
        }
    )+};
}

impl_embed_to_string!(
    String, &str, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64, bool, char
);

impl Embed for Json {
    #[inline(always)]
    fn embed(&self, dst: &mut TextEmbedder) -> Result<(), EmbedError> {
        dst.embed(serde_json::to_string(self).map_err(EmbedError::new)?);
        Ok(())
    }
}

impl<T: Embed> Embed for &T {
    #[inline(always)]
    fn embed(&self, dst: &mut TextEmbedder) -> Result<(), EmbedError> {
        (*self).embed(dst)
    }
}

impl<T: Embed> Embed for Vec<T> {
    #[inline]
    fn embed(&self, dst: &mut TextEmbedder) -> Result<(), EmbedError> {
        for item in self {
            item.embed(dst)?
        }
        Ok(())
    }
}
