//! The module defines the [`ToolSchema`] struct, an embeddable wrapper for any
//! value implementing [`crate::tool::ToolEmbedding`].  It allows tools to be
//! vector-indexed just like plain text so they can participate in RAG flows.
//!
//! A `ToolSchema` captures three things:
//!   • the tool's unique `name`  (mirrors `Tool::NAME`),
//!   • a JSON *context* blob returned by the tool – useful metadata for the
//!     agent when picking among candidate tools, and
//!   • *embedding_docs* – free-form strings that describe *what* the tool does.
//!
//! Those strings are fed into the embedding pipeline via the regular `Embed`
//! trait so they slot seamlessly into the overarching `EmbeddingsBuilder`
//! workflow.
//!
//! # Usage
//! ```rust
//! # use rig::embeddings::tool::ToolSchema;
//! # use rig::tool::{Tool, ToolEmbedding};
//! # use serde_json::json;
//! # use rig::completion::ToolDefinition;
//! # #[derive(Debug, thiserror::Error)]
//! # enum Err { Stub }
//! # struct Ping;
//! impl Tool for Ping {
//!     const NAME: &'static str = "ping";
//!     type Error  = Err;
//!     type Args   = ();
//!     type Output = String;
//!     async fn definition(&self, _: String) -> ToolDefinition {
//!         serde_json::from_value(json!({
//!             "name":        "ping",
//!             "description": "Returns 'pong'.",
//!             "parameters":  {}
//!         })).unwrap()
//!     }
//!     async fn call(&self, _: Self::Args) -> Result<Self::Output, Self::Error> {
//!         Ok("pong".into())
//!     }
//! }
//! impl ToolEmbedding for Ping {
//!     type InitError = Err;
//!     type Context   = serde_json::Value;
//!     type State     = ();
//!     fn init(_: Self::State, _: Self::Context) -> Result<Self, Self::InitError> { Ok(Ping) }
//!     fn embedding_docs(&self) -> Vec<String> { vec!["Responds with 'pong'.".into()] }
//!     fn context(&self) -> Self::Context { json!({ "category": "test" }) }
//! }
//! let schema: ToolSchema = ToolSchema::try_from(&Ping).unwrap();
//! assert_eq!(schema.name, "ping");
//! ```

use serde::Serialize;

use crate::{
    domain::tool::ToolEmbeddingDyn,
    embedding::embed::{Embed, EmbedError, TextEmbedder}};

/// Embeddable representation of a [`Tool`](crate::tool::Tool).
#[derive(Clone, Serialize, Default, Eq, PartialEq)]
pub struct ToolSchema {
    /// Unique tool identifier (mirrors `Tool::NAME`).
    pub name: String,
    /// Arbitrary JSON context emitted by the tool implementation.
    pub context: serde_json::Value,
    /// Human-readable documentation chunks plugged into the vector store.
    pub embedding_docs: Vec<String>}

// -------------------------------------------------------------------------
// Embed implementation – feed docs into `TextEmbedder` untouched.
// ----------------------------------------------------------------------
impl Embed for ToolSchema {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        for doc in &self.embedding_docs {
            embedder.embed(doc.clone());
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------
// Convenient conversion from any dynamic `ToolEmbedding` implementor.
// ----------------------------------------------------------------------
impl ToolSchema {
    /// Build a `ToolSchema` from a dynamic `ToolEmbedding` value.
    #[inline]
    pub fn try_from(tool: &dyn ToolEmbeddingDyn) -> Result<Self, EmbedError> {
        Ok(Self {
            name: tool.name(),
            context: tool.context().map_err(EmbedError::new)?,
            embedding_docs: tool.embedding_docs()})
    }
}

// --------------------------------------------------------------------------
// End of file
// --------------------------------------------------------------------------
