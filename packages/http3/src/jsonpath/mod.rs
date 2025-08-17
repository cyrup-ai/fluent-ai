//! High-performance JSONPath streaming deserializer for Http3
//!
//! This module provides blazing-fast, zero-allocation JSONPath expression evaluation
//! over streaming HTTP responses. It enables streaming individual array elements from
//! nested JSON structures like OpenAI's `{"data": [...]}` format.
//!
//! # Features
//!
//! - Full JSONPath specification support
//! - Zero-allocation streaming deserialization
//! - Lock-free concurrent processing
//! - Comprehensive error handling and recovery
//! - Integration with Http3 builder pattern
//!
//! # Examples
//!
//! ```rust
//! use fluent_ai_http3::Http3;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Model {
//!     id: String,
//!     object: String,
//! }
//!
//! // Stream individual models from OpenAI's {"data": [...]} response
//! Http3::json()
//!     .array_stream("$.data[*]")
//!     .bearer_auth(&api_key)
//!     .get("https://api.openai.com/v1/models")
//!     .on_chunk(|model: Model| {
//!         Ok => model.into(),
//!         Err(e) => BadChunk::from_err(e)
//!     })
//!     .collect_or_else(|error| Model::default());
//! ```

pub mod buffer;
pub mod core_evaluator;
pub mod deserializer;
pub mod error;
pub mod filter;
pub mod functions;
pub mod json_array_stream;
pub mod normalized_paths;
pub mod null_semantics;
pub mod parser;
pub mod safe_parsing;
pub mod state_machine;
pub mod stats;
pub mod stream_processor;
pub mod type_system;

// Decomposed parser modules
pub mod ast;
pub mod compiler;
pub mod expression;
pub mod filter_parser;
pub mod selector_parser;
pub mod tokenizer;
pub mod tokens;

pub use self::{
    buffer::{JsonBuffer, StreamBuffer},
    core_evaluator::CoreJsonPathEvaluator,
    deserializer::{JsonPathDeserializer, JsonPathIterator, StreamingDeserializer},
    error::{JsonPathError, JsonPathResult},
    filter::FilterEvaluator,
    functions::FunctionEvaluator,
    json_array_stream::{JsonArrayStream, StreamStats},
    parser::{
        ComparisonOp,
        ComplexityMetrics,
        FilterExpression,
        FilterValue,
        FunctionSignature,
        // RFC 9535 Implementation Types
        FunctionType,
        JsonPathExpression,
        JsonPathParser,
        JsonSelector,
        NormalizedPath,
        NormalizedPathProcessor,
        NullSemantics,
        PathSegment,
        PropertyAccessResult,
        TypeSystem,
        TypedValue,
    },
    safe_parsing::{SafeParsingContext, SafeStringBuffer, Utf8Handler, Utf8RecoveryStrategy},
    state_machine::{JsonStreamState, StreamStateMachine},
    stream_processor::JsonStreamProcessor,
};

#[cfg(test)]
mod tests {
    // Tests for JSON path streaming functionality will be implemented here
}
