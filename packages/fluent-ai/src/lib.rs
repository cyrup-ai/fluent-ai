#![feature(negative_impls)]
#![feature(auto_traits)]

//! # CRITICAL: ASYNC FRAMEWORK USAGE
//!
//! ⚠️⚠️⚠️ DO NOT IMPORT FROM cyrup-agent - IT WILL BE DELETED! ⚠️⚠️⚠️
//!
//! ALL async operations must use fluent-ai's async primitives:
//! ```rust
//! use fluent_ai::{AsyncTask, AsyncStream, spawn_async};  // ✅ CORRECT
//! // use cyrup_agent::runtime::{AsyncTask, AsyncStream};  // ❌ WRONG - WILL BREAK!
//! ```
//!
//! Cyrup-agent's battle-tested async primitives have been COPIED INTO fluent-ai.
//! The cyrup-agent crate will be deleted after this conversion is complete.

pub mod async_task;
pub use async_task::{AsyncTask, AsyncStream, spawn_async, NotResult};

// Re-export cyrup_sugars crate items (explicit imports to avoid conflicts)
pub use cyrup_sugars::{ByteSize, ByteSizeExt, OneOrMany, ZeroOneOrMany};
pub use cyrup_sugars::builders::*;
pub use cyrup_sugars::r#async::{AsyncResult, AsyncResultChunk, FutureExt, StreamExt};

// mapmacro is included via sugars module

// collection_ext comes from cyrup_sugars

pub mod loaders;
pub mod prelude;

pub mod chat;
pub mod macros;
pub mod markdown;
pub mod memory;
pub mod streaming;
pub mod workflow;

// Re-export domain and provider types
pub use fluent_ai_domain as domain;
pub use fluent_ai_provider as provider;
pub mod engine;
pub mod fluent;

// High-performance provider implementations
pub mod completion;

// Vector store implementations
pub mod vector_store;

// Additional modules
pub mod agent;
pub mod audio_generation;
pub mod builders;
pub mod client;
pub mod embedding;
pub mod extractor;
pub mod http;
pub mod providers;
pub mod runtime;
pub mod transcription;

// Utility modules for provider implementations and core systems
pub mod util;

// Re-export utility modules at crate root for compatibility
pub mod json_util {
    pub use crate::util::json_util::*;
}

// Re-export message types from domain at crate root for compatibility
pub mod message {
    pub use crate::domain::message::*;
}

// Re-export embeddings module for compatibility
pub mod embeddings {
    pub use crate::embedding::*;
}

// Re-export OneOrMany as one_or_many module for compatibility
pub mod one_or_many {
    pub use crate::OneOrMany;
    pub use crate::ZeroOneOrMany;
    
    use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
    use std::convert::Infallible;
    use std::fmt;
    use std::marker::PhantomData;
    use std::str::FromStr;
    use serde::Deserialize;
    
    /// Serde deserializer for OneOrMany<T> that accepts either a single string or array
    pub fn string_or_one_or_many<'de, T, D>(deserializer: D) -> Result<OneOrMany<T>, D::Error>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
        D: Deserializer<'de>,
    {
        struct StringOrOneOrMany<T>(PhantomData<fn() -> T>);

        impl<'de, T> Visitor<'de> for StringOrOneOrMany<T>
        where
            T: Deserialize<'de> + FromStr<Err = Infallible> + Clone,
        {
            type Value = OneOrMany<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("string or array of strings")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(OneOrMany::from(T::from_str(value).map_err(|_| {
                    E::custom("Failed to parse string")
                })?))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                OneOrMany::try_from(items).map_err(|_| de::Error::custom("Cannot create OneOrMany with empty vector"))
            }
        }

        deserializer.deserialize_any(StringOrOneOrMany(PhantomData))
    }
}

// Re-export commonly used sugars items
// Re-export context marker types
pub use domain::context::{Directory, File, Files, Github};
// Re-export new Tool API
pub use domain::tool::ExecToText;
// Re-export domain types
pub use domain::{
    audio::{AudioMediaType, ContentFormat as AudioContentFormat},
    image::{ContentFormat as ImageContentFormat, ImageDetail, ImageMediaType},
    Audio, CompletionRequest, Document, Embedding, Image, Message, MessageChunk, MessageRole,
};
// Re-export traits from domain
pub use domain::{CompletionBackend, CompletionModel};
pub use domain::{Context, Library, NamedTool, Perplexity, Stdio, ToolV2 as Tool};
// Re-export engine types
pub use engine::{get_default_engine, get_engine, register_engine, registry, set_default_engine};
pub use engine::{
    AgentConfig, CompletionResponse, Engine, EngineRegistry, ExtractionConfig, Usage,
};
// Memory and workflow modules are already defined above as pub mod

// Macros with #[macro_export] are already available at crate root

// HashMap is available through collection_ext::prelude

// Master builder export
pub use fluent::{Ai, FluentAi};
// Re-export from fluent_ai_provider crate
pub use fluent_ai_provider::{Model, ModelInfoData, Models, Provider, Providers};
// Note: cyrup_sugars is already re-exported at the top of the file

#[cfg(test)]
mod tests {
    use crate::async_task::AsyncTask;
    use crate::{AsyncStream, CompletionRequest, Document};

    /// Test that Result types cannot be used in AsyncTask - this should fail to compile
    #[test]
    fn test_negative_bounds_prevent_result_in_async_task() {
        // This should work - String is NotResult
        let _task: AsyncTask<String> = AsyncTask::from_value("test".to_string());

        // These lines would cause compilation errors due to NotResult constraint:
        // let _task: AsyncTask<Result<String, String>> = AsyncTask::from_value(Ok("test".to_string()));
        // let _stream: AsyncStream<Result<String, String>> = AsyncStream::from_chunks(vec![Ok("test".to_string())]);
    }

    /// Test that polymorphic builders require error handling before terminal methods
    #[test]
    fn test_polymorphic_builder_requires_error_handling() {
        // Document builder requires on_error before terminal methods
        let _builder = Document::from_text("test content");
        // This would work: builder.on_error(|e| eprintln!("Error: {}", e)).load();
        // This would NOT compile: builder.load(); // <- missing on_error

        // Test that on_error is required
        let document_with_handler =
            Document::from_text("test content").on_error(|e| eprintln!("Error: {}", e));

        // Now terminal methods are available
        let _doc = document_with_handler.load_async();
    }

    /// Test Completion polymorphic builder
    #[test]
    fn test_completion_polymorphic_builder() {
        // CompletionRequest builder requires on_error before terminal methods
        let _builder = CompletionRequest::prompt("Complete this");

        // Test that on_error enables terminal methods
        let completion_with_handler = CompletionRequest::prompt("Complete this")
            .temperature(0.7)
            .on_error(|e| eprintln!("Error: {}", e));

        // Now terminal methods are available
        let _request = completion_with_handler.request();
    }

    /// Test that NotResult trait is correctly implemented
    #[test]
    fn test_not_result_trait_implementation() {
        use crate::async_task::NotResult;

        // Test that normal types implement NotResult
        fn assert_not_result<T: NotResult>() {}

        assert_not_result::<String>();
        assert_not_result::<i32>();
        assert_not_result::<Vec<u8>>();
        assert_not_result::<Option<String>>();

        // These would fail compilation:
        // assert_not_result::<Result<String, String>>();
        // assert_not_result::<std::result::Result<i32, &str>>();
    }

    /// Test streaming operations with chunks
    #[test]
    fn test_streaming_with_chunks() {
        use std::collections::HashMap;

        use crate::domain::chunk::{DocumentChunk, ImageChunk, VoiceChunk};

        // Test that chunk types work with AsyncStream (using default empty streams for tests)
        let _doc_stream: AsyncStream<DocumentChunk> = AsyncStream::default();
        let _image_stream: AsyncStream<ImageChunk> = AsyncStream::default();
        let _voice_stream: AsyncStream<VoiceChunk> = AsyncStream::default();

        // Test chunk type creation
        let _doc_chunk = DocumentChunk {
            path: None,
            content: "test content".to_string(),
            byte_range: None,
            metadata: HashMap::new(),
        };

        let _image_chunk = ImageChunk {
            data: vec![1, 2, 3],
            format: crate::domain::chunk::ImageFormat::PNG,
            dimensions: Some((100, 100)),
            metadata: HashMap::new(),
        };
    }
}
