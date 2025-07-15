#![feature(negative_impls)]
#![feature(auto_traits)]

pub mod async_task;
pub use async_task::*;

pub mod sugars;
pub use sugars::{ByteSize, ByteSizeExt};

// McpTool is now in domain module

// one_or_many is exported through sugars module
pub use sugars::ZeroOneOrMany;

// mapmacro is included via sugars module

pub mod collection_ext;
pub use collection_ext::prelude::*;

pub mod loaders;
pub mod prelude;

pub mod chat_loop;
pub mod macros;
pub mod markdown;
pub mod memory;
pub mod streaming;
pub mod workflow;

pub mod domain;
pub mod engine;
pub mod fluent;

// Re-export commonly used sugars items
pub use sugars::*;

// Re-export from fluent_ai_provider crate
pub use fluent_ai_provider::{Model, ModelInfoData, Models, Provider, Providers};
// Re-export domain types
pub use domain::{
    Audio, CompletionRequest, Document, Embedding, Image, Message, MessageChunk, MessageRole,
    audio::{AudioMediaType, ContentFormat as AudioContentFormat},
    image::{ContentFormat as ImageContentFormat, ImageDetail, ImageMediaType},
};

// Re-export traits from domain
pub use domain::{CompletionBackend, CompletionModel};
// Re-export new Tool API
pub use domain::tool_v2::ExecToText;
pub use domain::{Context, Library, NamedTool, Perplexity, Stdio, ToolV2 as Tool};
// Re-export context marker types
pub use domain::context::{Directory, File, Files, Github};

// Re-export engine types
pub use engine::{
    AgentConfig, CompletionResponse, Engine, EngineRegistry, ExtractionConfig, Usage,
};
pub use engine::{get_default_engine, get_engine, register_engine, registry, set_default_engine};

// Memory and workflow modules are already defined above as pub mod

// Macros with #[macro_export] are already available at crate root

// HashMap is available through collection_ext::prelude

// Master builder export
pub use fluent::{Ai, FluentAi};

#[cfg(test)]
mod tests {
    use crate::async_task::AsyncTask;

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
        use crate::domain::chunk::{DocumentChunk, ImageChunk, VoiceChunk};
        use std::collections::HashMap;

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
