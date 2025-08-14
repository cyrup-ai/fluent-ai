# ON_CHUNK_CONVERSION.md - Tracking File

## Files requiring on_chunk conversion to cyrup_sugars pattern

### async-stream package (Priority 1 - Core Implementation)
- [ ] ./packages/async-stream/src/builder.rs:91 - ChunkHandler trait implementation
- [ ] ./packages/async-stream/src/chunk_handler.rs - Remove local trait definition, use cyrup_sugars
- [ ] ./packages/async-stream/src/lib.rs:8 - Update lib documentation 
- [ ] ./packages/async-stream/src/macros.rs:19,24 - Update macro documentation
- [ ] ./packages/async-stream/examples/collect_or_else_pattern.rs:52 - Example usage
- [ ] ./packages/async-stream/examples/on_chunk_pattern.rs:1,3,59,70,72,76,82,105,149,150 - Multiple usages
- [ ] ./packages/async-stream/tests/on_chunk_pattern_test.rs:1,3,28,49,124,140 - Test usages

### fluent-ai package (Priority 2 - Main API)
- [ ] ./packages/fluent-ai/src/builders/agent_role.rs:124,353,412 - AgentRoleBuilder implementations
- [ ] ./packages/fluent-ai/src/builders/loader.rs:45,199 - LoaderBuilder implementations
- [ ] ./packages/fluent-ai/src/builders/document.rs:70,298 - DocumentBuilder implementations
- [ ] ./packages/fluent-ai/src/builders/extractor.rs:40,146 - ExtractorBuilder implementations
- [ ] ./packages/fluent-ai/src/builders/embedding.rs:31,119 - EmbeddingBuilder implementations
- [ ] ./packages/fluent-ai/src/builders/audio.rs:32,139 - AudioBuilder implementations
- [ ] ./packages/fluent-ai/src/builders/image.rs:41,173 - ImageBuilder implementations
- [ ] ./packages/fluent-ai/src/agent/builder.rs:388 - Agent builder implementation
- [ ] ./packages/fluent-ai/examples/agent_role_builder.rs:40 - Example usage
- [ ] ./packages/fluent-ai/tests/architecture_api_test.rs:57 - Test usage

### candle package (Priority 3 - Candle Integration)
- [ ] ./packages/candle/src/main.rs:99 - Main function on_chunk usage
- [ ] ./packages/candle/src/agent/builder.rs:15,376 - Agent builder
- [ ] ./packages/candle/src/builders/agent_role.rs:185,380,567,671,672,742,743 - AgentRole builder
- [ ] ./packages/candle/src/builders/loader.rs:47,201 - Loader builder
- [ ] ./packages/candle/src/builders/document.rs:69,298 - Document builder
- [ ] ./packages/candle/src/builders/embedding.rs:29,117 - Embedding builder
- [ ] ./packages/candle/src/builders/audio.rs:30,137 - Audio builder
- [ ] ./packages/candle/src/builders/extractor.rs:41,147 - Extractor builder
- [ ] ./packages/candle/src/builders/image.rs:39,171 - Image builder
- [ ] ./packages/candle/src/domain/http/responses/completion.rs:1702 - Domain completion test
- [ ] ./packages/candle/src/domain/chat/integrations.rs:190,196,470 - Chat integrations
- [ ] ./packages/candle/src/domain/chat/macros.rs:1029 - Chat macros
- [ ] ./packages/candle/src/domain/chat/message/message_processing.rs:18,25,38,41,129 - Message processing
- [ ] ./packages/candle/src/http/responses/completion.rs:1702 - HTTP completion test
- [ ] ./packages/candle/src/providers/qwen3_coder.rs:348,349 - Qwen provider
- [ ] ./packages/candle/examples/candle_agent_role_builder.rs:51,81,154 - Example

### http3 package (Priority 4 - HTTP Integration)
- [ ] ./packages/http3/src/lib.rs:16,56 - Library documentation
- [ ] ./packages/http3/src/builder/streaming.rs:106 - Streaming builder
- [ ] ./packages/http3/src/response/body.rs:58,59,87,91,109,110,111,113,133 - Response body on_chunk
- [ ] ./packages/http3/src/stream.rs:247 - Stream on_chunk
- [ ] ./packages/http3/src/json_path/mod.rs:32 - JSONPath documentation
- [ ] ./packages/http3/examples/fluent_builder.rs:372,373,380,388 - Builder examples
- [ ] ./packages/http3/tests/hyper/brotli.rs:170 - Hyper brotli test
- [ ] ./packages/http3/tests/hyper/deflate.rs:171 - Hyper deflate test
- [ ] ./packages/http3/tests/hyper/gzip.rs:172 - Hyper gzip test
- [ ] ./packages/http3/tests/hyper/zstd.rs:163,195 - Hyper zstd tests
- [ ] ./packages/http3/tests/rfc9535_streaming_behavior.rs:189,201,289,299 - RFC streaming tests

### provider package (Priority 5 - Provider Integration)
- [ ] ./packages/provider/src/streaming_completion_provider.rs:42,137 - Streaming provider
- [ ] ./packages/provider/src/completion_provider.rs:143,148 - Completion provider
- [ ] ./packages/provider/src/clients/anthropic/streaming.rs:141 - Anthropic streaming
- [ ] ./packages/provider/src/clients/anthropic/requests.rs:107 - Anthropic requests
- [ ] ./packages/provider/src/clients/anthropic/completion.rs:8,297 - Anthropic completion
- [ ] ./packages/provider/src/clients/anthropic/tools/core.rs:224,231 - Anthropic tools
- [ ] ./packages/provider/src/clients/openai/streaming.rs:321,345,728,729,825,826 - OpenAI streaming
- [ ] ./packages/provider/src/clients/openai/completion.rs:8,361 - OpenAI completion
- [ ] ./packages/provider/src/clients/xai/completion.rs:232,236,354,359,387 - XAI completion
- [ ] ./packages/provider/src/clients/mistral/completion.rs:515,833 - Mistral completion
- [ ] ./packages/provider/src/clients/huggingface/completion.rs:8,336 - HuggingFace completion
- [ ] ./packages/provider/src/clients/candle/client.rs:941,945 - Candle client
- [ ] ./packages/provider/src/clients/together/completion.rs:273 - Together completion

### domain package (Priority 6 - Domain Models)
- [ ] ./packages/domain/src/http/responses/completion.rs:1650 - HTTP completion response test
- [ ] ./packages/domain/src/chat/message/message_processing.rs:18,26,38,41 - Message processing functions

### simd package (Priority 7 - SIMD Optimizations)
- [ ] ./packages/simd/src/similarity/simd/aarch64/neon.rs:33,130 - NEON SIMD implementation

## Conversion Pattern Requirements

Based on cyrup_sugars documentation, the correct pattern should be:
- on_chunk takes Result<T, E> and returns T
- Uses cyrup_sugars conversion with .into()
- Error handling via on_error handlers
- MessageChunk types for streaming data

## Implementation Notes

1. **Add cyrup_sugars dependency**: Add `cyrup_sugars = { git = "https://github.com/cyrup-ai/cyrup-sugars", branch = "main", features = ["all"] }` to Cargo.toml of all packages
2. **Remove local implementations**: Delete local MessageChunk and ChunkHandler trait definitions 
3. **Import from cyrup_sugars**: Use `use cyrup_sugars::prelude::*` to get MessageChunk and ChunkHandler traits
4. **Implement MessageChunk**: For each chunk type, implement MessageChunk with `bad_chunk()` and `error()` methods
5. **Convert builders**: Update all builders to use ChunkHandler<T, E> pattern from cyrup_sugars
6. **Update usage sites**: Change from macro patterns to Result-based closure patterns
7. **Update documentation**: Replace references to local implementations with cyrup_sugars usage
8. **Run tests**: Ensure all conversions compile and work correctly

## Conversion Pattern Examples

### Before (local implementation):
```rust
.on_chunk(|chunk| match chunk {
    Ok(data) => data,
    Err(e) => MyChunk::bad_chunk(e)
})
```

### After (cyrup_sugars):
```rust
use cyrup_sugars::prelude::*;

impl MessageChunk for MyChunk {
    fn bad_chunk(error: String) -> Self { /* ... */ }
    fn error(&self) -> Option<&str> { /* ... */ }
}

impl ChunkHandler<MyChunk> for MyBuilder {
    fn on_chunk<F>(mut self, handler: F) -> Self
    where F: Fn(Result<MyChunk, String>) -> MyChunk + Send + Sync + 'static
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}
```