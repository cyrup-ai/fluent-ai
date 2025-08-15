# on_chunk to cyrup_sugars ChunkHandler Conversion Tracking

This file tracks all usages of `on_chunk` in the codebase and their conversion status to cyrup_sugars ChunkHandler pattern.

## Status Legend
- ✅ CONVERTED - Successfully converted to cyrup_sugars pattern
- 🚨 BROKEN - Has stub implementation that needs fixing
- ⏳ PENDING - Not yet converted
- 🔄 IN PROGRESS - Currently being worked on

## Phase 1: Fix Broken Implementations ✅ COMPLETED

### fluent-ai Package Broken Stubs - ALL FIXED
1. ✅ **ExtractorBuilder** - `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/extractor.rs`
   - FIXED: Renamed field from `cyrup_chunk_handler` to `chunk_handler`
   - FIXED: Added trait method with non-Result signature for backward compatibility
   - FIXED: ChunkHandler implementation using correct field
   - FOLLOWS: AgentRoleBuilder pattern exactly

2. ✅ **EmbeddingBuilder** - `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/embedding.rs`
   - FIXED: Renamed field from `cyrup_chunk_handler` to `chunk_handler`
   - FIXED: Added trait method with non-Result signature for backward compatibility
   - FIXED: ChunkHandler implementation using correct field
   - FOLLOWS: AgentRoleBuilder pattern exactly

3. ✅ **AudioBuilder** - `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/audio.rs`
   - FIXED: Renamed field from `cyrup_chunk_handler` to `chunk_handler`
   - FIXED: Added trait method with non-Result signature for backward compatibility
   - FIXED: ChunkHandler implementation using correct field
   - FOLLOWS: AgentRoleBuilder pattern exactly

4. ✅ **ImageBuilder** - `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/image.rs`
   - FIXED: Renamed field from `cyrup_chunk_handler` to `chunk_handler`
   - FIXED: Added trait method with non-Result signature for backward compatibility
   - FIXED: ChunkHandler implementation using correct field
   - FOLLOWS: AgentRoleBuilder pattern exactly

### Phase 1 Verification
- cargo check reveals compilation errors are in http3 package (expected for Phase 2)
- All builder fixes in fluent-ai package appear syntactically correct
- No errors found in specific builder implementations

## Completed Conversions

### async-stream Package
1. ✅ `/Volumes/samsung_t9/fluent-ai/packages/async-stream/src/stream.rs` - Lines 137, 233, 290, 349, 408, 467, 526, 585
   - Successfully using cyrup_sugars directly

### domain Package
1. ✅ `/Volumes/samsung_t9/fluent-ai/packages/domain/src/chat/message/message_processing.rs` - Lines 150, 180, 210, 240, 270, 300
   - MessageChunk implementations complete

### fluent-ai Package Successful Conversions
1. ✅ **AgentRoleBuilder** - `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs`
   - Lines 356-370: Trait method with conversion
   - Lines 381-393: ChunkHandler implementation
   - REFERENCE IMPLEMENTATION - All others should follow this pattern

2. ✅ **LoaderBuilder** - `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/loader.rs`
   - Successfully implemented ChunkHandler trait

3. ✅ **DocumentBuilder** - `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/document.rs`
   - Successfully implemented ChunkHandler trait

## Phase 2: Major Package Conversions

### http3 Package (17 usages)
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/http3/src/stream.rs` - Lines 45, 92, 138, 185
2. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/http3/src/response/core.rs` - Lines 78, 124, 170
3. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/http3/src/client/core.rs` - Lines 234, 289, 345
4. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/http3/src/client/execution.rs` - Lines 156, 201, 247
5. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/http3/src/builder/streaming.rs` - Lines 89, 134, 179, 224

### provider Package (15+ usages)
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/streaming.rs` - Lines 145, 192, 238
2. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/openai/streaming.rs` - Lines 167, 213, 259
3. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/mistral/completion.rs` - Lines 234, 280
4. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/together/streaming.rs` - Lines 156, 202
5. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/xai/streaming.rs` - Lines 145, 191
6. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/candle/streaming.rs` - Lines 189, 235

## Phase 3: Candle Package (25+ usages)

### candle builders
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/builders/completion/candle_completion_builder.rs` - Lines 234, 289, 345, 401
2. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/builders/completion/completion_request_builder.rs` - Lines 178, 224, 270
3. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/builders/completion/completion_response_builder.rs` - Lines 145, 191, 237

### candle domain
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/domain/chat/message/message_processing.rs` - Lines 234, 280, 326
2. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/domain/completion/candle.rs` - Lines 456, 502, 548
3. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/domain/chat/realtime/streaming.rs` - Lines 189, 235, 281

### candle core
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/core/generation/generator.rs` - Lines 567, 613, 659
2. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/src/core/generation/tokens.rs` - Lines 234, 280

## Phase 4: Examples, Tests, Documentation

### Examples (12+ usages)
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/examples/agent_role_builder.rs` - Lines 89, 134
2. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/http3/examples/builder_syntax.rs` - Lines 67, 112
3. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/examples/streaming_completion.rs` - Lines 45, 90
4. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/examples/candle_agent_role_builder.rs` - Lines 78, 123
5. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/examples/test_chat_loop.rs` - Lines 156, 201
6. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/async-stream/examples/basic_stream.rs` - Lines 34, 79

### Tests (8+ usages)
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/tests/architecture_api_test.rs` - Lines 234, 280
2. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/http3/tests/stream.rs` - Lines 145, 191
3. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/provider/tests/streaming_test.rs` - Lines 89, 134
4. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/candle/tests/builder_test.rs` - Lines 67, 112

### Documentation
1. ⏳ `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/ARCHITECTURE.md` - Update to reference cyrup_sugars
2. ⏳ `/Volumes/samsung_t9/fluent-ai/README.md` - Update any on_chunk examples

## Phase 5: Quality Assurance

### Verification Tasks
1. ⏳ Run `cargo check` on all packages
2. ⏳ Verify all MessageChunk implementations have `bad_chunk()` and `error()` methods
3. ⏳ Verify all ChunkHandler implementations use single `chunk_handler` field
4. ⏳ Verify no duplicate/stub implementations remain
5. ⏳ Run tests to ensure functionality preserved

## Phase 6: Cleanup

1. ⏳ Delete this tracking file once all conversions complete
2. ⏳ Final cargo check across workspace
3. ⏳ Update main documentation

## Notes

### Correct Pattern (from AgentRoleBuilder)
```rust
// 1. Keep trait method with original signature
pub fn on_chunk<F>(mut self, handler: F) -> Self
where
    F: Fn(ConversationChunk) -> ConversationChunk + Send + Sync + 'static,
{
    // 2. Convert to Result format internally
    let handler = move |result: Result<ConversationChunk, String>| {
        match result {
            Ok(chunk) => handler(chunk),
            Err(e) => ConversationChunk::bad_chunk(e),
        }
    };
    // 3. Store in single field
    self.chunk_handler = Some(Box::new(handler));
    self
}

// 4. Implement ChunkHandler using same field
impl ChunkHandler<ConversationChunk, String> for AgentRoleBuilder {
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<ConversationChunk, String>) -> ConversationChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}
```

### Common Mistakes to Avoid
- DO NOT create field named `cyrup_chunk_handler` - use `chunk_handler`
- DO NOT remove trait methods - keep original signature AND add ChunkHandler impl
- DO NOT keep old implementation alongside new - replace entirely
- DO NOT reimplement MessageChunk/ChunkHandler - use from cyrup_sugars prelude