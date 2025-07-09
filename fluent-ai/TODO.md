# TODO.md - Complete Implementation Plan for Fluent-AI

## Overview
This document maps EVERY single action method for every fluent chain in fluent-ai, including all chunk types and polymorphic builders. FileLoader is removed as Document API handles all loading scenarios.

## 1. Chunk Types to Implement

### Core Chunk Types
- **DocumentChunk** - for file/document content
  - path: Option<PathBuf>
  - content: String
  - metadata: HashMap<String, Value>
  - byte_range: Option<(usize, usize)>
  
- **ImageChunk** - for image data
  - data: Vec<u8>
  - format: ImageFormat (PNG, JPEG, GIF, WebP)
  - dimensions: Option<(u32, u32)>
  - metadata: HashMap<String, Value>
  
- **VoiceChunk** - for audio/voice data
  - audio_data: Vec<u8>
  - format: AudioFormat
  - duration_ms: Option<u64>
  - sample_rate: Option<u32>
  - metadata: HashMap<String, Value>
  
- **ChatMessageChunk** - for streaming chat responses (renamed to avoid conflict with existing MessageChunk)
  - content: String
  - role: MessageRole
  - is_final: bool
  - metadata: HashMap<String, Value>
  
- **CompletionChunk** - for streaming completions
  - text: String
  - finish_reason: Option<FinishReason>
  - usage: Option<Usage>
  
- **EmbeddingChunk** - for streaming embeddings
  - embeddings: Vec<f32>
  - index: usize
  - metadata: HashMap<String, Value>

## 2. Document API Enhancements

### Remove FileLoader - Document handles all loading
```rust
// File operations
Document::from_file("path/to/file.txt")
    .on_error(|e| eprintln!("Error: {}", e))
    .load_async() -> AsyncTask<Document>

Document::from_glob("**/*.md")
    .on_error(|e| eprintln!("Error: {}", e))
    .stream() -> AsyncStream<Document>

// URL operations
Document::from_url("https://example.com/data.json")
    .on_error(|e| eprintln!("Error: {}", e))
    .load_async() -> AsyncTask<Document>

// GitHub operations
Document::from_github("owner/repo/path/to/file.rs")
    .on_error(|e| eprintln!("Error: {}", e))
    .load_async() -> AsyncTask<Document>

// Streaming operations
Document::from_glob("**/*.txt")
    .on_error(|e| eprintln!("Error: {}", e))
    .stream_chunks(1024) -> AsyncStream<DocumentChunk>
```

## 3. Polymorphic Builder Pattern Requirements

### Core Principle
Users MUST provide error handlers (on_result/on_error/on_chunk) BEFORE accessing terminal methods.

### Implementation Pattern
```rust
// Stage 1: Initial builder - NO terminal methods available
let builder = Document::from_glob("*.txt")?;

// Stage 2: After error handler - terminal methods NOW available
let stream = builder
    .on_error(|e| eprintln!("Error: {}", e))
    .stream(); // Returns AsyncStream<Document>
```

## 4. Complete Action Method Mapping

### Document Action Methods
- **Terminal methods (require error handler first)**:
  - `load() -> Document` (sync)
  - `load_async() -> AsyncTask<Document>`
  - `stream() -> AsyncStream<Document>`
  - `stream_chunks(size: usize) -> AsyncStream<DocumentChunk>`
  - `stream_lines() -> AsyncStream<DocumentChunk>`

### Image Action Methods
- **Terminal methods (require error handler first)**:
  - `load_async() -> AsyncTask<Image>`
  - `stream() -> AsyncStream<ImageChunk>`
  - `stream_thumbnails(size: (u32, u32)) -> AsyncStream<ImageChunk>`
  - `detect_objects() -> AsyncTask<Vec<DetectedObject>>`

### Audio Action Methods
- **Terminal methods (require error handler first)**:
  - `load_async() -> AsyncTask<Audio>`
  - `stream() -> AsyncStream<VoiceChunk>`
  - `transcribe() -> AsyncTask<String>`
  - `stream_transcription() -> AsyncStream<DocumentChunk>`

### Agent Action Methods
- **Direct terminal methods**:
  - `chat(message: &str) -> AsyncStream<MessageChunk>`
  - `stream_completion(prompt: &str) -> AsyncStream<CompletionChunk>`
  - `on_response<F>(message: &str, handler: F) -> AsyncTask<String>`
  
- **Builder methods** (return new builders):
  - `completion(prompt: &str) -> CompletionRequestBuilder`
  - `conversation() -> ConversationBuilder<M>`

### CompletionRequestBuilder Action Methods
- **Terminal methods (with polymorphic error handling)**:
  - `complete<F>(handler: F) -> AsyncStream<CompletionChunk>`
  - `on_completion<F>(handler: F) -> AsyncTask<String>`
  - `stream() -> AsyncStream<CompletionChunk>`
  - `request() -> CompletionRequest` (builds request object)

### Extractor<T> Action Methods
- **Terminal methods**:
  - `extract(text: &str) -> AsyncTask<T>`
  - `extract_with_context(text: &str, context: Vec<Document>) -> AsyncTask<T>`
  - `on_extraction<F>(text: &str, handler: F) -> AsyncTask<T>`
  - `stream_extract(text: &str) -> AsyncStream<T>`

### Memory Action Methods
- **Direct action methods (not builders)**:
  - `create_memory(node: MemoryNode) -> AsyncTask<MemoryNode>`
  - `get_memory(id: &str) -> AsyncTask<Option<MemoryNode>>`
  - `update_memory(node: MemoryNode) -> AsyncTask<MemoryNode>`
  - `delete_memory(id: &str) -> AsyncTask<bool>`
  - `query_all() -> AsyncStream<MemoryNode>`
  - `query_by_type(memory_type: MemoryType) -> AsyncStream<MemoryNode>`
  - `search_by_content(query: &str) -> AsyncStream<MemoryNode>`
  - `search_by_vector(embedding: Vec<f32>, limit: usize) -> AsyncStream<MemoryNode>`

### Workflow<In, Out> Action Methods
- **Builder methods** (after on_error):
  - `build() -> Workflow<In, Out>` (creates reusable workflow)
  
- **Workflow execution methods**:
  - `execute(input: In) -> AsyncTask<Out>`
  - `stream(inputs: AsyncStream<In>) -> AsyncStream<Out>`
  - `then<Out2>(other: Workflow<Out, Out2>) -> Workflow<In, Out2>` (composition)

### EmbeddingModel Action Methods
- **Terminal methods**:
  - `embed(text: &str) -> AsyncTask<Vec<f32>>`
  - `embed_batch(texts: Vec<String>) -> AsyncStream<EmbeddingChunk>`
  - `on_embedding<F>(text: &str, handler: F) -> AsyncTask<Vec<f32>>`

## 5. Implementation Order

### Phase 1: Core Infrastructure
- [x] NotResult trait preventing Result in AsyncTask/AsyncStream
- [ ] Remove FileLoader completely
- [ ] Implement chunk types in domain module
- [ ] Update Document API with all loading methods

### Phase 2: Document System
- [ ] Document::from_file with error handling
- [ ] Document::from_glob with streaming
- [ ] Document::from_url with async loading  
- [ ] Document::from_github integration
- [ ] Document streaming methods (chunks, lines)

### Phase 3: Media Types
- [ ] Image loading and streaming
- [ ] Audio loading and streaming
- [ ] Video support (future)

### Phase 4: AI Operations
- [ ] Agent chat streaming with MessageChunk
- [ ] Completion streaming with CompletionChunk
- [ ] Extractor with proper error handling
- [ ] Embedding operations

### Phase 5: Advanced Features
- [ ] Memory operations
- [ ] Workflow execution
- [ ] MCP tool integration
- [ ] Conversation management

## 6. Key Constraints

1. **NotResult trait is enforced** - AsyncTask<T> and AsyncStream<T> cannot have T be a Result type
2. **All errors MUST be handled through builder methods** - on_error, on_result, on_chunk
3. **Terminal methods are only available after error handling is specified**
4. **All async operations return AsyncTask<T> or AsyncStream<T>** where T is a clean type, never Result
5. **FileLoader is removed** - all file operations go through Document API

## 7. Testing Strategy

1. **Compile-time tests**:
   - Ensure Result types cannot be used in AsyncTask/AsyncStream
   - Verify terminal methods require error handlers

2. **Integration tests**:
   - Document loading from various sources
   - Streaming operations
   - Error handling paths

3. **End-to-end tests**:
   - Complete fluent chains
   - Real API interactions
   - Performance benchmarks

## 8. Migration Guide

### From FileLoader to Document
```rust
// OLD (FileLoader)
FileLoader::with_glob("*.txt")?
    .read_async()

// NEW (Document)  
Document::from_glob("*.txt")
    .on_error(|e| eprintln!("Error: {}", e))
    .stream()
```

## Current Status
- NotResult trait: ✅ Implemented
- FileLoader removal: ✅ Completed
- Document API enhancement: ✅ Implemented (with polymorphic error handling)
- Chunk types: ✅ Implemented
- Tool import conflicts: ✅ Fixed (disambiguated trait Tool vs struct Tool)
- AgentRole context field: ✅ Made optional (Optional<ZeroOneOrMany<Document>>)
- Compilation errors: ✅ All fixed - builds successfully
- Polymorphic builders: 🔄 In Progress (Document done, others pending)

## Recent Changes
- Removed FileLoader completely from the system
- Enhanced Document API with from_file, from_glob, from_url, from_github
- Implemented all chunk types (DocumentChunk, ImageChunk, VoiceChunk, ChatMessageChunk, CompletionChunk, EmbeddingChunk)
- Added polymorphic builder pattern to Document requiring on_error() before async operations
- Fixed compilation errors related to chunk types and async operations
- Completely redesigned Workflow API:
  - Workflow<In, Out> is now a reusable, composable domain object
  - WorkflowStep<In, Out> for individual transformations
  - ParallelStepsBuilder for parallel execution
  - Polymorphic builder pattern requiring on_error()
  - Clean API with no exposed Box types or Result types