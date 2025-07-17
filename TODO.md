# Updated TODO.md - Clean Streaming Completion Architecture

## OBJECTIVE
Achieve 0 compilation errors and 0 warnings by implementing clean, elegant streaming completions using cyrup_sugars typestate builders, ModelInfo defaults, and fluent_ai_http3.

## ARCHITECTURE NOTES
- **Clean Syntax**: `client.completion_model("gpt-4").system_prompt("You are helpful").temperature(0.8).prompt("Hello")` 
- **ModelInfo Defaults**: ALL properties (max_tokens, temperature, system_prompt, etc.) default from enumerations
- **Terminal Action**: `.prompt()` is the action method that returns unwrapped AsyncStream<CompletionChunk>
- **Optional Callbacks**: `.on_chunk()` is optional with env_logger defaults
- **HTTP3 Direct**: Use fluent_ai_http3 directly, no wrappers
- **Streaming First**: AsyncStream<CompletionChunk> with .collect() option
- **Complete Parameters**: system_prompt, chat_history, documents, tools, temperature, max_tokens, top_p, etc.

---

## Phase 1: ModelInfo Integration & Defaults

### 1. Implement ModelInfo Default Loading System
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/model_info.rs`
**Lines**: Add method to load defaults from model enumerations
**Implementation**: Create `ModelInfo::load_defaults(model: &str) -> ModelConfig` that reads max_tokens, temperature, top_p, default_system_prompt from generated model enumerations. Use domain ModelInfoData structure.
**Architecture**: Centralized configuration loading from dynamically generated enumerations ensures consistent defaults across all providers.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 2. QA: Validate ModelInfo Default System
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify ModelInfo correctly loads defaults from enumerations. Rate default loading implementation quality 1-10. Ensure no unwrap() usage.

---

## Phase 2: OpenAI Reference Implementation

### 3. Implement Clean OpenAI CompletionBuilder
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/openai/completion.rs`
**Lines**: Replace entire file content (lines 1-end)
**Implementation**: 
- Create `OpenAICompletionBuilder<State>` with typestate: `NeedsPrompt` → `Ready`
- Implement `.completion_model(model: &str)` that loads ModelInfo defaults
- Add optional methods with ModelInfo defaults:
  - `.system_prompt(prompt: &str)` - override default system prompt
  - `.temperature(temp: f64)` - override default temperature
  - `.max_tokens(tokens: u32)` - override default max_tokens
  - `.top_p(p: f64)` - override default top_p
  - `.frequency_penalty(penalty: f64)` - override default
  - `.presence_penalty(penalty: f64)` - override default
  - `.chat_history(history: Vec<Message>)` - add conversation context
  - `.documents(docs: Vec<Document>)` - add RAG context
  - `.tools(tools: Vec<ToolDefinition>)` - add function calling
  - `.additional_params(params: Value)` - provider-specific overrides
- Add optional `.on_chunk<F>(callback: F)` that captures user callback
- Implement terminal `.prompt(text: &str) -> AsyncStream<CompletionChunk>`
- Use fluent_ai_http3::HttpClient directly for streaming SSE responses
- Parse OpenAI SSE into CompletionChunk format with no unwrap()
- Default .on_chunk() behavior: Ok(chunk) → chunk.into(), Err(e) → env_logger + BadChunk::from_err(e)
**Architecture**: Reference implementation demonstrating clean syntax with complete parameter support and ModelInfo defaults.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 4. QA: Validate OpenAI Clean Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify clean syntax works: `client.completion_model("gpt-4").system_prompt("Be helpful").prompt("Hello")`. Rate streaming implementation and parameter support 1-10. Ensure no unwrap() usage.

### 5. Update OpenAI Client Factory Methods
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/openai/client.rs`
**Lines**: Update client creation and model factory methods
**Implementation**: 
- Update `Client::new()` to use fluent_ai_http3::HttpClient with HttpConfig::ai_optimized()
- Implement `completion_model(model: &str) -> OpenAICompletionBuilder<NeedsPrompt>`
- Load ModelInfo defaults in completion_model() method
- Add proper error handling for authentication without unwrap()
**Architecture**: Client provides clean factory methods that return builders with ModelInfo defaults loaded.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 6. QA: Validate OpenAI Client Integration
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify client factory methods work with clean syntax. Rate ModelInfo integration and error handling 1-10.

---

## Phase 3: Pattern Replication

### 7. Implement HuggingFace Clean CompletionBuilder
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/huggingface/completion.rs`
**Lines**: Replace all existing completion logic (lines 1-end)
**Implementation**: Apply exact same clean pattern as OpenAI: typestate builder with full parameter support (system_prompt, temperature, max_tokens, chat_history, documents, tools, etc.), ModelInfo defaults, optional .on_chunk(), terminal .prompt(), fluent_ai_http3 streaming. Parse HuggingFace responses into CompletionChunk format.
**Architecture**: Identical clean syntax across all providers with provider-specific streaming response parsing.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 8. QA: Validate HuggingFace Clean Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify HuggingFace follows OpenAI pattern exactly with clean syntax and full parameter support. Rate consistency and streaming quality 1-10.

### 9. Implement DeepSeek Clean CompletionBuilder
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/deepseek/completion.rs`
**Lines**: Replace all completion logic (lines 1-end)
**Implementation**: Apply clean pattern with ModelInfo defaults and full parameter support (system_prompt, chat_history, documents, tools, temperature, max_tokens, etc.). Fix all type resolution errors by using fluent_ai_domain types. Replace DeepSeekCompletionModel with clean builder pattern. Use fluent_ai_http3 for streaming.
**Architecture**: Clean syntax with DeepSeek-specific API format adaptation and complete parameter support.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 10. QA: Validate DeepSeek Clean Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify DeepSeek resolves type errors and implements clean streaming syntax with full parameter support. Rate type safety and consistency 1-10.

### 11. Implement All Remaining Provider Clean Builders
**Files**: 
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/gemini/completion.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/groq/completion.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/mistral/completion.rs`
- All other client completion modules
**Implementation**: Apply identical clean pattern to all remaining providers: ModelInfo defaults, full parameter support (system_prompt, temperature, max_tokens, chat_history, documents, tools, top_p, frequency_penalty, presence_penalty, additional_params), optional .on_chunk(), terminal .prompt(), fluent_ai_http3 streaming, provider-specific response parsing.
**Architecture**: Consistent clean syntax with complete parameter support across entire provider ecosystem.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 12. QA: Validate All Provider Clean Implementations
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify all providers consistently implement clean syntax with ModelInfo defaults and full parameter support. Rate pattern consistency 1-10.

---

## Phase 4: Integration & Validation

### 13. Re-enable Client Factory with Clean Patterns
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/client_factory.rs`
**Lines**: Uncomment and update entire module
**Implementation**: Update factory to create clients that support clean completion_model() syntax with full parameter support. Use AsyncTask patterns. Map Provider enum variants to client instances with ModelInfo integration.
**Architecture**: Factory creates clients with consistent clean completion builder interface supporting all parameters.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 14. QA: Validate Client Factory Clean Integration
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify factory produces clients with clean syntax support and full parameter capabilities. Rate factory implementation quality 1-10.

### 15. Fix All Import Statements for Clean Architecture
**Files**: All client modules and lib.rs
**Implementation**: Ensure all imports reference fluent_ai_domain for types (Message, Document, ToolDefinition, etc.). Update client trait definitions to support clean completion_model() methods with full parameter support. Remove any remaining provider-local domain types.
**Architecture**: Clean separation with proper imports supporting clean completion syntax and all parameters.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 16. QA: Validate Clean Import Architecture
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify imports support clean completion syntax with full parameters. Rate import organization and architecture 1-10.

### 17. Comprehensive Compilation Validation
**Command**: `cargo check --workspace`
**Implementation**: Verify zero compilation errors with clean syntax implementation and full parameter support. Fix any remaining type issues, import problems, or trait mismatches. Ensure ModelInfo defaults work correctly for all parameters.
**Architecture**: Complete clean syntax architecture with full parameter support compiles successfully.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 18. QA: Validate Zero Compilation Errors
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify cargo check returns completely clean build with clean syntax and full parameter support. Rate compilation success 1-10.

### 19. Eliminate All 47 Warnings
**Implementation**: Address remaining warnings while preserving clean syntax functionality and full parameter support. Remove unused imports, dead code, missing docs. Ensure warning-free build with clean completion interface.
**Architecture**: Production-quality clean syntax implementation with full parameters and zero warnings.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 20. QA: Validate Zero Warnings Achievement
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify all warnings eliminated while maintaining clean syntax and full parameter support. Rate warning resolution quality 1-10.

### 21. End-to-End Clean Syntax Validation
**Implementation**: Create test demonstrating clean syntax with full parameters: `client.completion_model("gpt-4").system_prompt("You are helpful").temperature(0.8).chat_history(messages).documents(docs).tools(tools).prompt("Hello")`. Verify ModelInfo defaults load correctly for all parameters. Test streaming with optional .on_chunk(). Validate .collect() functionality.
**Architecture**: Complete clean syntax streaming architecture with full parameter support working end-to-end.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 22. QA: Validate End-to-End Clean Functionality
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify clean syntax works end-to-end with ModelInfo defaults, full parameter support, and streaming. Rate system quality 1-10.

---

## Phase 5: Intelligent Tool Selection Pipeline (Future Enhancement)

### 23. Implement BERT Classifier for Tool Relevance Scoring
**Files**: 
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/tool_selection/classifier.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/tool_selection/pipeline.rs`
**Implementation**: 
- Integrate BERT-based inference pipeline for tool relevance classification
- Create semantic similarity scoring between user prompt and tool descriptions
- Implement `ToolRelevanceClassifier` that scores tools 0.0-1.0 based on prompt context
- Add `classify_tool_relevance(prompt: &str, tools: &[ToolDefinition]) -> Vec<(ToolDefinition, f64)>`
- Support contextual filtering (e.g., "planning" requests exclude `github_commit` tool)
**Architecture**: Pre-completion tool filtering ensures only relevant tools are sent to AI model, staying within 32-tool limit while maximizing utility.

### 24. Implement Dynamic Tool Selection Pipeline
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/tool_selection/selector.rs`
**Implementation**:
- Create `ToolSelector` that reduces N tools to ≤32 based on BERT classifier scores
- Implement smart selection strategies:
  - **Relevance-based**: Top 32 tools by semantic similarity score
  - **Category-balanced**: Ensure representation across tool categories (code, planning, communication, etc.)
  - **Context-aware**: Boost tools based on conversation history and domain context
- Add `select_optimal_tools(prompt: &str, available_tools: Vec<ToolDefinition>) -> Vec<ToolDefinition>`
- Ensure deterministic selection within score tiers for reproducible behavior
**Architecture**: Intelligent tool selection maximizes model effectiveness while respecting 32-tool constraint.

### 25. Integrate Tool Selection into CompletionProvider
**Files**: All provider completion implementations
**Implementation**:
- Modify `tools()` method to accept unlimited tools but intelligently select ≤32
- Add `enable_tool_selection(bool)` configuration option (default: true)
- Implement tool selection pipeline in completion builders:
  ```rust
  fn tools(mut self, tools: ZeroOneOrMany<ToolDefinition>) -> Result<Self, CompletionError> {
      let selected_tools = if self.tool_selection_enabled {
          self.tool_selector.select_optimal_tools(&self.prompt_context, tools)
      } else {
          tools.into_iter().take(32).collect() // Fallback to first 32
      };
      // Store selected tools in ArrayVec<ToolDefinition, 32>
  }
  ```
- Add logging of tool selection decisions for transparency
**Architecture**: Seamless integration maintains existing API while adding intelligent tool curation.

### 26. Implement Tool Selection Caching and Performance
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/tool_selection/cache.rs`
**Implementation**:
- Add LRU cache for BERT inference results to avoid re-computing similar prompts
- Implement `ToolSelectionCache` with configurable TTL and memory limits
- Add batch inference optimization for multiple tool evaluations
- Create tool fingerprinting for efficient cache key generation
- Implement async tool classification to avoid blocking completion pipeline
**Architecture**: High-performance tool selection with sub-millisecond response times through intelligent caching.

### 27. Add Tool Selection Configuration and Observability
**Files**:
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/tool_selection/config.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/provider/src/tool_selection/metrics.rs`
**Implementation**:
- Add `ToolSelectionConfig` with tunable parameters:
  - `relevance_threshold: f64` - Minimum score for tool inclusion
  - `max_tools: usize` - Tool limit (default: 32)
  - `category_balance: bool` - Enable category-balanced selection
  - `cache_enabled: bool` - Enable/disable result caching
- Implement tool selection metrics and observability:
  - Tool selection latency tracking
  - Cache hit/miss rates
  - Tool relevance score distributions
  - Selection decision audit logs
- Add `ToolSelectionMetrics` for performance monitoring
**Architecture**: Production-ready tool selection with full observability and tunable performance characteristics.

### 28. QA: Validate Intelligent Tool Selection System
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify:
- BERT classifier accurately scores tool relevance for different prompt types
- Tool selection pipeline consistently selects appropriate tools within 32-tool limit
- Performance meets sub-millisecond requirements with caching
- Integration preserves existing CompletionProvider API compatibility
- Tool selection decisions are logged and auditable
Rate intelligent tool selection implementation quality 1-10.

---

## Anthropic Advanced Features Integration

### 29. Implement Prompt Caching for Anthropic Provider
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
**Lines**: Add cache_control field to AnthropicCompletionRequest struct and builder methods
**Implementation**: 
- Add `with_prompt_caching()` method to AnthropicCompletionBuilder that sets internal boolean flag
- Modify `build_request()` to include `cache_control: { type: "ephemeral" }` when caching enabled
- Auto-enable caching for large system prompts (>2048 tokens) and context documents
- Add cache_control to system prompt, documents, and tool definitions in request structure
- Handle cache status in streaming responses for observability
- Integrate with existing Context<File> API - large files auto-cached
**Architecture**: Transparent prompt caching that reduces costs for repeated content while maintaining clean builder syntax `.with_prompt_caching()` and seamless Context integration.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 30. QA: Validate Anthropic Prompt Caching Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify prompt caching integrates cleanly with builder syntax, properly structures cache_control in requests, handles large content auto-caching, and maintains cost transparency. Rate implementation quality 1-10.

### 31. Implement Extended Thinking for Anthropic Provider
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
**Lines**: Add thinking field to AnthropicCompletionRequest struct and builder methods
**Implementation**: 
- Add `with_thinking(budget_tokens: Option<u32>)` method to AnthropicCompletionBuilder
- Modify `build_request()` to include `thinking: { type: "enabled", budget_tokens: n }` when thinking enabled
- Default budget_tokens to 1024 when None provided, use custom value when Some(n)
- Extend `parse_sse_chunk()` to handle thinking content blocks in streaming responses
- Add CompletionChunk::Thinking variant or include thinking content in existing text chunks
- Preserve existing `on_chunk(|chunk| { Ok => chunk.into(), Err(e) => ... })` syntax unchanged
- Handle thinking signatures and verification in response parsing
**Architecture**: Transparent extended thinking that provides reasoning transparency while maintaining existing cyrup_sugars on_chunk syntax and streaming patterns.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 32. QA: Validate Anthropic Extended Thinking Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify extended thinking integrates cleanly with builder syntax, properly structures thinking requests, handles streaming thinking content, preserves existing on_chunk syntax, and provides reasoning transparency. Rate implementation quality 1-10.

### 33. Implement Citations for Anthropic Provider
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
**Lines**: Add citations field to AnthropicCompletionBuilder and request processing
**Implementation**: 
- Add `with_citations()` method to AnthropicCompletionBuilder that sets internal boolean flag
- Integrate with existing Context<File>, Context<Files>, Context<Directory> API for automatic source attribution
- Modify document processing in `build_request()` to include source metadata when citations enabled
- Extend `parse_sse_chunk()` to handle citation blocks in streaming responses
- Add citation information to CompletionChunk responses (source file, line numbers, URLs)
- Preserve existing `on_chunk(|chunk| { Ok => chunk.into(), Err(e) => ... })` syntax unchanged
- Map Context sources to Anthropic citation format in request structure
**Architecture**: Transparent citations that provide source attribution for document-based responses while leveraging existing Context API and maintaining clean builder syntax `.with_citations()`.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 34. QA: Validate Anthropic Citations Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify citations integrate cleanly with builder syntax, leverage existing Context API for source attribution, properly handle citation blocks in streaming, preserve existing on_chunk syntax, and provide accurate source metadata. Rate implementation quality 1-10.

### 35. Implement Search Results Processing for Anthropic Provider
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
**Lines**: Add search results field to AnthropicCompletionBuilder and context processing
**Implementation**: 
- Add `with_search_results()` method to AnthropicCompletionBuilder that sets internal boolean flag
- Extend context() method to accept ZeroOneOrMany<SearchResult> alongside existing Context types
- Create SearchResult domain type with title, snippet, url, relevance_score fields
- Modify `build_request()` to format search results into structured context when flag enabled
- Apply Anthropic-specific search result formatting (title, snippet, source URL structure)
- Integrate search results with existing citation system when both flags enabled
- Preserve existing `on_chunk(|chunk| { Ok => chunk.into(), Err(e) => ... })` syntax unchanged
- Handle search result metadata in streaming responses for source attribution
**Architecture**: Transparent search results processing that auto-formats search context using ZeroOneOrMany pattern while maintaining clean builder syntax `.with_search_results()` and existing context API.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 36. QA: Validate Anthropic Search Results Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify search results integrate cleanly with builder syntax, properly format search context, use ZeroOneOrMany pattern correctly, integrate with citation system, preserve existing on_chunk syntax, and provide structured search formatting. Rate implementation quality 1-10.

### 37. Implement Batch Processing for Anthropic Provider
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
**Lines**: Add batch processing methods and separate execution path
**Implementation**: 
- Add `batch_prompts(args...)` method to AnthropicCompletionBuilder using ZeroOneOrMany variadic pattern
- Implement `execute_batch()` method that uses Anthropic's `/v1/messages/batches` endpoint
- Create separate batch request structure with array of individual completion requests
- Handle batch response parsing with multiple completion results
- Preserve all existing builder options (model, temperature, system_prompt, etc.) for batch requests
- Maintain cost optimization through batch API pricing advantages
- Add batch status tracking and completion polling for async batch processing
- Preserve existing `on_chunk(|chunk| { Ok => chunk.into(), Err(e) => ... })` syntax for streaming batch results
**Architecture**: Efficient batch processing that leverages Anthropic's batch API for cost optimization while maintaining clean builder syntax `.batch_prompts("prompt1", "prompt2", "prompt3").execute_batch()` and ZeroOneOrMany variadic pattern.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 38. QA: Validate Anthropic Batch Processing Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify batch processing integrates cleanly with builder syntax, uses ZeroOneOrMany variadic pattern correctly, leverages Anthropic batch API properly, maintains cost optimization, preserves existing on_chunk syntax, and provides efficient batch execution. Rate implementation quality 1-10.

### 39. Implement Multi-lingual Support for Anthropic Provider
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
**Lines**: Add multilingual processing and UTF-8 robustness
**Implementation**: 
- Add `with_multilingual()` method to AnthropicCompletionBuilder that sets internal boolean flag
- Enhance UTF-8 handling throughout request building and response parsing
- Add character encoding validation for system prompts, user prompts, and context documents
- Implement proper Unicode normalization (NFC) for consistent text processing
- Add language detection hints in request metadata when multilingual flag enabled
- Enhance streaming response parsing to handle multi-byte UTF-8 sequences correctly
- Add proper byte boundary handling in SSE chunk parsing to avoid broken Unicode
- Preserve existing `on_chunk(|chunk| { Ok => chunk.into(), Err(e) => ... })` syntax unchanged
- Ensure all text processing maintains Unicode correctness and character integrity
**Architecture**: Robust multilingual support that ensures proper UTF-8 handling and language processing while maintaining clean builder syntax `.with_multilingual()` and existing streaming patterns.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 40. QA: Validate Anthropic Multi-lingual Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify multilingual support integrates cleanly with builder syntax, properly handles UTF-8 encoding, maintains Unicode correctness, handles multi-byte sequences in streaming, preserves existing on_chunk syntax, and provides robust language processing. Rate implementation quality 1-10.

### 41. Implement Automatic Token Counting for Anthropic Provider
**File**: `/Volumes/samsung_t9/fluent-ai/packages/provider/src/clients/anthropic/completion.rs`
**Lines**: Add automatic token counting and cost transparency
**Implementation**: 
- Integrate Anthropic's `/v1/messages/count_tokens` endpoint for pre-request token counting
- Add automatic token counting in `build_request()` method before sending completion request
- Count tokens for system prompt, user prompt, chat history, documents, and tool definitions
- Add token usage tracking in streaming response parsing from usage metadata
- Implement cost calculation based on model pricing (input/output token rates)
- Add token budget validation against model limits with intelligent truncation
- Log token usage and cost information for transparency and optimization
- Preserve existing `on_chunk(|chunk| { Ok => chunk.into(), Err(e) => ... })` syntax unchanged
- Include token count and cost data in completion metadata and error messages
**Architecture**: Transparent automatic token counting that provides cost visibility and request optimization while maintaining existing builder syntax and requiring no user intervention.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### 42. QA: Validate Anthropic Token Counting Implementation
Act as an Objective QA Rust developer and rate the work performed previously on these requirements. Verify automatic token counting integrates seamlessly, provides accurate cost transparency, handles token budget validation, includes proper usage tracking, preserves existing on_chunk syntax, and requires no user syntax changes. Rate implementation quality 1-10.

---

## SUCCESS CRITERIA
- [ ] Clean syntax works: `client.completion_model("gpt-4").system_prompt("Be helpful").prompt("Hello")`
- [ ] All parameters supported: system_prompt, temperature, max_tokens, chat_history, documents, tools, top_p, frequency_penalty, presence_penalty, additional_params
- [ ] All properties default from ModelInfo enumerations
- [ ] .on_chunk() is optional with env_logger defaults
- [ ] .prompt() is terminal action returning unwrapped AsyncStream<CompletionChunk>
- [ ] All HTTP operations use fluent_ai_http3 directly
- [ ] `cargo check --workspace` returns 0 errors, 0 warnings
- [ ] Zero unwrap()/expect() in src/* code
- [ ] AsyncTask patterns throughout
- [ ] End-to-end streaming functionality with clean syntax and full parameter support