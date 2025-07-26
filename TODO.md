# TODO: Builder Trait Relocation Implementation

**APPROVED PLAN EXECUTION** - Relocate ALL Builder structs from domain crate to fluent-ai crate with trait-based patterns.

## CONSTRAINTS (CRITICAL)
- ✅ Zero allocation, blazing-fast performance
- ✅ No unsafe code, no unchecked operations  
- ✅ No locking mechanisms
- ✅ Elegant ergonomic code design
- ❌ NEVER use `unwrap()` or `expect()` in src/* code
- ✅ Full optimization implementation (no "future enhancements")
- ✅ All code complete with semantic error handling

---

## PHASE 2: Memory Builder Traits Implementation

### Task 2.1: Create Memory Builder Trait Foundation
**Files to Modify:**
- Create: `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/memory.rs`
- Modify: `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/mod.rs` (add memory module)

**Technical Specifications:**
- Create `MemoryNodeBuilder` trait with `impl Trait` returns following AgentRoleBuilder pattern
- Create `MemorySystemBuilder` trait with `impl Trait` returns
- Hidden implementation structs: `MemoryNodeBuilderImpl`, `MemorySystemBuilderImpl`
- Use `ZeroOneOrMany<T>` for collections (never `Option<ZeroOneOrMany<T>>`)
- All methods return `Result<T, E>` with comprehensive error handling
- Preserve zero-allocation patterns with `Arc<str>` and atomic operations
- Inline all hot paths with `#[inline(always)]`

**Architecture Notes:**
- Public traits: `pub trait MemoryNodeBuilder { fn with_content(self, content: MemoryContent) -> impl MemoryNodeBuilder; }`
- Hidden structs: `struct MemoryNodeBuilderImpl { content: ZeroOneOrMany<MemoryContent>, ... }`
- Immutable builder pattern: `..self` syntax for state updates
- Domain type preservation: All `fluent_ai_domain::memory::*` types maintained

### Task 2.2: Implement MemoryNodeBuilder Trait Methods
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/memory.rs` (lines 50-200)

**Technical Specifications:**
- Convert from current struct methods (domain/src/memory/primitives/node.rs:594-730)
- Methods: `with_id()`, `with_memory_type()`, `with_content()`, `with_text()`, `with_embedding()`, `with_importance()`, `with_keyword()`, `with_tag()`, `with_custom_metadata()`, `build()`
- State management using `ZeroOneOrMany::with_pushed()` for collections
- Validation in `build()` method with `MemoryResult<MemoryNode>`
- Performance: Stack-allocated temporary vectors, zero-copy where possible

### Task 2.3: Implement MemorySystemBuilder Trait Methods  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/memory.rs` (lines 200-350)

**Technical Specifications:**
- Convert from current struct methods (domain/src/memory/mod.rs:159-213)
- Methods: `with_database_config()`, `with_vector_config()`, `with_llm_config()`, `with_cognitive()`, `with_compatibility_mode()`, `build()`
- Configuration validation in `build()` method
- Support all specialized configurations: `for_semantic_search()`, `for_realtime_chat()`, `for_large_scale()`

---

## PHASE 3: Chat Builder Traits Implementation

### Task 3.1: Create Chat Builder Directory Structure
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/mod.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/config.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/realtime.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/streaming.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/typing.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/templates.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/macros.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/search.rs`

**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/mod.rs` (add chat module)

### Task 3.2: Implement ConfigurationBuilder and PersonalityConfigBuilder Traits
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/config.rs`

**Technical Specifications:**
- Convert from domain struct patterns (domain/src/chat/config.rs:1374-1495)
- `ConfigurationBuilder` trait: `temperature()`, `max_tokens()`, `system_prompt()`, `model_name()`, `build()`
- `PersonalityConfigBuilder` trait: `personality_type()`, `response_style()`, `expertise_level()`, `creativity_level()`, `formality_level()`, `build()`
- Zero-allocation with `Arc<str>` for string storage
- Range validation for numeric parameters (temperature 0.0-2.0, creativity 0.0-1.0)

### Task 3.3: Implement RealTimeSystemBuilder Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/realtime.rs`

**Technical Specifications:**
- Convert from domain struct (domain/src/chat/realtime.rs:1255-1369)
- Methods: `buffer_size()`, `max_concurrent_streams()`, `heartbeat_interval()`, `message_queue_size()`, `auto_reconnect()`, `build()`
- Use `Duration` for time-based configurations
- `AtomicUsize` for concurrent counters with relaxed ordering
- Validate buffer sizes and queue limits

### Task 3.4: Implement LiveMessageStreamerBuilder Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/streaming.rs`

**Technical Specifications:**
- Convert from domain struct (domain/src/chat/realtime/streaming.rs:622-685)
- Methods: `chunk_size()`, `stream_buffer_size()`, `enable_compression()`, `backpressure_threshold()`, `build()`
- `AsyncStream<T>` integration for streaming patterns
- Backpressure handling with atomic thresholds

### Task 3.5: Implement TypingIndicatorBuilder Trait  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/typing.rs`

**Technical Specifications:**
- Convert from domain struct (domain/src/chat/realtime/typing.rs:476-535)
- Methods: `show_typing()`, `typing_timeout()`, `auto_hide()`, `custom_indicator()`, `build()`
- `Duration` for timeout configurations
- Custom indicator validation

### Task 3.6: Implement TemplateBuilder Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/templates.rs`

**Technical Specifications:**
- Convert from domain struct (domain/src/chat/templates/mod.rs:102-189)
- Methods: `name()`, `content()`, `variables()`, `category()`, `build()`
- Convert `Vec<String>` to `ZeroOneOrMany<TemplateVariable>`
- Template validation with variable placeholder checking

### Task 3.7: Implement MacroBuilder Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/macros.rs`

**Technical Specifications:**
- Convert from domain struct (domain/src/chat/macros.rs:764-852)
- Methods: `name()`, `description()`, `actions()`, `triggers()`, `conditions()`, `dependencies()`, `execution_config()`, `build()`
- Convert multiple `Vec<T>` fields to `ZeroOneOrMany<T>`
- Macro action validation and circular dependency detection

### Task 3.8: Implement HistoryManagerBuilder Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/chat/search.rs`

**Technical Specifications:**
- Convert from domain pattern (domain/src/chat/search.rs)
- Methods: `max_history_size()`, `search_indexing()`, `compression_enabled()`, `retention_policy()`, `build()`
- Search indexing configuration with performance tuning

### Task 3.9: Merge ConversationBuilder with Existing Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/conversation.rs` (merge functionality)

**Technical Specifications:**
- Analyze existing fluent-ai conversation builder (fluent-ai/src/builders/conversation.rs)
- Merge domain ConversationBuilder functionality (domain/src/chat/conversation/mod.rs:502-597)
- Preserve trait-based pattern while adding domain functionality
- Convert `Vec<(String, MessageRole)>` to `ZeroOneOrMany<(String, MessageRole)>`
- Maintain backward compatibility

---

## PHASE 4: Completion Builder Traits Implementation

### Task 4.1: Create Completion Builder Trait Foundation with Lifetime Support
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/completion.rs`

**Technical Specifications:**
- Handle complex lifetime parameters from analysis
- `CompletionResponseBuilder<'a>` trait with lifetime preservation
- `CompletionCoreRequestBuilder<'a>` trait with `SmallVec<&'a str, MAX_STOP_TOKENS>` support
- Associated types for lifetime-bound returns
- Zero-allocation performance preservation for Core builders

### Task 4.2: Implement CompletionRequestBuilder Trait
**Files to Modify:**  
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/completion.rs` (lines 50-150)

**Technical Specifications:**
- Convert from domain struct (domain/src/completion/request.rs:40-202)
- Methods: `system_prompt()`, `chat_history()`, `documents()`, `tools()`, `temperature()`, `max_tokens()`, `chunk_size()`, `additional_params()`, `build()`
- Preserve `ZeroOneOrMany<T>` usage (already correct in domain)
- Validation with `CompletionRequestError`

### Task 4.3: Implement CompletionResponseBuilder<'a> Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/completion.rs` (lines 150-250)

**Technical Specifications:**
- Convert from domain struct (domain/src/completion/response.rs:34-162)
- Preserve `<'a>` lifetime parameter throughout trait definition
- Methods: `text()`, `model()`, `provider()`, `usage()`, `finish_reason()`, `build()`
- Handle `Cow<'a, str>` zero-copy patterns
- Lifetime propagation in `impl Trait` returns

### Task 4.4: Implement CompactCompletionResponseBuilder Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/completion.rs` (lines 250-350)

**Technical Specifications:**
- Convert from domain struct (domain/src/completion/response.rs:181-280)
- Methods: `content()`, `model()`, `provider()`, `tokens_used()`, `finish_reason()`, `response_time_ms()`, `build()`
- **CRITICAL**: `build()` returns `AsyncStream<CompactCompletionResponse>` 
- Use `fluent_ai_async::AsyncStream::with_channel` pattern
- `Arc<str>` for shared ownership

### Task 4.5: Implement CompletionCoreRequestBuilder<'a> Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/completion.rs` (lines 350-450)

**Technical Specifications:**
- Convert from domain struct (domain/src/completion/candle.rs:122-245)
- **CRITICAL**: Zero-allocation with `ArrayVec`, `SmallVec` preservation
- Complex lifetime: `SmallVec<&'a str, MAX_STOP_TOKENS>` handling
- Methods: `prompt()`, `max_tokens()`, `temperature()`, `top_k()`, `top_p()`, `stop_tokens()`, `stream()`, `model_params()`, `seed()`, `build()`
- All methods `#[inline(always)]` for performance
- Stack allocation validation

### Task 4.6: Implement CompletionCoreResponseBuilder Trait
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/completion.rs` (lines 450-550)

**Technical Specifications:**
- Convert from domain struct (domain/src/completion/candle.rs:328-417)
- Zero-allocation with `ArrayVec` containers
- Methods: `text()`, `tokens_generated()`, `generation_time_ms()`, `tokens_per_second()`, `finish_reason()`, `model()`, `build()`
- Atomic operations integration
- Stack allocation validation

---

## PHASE 5: Agent/Model Builder Merge Implementation

### Task 5.1: Analyze and Plan AgentBuilder Merge Strategy
**Files to Analyze:**
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/agent/builder.rs` (domain struct)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs` (existing traits)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/builder.rs` (typestate builder)

**Technical Specifications:**
- Create unified trait hierarchy preserving trait-based patterns
- Hybrid approach: trait-based public API + struct-based performance implementation
- Preserve const generic parameters for zero-allocation
- Resolve conflicts between static trait objects vs generic types

### Task 5.2: Implement Unified AgentBuilder Trait System
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs` (enhance existing traits)

**Technical Specifications:**
- Merge domain's `ArrayVec<McpToolData, TOOLS_CAPACITY>` with fluent-ai flexibility
- Preserve atomic statistics and lock-free patterns
- Adapter pattern for API compatibility
- Error handling unification with `AgentBuilderError`

### Task 5.3: Create ModelInfoBuilder Trait
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/model.rs`

**Technical Specifications:**
- Convert from domain struct (domain/src/model/info.rs)
- Merge with existing fluent-ai model builder (fluent-ai/src/builders/model.rs)
- Methods: `provider_name()`, `name()`, `capabilities()`, `pricing()`, `yaml_compatibility()`, `validation()`, `build()`
- Preserve YAML deserialization compatibility
- Unified error handling with `ModelError`

---

## PHASE 6: ToolBuilder Typestate Implementation

### Task 6.1: Create ToolBuilder Typestate Foundation
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/tool.rs`

**Technical Specifications:**
- Sophisticated typestate pattern with compile-time safety
- State types: `Named`, `WithDependency`, `WithRequestSchema`, `WithResultSchema`, `WithInvocation`
- Public trait: `pub trait ToolBuilder<State> { fn named(name: impl Into<String>) -> impl ToolBuilder<Named>; }`
- Hidden implementation: `struct ToolBuilderImpl<State> { _phantom: PhantomData<State>, ... }`
- Type-safe state transitions

### Task 6.2: Implement ToolBuilder Core Methods
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/tool.rs` (lines 100-200)

**Technical Specifications:**
- Follow exact API from domain/tests/toolregistry_tests.rs
- Methods: `named()`, `description()`, `with<D>()`, `request_schema<T>()`, `result_schema<T>()`
- Support `SchemaType` variants from domain
- `ZeroOneOrMany<T>` for dependency collections
- Compile-time state validation

### Task 6.3: Implement ToolBuilder Invocation and Build Methods
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/tool.rs` (lines 200-300)

**Technical Specifications:**
- Methods: `on_invocation<F>()`, `on_error<F>()`, `build()`, `into_typed_tool()`
- Support both sync and async invocation patterns from tests
- Create `TypedTool` return type with full functionality
- Comprehensive error handling with `Result<T, E>`

### Task 6.4: Implement ToolBuilder Registry Integration
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/tool.rs` (lines 300-400)

**Technical Specifications:**
- `TypedToolStorage` integration following domain/tests/toolregistry_tests.rs
- Tool registration with duplicate detection
- Memory management with zero-allocation patterns
- Registry functionality: add, remove, search, validate

---

## PHASE 7: Domain Cleanup and Re-export Updates

### Task 7.1: Remove Builder Re-exports from Domain
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/lib.rs` (remove builder re-exports)
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/memory/mod.rs` (remove MemoryNodeBuilder, MemorySystemBuilder)
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/chat/mod.rs` (remove all 9 chat builders)
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/completion/mod.rs` (remove all 5 completion builders)
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/agent/mod.rs` (remove AgentBuilder)
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/model/mod.rs` (remove ModelBuilder, ModelInfoBuilder)

**Technical Specifications:**
- Add deprecation warnings: `#[deprecated(since = "1.0.0", note = "Use fluent_ai::builders::* instead")]`
- Ensure domain contains only pure types and traits
- Clean architectural separation

### Task 7.2: Add Trait Re-exports to Fluent-AI
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/lib.rs` (add builder trait re-exports)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/mod.rs` (organize trait re-exports)

**Technical Specifications:**
- Logical organization by functional area: memory, chat, completion, model, tool
- Proper trait visibility: `pub use builders::memory::*;`
- External API access patterns

### Task 7.3: Update Global Import Statements
**Files to Search and Modify:**
- All crates in workspace: provider, async, http3, etc.
- Update: `use fluent_ai_domain::*Builder` → `use fluent_ai::builders::*Builder`
- Search pattern: `rg "fluent_ai_domain.*Builder" --type rust`

**Technical Specifications:**
- Comprehensive import updates across entire codebase
- Verify trait imports work correctly
- Update external dependencies

---

## PHASE 8: Testing and Validation

### Task 8.1: Update Domain Tests for Trait Usage
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/domain/tests/toolregistry_tests.rs` (update ToolBuilder import)
- All other test files using builders

**Technical Specifications:**
- Import ToolBuilder trait from fluent-ai
- Verify all trait-based builder usage
- Maintain exact functionality with trait patterns

### Task 8.2: Comprehensive Architecture Validation
**Technical Specifications:**
- Verify no circular dependencies: `cargo check --workspace`
- Dependency graph analysis
- Clean architectural separation verification
- Trait-based compilation order validation

### Task 8.3: Performance Benchmarking
**Technical Specifications:**
- Benchmark trait-based vs struct-based performance
- Zero-allocation verification
- Memory usage profiling
- Compilation time impact assessment

---

## IMPLEMENTATION ORDER

1. **Start with Memory Builders** (simplest, good foundation)
2. **Chat Builders** (moderate complexity, good learning)
3. **Completion Builders** (complex lifetime management)
4. **Agent/Model Merge** (most complex architectural work)
5. **ToolBuilder Typestate** (most sophisticated implementation)
6. **Domain Cleanup** (straightforward but critical)
7. **Testing and Validation** (comprehensive verification)

## SUCCESS CRITERIA

- ✅ All builders relocated from domain to fluent-ai crate
- ✅ All builders converted to trait-based patterns with `impl Trait` returns
- ✅ Zero-allocation performance maintained
- ✅ No circular dependencies
- ✅ All tests pass
- ✅ Full functionality preservation
- ✅ Clean architectural separation achieved