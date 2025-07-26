# FLUENT-AI WARNINGS AND ERRORS

## CRITICAL ERRORS (MUST FIX)

1. **fluent_ai_http3**
   - [ ] E0284: Type annotations needed for `AsyncStreamSender<T, _>` in `builder.rs` (multiple locations)
   - [ ] E0277: The trait bound `AsyncStreamSender<T, _>: Send` is not satisfied in `builder.rs`
   - [ ] E0308: Mismatched types in AsyncStream closure parameters
   - [ ] E0283: Type annotations needed for const generic parameter `CAP` in AsyncStream

2. **model-info build.rs**
   - [ ] E0432: Unresolved import `proc_macro2`
   - [ ] E0599: `HttpStream` is not an iterator (multiple locations)
   - [ ] E0308: Mismatched types in HashMap (HeaderName vs String)

3. **fluent-ai-candle**
   - [ ] E0432: Unresolved import `crate::domain::completion::model`

## WARNINGS (MUST FIX)

1. **fluent_ai_domain** (185 warnings)
   - [ ] Missing documentation for structs and fields
   - [ ] Missing documentation for methods and associated functions
   - [ ] Missing documentation for variants

2. **fluent-ai-candle**
   - [ ] Missing documentation for public items
   - [ ] Unused imports and variables
   - [ ] Dead code that needs implementation

## FLUENT-AI ARCHITECTURAL IMPROVEMENTS

**CRITICAL ARCHITECTURE UPGRADE** - Zero-allocation, zero-locking, zero-`dyn` implementation

## CORE PRINCIPLES

- **ZERO ALLOCATION**: No heap allocations in hot paths
- **ZERO LOCKING**: No mutexes, RwLocks, or atomic operations in hot paths
- **ZERO `dyn`**: No dynamic dispatch in hot paths
- **ZERO UNSAFE**: No `unsafe` blocks outside of well-audited system code
- **ZERO MACROS**: No procedural macros in public API
- **ZERO UNWRAP**: No `unwrap()` or `expect()` in production code

## FLUENT-AI-CANDLE: PRODUCTION-READY IMPROVEMENTS

### Memory Management Optimizations

#### Memory Management - Implementation Files

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/core.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/agent.rs`

#### Memory Management - Technical Specifications

1. Replace `Arc<dyn Model>` with generic type parameter `M: Model` in `Agent` struct
2. Implement arena allocation for short-lived objects in hot paths
3. Add memory pooling for frequent allocations
4. Implement zero-copy deserialization for model loading
5. Add memory usage tracking with `std::alloc::GlobalAlloc` wrapper

### 1. Async Stream Improvements (Core Implementation)

#### 1.1 Core Async Stream - Rust Implementation Files

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/completion.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/streaming.rs`

#### 1.2 Core Async Stream - Technical Implementation

1. Implement backpressure handling with bounded channels
2. Add stream batching for small chunks
3. Implement stream cancellation
4. Add stream metrics and monitoring
5. Optimize for zero-copy chunk processing

### 2. Documentation and Examples for Fluent-AI-Candle

#### 2.1 Documentation - Files to Update

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/examples/`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/README.md`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/lib.rs`

#### 2.2 Documentation - Enhancement Details

1. Add comprehensive API documentation with examples
2. Create usage examples for common scenarios
3. Add performance characteristics and benchmarks
4. Document memory management patterns
5. Add error handling guidelines

---

## PHASE 0: CRITICAL COMPILATION FIXES (URGENT - MUST EXECUTE FIRST)

### Task 0.1: Fix Missing Struct Fields in AgentRoleBuilderImpl

#### AgentRoleBuilder - Rust Implementation Files (Phase 0.1)

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs` (lines 104-116)

#### AgentRoleBuilder Implementation Details (Phase 0.1)

- **COMPILATION ERROR**: Lines 250, 350 reference non-existent struct fields
- Add missing fields to `AgentRoleBuilderImpl` struct using const generics:

```rust
struct AgentRoleBuilderImpl<const MAX_CONTEXTS: usize = 8, const MAX_TOOLS: usize = 8> {
    name: ArrayString<64>, // Stack-allocated string
    completion_provider: Option<CompletionProvider>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    system_prompt: Option<ArrayString<1024>>,
    contexts: ArrayVec<String, MAX_CONTEXTS>,
    tools: ArrayVec<String, MAX_TOOLS>,
    mcp_servers: Option<ArrayVec<McpServerConfig, 4>>,
    additional_params: Option<phf::Map<&'static str, &'static str>>,
    memory: Option<ArrayString<128>>,
    metadata: Option<phf::Map<&'static str, &'static str>>,
    on_tool_result_handler: Option<fn(&str) -> ()>,
    on_conversation_turn_handler: Option<fn(&AgentConversation, &AgentRoleAgent) -> ()>,
}
```

#### Error Handling

- Use `Option` for all optional fields
- Replace `Box<dyn Fn...>` with function pointers
- Use `ArrayString` and `ArrayVec` for stack allocation
- Use `phf` for compile-time maps

### Task 0.2: Fix Constructor Field Initialization

#### Phase 0.2 - Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs` (lines 118-135)

#### Phase 0.2 - Technical Specifications

Initialize all fields in `AgentRoleBuilderImpl::new()`:

```rust
fn new(name: impl Into<ArrayString<64>>) -> Self {
    Self {
        name: name.into(),
        completion_provider: None,
        temperature: None,
        max_tokens: None,
        system_prompt: None,
        contexts: ArrayVec::new(),
        tools: ArrayVec::new(),
        mcp_servers: None,
        additional_params: None,
        memory: None,
        metadata: None,
        on_tool_result_handler: None,
        on_conversation_turn_handler: None,
    }
}
```

## PHASE 1: ZERO-ALLOCATION IMPLEMENTATION

### Task 1.1: Replace Dynamic Dispatch with Static Dispatch

#### Static Dispatch - Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/builder.rs`

#### Static Dispatch - Technical Specifications

1. Replace all trait objects with generic parameters
2. Use `const` generics for array sizes
3. Replace `Box<dyn Error>` with custom error enums
4. Use `ArrayString` and `ArrayVec` from `arrayvec` crate
5. Replace `HashMap` with `phf` for compile-time maps

### Task 1.2: Remove All Locks and Atomics

#### Lock Removal - Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/agent.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/builder.rs`

#### Lock Removal - Technical Specifications

1. Replace `Mutex`/`RwLock` with message passing
2. Use `crossbeam-channel` for inter-thread communication
3. Implement work-stealing for parallel processing
4. Use `parking_lot` for any required synchronization

## PHASE 2: PERFORMANCE OPTIMIZATIONS

### Task 2.1: Implement Zero-Copy Parsing

#### Zero-Copy Parsing - Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/prompt.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/completion.rs`

#### Zero-Copy Parsing - Technical Specifications

1. Use `bytes::Bytes` for zero-copy parsing
2. Implement `Borrow<str>` for string types
3. Use `Cow<'_, str>` for string operations
4. Implement `From<&str>` for owned types

### Task 2.2: Optimize Memory Layout

#### Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/agent.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/builder.rs`

#### Technical Specifications

1. Use `#[repr(C)]` for FFI compatibility
2. Implement `Copy` and `Clone` for small types
3. Use `#[inline(always)]` for hot functions
4. Implement `Default` for all config types

## PHASE 3: ERROR HANDLING

### Task 3.1: Implement Comprehensive Error Types

#### Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/error.rs`

#### Technical Specifications

1. Define error enums with `thiserror`
2. Implement `From` for error conversions
3. Add context to all errors
4. Implement `Display` and `Error` for all error types

### Task 3.2: Add Error Recovery

#### Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/agent.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/agent/builder.rs`

#### Technical Specifications

1. Implement retry logic for recoverable errors
2. Add circuit breakers for external services
3. Implement backoff strategies
4. Add metrics for error rates

## PHASE 4: TESTING AND BENCHMARKING

### Task 4.1: Add Comprehensive Tests

#### Test Files to Create/Modify (Phase 4.1)

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/tests/agent_tests.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/benches/agent_benchmarks.rs`

#### Test Implementation Specifications (Phase 4.1)

1. Unit tests for all public APIs
2. Integration tests for end-to-end flows
3. Benchmarks for hot paths
4. Fuzz testing for input validation

### Task 4.2: Performance Profiling

#### Profiling Scripts to Create/Modify (Phase 4.2)

- `/Volumes/samsung_t9/fluent-ai/scripts/profile.sh`
- `/Volumes/samsung_t9/fluent-ai/scripts/benchmark.sh`

#### Profiling Implementation Specifications (Phase 4.2)

1. Add flamegraph generation
2. Add memory profiling
3. Add CPU profiling
4. Add I/O profiling

## DEPENDENCY UPDATES

### Task 5.1: Update Cargo.toml

#### Files to Modify

- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/Cargo.toml`

#### Technical Specifications

1. Add required dependencies:

```toml
[dependencies]
arrayvec = { version = "0.7", features = ["serde"] }
phf = { version = "0.11", features = ["macros"] }
crossbeam-channel = "0.5"
parking_lot = "0.12"
bytes = "1.0"
thiserror = "1.0"
```

1. Add dev dependencies:

```toml
[dev-dependencies]
criterion = { version = "0.4", features = ["html_reports"] }
proptest = "1.0"
```

## DOCUMENTATION

### Task 6.1: Update Documentation

#### Documentation - Files to Modify

- `/Volumes/samsung_t9/fluent-ai/ARCHITECTURE.md`
- `/Volumes/samsung_t9/fluent-ai/README.md`

#### Documentation - Technical Specifications

1. Document zero-allocation guarantees
2. Document thread safety
3. Add performance characteristics
4. Add examples of proper usage

## VALIDATION

### Task 7.1: Static Analysis

#### Static Analysis - Commands to Run

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo miri test
cargo udeps
```

### Task 7.2: Dynamic Analysis

#### Dynamic Analysis - Commands to Run

```bash
cargo test --all-features
cargo bench
valgrind --tool=memcheck --leak-check=full target/debug/fluent-ai
```

## DELIVERY

### Task 8.1: Create Release Notes
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/RELEASES.md`

**Technical Specifications:**
1. Document all breaking changes
2. List all new features
3. Document performance improvements
4. Include upgrade instructions

### Task 8.2: Update Examples
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/examples/chat_loop_example.rs`
- `/Volumes/samsung_t9/fluent-ai/examples/agent_builder_example.rs`

**Technical Specifications:**
1. Update examples to use new APIs
2. Add error handling
3. Add performance tips
4. Add best practices

## POST-RELEASE

### Task 9.1: Monitor Performance
**Tools to Use:**
- Prometheus
- Grafana
- Jaeger

**Metrics to Track:**
1. Memory usage
2. CPU usage
3. Error rates
4. Latency percentiles

### Task 9.2: Gather Feedback
**Channels to Monitor:**
- GitHub Issues
- Discord
- User surveys
- Performance benchmarks

**Areas of Focus:**
1. Performance regressions
2. API ergonomics
3. Documentation gaps
4. Feature requests

## PHASE 0: CRITICAL COMPILATION FIXES (URGENT - MUST EXECUTE FIRST)

### Task 0.1: Fix Missing Struct Fields in AgentRoleBuilderImpl
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs` (lines 104-116)

**Technical Specifications:**
- **COMPILATION ERROR**: Lines 250, 350 reference non-existent struct fields
- Add missing fields to `AgentRoleBuilderImpl` struct:
  ```rust
  struct AgentRoleBuilderImpl {
      name: String,
      completion_provider: Option<String>,
      temperature: Option<f64>,
      max_tokens: Option<u64>,
      system_prompt: Option<String>,
      contexts: Vec<String>,
      tools: Vec<String>,
      mcp_servers: Option<ZeroOneOrMany<McpServerConfig>>,
      additional_params: Option<HashMap<String, Value>>,
      memory: Option<String>,
      metadata: Option<HashMap<String, Value>>,
      // ADD THESE MISSING FIELDS:
      on_tool_result_handler: Option<Box<dyn FnMut(String) + Send + 'static>>,
      on_conversation_turn_handler: Option<Box<dyn Fn(&AgentConversation, &AgentRoleAgent) + Send + Sync + 'static>>,
  }
  ```

**Error Handling:**
- Initialize fields as `None` in `AgentRoleBuilderImpl::new()`
- Replace unwrap/expect with proper error propagation
- Use `Box<dyn Fn...>` temporarily for compilation fix (will be optimized in Phase 1)

### Task 0.2: Fix Constructor Field Initialization  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs` (lines 118-135)

**Technical Specifications:**
- Add missing field initialization in `AgentRoleBuilderImpl::new()`:
  ```rust
  Self {
      name: name.into(),
      completion_provider: None,
      temperature: None,
      max_tokens: None,
      system_prompt: None,
      contexts: Vec::new(),
      tools: Vec::new(),
      mcp_servers: None,
      additional_params: None,
      memory: None,
      metadata: None,
      // ADD THESE:
      on_tool_result_handler: None,
      on_conversation_turn_handler: None,
  }
  ```

### Task 0.3: Fix AsyncStream Collection Pattern in Examples
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/examples/chat_loop_example.rs` (line 50)

**Technical Specifications:**
- **SYNTAX MATCH FIX**: Current code correctly uses `.collect()` - NO CHANGE NEEDED
- Verify ARCHITECTURE.md examples use `.collect()` pattern consistently  
- Current: `.chat("Hello") // AsyncStream<MessageChunk>.collect();` - CORRECT
- Ensure no `.?` operator usage with AsyncStream returns

### Task 0.4: Verify Domain Import Dependencies  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai/src/builders/agent_role.rs` (lines 7-16)

**Technical Specifications:**
- **CRITICAL**: Ensure all required domain types are imported
- Add missing imports if referenced in lines 250, 350:
  ```rust
  use fluent_ai_domain::{
      AgentConversation, AgentConversationMessage, AgentRole, AgentRoleAgent, AgentRoleImpl,
      ChatMessageChunk, CompletionProvider, Context, Tool, McpServer, Memory, 
      AdditionalParams, Metadata, Conversation, MessageRole,
      ZeroOneOrMany, AsyncStream
  };
  ```
- Verify all type references resolve correctly

## CONSTRAINTS (CRITICAL)
- Zero allocation, blazing-fast performance
- No unsafe code, no unchecked operations  
- No locking mechanisms
- Elegant ergonomic code design
- NEVER use `unwrap()` or `expect()` in src/* code
- Full optimization implementation (no "future enhancements")
- All code complete with semantic error handling

---

## CRITICAL REMAINING TASKS

### Task R.1: Fix ContextArgs trait signature in domain package
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/agent/types.rs` (line 61)

**Technical Specifications:**
- Change `fn add_to(self, contexts: &mut Vec<String>)` to `fn add_to(self, contexts: &mut ZeroOneOrMany<Context>)`
- Add required import: `use crate::context::Context`

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task R.2: Fix ToolArgs trait signature in domain package  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/domain/src/agent/types.rs` (line 67)

**Technical Specifications:**
- Change `fn add_to(self, tools: &mut Vec<String>)` to `fn add_to(self, tools: &mut ZeroOneOrMany<Tool>)`
- Add required import: `use crate::tool::Tool`

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task R.3: Test ARCHITECTURE.md compilation
**Technical Specifications:**
- Create test to verify all three ARCHITECTURE.md examples compile exactly as written
- Verify zero compilation errors or warnings
- Confirm all syntax works including =>, {}, variadic args

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### Task R.4: Final verification and QA
**Technical Specifications:**
- Verify perfect line-by-line syntax compliance with ARCHITECTURE.md
- Confirm all domain objects are used correctly throughout  
- Validate zero architectural violations remain
- Check that user requirements are 100% satisfied

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

---

# FLUENT-AI-CANDLE STANDALONE PACKAGE IMPLEMENTATION

**CRITICAL OBJECTIVE** - Create production-quality standalone fluent-ai-candle package with complete object graph copy, systematic Candle prefixing, and kimi_k2 model integration.

## CANDLE PACKAGE CORE PRINCIPLES
- **COMPLETE INDEPENDENCE**: Zero dependencies on fluent-ai-domain or fluent-ai packages
- **SYSTEMATIC PREFIXING**: ALL domain objects and builders renamed with "Candle" prefix
- **ARCHITECTURE PRESERVATION**: Maintain exact trait-based zero-Box architecture
- **SYNTAX COMPATIBILITY**: Preserve exact ARCHITECTURE.md syntax patterns with Candle prefixes
- **KIMI_K2 INTEGRATION**: Working local model inference with Candle ML framework

---

## PHASE C1: SYSTEMATIC CANDLE PREFIX RENAMING (IN PROGRESS)

### Task C1.1: Complete Message Types Candle Renaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/message/mod.rs` (lines 100-300)

**Technical Specifications:**
- **STATUS**: Partially complete - CandleMessage, CandleMessageRole, CandleMessageChunk done
- **REMAINING**: Complete media types, error types, processing functions with Candle prefixes:
  ```rust
  // Update media types (lines 105-242)
  pub enum CandleMediaType { ... }
  pub enum CandleImageMediaType { ... }
  pub enum CandleDocumentMediaType { ... }
  pub enum CandleAudioMediaType { ... }
  pub enum CandleVideoMediaType { ... }
  
  // Update error types (lines 245-270)  
  pub enum CandleMessageError { ... }
  
  // Update processing functions (lines 273-295)
  pub fn candle_process_message(...) -> Result<(), CandleMessageError>
  pub fn candle_format_message(...) -> String
  
  // Update final re-exports (lines 298-306)
  pub use types::{CandleMessage, CandleMessageChunk, CandleMessageRole, CandleMessageType, CandleSearchChatMessage};
  pub use error::CandleMessageError;
  pub use media::{CandleMediaType, CandleImageMediaType, CandleDocumentMediaType, CandleAudioMediaType, CandleVideoMediaType};
  
  // Update test functions (lines 312-335)
  CandleMessageRole::User => "Hello, world!" patterns in tests
  ```

**Performance Requirements:**
- Zero allocation in hot paths
- No unwrap() or expect() usage
- Maintain all existing functionality with Candle prefixes

### Task C1.2: Complete Chat Module Candle Renaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/config.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/commands/mod.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/conversation/mod.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/export.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/formatting.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/integrations.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/macros.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/realtime.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/search.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/templates/mod.rs`

**Technical Specifications:**
- Rename ALL public structs, enums, traits with "Candle" prefix
- Update all impl blocks to use Candle-prefixed types  
- Update all function names with candle_ prefix where appropriate
- Maintain exact same API surface with Candle naming
- Example patterns:
  ```rust
  // config.rs
  pub struct CandleChatConfig { ... }
  pub struct CandlePersonalityConfig { ... }
  
  // commands/mod.rs  
  pub trait CandleCommandExecutor { ... }
  pub struct CandleCommandRegistry { ... }
  pub struct CandleImmutableChatCommand { ... }
  
  // conversation/mod.rs
  pub trait CandleConversation { ... }
  pub struct CandleConversationImpl { ... }
  ```

### Task C1.3: Complete Completion Module Candle Renaming  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/completion/core.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/completion/request.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/completion/response.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/completion/types.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/completion/candle.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/completion/mod.rs`

**Technical Specifications:**
- **CRITICAL**: Update CompletionModel trait → CandleCompletionModel
- Update all completion request/response types:
  ```rust
  // core.rs
  pub trait CandleCompletionModel { ... }
  pub trait CandleCompletionProvider { ... }
  
  // request.rs
  pub struct CandleCompletionRequest { ... }
  pub struct CandleStreamingCompletionRequest { ... }
  
  // response.rs  
  pub struct CandleCompletionResponse { ... }
  pub struct CandleCompletionChunk { ... }
  pub struct CandleStreamingCompletionResponse { ... }
  
  // types.rs
  pub enum CandleCompletionStatus { ... }
  pub struct CandleUsage { ... }
  ```

### Task C1.4: Complete Context Module Candle Renaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/context/chunk.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/context/document.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/context/loader.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/context/provider.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/context/extraction/mod.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/context/mod.rs`

**Technical Specifications:**
- **CRITICAL**: Update Context trait → CandleContext (used in ARCHITECTURE.md)
- Update all document and chunk types:
  ```rust
  // document.rs
  pub struct CandleDocument { ... }
  pub trait CandleDocumentLoader { ... }
  pub enum CandleContentFormat { ... }
  pub enum CandleDocumentMediaType { ... }
  
  // chunk.rs
  pub struct CandleChunk { ... }
  pub struct CandleDocumentChunk { ... }
  
  // provider.rs
  pub trait CandleContext<T> { ... }
  pub struct CandleContextProvider { ... }
  
  // loader.rs
  pub trait CandleLoader<T> { ... }
  ```

### Task C1.5: Complete Model Module Candle Renaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/capabilities.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/info.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/models.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/provider.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/registry.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/traits.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/usage.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/validation.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/model/mod.rs`

**Technical Specifications:**
- **CRITICAL**: Update Model trait → CandleModel (core to everything)
- Update all model-related types:
  ```rust
  // traits.rs  
  pub trait CandleModel { ... }
  pub trait CandleModelProvider { ... }
  
  // info.rs
  pub struct CandleModelInfo { ... }
  pub struct CandleModelPerformance { ... }
  
  // capabilities.rs
  pub struct CandleModelCapabilities { ... }
  pub enum CandleCapability { ... }
  pub enum CandleUseCase { ... }
  
  // usage.rs
  pub struct CandleUsage { ... }
  
  // validation.rs
  pub struct CandleValidationReport { ... }
  pub enum CandleValidationError { ... }
  pub enum CandleValidationSeverity { ... }
  ```

### Task C1.6: Complete Tool Module Candle Renaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/tool/core.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/tool/mcp.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/tool/traits.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/tool/types.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/tool/mod.rs`

**Technical Specifications:**
- **CRITICAL**: Update Tool trait → CandleTool (used in ARCHITECTURE.md)
- Update all tool-related types:
  ```rust
  // traits.rs
  pub trait CandleTool { ... }
  pub trait CandleToolRegistry { ... }
  
  // types.rs
  pub struct CandleToolCall { ... }
  pub struct CandleToolResult { ... }
  
  // mcp.rs
  pub struct CandleMcpTool { ... }
  pub struct CandleMcpServer { ... }
  ```

## PHASE C2: BUILDER SYSTEM CANDLE RENAMING

### Task C2.1: Complete AgentRole Builder Candle Renaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/agent_role.rs`

**Technical Specifications:**
- **CRITICAL**: Update entry point FluentAi → CandleFluentAi
- Update all trait and impl names:
  ```rust
  // Main entry point (used in ARCHITECTURE.md)
  pub struct CandleFluentAi;
  impl CandleFluentAi {
      pub fn agent_role(name: &str) -> impl CandleAgentRoleBuilder { ... }
  }
  
  // Builder trait
  pub trait CandleAgentRoleBuilder: Sized {
      fn completion_provider<P>(self, provider: P) -> impl CandleAgentRoleBuilder;
      fn temperature(self, temp: f64) -> impl CandleAgentRoleBuilder;
      fn max_tokens(self, tokens: u32) -> impl CandleAgentRoleBuilder;
      fn system_prompt(self, prompt: &str) -> impl CandleAgentRoleBuilder;
      fn context<C>(self, context: C) -> impl CandleAgentRoleBuilder;
      fn tools<T>(self, tools: T) -> impl CandleAgentRoleBuilder;
      fn conversation_history(self, ...) -> impl CandleAgentRoleBuilder;
      fn into_agent(self) -> CandleAgent;
      fn chat(self, input: &str) -> AsyncStream<CandleMessageChunk>;
  }
  
  // Implementation struct
  struct CandleAgentRoleBuilderImpl<F1, F2, F3> { ... }
  ```

### Task C2.2: Complete All Other Builder Candle Renaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/completion.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/embedding.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/document.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/extractor.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/loader.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/memory.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/memory_node.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/memory_system.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/memory_workflow.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/workflow.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/image.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/audio.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/mod.rs`

**Technical Specifications:**
- Rename ALL builder traits with Candle prefix:
  ```rust
  // Pattern for each builder file
  pub trait CandleXxxBuilder: Sized {
      fn method(self, param: T) -> impl CandleXxxBuilder;
  }
  
  struct CandleXxxBuilderImpl<F1, F2, F3> { ... }
  ```
- Update mod.rs re-exports to use Candle-prefixed names
- Maintain exact trait-based zero-Box architecture

## PHASE C3: KIMI_K2 MODEL INTEGRATION

### Task C3.1: Create Candle Model Provider
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/providers/mod.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/providers/kimi_k2.rs`

**Technical Specifications:**
- Implement CandleCompletionModel for kimi_k2:
  ```rust
  // providers/kimi_k2.rs
  use candle_core::{Device, Tensor};
  use candle_nn::VarBuilder;
  use candle_transformers::models::kimi::KimiModel;
  
  pub struct CandleKimiK2Provider {
      model: KimiModel,
      device: Device,
      tokenizer: CandleTokenizer,
  }
  
  impl CandleCompletionModel for CandleKimiK2Provider {
      fn complete(&self, request: CandleCompletionRequest) -> AsyncStream<CandleCompletionChunk> {
          // Implementation using Candle ML framework
      }
  }
  ```

### Task C3.2: Create Candle Tokenizer Integration  
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/providers/tokenizer.rs`

**Technical Specifications:**
- Zero-allocation tokenizer wrapper for kimi_k2
- Streaming token generation support
- Memory-efficient tensor operations

## PHASE C4: ARCHITECTURE EXAMPLE ADAPTATION

### Task C4.1: Create Candle ARCHITECTURE Example
**Files to Create:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/examples/candle_agent_role_builder.rs`

**Technical Specifications:**
- **CRITICAL**: Exact copy of ARCHITECTURE.md with Candle prefixes:
  ```rust
  let stream = CandleFluentAi::agent_role("rusty-squire")
      .completion_provider(CandleMistral::MagistralSmall)
      .temperature(1.0)
      .max_tokens(8000)
      .system_prompt("Act as a Rust developers 'right hand man'...")
      .context(
          CandleContext<File>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
          CandleContext<Files>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
          CandleContext<Directory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
          CandleContext<Github>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
      )
      .tools(
          CandleTool<Perplexity>::new({
              "citations" => "true"
          }),
          CandleTool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
      )
      .into_agent()
      .conversation_history(
          CandleMessageRole::User => "What time is it in Paris, France",
          CandleMessageRole::System => "The USER is inquiring about the time in Paris, France...",
          CandleMessageRole::Assistant => "It's 1:45 AM CEST on July 7, 2025, in Paris, France..."
      )
      .chat("Hello")
      .collect();
  ```

### Task C4.2: Verify Syntax Compatibility
**Technical Specifications:**
- **CRITICAL**: Ensure CandleMessageRole::User => "content" syntax works exactly like original
- Test all ARCHITECTURE.md patterns with Candle prefixes
- Verify zero compilation errors

## PHASE C5: FINAL INTEGRATION AND TESTING

### Task C5.1: Update Main Lib.rs Re-exports
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/lib.rs`

**Technical Specifications:**
- Re-export all Candle-prefixed types from domain and builders
- Create prelude module with commonly used Candle types
- Ensure complete API coverage

### Task C5.2: Test Complete Package Compilation
**Commands to Run:**
```bash
cd /Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle
cargo check --all-features
cargo test --all-features  
cargo run --example candle_agent_role_builder
```

**Success Criteria:**
- Zero compilation errors
- Zero compilation warnings
- Example runs successfully with kimi_k2 model
- All ARCHITECTURE.md syntax patterns work with Candle prefixes

## CANDLE PACKAGE CONSTRAINTS (CRITICAL)
- ✅ Complete independence from fluent-ai packages (except http3/async)
- ✅ Zero Box<dyn> usage - maintain trait-based architecture
- ✅ All types systematically renamed with Candle prefix
- ✅ Working kimi_k2 model integration
- ✅ ARCHITECTURE.md syntax preserved exactly with Candle prefixes
- ❌ NEVER use unwrap() or expect() in src/* code
- ✅ Zero allocation, blazing-fast performance
- ✅ All code complete with semantic error handling

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
---

# CRITICAL STUB FIXES FOR FLUENT-AI-CANDLE PACKAGE
**URGENT** - These stubs were left in the codebase and MUST be fixed immediately.

## PHASE S1: CRITICAL BUILDER STUBS (HIGHEST PRIORITY)

### Task S1.1: Fix Agent Role Builder Handler Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/agent_role.rs` (lines 344, 354)

**Technical Specifications:**
- **STUB FOUND**: "Store error handler (implementation simplified for now)" 
- **STUB FOUND**: "Store chunk handler (implementation simplified for now)"
- **FIX REQUIRED**: Implement proper handler storage using zero-allocation patterns:
  ```rust
  // Replace simplified error handler with proper generic storage
  fn on_error<F>(self, error_handler: F) -> impl CandleAgentRoleBuilder
  where
      F: FnMut(String) + Send + 'static,
  {
      CandleAgentRoleBuilderImpl {
          name: self.name,
          completion_provider: self.completion_provider,
          temperature: self.temperature,
          max_tokens: self.max_tokens,
          system_prompt: self.system_prompt,
          contexts: self.contexts,
          tools: self.tools,
          mcp_servers: self.mcp_servers,
          additional_params: self.additional_params,
          memory: self.memory,
          metadata: self.metadata,
          on_tool_result_handler: self.on_tool_result_handler,
          on_conversation_turn_handler: self.on_conversation_turn_handler,
          error_handler: Some(error_handler),
      }
  }

  // Replace simplified chunk handler with proper generic storage
  fn on_chunk<F>(self, handler: F) -> impl CandleAgentRoleBuilder
  where
      F: Fn(CandleChatMessageChunk) -> CandleChatMessageChunk + Send + Sync + 'static,
  {
      CandleAgentRoleBuilderImpl {
          name: self.name,
          completion_provider: self.completion_provider,
          temperature: self.temperature,
          max_tokens: self.max_tokens,
          system_prompt: self.system_prompt,
          contexts: self.contexts,
          tools: self.tools,
          mcp_servers: self.mcp_servers,
          additional_params: self.additional_params,
          memory: self.memory,
          metadata: self.metadata,
          on_tool_result_handler: self.on_tool_result_handler,
          on_conversation_turn_handler: self.on_conversation_turn_handler,
          chunk_handler: Some(handler),
      }
  }
  ```

**Error Handling:**
- Add `error_handler` and `chunk_handler` fields to `CandleAgentRoleBuilderImpl` struct
- Use generic type parameters instead of function stubs
- Implement zero-allocation builder state transitions

### Task S1.2: Fix Chat Method Empty Stream Stub
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/agent_role.rs` (line 499)

**Technical Specifications:**
- **STUB FOUND**: `AsyncStream::empty()` - completely non-functional!
- **FIX REQUIRED**: Implement real chat functionality with kimi_k2 integration:
  ```rust
  fn chat(&self, message: impl Into<String>) -> AsyncStream<CandleMessageChunk> {
      let message_text = message.into();
      let provider = self.inner.completion_provider.clone();
      let temperature = self.inner.temperature;
      let max_tokens = self.inner.max_tokens;
      let system_prompt = self.inner.system_prompt.clone();
      let conversation_history = self.conversation_history.clone();
      
      AsyncStream::with_channel(move |sender| {
          Box::pin(async move {
              // Build completion request from builder state
              let request = CandleCompletionRequest {
                  messages: build_messages_from_history(&conversation_history, &message_text, &system_prompt),
                  temperature: temperature.unwrap_or(0.7),
                  max_tokens: max_tokens.unwrap_or(2048),
                  stream: true,
              };
              
              // Use completion provider to generate response
              if let Some(provider) = provider {
                  let mut completion_stream = provider.stream_completion(request);
                  
                  while let Some(chunk) = completion_stream.next().await {
                      match chunk {
                          Ok(completion_chunk) => {
                              let message_chunk = CandleMessageChunk {
                                  content: completion_chunk.text,
                                  done: completion_chunk.done,
                              };
                              
                              if sender.send(message_chunk).await.is_err() {
                                  break;
                              }
                              
                              if completion_chunk.done {
                                  break;
                              }
                          }
                          Err(error) => {
                              // Send error as final chunk
                              let error_chunk = CandleMessageChunk {
                                  content: format!("Error: {}", error),
                                  done: true,
                              };
                              let _ = sender.send(error_chunk).await;
                              break;
                          }
                      }
                  }
              } else {
                  // Send error if no provider configured
                  let error_chunk = CandleMessageChunk {
                      content: "Error: No completion provider configured".to_string(),
                      done: true,
                  };
                  let _ = sender.send(error_chunk).await;
              }
              
              Ok(())
          })
      })
  }
  ```

**Performance Requirements:**
- Zero allocation in message processing
- Proper error handling without unwrap/expect
- Integration with kimi_k2 provider
- Real streaming chat functionality

### Task S1.3: Fix Agent Module Empty Stream Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/agent.rs` (lines 117, 209)

**Technical Specifications:**
- **STUB FOUND**: Multiple `AsyncStream::empty()` calls
- **STUB FOUND**: "simplified implementation that would need proper conversion"
- **FIX REQUIRED**: Implement proper agent stream operations with real functionality

## PHASE S2: DOMAIN LAYER STUB FIXES

### Task S2.1: Fix All JSON Utility Function Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/util/json_util.rs` (lines 40, 148, 233, 295, 306, 319, 329, 349, 369)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/util/json_util.rs` (lines 40, 148, 233, 295, 306, 319, 329, 349, 369)

**Technical Specifications:**
- **STUBS FOUND**: 9 TODO-marked function stubs in json_util.rs
- **FIX REQUIRED**: Implement complete JSON utility functions:
  ```rust
  #[inline(always)]
  pub fn merge(mut a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
      match (&mut a, b) {
          (serde_json::Value::Object(ref mut a_map), serde_json::Value::Object(b_map)) => {
              for (key, value) in b_map {
                  match a_map.entry(key) {
                      serde_json::map::Entry::Vacant(entry) => {
                          entry.insert(value);
                      }
                      serde_json::map::Entry::Occupied(mut entry) => {
                          *entry.get_mut() = merge(entry.get().clone(), value);
                      }
                  }
              }
          }
          (_, b_value) => a = b_value,
      }
      a
  }

  #[inline(always)]
  pub fn string_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
  where
      T: Deserialize<'de>,
      D: Deserializer<'de>,
  {
      use serde::de::{self, Visitor};
      use std::fmt;

      struct StringOrVec<T>(PhantomData<T>);

      impl<'de, T> Visitor<'de> for StringOrVec<T>
      where
          T: Deserialize<'de>,
      {
          type Value = Vec<T>;

          fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
              formatter.write_str("string or array of strings")
          }

          fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
          where
              E: de::Error,
          {
              Ok(vec![T::deserialize(value.into_deserializer())?])
          }

          fn visit_seq<S>(self, seq: S) -> Result<Self::Value, S::Error>
          where
              S: de::SeqAccess<'de>,
          {
              Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
          }
      }

      deserializer.deserialize_any(StringOrVec(PhantomData))
  }
  ```

**Performance Requirements:**
- Zero allocation where possible using in-place operations
- #[inline(always)] for hot path optimization
- Proper error handling without unwrap/expect

### Task S2.2: Fix Memory Operations Stub Functions
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/memory/ops.rs` (lines 206, 216, 223, 230)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/memory/ops.rs` (lines 206, 216, 223, 230)

**Technical Specifications:**
- **STUBS FOUND**: 4 TODO-marked function stubs in memory operations
- **FIX REQUIRED**: Implement complete memory statistics and SIMD operations:
  ```rust
  #[inline]
  pub fn get_memory_ops_stats() -> (u64, u64, u64) {
      (
          MEMORY_ALLOC_COUNT.load(Ordering::Relaxed),
          MEMORY_FREE_COUNT.load(Ordering::Relaxed),
          MEMORY_BYTES_ALLOCATED.load(Ordering::Relaxed),
      )
  }

  #[inline]
  pub fn should_use_stack_allocation(embedding_size: usize) -> bool {
      // Use stack allocation for embeddings up to 1KB
      embedding_size * std::mem::size_of::<f32>() <= 1024
  }

  #[inline]
  pub fn get_vector_pool_size() -> usize {
      VECTOR_POOL.len()
  }

  #[inline]  
  pub fn record_simd_operation() {
      SIMD_OPERATIONS.fetch_add(1, Ordering::Relaxed);
  }
  ```

### Task S2.3: Fix Memory Manager Pool Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/memory/manager.rs` (lines 89, 114, 159, 194, 215, 229, 248, 275, 295)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/memory/manager.rs` (lines 89, 114, 159, 194, 215, 229, 248, 275, 295)

**Technical Specifications:**
- **STUBS FOUND**: 9 TODO-marked functions in memory manager
- **FIX REQUIRED**: Implement complete lock-free memory pool operations with proper initialization, pooling, and statistics

### Task S2.4: Fix Cognitive Types Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/memory/cognitive/types.rs` (Multiple TODO lines)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/memory/cognitive/types.rs` (Multiple TODO lines)

**Technical Specifications:**
- **STUBS FOUND**: 30+ TODO-marked struct fields and function stubs
- **FIX REQUIRED**: Complete cognitive processing type implementations with SIMD optimization, quantum signature processing, atomic operations

## PHASE S3: STREAMING AND ASYNC STUBS

### Task S3.1: Fix Context Extraction Empty Streams
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/context/extraction/extractor.rs` (lines 70, 182)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/context/extraction/extractor.rs` (lines 70, 182)

**Technical Specifications:**
- **STUBS FOUND**: AsyncStream::with_channel with TODO comments
- **FIX REQUIRED**: Implement real document extraction with proper chunk processing

### Task S3.2: Fix Chat Search Statistics Stubs  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/search.rs` (lines 415, 966, 2081, 2222, 2261, 2290, 2317)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/chat/search.rs` (lines 415, 966, 2081, 2222, 2261, 2290, 2317)

**Technical Specifications:**
- **STUBS FOUND**: Multiple TODO-marked statistics operations
- **FIX REQUIRED**: Implement real search statistics collection and reporting

### Task S3.3: Fix Engine and Initialization Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/engine.rs` (line 407)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/engine.rs` (line 407)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/init/mod.rs` (line 36)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/init/mod.rs` (line 36)

**Technical Specifications:**
- **STUBS FOUND**: TODO comments in engine integration and initialization
- **FIX REQUIRED**: Implement proper provider system integration and memory manager initialization

## PHASE S4: BUILDER SYSTEM STUB FIXES

### Task S4.1: Fix All Builder TODO Field Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/agent_role.rs` (lines 401, 403)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/completion.rs` (line 403)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/extractor.rs` (line 162)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/image.rs` (line 223)

**Technical Specifications:**
- **STUBS FOUND**: TODO-marked struct fields and placeholder implementations
- **FIX REQUIRED**: Complete all builder implementations with proper field handling

### Task S4.2: Fix Tool Core Structure Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/tool/core.rs` (lines 52, 54, 100, 102, 104)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/tool/core.rs` (lines 52, 54, 100, 102, 104)

**Technical Specifications:**
- **STUBS FOUND**: TODO-marked struct fields in Tool and NamedTool
- **FIX REQUIRED**: Complete tool implementation with proper configuration handling

## PHASE S5: CHAT AND TEMPLATE STUBS

### Task S5.1: Fix Chat Command Execution Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/commands/execution.rs` (line 448)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/commands/types.rs` (line 748)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/chat/commands/execution.rs` (line 448)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/chat/commands/types.rs` (line 748)

**Technical Specifications:**
- **STUBS FOUND**: TODO comments in command execution and integration
- **FIX REQUIRED**: Implement real command system integration

### Task S5.2: Fix Template Processing Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/templates/core.rs` (line 429)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/templates/parser.rs` (line 191)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/chat/templates/core.rs` (line 429)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/chat/templates/parser.rs` (line 191)

**Technical Specifications:**
- **STUBS FOUND**: TODO comments in template processing
- **FIX REQUIRED**: Complete template parsing and processing implementations

### Task S5.3: Fix Chat Formatting Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/chat/formatting.rs` (lines 627, 786)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/chat/formatting.rs` (lines 627, 786)

**Technical Specifications:**
- **STUBS FOUND**: TODO comments in markdown and syntax highlighting integration
- **FIX REQUIRED**: Implement complete formatting with markdown parsing and syntax highlighting

## PHASE S6: AGENT AND ROLE STUBS

### Task S6.1: Fix Agent Role Field Stubs  
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/agent/role.rs` (lines 34-85)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/role.rs` (lines 34-85)

**Technical Specifications:**
- **STUBS FOUND**: 15+ TODO-marked struct fields using Box<dyn Any>
- **FIX REQUIRED**: Replace all Box<dyn Any> with proper typed implementations:
  ```rust
  // Replace stub fields with proper implementations
  pub struct CandleAgentRoleImpl<P, C, T, M> 
  where
      P: CandleCompletionProvider + Send + Sync + 'static,
      C: CandleContext + Send + Sync + 'static,
      T: CandleTool + Send + Sync + 'static,
      M: CandleMemory + Send + Sync + 'static,
  {
      name: String,
      completion_provider: Option<P>,
      temperature: Option<f64>,
      max_tokens: Option<u64>,
      system_prompt: Option<String>,
      api_key: Option<String>,
      contexts: Option<CandleZeroOneOrMany<C>>,
      tools: Option<CandleZeroOneOrMany<T>>,
      mcp_servers: Option<CandleZeroOneOrMany<CandleMcpServerConfig>>,
      additional_params: Option<HashMap<String, Value>>,
      memory: Option<M>,
      metadata: Option<HashMap<String, Value>>,
      on_tool_result_handler: Option<Box<dyn Fn(CandleZeroOneOrMany<Value>) + Send + Sync>>,
      on_conversation_turn_handler: Option<Box<dyn Fn(&CandleAgentConversation, &CandleAgentRoleAgent) + Send + Sync>>,
  }
  ```

### Task S6.2: Fix Agent Chat and Core Stubs
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/domain/agent/chat.rs` (lines 91, 125)
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/chat.rs` (lines 91, 125)

**Technical Specifications:**
- **STUBS FOUND**: Placeholder response and empty context returns
- **FIX REQUIRED**: Implement real chat processing and context retrieval

## CRITICAL EXECUTION CONSTRAINTS

### Performance Requirements:
- ✅ **ZERO ALLOCATION**: All hot paths must use stack allocation or pre-allocated pools
- ✅ **ZERO LOCKING**: Use lock-free data structures and channels for coordination  
- ✅ **ZERO `dyn`**: Replace all Box<dyn> with generic type parameters
- ✅ **ZERO UNSAFE**: All operations must be memory-safe
- ✅ **ZERO UNWRAP**: Never use unwrap() or expect() in src/* code

### Error Handling Requirements:
- All functions must return Result<T, E> with proper error types
- Use thiserror for error definition and context
- Implement comprehensive error recovery strategies
- No panic!() or unimplemented!() in production code

### Threading Requirements:
- Use AsyncStream with proper channel patterns
- All closures must be Send + 'static
- Use Arc for shared immutable data only
- Implement backpressure handling in streams

### Memory Requirements:
- Use object pools for frequent allocations
- Implement proper resource cleanup
- Use const generics for compile-time optimization
- Stack allocate small objects where possible

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required to complete full implementations.

## IMPLEMENTATION TASKS

### Task I1: Complete Agent Role Builder Handlers
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/agent_role.rs` (lines 340-360)

**Implementation Details:**
- Replace all `Box<dyn Any>` with proper generic type parameters
- Implement zero-allocation handler storage
- Add proper error types and conversions
- Ensure thread safety with `Send + Sync` bounds
- Add comprehensive documentation

**QA Check:** Verify all handlers are properly stored and invoked without allocation

### Task I2: Implement Chat Streaming
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/builders/agent_role.rs` (lines 490-550)

**Implementation Details:**
- Implement real kimi_k2 integration
- Add proper error handling for missing providers
- Ensure zero-copy message building
- Implement backpressure handling
- Add proper stream termination

**QA Check:** Verify chat streams responses correctly and handles errors

### Task I3: Complete Memory Management
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/memory/manager.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/vector/store.rs`

**Implementation Details:**
- Implement lock-free memory pool
- Add SIMD-optimized vector operations
- Implement proper cleanup on drop
- Add allocation tracking
- Add statistics collection

**QA Check:** Verify memory is properly recycled and tracked

### Task I4: Complete Error Handling
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/error.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/recovery.rs`

**Implementation Details:**
- Define comprehensive error types
- Implement proper error contexts
- Add retry logic for transient failures
- Implement circuit breakers
- Add proper logging

**QA Check:** Verify all error cases are properly handled

### Task I5: Implement Domain Models
**Files to Modify:**
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/agent/role.rs`
- `/Volumes/samsung_t9/fluent-ai/packages/fluent-ai-candle/src/chat/history.rs`

**Implementation Details:**
- Replace all `Box<dyn Any>` with proper types
- Implement proper validation
- Add serialization/deserialization
- Implement thread-safe operations
- Add comprehensive documentation

**QA Check:** Verify role configuration is correctly validated
