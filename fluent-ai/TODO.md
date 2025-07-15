## Current Status: DOMAIN PATTERN COMPLIANCE PROJECT

## SUCCESS CRITERIA: ALL DOMAIN OBJECTS FOLLOW TRAIT + IMPL + BUILDER PATTERN

---

# 🚨 CRITICAL ERRORS TO FIX (10 remaining)

## Error 1: Thread safety - loader.rs:70:26
**Issue**: `(dyn Iterator<Item = PathBuf> + std::marker::Send + 'static)` cannot be shared between threads safely
**Fix**: Change line 44 from `Box<dyn Iterator<Item = T> + Send>` to `Box<dyn Iterator<Item = T> + Send + Sync>`

## Error 2: Thread safety - loader.rs:118:30
**Issue**: Same as Error 1 - `(dyn Iterator<Item = PathBuf> + std::marker::Send + 'static)` cannot be shared between threads safely
**Fix**: Same fix as Error 1

## Error 3: ZeroOneOrMany handling - loader.rs:121:44
**Issue**: `mismatched types: expected '&PathBuf', found '&Vec<PathBuf>'` - treating ZeroOneOrMany like Vec
**Fix**: Handle ZeroOneOrMany properly by matching its variants or using proper iterator methods

## Error 4: ZeroOneOrMany handling - loader.rs:289:25
**Issue**: `mismatched types: expected '&T', found '&Vec<T>'` - same ZeroOneOrMany issue
**Fix**: Handle ZeroOneOrMany properly in this method too

# 🚨 CRITICAL: PLACEHOLDER IMPLEMENTATIONS TO FIX

## PLACEHOLDER 1: Agent completion streaming - agent.rs:349
**Location**: src/domain/agent.rs line 349
**Current**: `// TODO: Implement actual completion streaming` with empty stream
**Required**: Proper completion streaming implementation
**Questions**:
- What LLM API should be called for streaming completions?
- How should the CompletionRequest be converted to API calls?
- What error handling is needed for stream failures?
- Should this integrate with the agent's tools and memory?

## PLACEHOLDER 2: Agent chat logic with tool calling - agent.rs:390
**Location**: src/domain/agent.rs line 390
**Current**: `// TODO: Implement actual agent chat logic with tool calling loop` 
**Required**: Full agent conversation loop with tool calling
**Questions**:
- What's the expected tool calling loop flow?
- How should tools be selected and executed?
- How should tool results be integrated back into the conversation?
- What's the termination condition for the conversation loop?

## PLACEHOLDER 3: Agent completion with model - agent.rs:424
**Location**: src/domain/agent.rs line 424
**Current**: `// TODO: Implement actual completion with the model` returning placeholder
**Required**: Real model completion implementation
**Questions**:
- How should the agent's model be called for completions?
- Should this use the same mechanism as stream_completion?
- How should the agent's context and tools be included?
- What's the expected return format?

## PLACEHOLDER 4: Agent conversation handling - agent.rs:499
**Location**: src/domain/agent.rs line 499
**Current**: `// TODO: Implement actual conversation handling with message history`
**Required**: Conversation management with full message history
**Questions**:
- How should message history be maintained and passed to the model?
- Should this integrate with the agent's memory system?
- How should conversation context be managed across turns?
- What's the expected format for conversation state?

## PLACEHOLDER 5: MCP tool execution - mcp_tool.rs:66
**Location**: src/domain/mcp_tool.rs line 66
**Current**: `// This is a placeholder - real MCP tools would delegate to the MCP server`
**Required**: Actual MCP server communication
**Questions**:
- How should MCP tools connect to and communicate with MCP servers?
- What's the protocol for tool execution requests/responses?
- How should MCP server discovery and connection management work?
- What error handling is needed for MCP communication failures?

## PLACEHOLDER 6: Extractor model completion - extractor.rs:177
**Location**: src/domain/extractor.rs line 177
**Current**: `// TODO: Convert model to agent properly` with placeholder agent creation
**Required**: Proper model-to-agent conversion for extraction
**Questions**:
- How should completion models be converted to agents for extraction tasks?
- What system prompts should be used for different extraction types?
- How should the extracted type T be communicated to the model?
- Should extraction use structured output or parse from text?

## PLACEHOLDER 7: Extractor implementation - extractor.rs (multiple locations)
**Location**: src/domain/extractor.rs 
**Current**: `// TODO: Implement actual model completion` and other placeholders
**Required**: Full extraction pipeline implementation
**Questions**:
- What's the complete flow from text input to extracted structured data?
- How should different extraction models be handled?
- What validation should be done on extracted data?
- How should extraction failures be handled and retried?

## PLACEHOLDER 8: Memory operations - memory_ops.rs (multiple locations)
**Location**: src/domain/memory_ops.rs
**Current**: `// TODO: Generate embedding if enabled` and other memory placeholders
**Required**: Complete memory storage and retrieval system
**Questions**:
- What embedding models should be used for memory operations?
- How should memory persistence be implemented?
- What's the expected memory search and retrieval interface?
- How should memory relationships and context be maintained?

## PLACEHOLDER 9: Image processing - image.rs
**Location**: src/domain/image.rs
**Current**: `// TODO: Implement actual processing`
**Required**: Image processing and analysis implementation
**Questions**:
- What image processing capabilities are needed?
- Should this integrate with vision models for analysis?
- What image formats and transformations should be supported?
- How should processed images be stored and referenced?

## PLACEHOLDER 10: Context GitHub implementation - context.rs
**Location**: src/domain/context.rs
**Current**: `// TODO: Actual GitHub implementation would use GitHub API`
**Required**: Real GitHub API integration for context loading
**Questions**:
- What GitHub data should be loaded as context (repos, issues, PRs)?
- How should GitHub authentication and rate limiting be handled?
- What's the expected format for GitHub context data?
- Should this cache GitHub data locally?

## PLACEHOLDER 11: Workflow panic implementations - workflow.rs (multiple)
**Location**: src/workflow.rs
**Current**: `// For now, this is a placeholder that panics`
**Required**: Actual workflow execution logic
**Questions**:
- What's the expected workflow execution model?
- How should workflow steps be chained and data passed between them?
- What error handling and retry logic is needed for workflows?
- Should workflows support parallel execution of steps?

# 🚨 CRITICAL: COMPREHENSIVE ERROR HANDLING SYSTEM IMPLEMENTATION

## OBJECTIVE: "Nothing Returns Result" - All Results Unwrapped at Handler Level

User guidance: "nothing returns Result (all unwrapped by `on_result` or `on_chunk` or `on_error` all Streams are unwrapped fully)"

## CURRENT COMPILATION ERRORS TO FIX:
- embedding.rs:24 - `Result<ZeroOneOrMany<f32>, RecvError>` vs `ZeroOneOrMany<f32>` mismatch
- loader.rs:136,137,138 - `Result<ZeroOneOrMany<PathBuf>, ...>` vs `ZeroOneOrMany<_>` mismatches
- loader.rs:321,322,323 - `Result<ZeroOneOrMany<T>, RecvError>` vs `ZeroOneOrMany<_>` mismatches  
- extractor.rs:156 - trait object NotResult issues
- loader.rs:280 - trait object NotResult issues

## TASK 1: Create Error Handler Infrastructure
**File**: src/async_task/error_handlers.rs (new file)
**Action**: Create BadTraitImpl pattern for futures, BadAppleChunk pattern for streams, default error handlers with env_logger integration
**Implementation**: 
- BadTraitImpl trait that returns default implementations when errors occur
- BadAppleChunk trait that returns default chunks when stream errors occur
- Default error handlers that log with env_logger and return appropriate defaults
**Lines**: Entire new file
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## TASK 2: Update AsyncTask for Result Unwrapping
**File**: src/async_task/task.rs
**Action**: Update AsyncTask::from_future to accept Future<Output = Result<T, E>> and unwrap with error handlers
**Implementation**: 
- Change from_future to accept Result<T, E> and apply error handler internally
- Apply error handler to unwrap Result -> T before storing in AsyncTask
- Ensure all AsyncTask operations return unwrapped types
**Lines**: Around 44-60 where from_future is defined
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## TASK 3: Fix embedding.rs Result Mismatch
**File**: src/domain/embedding.rs
**Action**: Fix line 24 where embed_task.await returns Result but ZeroOneOrMany expected
**Implementation**: 
- Apply error handler to unwrap Result<ZeroOneOrMany<f32>, RecvError> -> ZeroOneOrMany<f32>
- Use default error handler that logs error and returns ZeroOneOrMany::None
**Lines**: 24 - handler(embedding) should receive unwrapped ZeroOneOrMany
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## TASK 4: Fix loader.rs Result Mismatches
**File**: src/domain/loader.rs
**Action**: Fix lines 136,137,138,321,322,323 where AsyncTask expects Result but gets ZeroOneOrMany
**Implementation**: 
- Apply error handlers to unwrap Results in AsyncTask operations
- Use default error handlers that log errors and return appropriate defaults
- Ensure all match arms return unwrapped ZeroOneOrMany types
**Lines**: 136,137,138 (match arms), 321,322,323 (match arms)
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## TASK 5: Fix Trait Object NotResult Issues
**File**: src/domain/extractor.rs:156, src/domain/loader.rs:280
**Action**: Restructure trait objects to avoid NotResult constraint violations
**Implementation**: 
- Change return types from trait objects to concrete types where possible
- Use type erasure patterns that are compatible with NotResult
- Ensure all AsyncTask returns are NotResult compatible
**Lines**: extractor.rs:156 (build_async method), loader.rs:280 (trait object usage)
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## TASK 6: Add Error Handler Methods to All Builders ✅ COMPLETED

### TASK 6.1: AgentBuilder Error Handling ✅ COMPLETED
**File**: src/domain/agent.rs
**Action**: Added on_error, on_result, on_chunk methods to AgentBuilder
**Completed Changes**:
- ✅ Added result_handler and chunk_handler fields to AgentBuilderWithHandler struct (lines 46-47)
- ✅ Added on_result method for Agent handling (lines 186-203)
- ✅ Added on_chunk method for streaming agent operations (lines 205-222)
- ✅ Updated on_error constructor to initialize new fields (lines 171-183)

### TASK 6.2: LoaderBuilder Error Handling ✅ COMPLETED
**File**: src/domain/loader.rs
**Action**: Added on_error, on_result, on_chunk methods to LoaderBuilder
**Completed Changes**:
- ✅ Added result_handler and chunk_handler fields to LoaderBuilderWithHandler struct (lines 212-213)
- ✅ Added on_result method for ZeroOneOrMany<T> handling (lines 271-283)
- ✅ Added on_chunk method for streaming loader operations (lines 285-297)
- ✅ Updated on_error constructor to initialize new fields (lines 261-268)

### TASK 6.3: ExtractorBuilder Error Handling ✅ COMPLETED
**File**: src/domain/extractor.rs
**Action**: Added on_result, on_chunk methods to ExtractorBuilder (on_error already existed)
**Completed Changes**:
- ✅ Added result_handler and chunk_handler fields to ExtractorBuilderWithHandler struct (lines 97-98)
- ✅ Added on_result method for T handling (lines 141-153)
- ✅ Added on_chunk method for streaming extractor operations (lines 155-167)
- ✅ Updated on_error constructor to initialize new fields (lines 131-138)

### TASK 6.4: CompletionRequestBuilder Error Handling ✅ COMPLETED
**File**: src/domain/completion.rs
**Action**: Added on_error, on_result, on_chunk methods to CompletionRequestBuilder
**Completed Changes**:
- ✅ Added result_handler and chunk_handler fields to CompletionRequestBuilderWithHandler struct (lines 65-66)
- ✅ Added on_result method for CompletionRequest handling (lines 249-267)
- ✅ Added on_chunk method for streaming completion operations (lines 269-287)
- ✅ Updated on_error constructor to initialize new fields (lines 233-246)

### TASK 6.5: ConversationBuilder Error Handling ✅ COMPLETED
**File**: src/domain/conversation.rs
**Action**: Added on_error, on_result, on_chunk methods to ConversationBuilder
**Completed Changes**:
- ✅ Added result_handler and chunk_handler fields to ConversationBuilderWithHandler struct (lines 89-90)
- ✅ Added on_result method for ConversationImpl handling (lines 121-131)
- ✅ Added on_chunk method for streaming conversation operations (lines 133-143)
- ✅ Updated on_error constructor to initialize new fields (lines 113-118)

### TASK 6.6: EmbeddingBuilder Error Handling ✅ COMPLETED
**File**: src/domain/embedding.rs
**Action**: Added on_result, on_chunk methods to EmbeddingBuilder (on_error already existed)
**Completed Changes**:
- ✅ Added result_handler and chunk_handler fields to EmbeddingBuilderWithHandler struct (lines 43-44)
- ✅ Added on_result method for Result<ZeroOneOrMany<f32>, RecvError> handling (lines 82-93)
- ✅ Added on_chunk method for streaming embedding operations (lines 95-106)
- ✅ Updated constructor to initialize new fields (lines 77-79)

### TASK 6.7: MemoryBuilder Error Handling
**File**: src/domain/memory.rs
**Action**: Add on_error, on_result, on_chunk methods to VectorQueryBuilder and MemoryBuilder
**Specific Changes**:
- Find VectorQueryBuilder struct (estimated lines 50-100)
- Find MemoryBuilder struct (estimated lines 150-200)
- Add error handler fields to MemoryBuilderWithHandler struct
- Add on_error method after existing builder methods
- Add on_result method for Result<Memory, MemoryError> handling
- Add on_chunk method for streaming memory operations
- Update retrieve and build terminal methods to use error handlers
**Lines**: Find exact lines by searching for "pub struct VectorQueryBuilder", "pub struct MemoryBuilder", and their implementations
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### TASK 6.8: McpToolBuilder Error Handling ✅ COMPLETED
**File**: src/domain/mcp_tool.rs
**Action**: Added on_error, on_result, on_chunk methods to McpToolBuilder
**Completed Changes**:
- ✅ Added result_handler and chunk_handler fields to McpToolBuilderWithHandler struct (lines 139-140)
- ✅ Added on_result method for McpTool handling (lines 195-207)
- ✅ Added on_chunk method for streaming MCP tool operations (lines 210-222)
- ✅ Updated on_error constructor to initialize new fields (lines 184-192)

## TASK 7: Update AsyncStream for Result Unwrapping
**File**: src/async_task/stream.rs
**Action**: Update AsyncStream to handle Result<T, E> items with on_chunk handlers
**Implementation**: 
- Add on_chunk pattern with Ok/Err match arms
- Apply chunk handlers to unwrap Results in streams
- Use default chunk handlers that log errors and return BadAppleChunk
**Lines**: AsyncStream implementation and constructor methods
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

## SUCCESS CRITERIA:
- cargo check -p fluent-ai shows 0 compilation errors
- All AsyncTask/AsyncStream operations return unwrapped types (no Result types)
- Error handlers properly integrated into all builders
- Default handlers log errors with env_logger and return BadTraitImpl/BadAppleChunk patterns
- All Results unwrapped via error handlers in builders

---

## ✅ COMPLETED: Convert ALL Vec/array returns to ZeroOneOrMany (12 methods identified)

### ✅ EXECUTION RESULTS:
**Phase 1 - Core Traits**: ✅ COMPLETED
- ✅ embedding.rs Lines 9, 15 (EmbeddingModel trait) - Converted to ZeroOneOrMany<f32>
- ✅ memory.rs Lines 20, 25 (VectorStoreIndexDyn trait) - Converted to ZeroOneOrMany<(f64, String, Value)>
- ✅ conversation.rs Line 20 (Conversation trait) - Converted to ZeroOneOrMany<String>

**Phase 2 - Operation Types**: ✅ COMPLETED
- ✅ memory_ops.rs Lines 76, 109, 210 (Op trait implementations) - Converted to ZeroOneOrMany<MemoryNode>
- ✅ completion.rs Lines 25-27 (CompletionRequest struct fields) - Converted to ZeroOneOrMany<T>

**Phase 3 - Builder Terminal Methods**: ✅ COMPLETED
- ✅ agent.rs Line 426 (ConversationBuilder.history) - Converted to ZeroOneOrMany<Message>
- ✅ memory.rs Lines 63, 72 (VectorQueryBuilder methods) - Converted to ZeroOneOrMany<T>

**Phase 4 - Downstream Code**: ✅ COMPLETED
- ✅ completion.rs builder methods - Updated to handle ZeroOneOrMany properly
- ✅ memory_workflow.rs - Updated to handle ZeroOneOrMany search results
- ✅ fluent_engine.rs - Updated CompletionRequest initialization

### ⚠️ REMAINING ISSUES:
- Some Result<ZeroOneOrMany<T>, Error> vs ZeroOneOrMany<T> mismatches (need error handling at builder level)
- NotResult trait bound errors (covered by separate TODO task)

## TASK: Convert ALL Vec/array returns to ZeroOneOrMany (12 methods identified) - STATUS: ✅ COMPLETED

### 1. agent.rs - Line 426
**Method**: `pub fn history(self) -> Vec<Message>`
**Fix**: Change to `pub fn history(self) -> ZeroOneOrMany<Message>`
**Context**: Terminal method in ConversationBuilder

### 2. embedding.rs - Line 9  
**Method**: `fn embed(&self, text: &str) -> AsyncTask<Vec<f32>>`
**Fix**: Change to `fn embed(&self, text: &str) -> AsyncTask<ZeroOneOrMany<f32>>`
**Context**: Core trait method in EmbeddingModel

### 3. embedding.rs - Line 15
**Method**: `fn on_embedding<F>(&self, text: &str, handler: F) -> AsyncTask<Vec<f32>>`
**Fix**: Change to `fn on_embedding<F>(&self, text: &str, handler: F) -> AsyncTask<ZeroOneOrMany<f32>>`
**Context**: Handler method in EmbeddingModel trait

### 4. memory.rs - Line 63
**Method**: `pub fn retrieve(self) -> AsyncTask<Vec<(f64, String, Value)>>`
**Fix**: Change to `pub fn retrieve(self) -> AsyncTask<ZeroOneOrMany<(f64, String, Value)>>`
**Context**: Terminal method in VectorQueryBuilder

### 5. memory.rs - Line 72
**Method**: `pub fn retrieve_ids(self) -> AsyncTask<Vec<(f64, String)>>`
**Fix**: Change to `pub fn retrieve_ids(self) -> AsyncTask<ZeroOneOrMany<(f64, String)>>`
**Context**: Terminal method in VectorQueryBuilder

### 6. memory.rs - Line 20
**Method**: `fn top_n(&self, query: &str, n: usize) -> BoxFuture<Result<Vec<(f64, String, Value)>, VectorStoreError>>`
**Fix**: Change to `fn top_n(&self, query: &str, n: usize) -> BoxFuture<Result<ZeroOneOrMany<(f64, String, Value)>, VectorStoreError>>`
**Context**: Core trait method in VectorStoreIndexDyn

### 7. memory.rs - Line 25
**Method**: `fn top_n_ids(&self, query: &str, n: usize) -> BoxFuture<Result<Vec<(f64, String)>, VectorStoreError>>`
**Fix**: Change to `fn top_n_ids(&self, query: &str, n: usize) -> BoxFuture<Result<ZeroOneOrMany<(f64, String)>, VectorStoreError>>`
**Context**: Core trait method in VectorStoreIndexDyn

### 8. memory_ops.rs - Line 76
**Method**: `type Output = Result<Vec<MemoryNode>, MemoryError>`
**Fix**: Change to `type Output = Result<ZeroOneOrMany<MemoryNode>, MemoryError>`
**Context**: Op trait implementation for RetrieveMemories

### 9. memory_ops.rs - Line 109
**Method**: `type Output = Result<Vec<MemoryNode>, MemoryError>`
**Fix**: Change to `type Output = Result<ZeroOneOrMany<MemoryNode>, MemoryError>`
**Context**: Op trait implementation for SearchMemories

### 10. memory_ops.rs - Line 210
**Method**: `type Output = Result<(MemoryNode, Vec<MemoryRelationship>), MemoryError>`
**Fix**: Change to `type Output = Result<(MemoryNode, ZeroOneOrMany<MemoryRelationship>), MemoryError>`
**Context**: Op trait implementation for StoreWithContext

### 11. completion.rs - Lines 25-27
**Fields**: `pub chat_history: Vec<crate::domain::Message>`, `pub documents: Vec<crate::domain::Document>`, `pub tools: Vec<ToolDefinition>`
**Fix**: Change to `ZeroOneOrMany<T>` for all collection fields
**Context**: Fields in CompletionRequest struct

### 12. conversation.rs - Line 20
**Method**: `fn messages(&self) -> &[String]`
**Fix**: Change to `fn messages(&self) -> &ZeroOneOrMany<String>`
**Context**: Core trait method in Conversation trait

**CONSTRAINTS**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope. Never use unwrap() or expect() in src/* files.

**ACHIEVEMENT**: Successfully converted all 12 identified methods from Vec/array returns to ZeroOneOrMany, updated all downstream code to handle ZeroOneOrMany properly, and maintained zero-allocation patterns throughout.

## NotResult Errors: Remove explicit NotResult constraints

**Problem**: `NotResult` is an auto trait that should NOT be explicitly added to type bounds. It's automatically implemented for all non-Result types.

**Files to fix** (remove `+ crate::async_task::NotResult` from ALL type bounds):

1. **src/domain/extractor.rs**: 
   - Line 11: `ExtractorBuilder` trait bounds
   - Line 31: `ExtractorImpl` struct bounds
   - Line 37: `ExtractorImpl` impl bounds
   - Line 83: `ExtractorBuilder` struct bounds
   - Line 90: `ExtractorBuilderWithHandler` struct bounds
   - Line 97: `ExtractorImpl` impl bounds
   - Line 108: `ExtractorBuilderWithHandler` impl bounds
   - Line 136: `ExtractorBuilderWithHandler` impl bounds

2. **src/domain/loader.rs**:
   - Line 8: `Loader` trait bounds
   - Line 26: `load_with_processor` method bounds
   - Line 41: `LoaderImpl` struct bounds
   - Line 48: `LoaderImpl` Debug impl bounds
   - Line 59: `LoaderImpl` Clone impl bounds
   - Line 116: `load_with_processor` method bounds
   - Line 149: `LoaderImpl` impl bounds
   - Line 165: `LoaderBuilder` struct bounds
   - Line 172: `LoaderBuilderWithHandler` struct bounds
   - Line 190: `LoaderBuilderWithHandler` impl bounds
   - Line 210: `LoaderBuilderWithHandler` impl bounds
   - Line 233: `LoaderBuilderWithHandler` impl bounds
   - Line 273: `LoaderBuilderWithHandler` impl bounds

3. **src/workflow.rs**:
   - Line 49: `Workflow` struct bounds
   - Line 66: `WorkflowStep` impl bounds
   - Line 67: `WorkflowStep` impl bounds
   - Line 439: `WorkflowStep` impl bounds
   - Line 452: `WorkflowStep` impl bounds

4. **src/domain/conversation.rs**:
   - Line 117: `ConversationBuilderWithHandler` impl bounds

5. **src/domain/embedding.rs**:
   - Line 18: `EmbeddingImpl` impl bounds

6. **src/domain/memory.rs**:
   - Line 84: `MemoryBuilderWithHandler` impl bounds

7. **src/domain/mcp_tool.rs**:
   - Line 209: `McpToolBuilderWithHandler` impl bounds

8. **src/fluent.rs**:
   - Line 26: `FluentBuilder` impl bounds

9. **src/async_task/emitter_builder.rs**:
   - Line 20: `EmitterBuilder` impl bounds

**Action**: Remove `+ crate::async_task::NotResult` from all the above locations. The auto trait will handle this automatically.

**Note**: Keep the `NotResult` trait definition in `async_task/task.rs` and the imports in `async_task/stream.rs` - only remove explicit constraints in type bounds.

---

# 🎯 DOMAIN OBJECT COMPLIANCE PROJECT

## OVERALL STATUS: 9/21 COMPLIANT (43%)

### ✅ COMPLIANT (9): Agent, Document, Completion, Conversation, Embedding, Image, Audio, Memory, AgentRole
### ⚠️ PARTIAL (3): Extractor, Loader, Tool v2
### ❌ NON-COMPLIANT (9): Context, Prompt, Message, Workflow, Memory Ops, Library, Model Info Provider, MCP, Chunk

### 🗑️ DELETED: Tool v1 (legacy system removed)

---

# 📋 SEQUENTIAL EXECUTION PLAN

## PHASE 1: FIX PARTIAL COMPLIANCE (HIGH PRIORITY) 🚨

### 1.1 AgentRole Pattern Compliance ✅ COMPLETED
- **STATUS**: ✅ COMPLIANT - Uses .on_chunk() polymorphic error handling (correct pattern)
- **PATTERN**: AgentRoleBuilder → .on_chunk() → AgentRoleBuilderWithChunkHandler → .into_agent()
- **CORRECTLY IMPLEMENTED**: Follows ARCHITECTURE.md exactly with chunk-based error handling
- **FILES**: `src/domain/agent_role.rs`
- **CONCLUSION**: AgentRole was already compliant - uses .on_chunk() instead of .on_error() which is the correct pattern

### 1.2 Tool v1 System 🗑️ DELETED
- **STATUS**: 🗑️ REMOVED - Legacy system deleted per user request
- **ACTION TAKEN**: Deleted `src/domain/tool.rs` and legacy `src/agent.rs`
- **CURRENT**: Only Tool v2 system remains (`src/domain/tool_v2.rs`)
- **IMPACT**: Simplified architecture to single tool system

### 1.3 Extractor Pattern Verification
- **STATUS**: ⚠️ PARTIAL - Complex generics, verify compliance
- **CURRENT**: Has ExtractorBuilder with .on_error()
- **TARGET**: Verify trait + impl pattern with all properties
- **FILES**: `src/domain/extractor.rs`
- **STEPS**:
  1. ⚠️ Review current Extractor pattern
  2. ⚠️ Verify Extractor trait exists with all operations
  3. ⚠️ Check ExtractorImpl implementation
  4. ⚠️ Verify all properties are on trait
  5. ⚠️ Test builder returns impl Extractor
  6. ⚠️ Fix any missing trait methods

### 1.4 Loader Pattern Compliance
- **STATUS**: ⚠️ PARTIAL - Missing polymorphic error handling
- **CURRENT**: Has FileLoaderBuilder but no .on_error()
- **TARGET**: Add .on_error() polymorphic error handling
- **FILES**: `src/domain/loader.rs`
- **STEPS**:
  1. ⚠️ Review current FileLoaderBuilder
  2. ⚠️ Add .on_error() method
  3. ⚠️ Create FileLoaderBuilderWithHandler
  4. ⚠️ Move terminal methods to WithHandler
  5. ⚠️ Verify Loader trait exists
  6. ⚠️ Test polymorphic error handling

## PHASE 2: REDESIGN NON-COMPLIANT (MEDIUM PRIORITY) 🔧

### 2.1 Tool v2 Complete Redesign
- **STATUS**: ❌ NON-COMPLIANT - Marker trait approach
- **TARGET**: Full trait + impl + builder pattern
- **FILES**: `src/domain/tool_v2.rs`

### 2.2 Context Complete Redesign  
- **STATUS**: ❌ NON-COMPLIANT - Marker trait approach
- **TARGET**: Full trait + impl + builder pattern
- **FILES**: `src/domain/context.rs`

### 2.3 Prompt Enhancement
- **STATUS**: ❌ NON-COMPLIANT - Simple struct
- **TARGET**: Add trait + impl + builder (if complex operations needed)
- **FILES**: `src/domain/prompt.rs`

## PHASE 3: EVALUATE REMAINING (LOW PRIORITY) 📝

### 3.1 Message, Workflow, Library, etc.
- **DECISION**: Evaluate if full pattern needed or appropriate as-is
- **CRITERIA**: Complexity of operations, user-facing API importance

---

# 🔍 PATTERN REQUIREMENTS CHECKLIST

For each domain object, verify:

### ✅ TRAIT DEFINITION
```rust
pub trait DomainTrait: Send + Sync + Debug + Clone {
    // ALL properties as getters
    fn property_name(&self) -> &Type;
    
    // ALL operations as methods  
    fn operation_method(&mut self, param: Type);
    
    // Constructor
    fn new(param: Type) -> Self;
}
```

### ✅ IMPLEMENTATION STRUCT
```rust
#[derive(Debug, Clone)]
pub struct DomainImpl {
    // ALL properties stored here
    property_name: Type,
}

impl DomainTrait for DomainImpl {
    // ALL trait methods implemented
}
```

### ✅ BUILDER PATTERN
```rust
pub struct DomainBuilder { /* config fields */ }
pub struct DomainBuilderWithHandler { 
    /* config fields + error_handler */ 
}

impl DomainBuilder {
    pub fn configuration_method(mut self, param: Type) -> Self { self }
    pub fn on_error<F>(self, handler: F) -> DomainBuilderWithHandler where F: FnMut(String) + Send + 'static { }
}

impl DomainBuilderWithHandler {
    pub fn terminal_method(self) -> impl DomainTrait { }
    pub fn async_terminal(self) -> AsyncTask<impl DomainTrait> { }
}
```

### ✅ FLUENT API INTEGRATION
```rust
impl FluentAi {
    pub fn domain_object() -> DomainBuilder { }
}
```

---

# 📊 TRACKING METRICS

- **Total Domain Objects**: 21 (after cleanup)
- **Currently Compliant**: 9 (43%)
- **Target After Phase 1**: 12 (57%)
- **Target After Phase 2**: 15 (71%)
- **Final Target**: Evaluate remaining for 18+ (86%+)

---

## LEGACY WARNINGS TRACKING

## Current Status: 145 WARNINGS, 0 ERRORS

## SUCCESS CRITERIA: 0 WARNINGS, 0 ERRORS

---

## CATEGORY A: CRITICAL UNREACHABLE PATTERNS (85 warnings) 🚨

### A1. Fix providers.rs unreachable pattern at line 1322
### A2. Fix providers.rs unreachable pattern at line 1333
### A3. Fix providers.rs unreachable pattern at line 1344
### A4. Fix providers.rs unreachable pattern at line 1377
### A5. Fix providers.rs unreachable pattern at line 1388
### A6. Fix providers.rs unreachable pattern at line 1399
### A7. Fix providers.rs unreachable pattern at line 1410
### A8. Fix providers.rs unreachable pattern at line 1421
### A9. Fix providers.rs unreachable pattern at line 1432
### A10. Fix providers.rs unreachable pattern at line 1454
### A11. Fix providers.rs unreachable pattern at line 2829
### A12. Fix providers.rs unreachable pattern at line 2840
### A13. Fix providers.rs unreachable pattern at line 3049
### A14. Fix providers.rs unreachable pattern at line 3060
### A15. Fix providers.rs unreachable pattern at line 3071
### A16. Fix providers.rs unreachable pattern at line 3082
### A17. Fix providers.rs unreachable pattern at line 3093
### A18. Fix providers.rs unreachable pattern at line 3104
### A19. Fix providers.rs unreachable pattern at line 3115
### A20. Fix providers.rs unreachable pattern at line 3126
### A21. Fix providers.rs unreachable pattern at line 3137
### A22. Fix providers.rs unreachable pattern at line 3148
### A23. Fix providers.rs unreachable pattern at line 3159
### A24. Fix providers.rs unreachable pattern at line 3170
### A25. Fix providers.rs unreachable pattern at line 3225
### A26. Fix providers.rs unreachable pattern at line 3236
### A27. Fix providers.rs unreachable pattern at line 3269
### A28. Fix providers.rs unreachable pattern at line 3379
### A29. Fix providers.rs unreachable pattern at line 3390
### A30. Fix providers.rs unreachable pattern at line 3401
### A31. Fix providers.rs unreachable pattern at line 3456
### A32. Fix providers.rs unreachable pattern at line 3489
### A33. Fix providers.rs unreachable pattern at line 3522
### A34. Fix providers.rs unreachable pattern at line 3632
### A35. Fix providers.rs unreachable pattern at line 3775
### A36. Fix providers.rs unreachable pattern at line 3776
### A37. Fix providers.rs unreachable pattern at line 3777
### A38. Fix providers.rs unreachable pattern at line 3780
### A39. Fix providers.rs unreachable pattern at line 3781
### A40. Fix providers.rs unreachable pattern at line 3782
### A41. Fix providers.rs unreachable pattern at line 3783
### A42. Fix providers.rs unreachable pattern at line 3784
### A43. Fix providers.rs unreachable pattern at line 3785
### A44. Fix providers.rs unreachable pattern at line 3787
### A45. Fix providers.rs unreachable pattern at line 3932
### A46. Fix providers.rs unreachable pattern at line 3933
### A47. Fix providers.rs unreachable pattern at line 3952
### A48. Fix providers.rs unreachable pattern at line 3953
### A49. Fix providers.rs unreachable pattern at line 3954
### A50. Fix providers.rs unreachable pattern at line 3955
### A51. Fix providers.rs unreachable pattern at line 3956
### A52. Fix providers.rs unreachable pattern at line 3957
### A53. Fix providers.rs unreachable pattern at line 3958
### A54. Fix providers.rs unreachable pattern at line 3959
### A55. Fix providers.rs unreachable pattern at line 3960
### A56. Fix providers.rs unreachable pattern at line 3961
### A57. Fix providers.rs unreachable pattern at line 3962
### A58. Fix providers.rs unreachable pattern at line 3963
### A59. Fix providers.rs unreachable pattern at line 3968
### A60. Fix providers.rs unreachable pattern at line 3969
### A61. Fix providers.rs unreachable pattern at line 3972
### A62. Fix providers.rs unreachable pattern at line 3984
### A63. Fix providers.rs unreachable pattern at line 3987
### A64. Fix providers.rs unreachable pattern at line 3988
### A65. Fix providers.rs unreachable pattern at line 3989
### A66. Fix providers.rs unreachable pattern at line 3990
### A67. Fix providers.rs unreachable pattern at line 3991
### A68. Fix providers.rs unreachable pattern at line 3992
### A69. Fix providers.rs unreachable pattern at line 3993
### A70. Fix providers.rs unreachable pattern at line 3996
### A71. Fix providers.rs unreachable pattern at line 4001
### A72. Fix providers.rs unreachable pattern at line 4011

---

## CATEGORY B: UNUSED IMPORTS (8 warnings) 📦

### B1. Remove unused import `ToolDefinition` from fluent_engine.rs:1
### B2. Remove unused import `std::collections::HashMap` from fluent_engine.rs:4
### B3. Remove unused import `std::collections::HashMap` from providers.rs:6
### B4. Remove unused import `macros::*` from lib.rs:63
### B5. Remove unused import `futures::StreamExt` from async_task/stream.rs:5
### B6. Remove unused import `Conversation` from domain/agent.rs:5

---

## CATEGORY C: UNUSED VARIABLES (15 warnings) 🔧

### C1. Fix unused variable `task` in workflow.rs line 428
### C2. Fix unused variable `stream` in workflow.rs line 441
### C3. Fix unused variable `agent` in domain/agent.rs line 212
### C4. Fix unused variable `request` in domain/agent.rs line 239
### C5. Fix unused variable `agent` in domain/agent.rs line 281
### C6. Fix unused variable `chunk_size` in domain/agent.rs line 270
### C7. Fix unused variable `agent` in domain/agent.rs line 321
### C8. Fix unused variable `message` in domain/agent.rs line 317
### C9. Fix unused variable `last_user_message` in domain/agent.rs line 374
### C10. Fix unused variable `handler` in domain/completion.rs line 209
### C11. Fix unused variable `pattern` in domain/document.rs line 393
### C12. Fix unused variable `f` in domain/image.rs line 165
### C13. Fix unused variable `model_name` in fluent_engine.rs line 100
### C14. Fix unused variable `default_temperature` in fluent_engine.rs line 101
### C15. Fix unused variable `default_max_tokens` in fluent_engine.rs line 102

---

## CATEGORY D: UNUSED FIELDS (20 warnings) 📋

### D1. Fix unused fields `stream` and `f` in sugars.rs line 312
### D2. Fix unused fields `stream` and `f` in sugars.rs line 317
### D3. Fix unused field `error_handler` in domain/agent.rs line 42
### D4. Fix unused field `agent` in domain/agent.rs line 333
### D5. Fix unused field `name` in domain/agent_role.rs line 11
### D6. Fix unused fields `server_type`, `bin_path`, and `init_command` in domain/agent_role.rs line 29
### D7. Fix unused field `inner` in domain/agent_role.rs line 251
### D8. Fix unused fields `format` and `error_handler` in domain/audio.rs line 36
### D9. Fix unused field `error_handler` in domain/completion.rs line 61
### D10. Fix unused field `source` in domain/context.rs line 22
### D11. Fix unused field `pattern` in domain/context.rs line 132
### D12. Fix unused field `error_handler` in domain/document.rs line 48
### D13. Fix unused field `error_handler` in domain/embedding.rs line 44
### D14. Fix unused field `error_handler` in domain/extractor.rs line 22
### D15. Fix unused field `error_handler` in domain/image.rs line 49
### D16. Fix unused field `config` in domain/tool_v2.rs line 13
### D17. Fix unused field `name` in domain/tool_v2.rs line 31

---

## CATEGORY E: UNUSED FUNCTIONS/STRUCTS (5 warnings) 🏗️

### E1. Fix unused function `passthrough` in domain/memory_workflow.rs line 39
### E2. Fix unused function `run_both` in domain/memory_workflow.rs line 59
### E3. Fix unused function `new` in domain/memory_workflow.rs line 25
### E4. Fix unused struct `WorkflowBuilder` in domain/memory_workflow.rs line 29
### E5. Fix unused method `chain` in domain/memory_workflow.rs line 32

---

## CATEGORY F: MISC STYLE/LINT WARNINGS (12 warnings) 🎨

### F1. Fix ambiguous glob re-export `ContentFormat` in domain/mod.rs line 26
### F2. Fix ambiguous glob re-export `Prompt` in domain/mod.rs line 39
### F3. Fix ambiguous glob re-export `Tool` in domain/mod.rs line 42
### F4. Fix unnecessary mutable variable in collection_ext.rs line 510
### F5. Fix unnecessary mutable variable in domain/agent_role.rs line 212
### F6. Fix confusing lifetime flow in sugars.rs line 67
### F7. Fix confusing lifetime flow in domain/memory.rs line 47
### F8. Fix unused import `std::collections::HashMap` in fluent_engine.rs line 4
### F9. Fix unused import `std::collections::HashMap` in providers.rs line 6
### F10. Fix unused import `macros::*` in lib.rs line 63
### F11. Fix unused import `futures::StreamExt` in async_task/stream.rs line 5
### F12. Fix unused import `Conversation` in domain/agent.rs line 5