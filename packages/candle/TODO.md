# Fluent AI Candle - ERROR AND WARNING FIXES

## OBJECTIVE: FIX ALL 222 ERRORS + 18 WARNINGS = 240 TOTAL ISSUES 🎯

### SUCCESS CRITERIA: 0 (Zero) errors and 0 (Zero) warnings

---

## CURRENT STATUS 📊
- **ERRORS: 222** 🔥
- **WARNINGS: 18** ⚠️  
- **TOTAL: 240 issues**

---

## CRITICAL ERRORS - GENERIC TYPE PARAMETERS (High Impact)

### 1. Fix missing generics for CandleContext<T> struct
**Pattern**: `error[E0107]: missing generics for struct domain::context::provider::CandleContext`
**Impact**: ~60+ errors
**Files**: Multiple context provider files

### 2. QA: Rate CandleContext<T> generic fix quality (1-10) with specific feedback

### 3. Fix missing generics for CandleTool<T> struct  
**Pattern**: `error[E0107]: missing generics for struct domain::tool::core::CandleTool`
**Impact**: ~40+ errors
**Files**: Multiple tool files

### 4. QA: Rate CandleTool<T> generic fix quality (1-10) with specific feedback

### 5. Fix missing generics for CandleClient<T> struct
**Pattern**: `error[E0107]: missing generics for struct CandleClient`
**Impact**: ~20+ errors
**Files**: Client-related files

### 6. QA: Rate CandleClient<T> generic fix quality (1-10) with specific feedback

---

## TRAIT VS TYPE CONFUSION ERRORS

### 7. Fix "expected a type, found a trait" errors
**Pattern**: `error[E0782]: expected a type, found a trait`
**Impact**: ~25+ errors
**Description**: Trait objects being used as concrete types

### 8. QA: Rate trait vs type confusion fixes quality (1-10) with specific feedback

---

## IMPORT AND NAMING ERRORS

### 9. Fix CandlePrompt import errors
**Pattern**: `cannot find type CandlePrompt in this scope`
**Files**: providers/kimi_k2.rs and others

### 10. QA: Rate CandlePrompt import fixes quality (1-10) with specific feedback

### 11. Fix CandleCompletionParams import errors
**Pattern**: `cannot find type CandleCompletionParams in this scope`
**Files**: Multiple provider files

### 12. QA: Rate CandleCompletionParams import fixes quality (1-10) with specific feedback

### 13. Fix additional_types module missing in domain/agent
**Pattern**: `CandleAdditionalParams, CandleMetadata imports failing`

### 14. QA: Rate additional_types module fix quality (1-10) with specific feedback

### 15. Fix CandleCompletionProvider vs CandleCompletionModel trait confusion
**Pattern**: Trait inheritance and implementation confusion

### 16. QA: Rate completion provider/model trait fixes quality (1-10) with specific feedback

### 17. Fix missing CandleMcpServer type imports
**Pattern**: MCP server type not found

### 18. QA: Rate MCP server import fixes quality (1-10) with specific feedback

### 19. Fix missing CandleChatLoop type
**Pattern**: Chat loop type not found

### 20. QA: Rate chat loop type fixes quality (1-10) with specific feedback

### 21. Fix method complete not member of trait CandleCompletionModel
**Pattern**: Method signature mismatch in kimi_k2.rs

### 22. QA: Rate completion model method fixes quality (1-10) with specific feedback

---

## ASYNCSTREAM PATTERN FIXES

### 23. Apply AsyncStream patterns from fluent-ai-async to ALL streaming operations
**Requirements**: NO FUTURES, ALL UNWRAPPED, ALL STREAMS
**Pattern**: `AsyncStream::with_channel(|sender| { ... })`

### 24. QA: Rate AsyncStream pattern implementation quality (1-10) with specific feedback

### 25. Fix AsyncStream usage in completion providers
**Files**: All provider files using async/await patterns

### 26. QA: Rate completion provider AsyncStream fixes quality (1-10) with specific feedback

### 27. Fix AsyncStream usage in context providers
**Files**: Context provider files

### 28. QA: Rate context provider AsyncStream fixes quality (1-10) with specific feedback

### 29. Fix AsyncStream usage in tool implementations
**Files**: Tool implementation files

### 30. QA: Rate tool AsyncStream fixes quality (1-10) with specific feedback

---

## HTTP3 PATTERN FIXES

### 31. Apply HTTP3 streaming patterns for HTTP operations
**Requirements**: Use .collect(), .collect_or_else() patterns
**Pattern**: `Http3::json().body(&request).post(url).collect::<T>()`

### 32. QA: Rate HTTP3 pattern implementation quality (1-10) with specific feedback

---

## WARNING FIXES (18 Total)

### 33. Fix unused import: `self`
**Files**: Various import statements

### 34. QA: Rate unused import cleanup quality (1-10) with specific feedback

### 35. Fix unused imports: `CommandRegistryStats as TypesCommandRegistryStats`, `ImmutableChatCommand`
**Files**: Command system files

### 36. QA: Rate command system import cleanup quality (1-10) with specific feedback

### 37. Fix unused import: `NonZeroU8`
**Files**: Numeric type files

### 38. QA: Rate numeric import cleanup quality (1-10) with specific feedback

### 39. Fix unused imports: `CognitiveMemoryConfig` and `CognitiveProcessorConfig`
**Files**: Memory system files

### 40. QA: Rate memory system import cleanup quality (1-10) with specific feedback

### 41. Fix unused import: `CandleCompletionRequest`
**Files**: Completion system files

### 42. QA: Rate completion system import cleanup quality (1-10) with specific feedback

### 43. Fix unused imports: `AsyncTask` and `spawn_task`
**Files**: Async system files

### 44. QA: Rate async system import cleanup quality (1-10) with specific feedback

### 45. Fix unnecessary parentheses around `match` scrutinee expression
**Files**: Pattern matching code

### 46. QA: Rate code style fix quality (1-10) with specific feedback

### 47. Fix unused import: `Encoding`
**Files**: Encoding-related files

### 48. QA: Rate encoding import cleanup quality (1-10) with specific feedback

---

## LIBRARY VERSION UPDATES

### 49. Update ALL libraries to VERY LATEST versions via `cargo search`
**Requirement**: Ensure latest dependency versions

### 50. QA: Rate library update implementation quality (1-10) with specific feedback

---

## FUNCTIONALITY VERIFICATION

### 51. Test code functionality as end user after all fixes
**Requirement**: Verify code ACTUALLY WORKS

### 52. QA: Rate end-user functionality verification quality (1-10) with specific feedback

---

## ARCHITECTURAL ALIGNMENT - EXACT DOMAIN REPLICA WITH CANDLE PREFIXES

### 55. Fix ZeroOneOrMany import source violation
**File**: `src/domain/agent/role.rs` lines 11-12
**Issue**: Currently imports from `crate::domain::CandleZeroOneOrMany` - violates CLAUDE.md isolation
**Fix**: Import from `cyrup_sugars::ZeroOneOrMany` 
**Impact**: Critical CLAUDE.md compliance violation

### 56. QA: Rate ZeroOneOrMany import fix quality (1-10) with specific feedback

### 57. Fix struct field type mismatches in CandleAgentRoleImpl
**File**: `src/domain/agent/role.rs` lines 95-99
**Issues**: 
- `additional_params: hashbrown::HashMap<String, serde_json::Value>` → should be `HashMap<String, Value>`
- `metadata: hashbrown::HashMap<String, serde_json::Value>` → should be `HashMap<String, Value>`
- `on_tool_result_handler: Fn(ZeroOneOrMany<serde_json::Value>)` → should be `Fn(ZeroOneOrMany<Value>)`
- `on_conversation_turn_handler: Fn(&super::types::AgentConversation, &super::types::AgentRoleAgent)` → should be `Fn(&AgentConversation, &AgentRoleAgent)`

### 58. QA: Rate struct field type alignment quality (1-10) with specific feedback

### 59. Fix CandleAgentRole::new() field initialization errors
**File**: `src/domain/agent/role.rs` lines 160-169
**Issue**: Using `ZeroOneOrMany::None` for Option fields
**Fix**: Replace with `None` for all Option<ZeroOneOrMany<...>> fields:
- `completion_provider: None`
- `contexts: None`
- `tools: None`
- `mcp_servers: None`
- `additional_params: None`
- `memory: None`
- `metadata: None`

### 60. QA: Rate field initialization fix quality (1-10) with specific feedback

### 61. Fix impl block naming error
**File**: `src/domain/agent/role.rs` line 176
**Issue**: `impl AgentRoleImpl` should be `impl CandleAgentRoleImpl`
**Fix**: Update impl block name to match Candle struct

### 62. QA: Rate impl block naming fix quality (1-10) with specific feedback

### 63. Fix memory method signatures to match domain exactly
**File**: `src/domain/agent/role.rs` lines 185-201
**Issues**:
- Method name: `get_memory_tools()` → should be `get_memory_tool()`
- Return type: `&ZeroOneOrMany<CandleMemory>` → should be `Option<&dyn std::any::Any>`
- Method name: `with_memory_tools()` → should be `with_memory_tool()`
- Parameter type: `ZeroOneOrMany<CandleMemory>` → should be `Box<dyn std::any::Any + Send + Sync>`
**Fix**: Match domain AgentRoleImpl methods exactly

### 64. QA: Rate memory method signature alignment quality (1-10) with specific feedback

### 65. Fix trait argument architecture violations
**File**: `src/domain/agent/role.rs` lines 268-283
**Issues**:
- Missing Candle prefixes: `ContextArgs` → `CandleContextArgs`
- Missing Candle prefixes: `ToolArgs` → `CandleToolArgs`  
- Missing Candle prefixes: `ConversationHistoryArgs` → `CandleConversationHistoryArgs`
- Strongly-typed trait objects → should use `Box<dyn std::any::Any + Send + Sync>` like domain
- Parameter types: `&mut ZeroOneOrMany<Box<dyn CandleTool + Send + Sync>>` → should be `&mut Option<ZeroOneOrMany<Box<dyn std::any::Any + Send + Sync>>>`
- MessageRole consistency: use `CandleMessageRole` throughout

### 66. QA: Rate trait argument architecture fix quality (1-10) with specific feedback

### 67. Verify all HashMap imports use same source as domain
**File**: `src/domain/agent/role.rs` imports
**Issue**: Ensure consistent HashMap usage (domain uses `hashbrown::HashMap`)
**Fix**: Import `hashbrown::HashMap` and `serde_json::Value` like domain

### 68. QA: Rate HashMap import consistency quality (1-10) with specific feedback

### 69. Property-by-property verification against domain
**Requirement**: Every struct field, method signature, and trait must exactly match domain with Candle prefixes
**Files**: Compare `packages/domain/src/agent/role.rs` vs `packages/candle/src/domain/agent/role.rs`
**Method**: Line-by-line diff verification
**Success Criteria**: Zero architectural differences except Candle prefixes

### 70. QA: Rate property-by-property alignment completeness (1-10) with specific feedback

---

## FINAL VERIFICATION

### 71. Run final cargo check to achieve 0 errors and 0 warnings
**Success Criteria**: Clean `cargo check` output

### 72. QA: Rate final verification completeness (1-10) with specific feedback

---

## NOTES

- **NO SHORTCUTS**: Every error and warning must be properly fixed
- **PRODUCTION QUALITY**: All code must be production-ready
- **NO BLOCKING CODE**: Unless explicitly approved by David Maple
- **ZERO ALLOCATION**: Prefer zero-allocation patterns where possible
- **ASYNCSTREAM EVERYWHERE**: Use AsyncStream patterns throughout
- **TEST EVERYTHING**: Verify functionality works for end users

---

## CANDLE STANDALONE PACKAGE IMPLEMENTATION - ARCHITECTURE.md BUILDER PATTERN

### 73. Remove over-imported tool types that don't exist in domain
**Files**: All files importing tool types
**Issue**: Candle imports CandleToolDefinition, CandleToolParams, CandleToolResult, CandleToolStatus, CandleToolValidation, CandleToolCapabilities, CandleToolResources, MockCandleTool
**Domain Reality**: Domain only has `Tool` and `McpTool` traits
**Fix**: Remove all over-imported tool types, keep only CandleTool and CandleMcpTool traits
**Pattern**: Search and remove all non-existent tool type imports
**Requirement**: Exact replica of domain tool architecture with Candle prefixes

### 74. QA: Rate over-imported tool type removal quality (1-10) with specific feedback

### 75. Implement CandleFluentAi entry point struct
**File**: `src/lib.rs` 
**Requirement**: Enable exact ARCHITECTURE.md syntax: `CandleFluentAi::agent_role("rusty-squire")`
**Implementation**: 
```rust
pub struct CandleFluentAi;
impl CandleFluentAi {
    pub fn agent_role(name: impl Into<String>) -> CandleAgentRoleBuilder {
        CandleAgentRoleBuilder::new(name)
    }
}
```
**Constraints**: Zero allocation, no Box<dyn>, static dispatch only

### 76. QA: Rate CandleFluentAi entry point implementation quality (1-10) with specific feedback

### 77. Create candle builders module structure
**File**: `src/builders/mod.rs` (new file)
**Content**: Module declarations and re-exports for all builder types
**Files to create**:
- `src/builders/agent_role.rs` - CandleAgentRoleBuilder
- `src/builders/agent.rs` - CandleAgentBuilder
- `src/builders/mod.rs` - Module exports
**Requirement**: Mirror fluent-ai builder architecture but standalone with Candle prefixes

### 78. QA: Rate builders module structure quality (1-10) with specific feedback

### 79. Implement CandleAgentRoleBuilder with complete ARCHITECTURE.md support
**File**: `src/builders/agent_role.rs` (new file)
**Requirements**: Support all ARCHITECTURE.md builder methods:
- `.completion_provider(CandleKimiK2Provider::new(...))`
- `.temperature(1.0)`, `.max_tokens(8000)`, `.system_prompt("...")`
- `.context(...)` with variadic tuple support
- `.tools(...)` with variadic tuple support  
- `.mcp_server<CandleStdio>().bin(...).init(...)`
- `.additional_params({"beta" => "true"})` 
- `.memory(CandleLibrary::named(...))`
- `.metadata({"key" => "val", "foo" => "bar"})`
- `.on_tool_result(|results| { ... })`
- `.on_conversation_turn(|conversation, agent| { ... })`
- `.on_chunk(|chunk| { ... })`
- `.into_agent()`
**Constraints**: Zero allocation, no Box<dyn>, use cyrup_sugars::ZeroOneOrMany, static dispatch
**Architecture**: Exact replica of fluent-ai agent_role.rs but standalone

### 80. QA: Rate CandleAgentRoleBuilder implementation quality (1-10) with specific feedback

### 81. Implement CandleAgentBuilder with chat loop support
**File**: `src/builders/agent.rs` (new file)
**Requirements**: Support ARCHITECTURE.md agent methods:
- `.conversation_history(CandleMessageRole::User => "content", CandleMessageRole::System => "content", ...)`
- `.chat("Hello")` returning AsyncStream<CandleMessageChunk>
- `.chat(|conversation| CandleChatLoop)` with closure support
- `.collect()` method for stream collection
**Return Types**: AsyncStream<CandleMessageChunk> for all chat operations
**Constraints**: AsyncStream-only architecture, no Result wrapping, no futures

### 82. QA: Rate CandleAgentBuilder implementation quality (1-10) with specific feedback

### 83. Implement CandleKimiK2Provider completion provider
**File**: `src/domain/completion/providers.rs` (new file)
**Requirements**: 
```rust
pub struct CandleKimiK2Provider {
    model_path: String,
}
impl CandleKimiK2Provider {
    pub fn new(path: impl Into<String>) -> Self { ... }
}
impl CandleCompletionProvider for CandleKimiK2Provider { ... }
```
**Integration**: Must work with `.completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))`
**Constraints**: Zero allocation, HTTP3 streaming patterns, AsyncStream returns

### 84. QA: Rate CandleKimiK2Provider implementation quality (1-10) with specific feedback

### 85. Implement CandleContext system for document loading
**File**: `src/domain/context/mod.rs` (new file)
**Requirements**: Support ARCHITECTURE.md context syntax:
```rust
.context(
    CandleContext<CandleFile>::of("/path/to/file.pdf"),
    CandleContext<CandleFiles>::glob("/path/**/*.{md,txt}"),
    CandleContext<CandleDirectory>::of("/path/to/dir"),
    CandleContext<CandleGithub>::glob("/repo/**/*.{rs,md}")
)
```
**Types needed**: CandleContext<T>, CandleFile, CandleFiles, CandleDirectory, CandleGithub
**Architecture**: Generic context system with type-safe builders

### 86. QA: Rate CandleContext system implementation quality (1-10) with specific feedback

### 87. Implement CandleTool system for function calling
**File**: `src/domain/tool/core.rs` (new file)
**Requirements**: Support ARCHITECTURE.md tool syntax:
```rust
.tools(
    CandleTool<CandlePerplexity>::new({"citations" => "true"}),
    CandleTool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
)
```
**Types needed**: CandleTool trait, CandleMcpTool trait (exact replicas of domain)
**Integration**: Variadic tuple support, ZeroOneOrMany storage
**Constraints**: Only implement tools that exist in domain - no over-engineering

### 88. QA: Rate CandleTool system implementation quality (1-10) with specific feedback

### 89. Implement CandleChatLoop and streaming types
**File**: `src/domain/chat/loop.rs` (new file)
**Requirements**: Support ARCHITECTURE.md chat loop syntax:
```rust
.chat(|conversation| {
    match conversation.latest_user_message().to_lowercase().as_str() {
        "quit" | "exit" => CandleChatLoop::Break,
        _ => CandleChatLoop::Reprompt("response".to_string())
    }
})
```
**Types needed**: 
- CandleChatLoop enum (Break, Reprompt, UserPrompt)
- CandleMessageChunk struct for streaming
- CandleAgentConversation with latest_user_message() method
**Architecture**: Exact replica of domain chat types with Candle prefixes

### 90. QA: Rate CandleChatLoop and streaming types quality (1-10) with specific feedback

### 91. Update src/lib.rs with complete type exports
**File**: `src/lib.rs`
**Requirements**: Export all types needed for ARCHITECTURE.md syntax:
- CandleFluentAi entry point
- All builder types
- All domain types (providers, contexts, tools, chat)
- All streaming types
**Organization**: Clear module structure with re-exports
**Verification**: No missing type errors when using ARCHITECTURE.md syntax

### 92. QA: Rate lib.rs type exports completeness quality (1-10) with specific feedback

### 93. Verify Cargo.toml dependencies are standalone compliant
**File**: `Cargo.toml`
**Requirements**: Ensure ONLY allowed dependencies:
- fluent_ai_async = { path = "../async-stream" }
- fluent-ai-http3 = { path = "../http3" }  
- cyrup_sugars = { git = "https://github.com/cyrup-ai/cyrup-sugars", features = ["all"] }
**Forbidden**: Any imports from domain, fluent-ai, provider, memory packages
**Verification**: Package compiles standalone

### 94. QA: Rate Cargo.toml standalone compliance quality (1-10) with specific feedback

### 95. Create ARCHITECTURE.md compilation test example
**File**: `examples/architecture_test.rs` (new file)
**Requirements**: Copy exact ARCHITECTURE.md code and verify compilation:
```rust
let stream = CandleFluentAi::agent_role("rusty-squire")
    .completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))
    .temperature(1.0)
    .max_tokens(8000)
    .system_prompt("Act as a Rust developers 'right hand man'...")
    .context(...)
    .tools(...)
    .into_agent()
    .conversation_history(...)
    .chat(|conversation| { ... })
    .collect();
```
**Success Criteria**: Compiles without errors, demonstrates working builder pattern

### 96. QA: Rate ARCHITECTURE.md compilation test quality (1-10) with specific feedback

### 97. Run comprehensive candle package compilation test
**Command**: `cd packages/candle && cargo check`
**Requirements**: 
- Zero compilation errors
- Zero warnings
- Standalone package verification (no external dependencies)
- All ARCHITECTURE.md syntax supported
**Success Criteria**: Clean cargo check output

### 98. QA: Rate candle package compilation test quality (1-10) with specific feedback

### 99. Verify ARCHITECTURE.md builder pattern functionality
**Requirements**: Test actual builder method chaining:
- Entry point works: CandleFluentAi::agent_role()
- All builder methods chain correctly
- AsyncStream patterns work
- Chat loop functionality works
- Type safety maintained throughout
**Method**: Create comprehensive functionality test

### 100. QA: Rate ARCHITECTURE.md functionality verification quality (1-10) with specific feedback

---

**Remember: Any QA item scoring less than 9 should be redone! 🎯**