# Production Readiness TODO

This document contains detailed analysis and solutions for all non-production patterns found in the codebase.

## Critical Issues (Must Fix Immediately)

### 1. MockLLMProvider Replacement
**File:** `packages/memory/src/cognitive/manager.rs:66-87`  
**Violation:** Fake LLM provider returning hardcoded responses
**Impact:** Incorrect intent analysis, fake embeddings, unreliable cognitive processing

**Technical Analysis:**
- `analyze_intent()` always returns `QueryIntent::Retrieval`
- `embed()` returns fake embedding `vec![0.1; 512]` 
- `generate_hints()` returns hardcoded `["general"]`
- Causes incorrect similarity calculations and memory retrieval

**Production Solution:**
Replace with `ProductionLLMProvider` that:
- Integrates with real providers (OpenAI, Anthropic, local models)
- Uses actual embedding services via existing provider system
- Implements proper intent analysis using NLP patterns + ML
- Includes zero-allocation streaming and lock-free caching
- Supports configuration-driven provider selection
- Has circuit breaker patterns for resilience

### 2. Placeholder Embedding Service
**File:** `packages/memory/src/cognitive/manager.rs:452`  
**Violation:** `let memory_embedding = vec![0.1; 512]; // Placeholder embedding`
**Impact:** Broken similarity calculations, incorrect attention weights

**Technical Analysis:**
Fake embeddings cause:
- Incorrect memory similarity matching
- Wrong attention weight calculations  
- Poor cognitive memory retrieval
- Invalid vector operations

**Production Solution:**
- Integrate with actual embedding APIs (text-embedding-3-small, etc.)
- Use `fluent_ai_http3` for HTTP/3 performance
- Implement semantic caching with LRU + TTL
- Support multiple embedding dimensions (512, 768, 1536)
- Add zero-allocation streaming for large text processing
- Include embedding model hot-swapping

### 3. Blocking Code Violation
**File:** `packages/domain/src/memory.rs:580`  
**Violation:** Uses `tokio::task::block_in_place` and `handle.block_on`
**Impact:** Thread pool exhaustion, performance bottlenecks, violates async-only constraint

**Technical Analysis:**
Current code blocks async runtime:
```rust
tokio::task::block_in_place(|| {
    handle.block_on(async { /* async work */ })
})
```
This causes:
- Thread pool starvation
- Deadlock potential  
- Performance degradation
- Violates production async constraints

**Production Solution:**
- Remove all blocking calls completely
- Redesign with dependency injection pattern
- Use pre-initialized async components
- Implement async factory patterns
- Create proper async initialization sequences
- Use async channels for coordination

### 4. create_llm_provider Placeholder
**File:** `packages/memory/src/cognitive/manager.rs:149-152`  
**Violation:** `// Placeholder - would create actual provider based on settings.llm_provider`
**Impact:** Always returns MockLLMProvider regardless of configuration

**Production Solution:**
```rust
fn create_llm_provider(settings: &CognitiveSettings) -> Result<Arc<dyn LLMProvider>> {
    match settings.llm_provider.as_str() {
        "openai" => Ok(Arc::new(OpenAILLMProvider::new(&settings.api_key)?)),
        "anthropic" => Ok(Arc::new(AnthropicLLMProvider::new(&settings.api_key)?)),
        "local" => Ok(Arc::new(LocalLLMProvider::new(&settings.model_path)?)),
        _ => Err(Error::Config(format!("Unsupported provider: {}", settings.llm_provider)))
    }
}
```

## Panic Safety Issues

### 5. unwrap() Calls (200+ instances)
**Violation:** 200+ `unwrap()` calls in src/ can cause panics
**Files:** Throughout codebase (see search results)
**Impact:** Production panics, service crashes, data corruption

**Production Solution:**
Replace all `unwrap()` with proper error handling:
```rust
// Instead of: value.unwrap()
value.map_err(|e| MyError::InvalidValue(e.to_string()))?

// For Option types:
value.ok_or_else(|| MyError::MissingValue)?
```

### 6. expect() Calls (50+ instances)  
**Violation:** 50+ `expect()` calls in src/ can cause panics
**Files:** Throughout packages/*/src/**/*.rs
**Impact:** Production panics with descriptive messages

**Production Solution:**
Replace with descriptive error types:
```rust
// Instead of: value.expect("failed to parse")
value.map_err(|e| ParseError::InvalidFormat { 
    input: input.to_string(), 
    reason: e.to_string() 
})?
```

## Large File Decomposition

### 7. packages/provider/src/clients/openrouter/streaming.rs (1769 lines)
**Violation:** Monolithic streaming implementation
**Decomposition Plan:**
- `streaming/core.rs` - Core streaming types and traits
- `streaming/events.rs` - SSE event handling
- `streaming/parsers.rs` - Response parsing logic  
- `streaming/handlers.rs` - Stream processing handlers
- `streaming/errors.rs` - Error handling and recovery
- `streaming/buffer.rs` - Buffering and flow control
- `streaming/metrics.rs` - Performance monitoring

### 8. packages/provider/src/clients/anthropic/tools.rs (1749 lines)
**Violation:** All tool handling in single file
**Decomposition Plan:**
- `tools/core.rs` - Tool trait definitions and base types
- `tools/function_calling.rs` - Function call handling
- `tools/validation.rs` - Input/output validation
- `tools/execution.rs` - Tool execution engine
- `tools/registry.rs` - Tool registration and discovery
- `tools/serialization.rs` - JSON schema handling
- `tools/computer_use.rs` - Computer use tool specifics

### 9. packages/provider/src/clients/gemini/completion_old.rs (1734 lines)
**Violation:** Legacy completion implementation  
**Action:** Remove after migrating to new implementation or decompose into:
- `completion/core.rs` - Base completion types
- `completion/streaming.rs` - Streaming implementation
- `completion/legacy_compat.rs` - Backward compatibility
- `completion/error_handling.rs` - Error handling

### 10. packages/provider/src/model_info.rs (1619 lines)
**Violation:** All model information in single file
**Decomposition Plan:**
- `model_info/registry.rs` - Model registry and lookup
- `model_info/capabilities.rs` - Model capability definitions
- `model_info/pricing.rs` - Model pricing information
- `model_info/validation.rs` - Model validation logic
- `model_info/generated.rs` - Auto-generated model data

### 11. packages/domain/src/chat/templates.rs (1606 lines)
**Violation:** All template handling in single file
**Decomposition Plan:**
- `templates/core.rs` - Template engine and base types
- `templates/rendering.rs` - Template rendering logic
- `templates/variables.rs` - Variable substitution  
- `templates/validation.rs` - Template validation
- `templates/cache.rs` - Template caching system
- `templates/loader.rs` - Template loading from various sources

### 12. packages/domain/src/chat/search.rs (1541 lines)  
**Violation:** All search functionality in single file
**Decomposition Plan:**
- `search/core.rs` - Search engine and types
- `search/indexing.rs` - Content indexing logic
- `search/ranking.rs` - Search result ranking
- `search/filters.rs` - Search filtering and faceting
- `search/semantic.rs` - Semantic search capabilities
- `search/cache.rs` - Search result caching

## Test Extraction

### 13. Extract Tests from Source Files
**Violation:** 70+ `#[cfg(test)]` blocks embedded in src/
**Files:** See search results for `#[cfg(test)]`
**Impact:** Tests mixed with production code, harder maintenance

**Extraction Plan:**
Create `tests/` directory structure:
```
tests/
├── memory/
│   ├── cognitive_tests.rs
│   ├── quantum_tests.rs  
│   ├── manager_tests.rs
│   └── integration/
├── provider/
│   ├── client_tests.rs
│   ├── streaming_tests.rs
│   └── integration/
├── domain/
│   ├── chat_tests.rs
│   ├── memory_tests.rs
│   └── integration/
└── integration/
    └── full_system_tests.rs
```

### 14. Bootstrap nextest
**Action:** Set up nextest for faster test execution
**Steps:**
1. Add nextest to CI/CD pipeline
2. Configure test partitioning
3. Set up test result reporting
4. Verify all tests pass after extraction

## Temporary Solutions (Medium Priority)

### 15. "for now" Patterns (40+ instances)
**Violation:** Temporary implementations marked with "for now"
**Files:** Throughout codebase
**Action:** Review each instance and implement proper solution

### 16. "in a real" Patterns (25+ instances)  
**Violation:** Mock/fake implementations noted with "in a real"
**Files:** Throughout codebase
**Action:** Replace with production implementations

### 17. TODO Items (100+ instances)
**Violation:** Unfinished work marked with TODO
**Files:** Throughout codebase  
**Action:** Prioritize and implement missing functionality

### 18. Legacy Code (80+ instances)
**Violation:** Backward compatibility code that may need updating
**Files:** Throughout codebase
**Action:** Evaluate if legacy support is still needed

## Implementation Priority

1. **Critical Issues (1-4)** - Fix immediately, block release
2. **Panic Safety (5-6)** - Fix before production deployment  
3. **Large Files (7-12)** - Improve maintainability and performance
4. **Test Extraction (13-14)** - Improve development workflow
5. **Temporary Solutions (15-18)** - Clean up technical debt

## Production Constraints Applied

All solutions implement:
- ✅ Zero allocation where possible
- ✅ Blazing-fast performance optimizations
- ✅ No unsafe code
- ✅ No unchecked operations  
- ✅ No locking (lock-free data structures)
- ✅ Elegant ergonomic APIs
- ✅ Comprehensive error handling
- ✅ No `unwrap()` or `expect()` in production code
- ✅ Tests separated from source code
- ✅ Sequential thinking applied to architecture