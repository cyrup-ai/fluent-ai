# Candle Package TODO - Production Readiness

## CRITICAL ISSUES - Non-Production Code Patterns

### STATUS: PLANNED - Fix All Placeholder Code

**Files with placeholder/stub implementations:**

1. **engine.rs:459** - Placeholder error handling
   - **Issue**: Using generic placeholder error for unimplemented providers
   - **Solution**: Implement proper provider registry with factory pattern, return specific error types
   - **Technical Notes**: Create `ProviderRegistry` trait with `get_provider()` method, implement for each provider type

2. **domain/init/mod.rs:16-70** - Multiple placeholder implementations
   - **Issue**: Entire initialization module is placeholder code
   - **Solution**: Implement proper initialization with configuration validation, dependency injection
   - **Technical Notes**: Create `CandleInitializer` with proper error handling, configuration loading from files/env

3. **domain/http/auth.rs:257,258,260,798** - Auth placeholder implementations
   - **Issue**: Authentication methods return placeholder responses
   - **Solution**: Implement real OAuth2/JWT/API key authentication with proper token validation
   - **Technical Notes**: Use `jsonwebtoken` crate, implement `AuthProvider` trait with different auth strategies

### STATUS: PLANNED - Remove ALL unwrap() Calls (Production Critical)

**High-priority unwrap() removals in core paths:**

1. **providers/kimi_k2.rs:287,290,291** - Model initialization unwraps
   - **Issue**: Model loading can panic on missing files
   - **Solution**: Return proper `Result<Model, ModelError>` with detailed error messages
   - **Technical Notes**: Check file existence, validate model format, graceful fallback

2. **main.rs:135** - CLI argument unwrap
   - **Issue**: Can panic on invalid CLI input
   - **Solution**: Use proper error handling with user-friendly messages
   - **Technical Notes**: Implement custom error types for CLI validation

3. **domain/model/registry.rs:408,413,418,432,438,444** - Model registry unwraps
   - **Issue**: Model registration can panic
   - **Solution**: Use `try_register()` pattern with rollback on failure
   - **Technical Notes**: Implement transaction-like model registration

### STATUS: PLANNED - Remove ALL expect() Calls (Production Critical)

**Critical expect() calls in production paths:**

1. **domain/http/requests/completion.rs:1183-1303** - HTTP request building expects
   - **Issue**: Network operations can fail, causing panics
   - **Solution**: Return `Result<Request, HttpError>` with retry logic
   - **Technical Notes**: Implement exponential backoff, circuit breaker pattern

2. **domain/http/responses/completion.rs:1624-1836** - Response parsing expects
   - **Issue**: Malformed responses cause panics
   - **Solution**: Robust JSON parsing with schema validation
   - **Technical Notes**: Use `serde_json::from_str()` with custom error handling

### STATUS: PLANNED - Remove "for now" Temporary Code

**Temporary implementations that need completion:**

1. **engine.rs:582,604** - Fallback sampling logic
   - **Issue**: "For now, fall back to greedy" - incomplete sampling
   - **Solution**: Implement full sampling algorithms (nucleus, top-k, temperature)
   - **Technical Notes**: Use proper probability distributions, configurable sampling strategies

2. **domain/chat/search.rs:398,1491,2087** - Search functionality stubs
   - **Issue**: Multiple "for now" search implementations
   - **Solution**: Implement full-text search with ranking, filtering, faceting
   - **Technical Notes**: Use `tantivy` or `meilisearch` for high-performance search

### STATUS: PLANNED - Complete TODO Items

**High-impact TODO items:**

1. **main.rs:108,275** - Model download and statistics
   - **Issue**: TODO comments for progresshub integration
   - **Solution**: Implement model downloading with progress bars, caching
   - **Technical Notes**: Use `indicatif` for progress, `tokio::fs` for async I/O

2. **domain/memory/manager.rs:66-249** - Memory management TODOs
   - **Issue**: 15+ TODO items in core memory system
   - **Solution**: Complete memory pool implementation with proper cleanup
   - **Technical Notes**: Implement `Drop` trait, reference counting, garbage collection

## ARCHITECTURE DECOMPOSITION - Large Files (>300 lines)

### STATUS: PLANNED - Decompose Oversized Modules

**Files requiring modular decomposition:**

1. **chat/search.rs (2858 lines)** 
   - **Decomposition Plan**:
     - `search/query.rs` - Query parsing and validation
     - `search/index.rs` - Search index management
     - `search/ranking.rs` - Result ranking algorithms
     - `search/filters.rs` - Search filtering logic
     - `search/facets.rs` - Faceted search implementation
   - **Technical Notes**: Each module should be <400 lines, clear separation of concerns

2. **domain/chat/commands/types.rs (2169 lines)**
   - **Decomposition Plan**:
     - `commands/basic.rs` - Basic command types
     - `commands/advanced.rs` - Advanced command implementations
     - `commands/validation.rs` - Command validation logic
     - `commands/execution.rs` - Command execution engine
     - `commands/response.rs` - Response handling
   - **Technical Notes**: Use trait objects for command polymorphism

3. **domain/chat/realtime.rs (1903 lines)**
   - **Decomposition Plan**:
     - `realtime/connection.rs` - WebSocket connection management
     - `realtime/events.rs` - Event handling and dispatch
     - `realtime/streaming.rs` - Real-time streaming logic
     - `realtime/sync.rs` - State synchronization
   - **Technical Notes**: Use actor pattern for concurrent event processing

4. **http/responses/completion.rs (1846 lines)**
   - **Decomposition Plan**:
     - `responses/completion/types.rs` - Response type definitions
     - `responses/completion/parser.rs` - Response parsing logic
     - `responses/completion/streaming.rs` - Streaming response handling
     - `responses/completion/validation.rs` - Response validation
   - **Technical Notes**: Zero-allocation parsing with `nom` or custom parsers

5. **memory/cognitive/types.rs (1493 lines)**
   - **Decomposition Plan**:
     - `cognitive/reasoning.rs` - Reasoning algorithm implementations
     - `cognitive/memory.rs` - Memory storage and retrieval
     - `cognitive/planning.rs` - Planning and goal-setting
     - `cognitive/learning.rs` - Learning and adaptation
   - **Technical Notes**: Use trait-based design for pluggable cognitive modules

## TEST EXTRACTION

### STATUS: PLANNED - Extract Tests to ./tests/

**Files with embedded tests (move to ./tests/):**

1. **Extract from providers/kimi_k2.rs**:
   - Move tests to `./tests/providers/test_kimi_k2.rs`
   - Add integration tests for model loading, inference

2. **Extract from main.rs**:
   - Move tests to `./tests/integration/test_main.rs`
   - Add CLI integration tests with temp directories

3. **Extract from domain/http/requests/completion.rs**:
   - Move tests to `./tests/http/test_completion_requests.rs`
   - Add mock server tests for HTTP requests

4. **Extract from all other files with #[cfg(test)]**:
   - Systematic extraction to `./tests/` with proper module structure
   - Maintain test coverage, add property-based tests where appropriate

### STATUS: PLANNED - Bootstrap Nextest

**Nextest setup and configuration:**

1. Add `nextest.toml` configuration file
2. Configure parallel test execution
3. Set up test groups for unit/integration/performance tests
4. Add CI/CD integration with test reporting

## LOGGING IMPROVEMENTS

### STATUS: PLANNED - Replace println!/eprintln! with env_logger

**Logging cleanup:**

1. Replace all `println!` with proper `log::info!` macros
2. Replace all `eprintln!` with `log::error!` macros  
3. Add structured logging with `serde_json` for production
4. Configure log levels per module
5. Add request tracing with correlation IDs

## PERFORMANCE OPTIMIZATIONS

### STATUS: PLANNED - Zero-Allocation Implementations

**Performance-critical optimizations:**

1. **String handling**: Replace `String` with `Cow<str>` where possible
2. **Collections**: Use `SmallVec` and `ArrayVec` for small collections
3. **Async**: Ensure all async code uses `AsyncStream` pattern (no futures)
4. **Memory pools**: Implement object pooling for frequently used types
5. **SIMD**: Use SIMD operations for tensor calculations where applicable

### STATUS: PLANNED - Eliminate Locking

**Lock-free implementations:**

1. Replace `Mutex` with lock-free data structures using `crossbeam`
2. Use atomic operations for simple state management
3. Implement channel-based communication instead of shared state
4. Use `RwLock` only where absolutely necessary (read-heavy workloads)

## ERROR HANDLING

### STATUS: PLANNED - Comprehensive Error Handling

**Error handling improvements:**

1. Define custom error types for each module with `thiserror`
2. Implement error context propagation with `anyhow` for debugging
3. Add error recovery strategies (retry, fallback, graceful degradation)
4. Implement structured error reporting for API consumers
5. Add error metrics and monitoring integration

## INTEGRATION REQUIREMENTS

### STATUS: PLANNED - Production Integration

**Integration with existing fluent-ai architecture:**

1. Ensure all HTTP calls use `fluent_ai_http3` package
2. Maintain `AsyncStream` pattern throughout (no Future trait usage)
3. Zero dependency on domain/fluent-ai packages (standalone requirement)
4. Proper integration with cyrup_sugars utilities