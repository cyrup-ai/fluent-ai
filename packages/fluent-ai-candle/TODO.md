# TODO: Complete Async Architecture Enforcement - Production Quality Plan

## OBJECTIVE
Systematically remove ALL async/await, async fn, Future, and Result types from streaming code throughout the entire fluent-ai-candle codebase, enforcing fluent-ai-async streaming primitives (AsyncStream<T>, AsyncTask, emit!, handle_error!) universally with zero-allocation, lock-free, production-quality implementation.

## CRITICAL ARCHITECTURE REQUIREMENTS
- ✅ **AsyncStream<T>** - pure values only, never AsyncStream<Result<T,E>>
- ✅ **emit!(value)** - for successful values
- ✅ **handle_error!(err, "context")** - for errors (terminates stream)
- ❌ **AsyncStream<Result<T,E>>** - strictly forbidden
- ❌ **async fn signatures** - zero allowed
- ❌ **.await patterns** - zero allowed in streaming contexts
- ❌ **Future types** - zero allowed in streaming contexts
- ❌ **tokio::spawn** - replaced with AsyncTask::spawn
- ✅ **Zero-allocation, lock-free architecture**

---

## PRIORITY 1: CORE STREAMING FUNCTIONS

### Task 1: Fix tokenizer.rs async violations
**File:** `src/tokenizer.rs`
**Lines:** 620 (async fn), 198 (.await)
**Implementation:** Replace async fn with synchronous function returning AsyncStream<Token>. Replace .await with emit! macro usage.
**Architecture:** Convert tokenization to streaming pattern using AsyncStream::with_channel, emit tokens via emit! macro, handle errors via handle_error! macro. Return AsyncStream<Token> not AsyncStream<Result<Token,E>>.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 2: QA Review tokenizer.rs async compliance
Act as an Objective QA Rust developer and confirm tokenizer.rs is free of async/await, Future, and Result types in streaming contexts. Verify only fluent-ai-async primitives are used and AsyncStream<T> pattern (not AsyncStream<Result<T,E>>). Rate the work performed on architectural compliance and production quality.

### Task 3: Fix streaming/flow_control.rs async violations
**File:** `src/streaming/flow_control.rs`
**Lines:** 367 (async fn), 661 (async fn), 369 (.await), 667 (.await)
**Implementation:** Replace async fn signatures with synchronous functions returning AsyncStream<FlowEvent>. Replace .await patterns with emit!/handle_error! macros.
**Architecture:** Convert flow control to streaming pattern using AsyncStream, implement backpressure via stream capacity limits, use AsyncTask::spawn for control loops. Return AsyncStream<FlowEvent> not AsyncStream<Result<FlowEvent,E>>.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 4: QA Review streaming/flow_control.rs async compliance
Act as an Objective QA Rust developer and confirm streaming/flow_control.rs is free of async/await, Future, and Result types. Verify streaming flow control uses only fluent-ai-async primitives and AsyncStream<T> pattern. Rate architectural compliance and zero-allocation patterns.

### Task 5: Fix model/loading/mod.rs async violations
**File:** `src/model/loading/mod.rs`
**Lines:** 407 (async fn)
**Implementation:** Replace async fn with synchronous function returning AsyncStream<LoadProgress> for model loading progress.
**Architecture:** Convert model loading to streaming pattern, emit loading progress via emit! macro, handle loading errors via handle_error! macro. Return AsyncStream<LoadProgress> not AsyncStream<Result<LoadProgress,E>>.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 6: QA Review model/loading/mod.rs async compliance
Act as an Objective QA Rust developer and confirm model/loading/mod.rs uses only fluent-ai-async streaming primitives and AsyncStream<T> pattern. Verify model loading streams progress correctly. Rate production quality and architectural compliance.

---

## PRIORITY 2: MODEL CACHING & KV CACHE

### Task 7: Fix model/cache/mod.rs Future violations
**File:** `src/model/cache/mod.rs`
**Lines:** 58 (future reference)
**Implementation:** Replace Future references with AsyncStream<CacheEvent> streaming patterns.
**Architecture:** Convert caching operations to streaming pattern using AsyncStream, implement cache updates via streaming, use zero-allocation cache management. Return AsyncStream<CacheEvent> not AsyncStream<Result<CacheEvent,E>>.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 8: QA Review model/cache/mod.rs async compliance
Act as an Objective QA Rust developer and confirm model/cache/mod.rs is free of Future types and uses only AsyncStream<T> pattern. Verify zero-allocation cache patterns. Rate architectural compliance.

### Task 9: ACTIVELY WORKING BY claude35 - Decompose kv_cache/mod.rs (1348 lines) into logical modules ≤300 lines each
**File:** `src/kv_cache/mod.rs`
**Lines:** 1348 total lines - needs decomposition into focused submodules
**Implementation:** Break into logical modules: cache_core.rs, eviction.rs, stats.rs, config.rs, builder.rs, entry.rs
**Architecture:** Maintain all functionality while creating focused modules ≤300 lines each. Preserve zero-allocation, lock-free design.
**Status:** ACTIVELY WORKING BY claude35 - Do not duplicate this work
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 9b: ACTIVELY WORKING BY claude82 - Decompose streaming/flow_control.rs (752 lines) into logical modules ≤300 lines each
**File:** `src/streaming/flow_control.rs`
**Lines:** 752 total lines - needs decomposition into focused submodules
**Implementation:** Break into logical modules: flow_core.rs, backpressure.rs, metrics.rs, strategies.rs
**Architecture:** Maintain all functionality while creating focused modules ≤300 lines each. Preserve zero-allocation, lock-free design with AsyncStream patterns.
**Status:** ACTIVELY WORKING BY claude82 - Do not duplicate this work
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 9c: ACTIVELY WORKING BY claude92 - Decompose types/candle_chat/chat/config.rs (1389 lines) into logical modules ≤300 lines each
**File:** `src/types/candle_chat/chat/config.rs`
**Lines:** 1389 total lines - needs decomposition into focused submodules  
**Implementation:** Break into logical modules: config_core.rs, validation.rs, builder.rs, serialization.rs, defaults.rs
**Architecture:** Maintain all functionality while creating focused modules ≤300 lines each. Preserve zero-allocation, lock-free design with AsyncStream patterns.
**Status:** ACTIVELY WORKING BY claude92 - Do not duplicate this work
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 9d: ACTIVELY WORKING BY claude66 - Decompose types/candle_chat/chat/formatting.rs (880 lines) into logical modules ≤300 lines each
**File:** `src/types/candle_chat/chat/formatting.rs`
**Lines:** 880 total lines - needs decomposition into focused submodules
**Implementation:** Break into logical modules: content.rs, styles.rs, rendering.rs, streaming.rs, processors.rs
**Architecture:** Maintain all functionality while creating focused modules ≤300 lines each. Preserve zero-allocation, lock-free design with immutable structures.
**Status:** ACTIVELY WORKING BY claude66 - Do not duplicate this work
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 10: QA Review kv_cache/mod.rs async compliance
Act as an Objective QA Rust developer and confirm kv_cache/mod.rs uses only fluent-ai-async streaming primitives and AsyncStream<T> pattern. Verify KV cache streaming operations. Rate zero-allocation patterns and production quality.

---

## PRIORITY 3: SAMPLING MODULES

### Task 11: ACTIVELY WORKING BY claude61 - Decompose types/candle_chat/chat/commands/types.rs (1760 lines) into logical modules ≤300 lines each
**File:** `src/types/candle_chat/chat/commands/types.rs`
**Lines:** 1760 total lines - massive monolith needs decomposition into focused submodules
**Implementation:** Break into logical modules: error.rs, parameter.rs, command.rs, events.rs, executor.rs, parser.rs, context.rs, output.rs, metrics.rs, handler.rs
**Architecture:** Maintain all functionality while creating focused modules ≤300 lines each. Preserve zero-allocation, lock-free design with AsyncStream patterns.
**Status:** ACTIVELY WORKING BY claude61 - Do not duplicate this work
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 11b: ACTIVELY WORKING BY claude98 - Decompose sampling/mirostat.rs (676 lines) into logical modules ≤300 lines each
**File:** `src/sampling/mirostat.rs`
**Lines:** 676 total lines - needs decomposition into focused submodules
**Implementation:** Break into logical modules: config.rs, perplexity.rs, processor.rs, stats.rs
**Architecture:** Maintain all functionality while creating focused modules ≤300 lines each. Preserve zero-allocation, lock-free design with atomic operations.
**Status:** ACTIVELY WORKING BY claude98 - Do not duplicate this work
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 11c: ✅ COMPLETED BY claude83 - Decompose types/candle_chat/commands/parsing.rs (732 lines) into logical modules ≤300 lines each
**File:** `src/types/candle_chat/commands/parsing.rs`
**Lines:** 732 total lines - successfully decomposed into focused submodules
**Implementation:** Successfully decomposed into: parser.rs (256 lines), command_parsers.rs (233 lines), registration.rs (206 lines), errors.rs (27 lines), registry.rs (35 lines), validators.rs (33 lines), mod.rs (17 lines)
**Architecture:** Maintained all functionality while creating focused modules ≤300 lines each. Preserved zero-allocation, lock-free design.
**Status:** ✅ COMPLETED BY claude83 - All parsing logic successfully decomposed and verified

### Task 11d: ✅ COMPLETED BY claude982 - Decompose processing/error.rs (673 lines) into logical modules ≤300 lines each
**File:** `src/processing/error.rs`
**Lines:** 673 total lines - SUCCESSFULLY decomposed into 5 focused submodules
**Implementation:** Successfully created 5 well-structured modules:
- error_types.rs (294 lines) - Core ProcessingError enum with constructors and classification methods
- context.rs (215 lines) - ErrorContext and ContextualError with comprehensive metadata support
- validation.rs (269 lines) - Utility functions, validators, and error handling helpers
- classification.rs (117 lines) - ErrorCategory and ErrorSeverity enums with Display implementations
- conversion.rs (94 lines) - Type aliases and From trait implementations for system integration
- mod.rs (185 lines) - Comprehensive re-exports and integration tests
**Architecture:** Successfully maintained all functionality while creating focused modules ≤300 lines each. Preserved zero-allocation, lock-free design with comprehensive error handling patterns. Enhanced functionality with detailed validation utilities and contextual error information.
**Status:** ✅ COMPLETED BY claude982 - Original 673-line error.rs backed up as error.rs.backup then removed after verification

### Task 11f: ACTIVELY WORKING BY claude965 - Decompose processing/context.rs (632 lines) into logical modules ≤300 lines each
**File:** `src/processing/context.rs`
**Lines:** 632 total lines - needs decomposition into focused submodules
**Implementation:** Break into logical modules: context_core.rs, context_builder.rs, context_validation.rs, context_serialization.rs, context_utils.rs
**Architecture:** Maintain all functionality while creating focused modules ≤300 lines each. Preserve zero-allocation, lock-free design with context management patterns.
**Status:** ACTIVELY WORKING BY claude965 - Do not duplicate this work
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 11e: ✅ COMPLETED BY claude976 - Decompose error.rs (623 lines) into logical modules ≤300 lines each
**File:** `src/error.rs`
**Lines:** 623 total lines - SUCCESSFULLY decomposed into 5 focused submodules
**Implementation:** Successfully created 5 well-structured modules:
- error_types.rs (149 lines) - Core CandleError enum with Display implementation
- error_helpers.rs (167 lines) - Helper functions and error utility methods with retry logic
- error_context.rs (85 lines) - ErrorContext and CandleErrorWithContext for enhanced debugging
- conversions.rs (190 lines) - From trait implementations and type conversions
- macros.rs (31 lines) - candle_error! macro for convenient error creation with context
- mod.rs (16 lines) - Module orchestration and re-exports
**Architecture:** Successfully maintained all functionality while creating focused modules ≤300 lines each. Preserved zero-allocation, inline design patterns with comprehensive error handling capabilities.
**Status:** ✅ COMPLETED BY claude976 - All error handling logic successfully decomposed and verified

### Task 11g: CRITICAL - Fix Search Index Decomposition Breaking Changes (claude916)
**File:** Multiple files in `src/types/candle_chat/search/index/` module
**Lines:** All decomposed modules created by claude916
**Issue:** Search index decomposition introduced breaking changes preventing safe deletion of original
**Architecture:** Achieve 100% functional parity with original implementation while maintaining modular structure
**Status:** ACTIVELY WORKING BY claude916 - Critical fixes required for production deployment

#### Sub-task 11g.1: Fix Search Stream API Contract (query_engine.rs:50-75)
**File:** `src/types/candle_chat/search/index/query_engine.rs`
**Lines:** 50-75 (search_stream method)
**Issue:** Current implementation parses `query.query` string, but original used `query.terms` directly
**Architecture:** Restore original API contract that expects pre-parsed terms in SearchQuery struct
**Implementation:** Modify search_stream to use `query.terms` directly instead of parsing `query.query` string. Ensure SearchQuery struct has terms field accessible.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.2: QA: Verify Search Stream API Compatibility
Act as an Objective QA Rust developer and rate the search stream API fix on a scale of 1-10. Verify that the search_stream method now uses the exact same API contract as the original implementation. Confirm that existing code calling this method will continue to work without changes. Test with sample SearchQuery structs to ensure identical behavior.

#### Sub-task 11g.3: Fix SortOrder Enum Compatibility (sorting.rs:12-50)
**File:** `src/types/candle_chat/search/index/sorting.rs`
**Lines:** 12-50 (sort_results method)
**Issue:** Original used `SortOrder::Relevance`, decomposed uses `SortOrder::RelevanceDescending/Ascending`
**Architecture:** Restore original SortOrder enum variant names to maintain backward compatibility
**Implementation:** Update sort_results method to handle original SortOrder::Relevance variant. Check types.rs file for correct enum definition and align implementation.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.4: QA: Verify SortOrder Enum Compatibility
Act as an Objective QA Rust developer and rate the SortOrder enum fix on a scale of 1-10. Verify that all original enum variants are supported and produce identical sorting behavior. Test each sort order variant against the original implementation to ensure no behavioral changes.

#### Sub-task 11g.5: Fix Search Proximity Algorithm (search_ops.rs:140-200)
**File:** `src/types/candle_chat/search/index/search_ops.rs`
**Lines:** 140-200 (search_proximity method)
**Issue:** Decomposed version uses cartesian_product method, original used direct proximity checking
**Architecture:** Restore original proximity checking algorithm to ensure identical search results
**Implementation:** Replace current cartesian_product approach with original direct proximity checking logic. Reference original lines 542-575 for exact algorithm. Ensure token position checking matches exactly.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.6: QA: Verify Search Proximity Algorithm
Act as an Objective QA Rust developer and rate the search proximity algorithm fix on a scale of 1-10. Verify that proximity search now produces identical results to the original implementation. Test with various term combinations and proximity distances to ensure behavioral parity.

#### Sub-task 11g.7: Verify Fuzzy Match Implementation (search_ops.rs:202-244)
**File:** `src/types/candle_chat/search/index/search_ops.rs`
**Lines:** 202-244 (fuzzy_match and levenshtein_distance methods)
**Issue:** Ensure fuzzy matching logic exactly matches original implementation
**Architecture:** Confirm fuzzy_match threshold calculation and levenshtein_distance implementation are identical
**Implementation:** Compare against original lines 477-518. Verify max_distance calculation formula and matrix implementation match exactly. Fix any discrepancies.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.8: QA: Verify Fuzzy Match Implementation
Act as an Objective QA Rust developer and rate the fuzzy match implementation on a scale of 1-10. Verify that fuzzy matching produces identical results to the original. Test with various string combinations and distance thresholds to ensure algorithmic parity.

#### Sub-task 11g.9: Fix Import Dependencies (mod.rs:1-17)
**File:** `src/types/candle_chat/search/index/mod.rs`
**Lines:** 1-17 (all module declarations and re-exports)
**Issue:** Ensure all imports and re-exports maintain original API surface
**Architecture:** Verify that external code can import and use ChatSearchIndex exactly as before
**Implementation:** Check that all public methods, structs, and traits are properly re-exported. Ensure IndexEntry and other types are accessible at the same import paths as original.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.10: QA: Verify Module Integration
Act as an Objective QA Rust developer and rate the module integration on a scale of 1-10. Verify that all imports work correctly and external code can use the decomposed modules exactly like the original file. Test import statements and API access patterns.

#### Sub-task 11g.11: Fix Cross-Module Method Access (all modules)
**File:** All modules in `src/types/candle_chat/search/index/`
**Lines:** Method declarations with pub(super) visibility
**Issue:** Ensure all methods are accessible where needed across modules
**Architecture:** Verify that search_and can call intersect_results, search_proximity can call check_proximity, etc.
**Implementation:** Review all method visibility declarations. Ensure methods called across module boundaries have proper pub(super) visibility. Fix any compilation errors related to method access.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.12: QA: Verify Cross-Module Method Access
Act as an Objective QA Rust developer and rate the cross-module method access on a scale of 1-10. Verify that all methods are properly accessible across modules and compilation succeeds. Test that method calls work correctly in all contexts.

#### Sub-task 11g.13: Verify SIMD Optimization Behavior (indexing.rs:107-117)
**File:** `src/types/candle_chat/search/index/indexing.rs`
**Lines:** 107-117 (SIMD threshold logic)
**Issue:** Ensure SIMD optimization triggers under same conditions as original
**Architecture:** Verify that simd_threshold loading and comparison logic matches original exactly
**Implementation:** Compare against original lines 234-236. Ensure >= comparison and threshold loading behavior is identical. Fix any discrepancies in condition evaluation.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.14: QA: Verify SIMD Optimization Behavior
Act as an Objective QA Rust developer and rate the SIMD optimization behavior on a scale of 1-10. Verify that SIMD processing triggers under identical conditions to the original. Test with various text sizes to ensure optimization thresholds work correctly.

#### Sub-task 11g.15: Verify Statistics Update Logic (indexing.rs:77-94)
**File:** `src/types/candle_chat/search/index/indexing.rs`
**Lines:** 77-94 (statistics update in add_message_stream)
**Issue:** Ensure statistics tracking behavior matches original exactly
**Architecture:** Verify counter increments, timestamp updates, and statistics calculation are identical
**Implementation:** Compare against original lines 180-189. Ensure document_count, index_update_counter, and statistics updates happen in same order with same values.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.16: QA: Verify Statistics Update Logic
Act as an Objective QA Rust developer and rate the statistics update logic on a scale of 1-10. Verify that statistics tracking produces identical results to the original. Test message indexing to ensure counters and timestamps update correctly.

#### Sub-task 11g.17: Comprehensive Compilation Verification
**File:** Entire `src/types/candle_chat/search/index/` module
**Lines:** All files
**Issue:** Ensure clean compilation with zero errors and zero warnings
**Architecture:** Verify all modules compile together cleanly with proper dependencies
**Implementation:** Run cargo check on the entire module. Fix any compilation errors, unused imports, or warnings. Ensure all method signatures are compatible across modules.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.18: QA: Verify Compilation Success
Act as an Objective QA Rust developer and rate the compilation verification on a scale of 1-10. Verify that the entire module compiles cleanly with zero errors and zero warnings. Test in both debug and release modes.

#### Sub-task 11g.19: Final Behavioral Verification Testing
**File:** All modules in decomposed search index
**Lines:** All implementations
**Issue:** Verify identical behavior through comprehensive testing
**Architecture:** Test all search operations, indexing, and statistics to ensure perfect parity
**Implementation:** Create test cases that compare original vs decomposed behavior. Test: tokenization, exact search, fuzzy search, proximity search, phrase search, sorting, and statistics. Verify identical results.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.20: QA: Verify Behavioral Parity
Act as an Objective QA Rust developer and rate the behavioral verification on a scale of 1-10. Verify that all search operations produce identical results to the original implementation. Confirm that the decomposition introduces zero functional changes.

#### Sub-task 11g.21: Safe Original File Deletion
**File:** `src/types/candle_chat/search/index_original.rs.backup`
**Lines:** N/A (file deletion)
**Issue:** Safely remove original file only after 100% verification
**Architecture:** Final cleanup step after complete verification
**Implementation:** Only delete the backup file after all previous verification tasks pass with 9/10 or higher ratings. Ensure decomposed modules are fully production-ready.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

#### Sub-task 11g.22: QA: Verify Safe Deletion
Act as an Objective QA Rust developer and rate the deletion safety on a scale of 1-10. Verify that the original file has been safely removed and that the decomposed modules provide complete functional replacement. Confirm that no functionality was lost in the process.

### Task 12: Fix sampling/mirostat.rs Future violations
**File:** `src/sampling/mirostat.rs`
**Lines:** 41, 45, 49 (future references)
**Implementation:** Replace Future references with AsyncStream<SampleResult> streaming patterns for Mirostat sampling.
**Architecture:** Convert Mirostat sampling to streaming pattern using AsyncStream, emit sampling results via emit! macro, handle sampling errors via handle_error! macro. Return AsyncStream<SampleResult> not AsyncStream<Result<SampleResult,E>>.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 13: QA Review sampling/mirostat.rs async compliance
Act as an Objective QA Rust developer and confirm sampling/mirostat.rs uses only fluent-ai-async streaming primitives and AsyncStream<T> pattern. Verify Mirostat sampling streams correctly. Rate production quality.

### Task 14: Fix sampling/gumbel.rs Future violations
**File:** `src/sampling/gumbel.rs`
**Lines:** 31 (future reference)
**Implementation:** Replace Future reference with AsyncStream<GumbelSample> streaming pattern for Gumbel sampling.
**Architecture:** Convert Gumbel sampling to streaming pattern using AsyncStream, implement streaming Gumbel distribution sampling. Return AsyncStream<GumbelSample> not AsyncStream<Result<GumbelSample,E>>.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 14: QA Review sampling/gumbel.rs async compliance
Act as an Objective QA Rust developer and confirm sampling/gumbel.rs uses only fluent-ai-async streaming primitives and AsyncStream<T> pattern. Verify Gumbel sampling architectural compliance.

### Task 15: Fix sampling/composite.rs Future violations
**File:** `src/sampling/composite.rs`
**Lines:** 22, 25, 202, 389, 391 (future references)
**Implementation:** Replace all Future references with AsyncStream<CompositeSample> streaming patterns for composite sampling.
**Architecture:** Convert composite sampling to streaming pattern, implement streaming composite sampling via AsyncStream, coordinate multiple sampling strategies via streaming. Return AsyncStream<CompositeSample> not AsyncStream<Result<CompositeSample,E>>.
DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

### Task 16: QA Review sampling/composite.rs async compliance
Act as an Objective QA Rust developer and confirm sampling/composite.rs uses only fluent-ai-async streaming primitives and AsyncStream<T> pattern. Verify composite sampling streams correctly. Rate zero-allocation patterns.
**Root Cause**: Incomplete error enum definition for handle_error! macro usage
**Architecture**: Add missing error variants to support comprehensive error handling
**Implementation**: Add `InternalError(String)` and `SubscriberNotFound(String)` to RealTimeError enum
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA.**

### TOKIO-6. Fix AtomicUsize clone issue in realtime.rs
**File**: `src/types/candle_chat/realtime.rs`
**Line**: 552
**Issue**: `AtomicUsize::clone()` method does not exist
**Root Cause**: Incorrect atomic usage - AtomicUsize doesn't implement Clone
**Architecture**: Use `AtomicUsize::load(Ordering::Acquire)` for atomic value access
**Implementation**: Change `self.queue_size_limit.clone()` to `self.queue_size_limit.load(Ordering::Acquire)`
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA.**

### TOKIO-7. Fix borrowed data escaping in candle_model.rs  
**File**: `src/model/core/candle_model.rs`
**Lines**: 187-203, 248-268
**Issue**: `self` reference escapes method body in AsyncStream requiring 'static lifetime
**Root Cause**: Using borrowed self in AsyncStream::with_channel closures
**Architecture**: Clone self data into owned values before AsyncStream closure
**Implementation**: Create self_clone with owned data before `AsyncStream::with_channel(move |sender| { ... })`
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA.**

---

## CRITICAL ASYNC ARCHITECTURE VIOLATIONS (ABSOLUTE PRIORITY 2) ⚡

### ASYNC-1. Fix async misuse in model_loader.rs (lines 27-33)
**File**: `src/model/core/model_loader.rs`
**Issue**: Box::pin(async move { ... }) pattern violates fluent-ai-async architecture
**Root Cause**: Using async/await patterns instead of AsyncTask streaming
**Architecture**: Replace with proper AsyncTask streaming implementation using spawn_task from fluent-ai-async
**Implementation**: Remove Box::pin(async move { ... }) and use AsyncTask::spawn with emit! and handle_error! macros
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.**

### ASYNC-1-QA. QA: Act as an Objective QA Rust developer and rate the async misuse fix in model_loader.rs on a scale of 1-10. Verify no Box::pin, async move, or .await patterns remain in streaming contexts. Confirm proper use of AsyncTask and emit! macros.

### ASYNC-2. Audit all streaming functions for async violations
**Files**: `src/generator.rs`, `src/types/candle_chat/macros.rs`, `src/model/core/*.rs`
**Issue**: Search for Box::pin, async move, .await patterns in streaming contexts
**Root Cause**: Potential async/await violations throughout streaming codebase
**Architecture**: Replace any violations with proper AsyncStream/AsyncTask patterns
**Implementation**: Comprehensive search and replace with fluent-ai-async primitives only
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA.**

### ASYNC-2-QA. QA: Act as an Objective QA Rust developer and rate the comprehensive async pattern audit on a scale of 1-10. Verify all streaming operations use only AsyncStream, AsyncTask, emit!, and handle_error! macros. Confirm zero async/await usage in streaming contexts.

### ASYNC-3. Replace all ? operator usage with handle_error! macro in streaming contexts
**Files**: `src/model/core/model_loader.rs`, `src/generator.rs`, any streaming functions
**Issue**: ? operators in streaming contexts violate fluent-ai-async architecture
**Root Cause**: Using Result propagation instead of streaming error handling
**Architecture**: Replace with handle_error! macro for proper streaming error handling
**Implementation**: Search for ? operators in streaming contexts and replace with handle_error! patterns
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA.**

### ASYNC-3-QA. QA: Act as an Objective QA Rust developer and rate the error handling architecture compliance on a scale of 1-10. Verify all streaming errors use handle_error! macro and no ? operators remain in streaming contexts. Confirm proper error propagation through streams.

---

## ERRORS (44 total) - CRITICAL PRIORITY

### 1. Registry API Mismatch - Multiple occurrences
**Files**: `src/types/candle_model/resolver.rs`
**Issue**: Registry.get() method signature mismatch - expects 1 arg but given 2
**Lines**: 269, 282, 296, 316, 351
**Root Cause**: Using wrong ModelRegistry API

### 2. Missing find_all method on ModelRegistry  
**File**: `src/types/candle_model/resolver.rs:402`
**Issue**: `no method named 'find_all' found for reference &model::registry::ModelRegistry`

### 3. Type conversion error - ModelError to CandleCompletionError
**File**: `src/types/candle_model/resolver.rs:423`
**Issue**: Returning wrong error type, needs .into() conversion

### 4. Engine type mismatch
**File**: `src/types/candle_engine.rs:265`
**Issue**: Expected `EngineResult<CandleCompletionResponse>` but got ValidationError

### 5. VarBuilder tensor_metadata field type mismatch
**Files**: `src/var_builder.rs`
**Lines**: 839, 934, 952
**Issue**: Expected `HashMap` but got `Arc<HashMap>` 

### 6. VarBuilder field type mismatches
**File**: `src/var_builder.rs`
**Lines**: 864 (u64 vs usize), 908 (Dtype vs DType)

### 7. Missing TensorView methods
**File**: `src/var_builder.rs`
**Lines**: 909, 910
**Issue**: `data_offsets()` method doesn't exist on TensorView

### 8. Vec iterator method missing
**File**: `src/var_builder.rs:925`
**Issue**: Vec doesn't have .map() - needs .into_iter().map()

### 9. VarBuilder Try trait not implemented
**Files**: `src/var_builder.rs`
**Lines**: 922, 948
**Issue**: Can't use ? operator on VarBuilderArgs

### 10. VarBuilder static method usage
**Files**: `src/var_builder.rs`
**Lines**: 938, 955
**Issue**: Using instance methods as static methods

### 11. Client lifetime and borrowing issues
**File**: `src/client.rs`
**Lines**: 687, 667
**Issue**: Generator doesn't live long enough + lifetime variance

### 12. CandleError string borrow issues
**File**: `src/logits.rs`
**Lines**: 143, 159, 166, 190
**Issue**: Temporary string values freed while borrowed

### 13. Model loader borrowing conflicts
**File**: `src/model/fluent/kimi_k2/loader.rs`
**Lines**: 109, 113, 127, 58
**Issue**: Multiple borrow/move conflicts with shards data

### 14. KimiK2 model topk results lifetime
**File**: `src/model/fluent/kimi_k2/model.rs:296`
**Issue**: indexed vec doesn't live long enough

### 15. Format string temporary value issues
**File**: `src/model/loading/mod.rs`
**Lines**: 388, 375
**Issue**: format!() temporary strings freed while borrowed

### 16. Composite processor mutable borrow
**File**: `src/sampling/composite.rs:144`
**Issue**: Cannot borrow immutable iterator reference as mutable

### 17. Chat search method lifetime escape
**File**: `src/types/candle_chat/search.rs:1510`
**Issue**: self reference escapes method body in AsyncStream

---

## WARNINGS (23 total) - MUST BE FIXED

### 18. Unreachable code after handle_error! macros
**File**: `src/types/candle_context/provider.rs`
**Lines**: 444, 468
**Issue**: Statements after handle_error! macro are unreachable

### 19. Unused import LogitsProcessor
**File**: `src/sampling/simd.rs:16`
**Issue**: Imported but never used

### 20. Unused variables in composite processor
**File**: `src/sampling/composite.rs`
**Lines**: 105 (token_ids), 106 (position)

### 21. Unused variable in gumbel sampling
**File**: `src/sampling/gumbel.rs:197`
**Issues**: unused one_hot variable + unnecessary mut

### 22. Unused/assigned but never used in mirostat
**File**: `src/sampling/mirostat.rs`
**Lines**: 413 (total_suppressed), 482 (context)

### 23. Multiple unused msg variables in SIMD
**File**: `src/sampling/simd.rs`
**Lines**: 36, 39, 40, 43, 46, 49 (all msg parameters)

### 24. Unused parameters in SIMD methods
**File**: `src/sampling/simd.rs`
**Lines**: 85 (logits), 86 (context), 129 (temperature), 216 (size), 217 (iterations)

### 25. Unused variables in chat search
**File**: `src/types/candle_chat/search.rs`  
**Lines**: 306 (index), 446 (query_time), 544 (fuzzy)

---

## IMPLEMENTATION APPROACH

1. **Research Phase**: Understand each error's context and all call sites
2. **Fix Phase**: Implement production-quality solutions
3. **QA Phase**: Rate each fix 1-10, redo if < 9
4. **Verify Phase**: Confirm fix with `cargo check`

## PERFORMANCE REQUIREMENTS
- Zero allocation where possible  
- Non-locking, asynchronous code
- Production-ready quality
- No mocking/fake implementations

---

## CRITICAL ASYNCSTREAM ARCHITECTURE COMPLIANCE

### 26. URGENT: Remove forbidden futures_util::StreamExt imports from client.rs
**Files**: `src/client.rs` lines 12, 306
**Issue**: Using FORBIDDEN futures_util::StreamExt imports - violates AsyncStream-only architecture  
**Architecture**: fluent-ai-async provides all necessary streaming functionality - futures_util is BANNED
**Implementation**: Remove both `use futures_util::StreamExt;` imports and replace any .next().await with AsyncStream patterns
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 27. Act as an Objective QA Rust developer - Rate futures_util removal. Verify all futures_util imports are removed and no futures ecosystem dependencies remain in client.rs.

### 28. URGENT: Convert client.rs complete_token_stream method from async fn to AsyncStream
**Files**: `src/client.rs` lines 392-424  
**Issue**: Method signature `async fn complete_token_stream(...) -> CandleResult<TokenOutputStream>` violates AsyncStream-only architecture
**Architecture**: Convert to `fn complete_token_stream(...) -> AsyncStream<TokenOutputStream>` using AsyncStream::with_channel
**Implementation**: Remove async fn, wrap logic in `AsyncStream::with_channel(move |sender| { /* sync code */ })`, use emit!(sender, token_stream)
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 29. Act as an Objective QA Rust developer - Rate complete_token_stream conversion. Verify method returns AsyncStream, uses emit! pattern, and contains no async/await patterns.

### 30. SYSTEMATIC: Convert all Box::pin(async move {}) patterns to AsyncStream::with_channel
**Files**: `src/types/candle_chat/macros.rs:517`, `src/types/candle_context/provider_backup.rs:3`, `src/types/candle_chat/chat/macros.rs:518`
**Issue**: Using forbidden Box::pin(async move {}) pattern instead of AsyncStream::with_channel
**Architecture**: Replace with `AsyncStream::with_channel(move |sender| { /* synchronous code */ })` pattern
**Implementation**: Remove all async/await, use pure synchronous code with sender.send() for value emission

### 27. SYSTEMATIC: Eliminate all tokio::spawn usage
**Files**: `src/generator.rs:832`, `src/client.rs:407`, `src/types/candle_context/provider_backup.rs:4`, `src/types/candle_chat/config.rs:863`, `src/types/candle_chat/realtime.rs:348,637,1006,1219,1768`, `src/types/candle_chat/chat/config.rs:863`, `src/types/candle_chat/chat/realtime.rs:348,637,1006,1219,1768`
**Issue**: Using forbidden tokio::spawn instead of AsyncStream::with_channel internal std::thread::spawn
**Architecture**: Convert to AsyncStream::with_channel pattern - AsyncStream uses std::thread::spawn internally
**Implementation**: Replace tokio::spawn(async move { ... }) with AsyncStream::with_channel(move |sender| { ... })

### 28. SYSTEMATIC: Eliminate all .await usage throughout codebase
**Files**: Multiple files with hundreds of .await calls (generator.rs, tokenizer.rs, client.rs, model/core/, streaming/, types/candle_chat/, types/candle_context/, types/candle_model/)
**Issue**: Using forbidden .await in AsyncStream-only architecture
**Architecture**: AsyncStream::with_channel uses pure synchronous code - NO async/await allowed
**Implementation**: Convert all async functions to return AsyncStream<T> and use synchronous code inside with_channel closures

### 29. SYSTEMATIC: Fix all current compilation errors using AsyncStream patterns
**Files**: As identified in items 1-17 above
**Issue**: Current errors caused by mixing async/await patterns with AsyncStream
**Architecture**: Apply AsyncStream::with_channel pattern to resolve borrowing, lifetime, and type issues
**Implementation**: Convert problematic async operations to pure synchronous AsyncStream patterns

### 30. PERFORMANCE: Ensure zero-allocation, lock-free AsyncStream usage
**Architecture**: AsyncStream uses crossbeam_queue::ArrayQueue for zero-allocation, lock-free operation
**Implementation**: Verify all AsyncStream::with_channel usage follows zero-allocation patterns, use emit! macro for optimal performance

---

## DETAILED ASYNCSTREAM CONVERSION IMPLEMENTATION

### 31. Convert HTTP Client Operations to fluent_ai_http3 streaming
**File**: `src/client.rs` lines 200-400
**Issue**: Replace any reqwest/hyper usage with fluent_ai_http3 streaming patterns
**Architecture**: Use `Http3::json().body(request).post(url)` for single requests, `Http3::json().body(request).post(url).stream()` for streaming
**Implementation**: Convert all HTTP operations to use fluent_ai_http3 with proper error handling via handle_error! macro
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 32. Act as an Objective QA Rust developer - Rate HTTP client conversion work. Verify all HTTP calls now use fluent_ai_http3 streaming patterns correctly and no async/await patterns remain.

### 33. Convert completion methods in src/client.rs to AsyncStream patterns
**File**: `src/client.rs` lines 650-750
**Issue**: Replace `async fn complete()` with `fn complete() -> AsyncStream<CompletionChunk>`
**Architecture**: Use `AsyncStream::with_channel(move |sender| { ... })` pattern with synchronous completion processing
**Implementation**: Convert completion logic to emit chunks via `emit!(sender, chunk)` and handle errors via `handle_error!(err, "completion failed")`
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 34. Act as an Objective QA Rust developer - Rate completion method conversion. Verify all completion operations now return AsyncStream and use proper emit!/handle_error! patterns.

### 35. Convert async generation methods in src/generator.rs to AsyncStream patterns
**File**: `src/generator.rs` lines 700-900
**Issue**: Replace `async fn generate()` with `fn generate() -> AsyncStream<GeneratedToken>`
**Architecture**: Use `AsyncStream::with_channel(move |sender| { ... })` for token generation streaming
**Implementation**: Convert token generation to emit tokens in real-time via `emit!(sender, token)` with proper error handling
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 36. Act as an Objective QA Rust developer - Rate generator conversion. Verify token generation now uses streaming patterns and emits tokens correctly without async/await.

### 37. Convert async model loading in src/model/loading/mod.rs to streaming patterns
**File**: `src/model/loading/mod.rs` lines 300-500
**Issue**: Replace `async fn load_model()` with `fn load_model() -> AsyncStream<ModelLoadEvent>`
**Architecture**: Use `AsyncStream::with_channel(move |sender| { ... })` for model loading progress streaming
**Implementation**: Stream loading events (start, weights_loaded, quantization_applied, ready) via `emit!(sender, ModelLoadEvent::Progress(percent))`
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 38. Act as an Objective QA Rust developer - Rate model loading conversion. Verify model loading now streams progress events and uses AsyncStream patterns correctly.

### 39. Convert async model loading in src/model/fluent/kimi_k2/loader.rs to streaming patterns
**File**: `src/model/fluent/kimi_k2/loader.rs` lines 100-200
**Issue**: Replace async loading patterns with `AsyncStream<ModelShard>` streaming
**Architecture**: Use `AsyncStream::with_channel(move |sender| { ... })` with file streaming for large model files
**Implementation**: Convert model shard loading to emit shards as loaded via `emit!(sender, ModelShard::new(data))`
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 40. Act as an Objective QA Rust developer - Rate KimiK2 loader conversion. Verify model shard loading now uses streaming patterns and handles large files efficiently.

### 41. Complete AsyncStream conversion in src/types/candle_context/provider.rs
**File**: `src/types/candle_context/provider.rs` lines 700-900
**Issue**: Finish converting remaining async patterns to `AsyncStream::with_channel`
**Architecture**: Complete the streaming context loading pattern with proper file I/O streaming
**Implementation**: Ensure all context loading operations use `AsyncStream::with_channel(move |sender| { ... })` with comprehensive error handling
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 42. Act as an Objective QA Rust developer - Rate context provider conversion completion. Verify all AsyncStream patterns are properly implemented and no async/await remains.

### 43. Convert all async file operations to AsyncStream<FileChunk> patterns
**Files**: Throughout codebase - any file I/O operations
**Issue**: Replace async file I/O with streaming patterns
**Architecture**: Use `AsyncStream::with_channel(move |sender| { ... })` with `BufReader` for efficient file streaming
**Implementation**: Convert file operations to emit chunks via `emit!(sender, FileChunk::new(buffer))` with proper error handling
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 44. Act as an Objective QA Rust developer - Rate file I/O conversion. Verify all file operations now use streaming patterns and emit chunks efficiently without blocking.

### 45. Convert async search methods in src/types/candle_chat/search.rs to streaming patterns
**File**: `src/types/candle_chat/search.rs` lines 1400-1600
**Issue**: Replace async search with `AsyncStream<SearchResult>` streaming
**Architecture**: Use `AsyncStream::with_channel(move |sender| { ... })` for search result streaming
**Implementation**: Convert search processing to emit results as found via `emit!(sender, SearchResult::new(...))`
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 46. Act as an Objective QA Rust developer - Rate search operations conversion. Verify search now streams results in real-time and handles large result sets efficiently.

### 47. Convert async chat processing in src/types/candle_chat/ to streaming patterns
**Files**: `src/types/candle_chat/commands/`, `src/types/candle_chat/config.rs`, `src/types/candle_chat/search.rs`
**Issue**: Convert async chat operations to `AsyncStream<ChatMessage>` streaming
**Architecture**: Use `AsyncStream::with_channel(move |sender| { ... })` for real-time chat message processing
**Implementation**: Convert chat operations to emit messages and progress via `emit!(sender, ChatMessage::new(...))`
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 48. Act as an Objective QA Rust developer - Rate chat operations conversion. Verify chat processing now uses streaming patterns for real-time message handling.

### 49. Remove async runtime dependencies from Cargo.toml
**File**: `Cargo.toml`
**Issue**: Remove tokio, async-std, or other async runtime dependencies
**Architecture**: Ensure only fluent-ai-async and std::thread are used for async operations
**Implementation**: Remove async runtime dependencies while ensuring build still works correctly
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 50. Act as an Objective QA Rust developer - Rate dependency cleanup. Verify no async runtime dependencies remain and the build still works correctly.

### 51. Find and convert all .await call sites to .collect() or streaming patterns
**Files**: Throughout codebase - search for ".await"
**Issue**: Replace all `.await` usage with `.collect()` for single results or proper streaming consumption
**Architecture**: Use `stream.collect()` for await-like behavior, `stream.try_next()` for non-blocking consumption
**Implementation**: Replace `result = async_call().await?` with `result = async_stream_call().collect()` and handle errors appropriately
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 52. Act as an Objective QA Rust developer - Rate .await conversion. Verify no .await calls remain and all replacements use correct AsyncStream/AsyncTask patterns.

### 53. Convert Result<T,E> propagation to emit!/handle_error! patterns in streaming contexts
**Files**: Throughout codebase - streaming operations with Result propagation
**Issue**: Replace `?` operator with explicit error handling in streaming contexts
**Architecture**: Use `handle_error!(err, "operation failed")` instead of returning errors in streams
**Implementation**: Ensure no `Result<T, E>` types inside `AsyncStream<Result<T, E>>` - use proper error emission patterns
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 54. Act as an Objective QA Rust developer - Rate error handling conversion. Verify all streaming error handling now uses emit!/handle_error! patterns correctly.

### 55. Run comprehensive grep audit for remaining async/await/Future patterns
**Files**: Entire codebase
**Issue**: Verify zero async/await/Future patterns remain
**Architecture**: Ensure complete AsyncStream-only architecture compliance
**Implementation**: Search for "async fn", ".await", "Future<" patterns and document any remaining patterns requiring conversion
**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

### 56.