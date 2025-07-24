# COMPREHENSIVE ERROR AND WARNING FIX TODO

## OBJECTIVE: 0 ERRORS, 0 WARNINGS ✨

**Current Status**: 44 ERRORS + 23 WARNINGS = 67 TOTAL ISSUES 😱

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

### 26. SYSTEMATIC: Convert all Box::pin(async move {}) patterns to AsyncStream::with_channel
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

**Next Steps**: Execute AsyncStream conversion systematically, then fix remaining compilation issues 🎯