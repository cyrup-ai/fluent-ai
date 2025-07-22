# TODO: Fix All Errors and Warnings

## OBJECTIVE: Achieve 0 (Zero) errors and 0 (Zero) warnings

**Current Status**: 195 errors + 59 warnings in fluent_ai_candle, plus warnings in other crates

---

## CRITICAL BLOCKING ERRORS (Must fix first)

### 1. Fix AsyncStream private import errors in fluent-ai-candle
- **File**: `packages/fluent-ai-candle/src/client.rs`
- **Line**: 15
- **Error**: `error[E0603]: struct import 'AsyncStream' is private`
- **Root Cause**: Importing AsyncStream from domain_stubs which imports from fluent_ai_async instead of canonical fluent-ai-async
- **Fix**: Update domain_stubs.rs to import AsyncStream from fluent-ai-async, ensure fluent-ai-async is added as dependency
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 2. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 3. Fix AsyncStream private import errors in fluent-ai-candle hub.rs
- **File**: `packages/fluent-ai-candle/src/hub.rs`
- **Line**: 28
- **Error**: `error[E0603]: struct import 'AsyncStream' is private`
- **Root Cause**: Same as above - importing from wrong location
- **Fix**: Update import to use canonical AsyncStream from fluent-ai-async via domain_stubs
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 4. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 5. Fix const function error in var_builder.rs
- **File**: `packages/fluent-ai-candle/src/var_builder.rs`
- **Line**: 1174
- **Error**: `error[E0015]: cannot call non-const associated function 'var_builder::VarBuilderConfig::new' in constant functions`
- **Root Cause**: Attempting to call non-const function in const context
- **Fix**: Either make VarBuilderConfig::new const or remove const from the calling function
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 6. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

---

## FLUENT-AI-CANDLE WARNINGS (59 total)

### 7. Fix unused import: tokio_stream::wrappers::UnboundedReceiverStream
- **File**: `packages/fluent-ai-candle/src/client.rs`
- **Line**: 4
- **Warning**: `unused import: 'tokio_stream::wrappers::UnboundedReceiverStream'`
- **Fix**: Remove unused import or implement the functionality that needs it
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 8. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 9. Fix unused import: std::pin::Pin
- **File**: `packages/fluent-ai-candle/src/client.rs`
- **Line**: 6
- **Warning**: `unused import: 'std::pin::Pin'`
- **Fix**: Remove unused import or implement the functionality that needs it
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 10. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 11. Fix unused import: ChatMessage in generator.rs
- **File**: `packages/fluent-ai-candle/src/generator.rs`
- **Line**: 13
- **Warning**: `unused import: 'ChatMessage'`
- **Fix**: Remove unused import or implement the functionality that needs it
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 12. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 13. Fix unused import: tokio::time::timeout in hub.rs
- **File**: `packages/fluent-ai-candle/src/hub.rs`
- **Line**: 25
- **Warning**: `unused import: 'tokio::time::timeout'`
- **Fix**: Remove unused import or implement the functionality that needs it
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 14. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 15. Fix unused import: HttpMethod in hub.rs
- **File**: `packages/fluent-ai-candle/src/hub.rs`
- **Line**: 26
- **Warning**: `unused import: 'HttpMethod'`
- **Fix**: Remove unused import or implement the functionality that needs it
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 16. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 17. Fix unused imports: CandleError and CandleResult in hub.rs
- **File**: `packages/fluent-ai-candle/src/hub.rs`
- **Line**: 31
- **Warning**: `unused imports: 'CandleError' and 'CandleResult'`
- **Fix**: Remove unused imports or implement the functionality that needs them
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 18. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 19. Fix unused import: Device in kv_cache/mod.rs
- **File**: `packages/fluent-ai-candle/src/kv_cache/mod.rs`
- **Line**: 78
- **Warning**: `unused import: 'Device'`
- **Fix**: Remove unused import or implement the functionality that needs it
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 20. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 21. Fix unused imports: ModelConfig and ModelType in model/loading/mod.rs
- **File**: `packages/fluent-ai-candle/src/model/loading/mod.rs`
- **Line**: 22
- **Warning**: `unused imports: 'ModelConfig' and 'ModelType'`
- **Fix**: Remove unused imports or implement the functionality that needs them
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 22. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 23. Fix unused imports: GenerationMetrics and ModelPerformanceStats in model/core/mod.rs
- **File**: `packages/fluent-ai-candle/src/model/core/mod.rs`
- **Line**: 25
- **Warning**: `unused imports: 'GenerationMetrics' and 'ModelPerformanceStats'`
- **Fix**: Remove unused imports or implement the functionality that needs them
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 24. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 25. Fix unnecessary parentheses in repetition_penalty.rs
- **File**: `packages/fluent-ai-candle/src/processing/processors/repetition_penalty.rs`
- **Line**: 197
- **Warning**: `unnecessary parentheses around assigned value`
- **Fix**: Remove unnecessary parentheses
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 26. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 27. Fix unused variables in progress/mod.rs (multiple)
- **File**: `packages/fluent-ai-candle/src/progress/mod.rs`
- **Lines**: 345, 346, 347, 348
- **Warnings**: Multiple unused variables: `_total_parameters`, `_quantized_layers`, `_total_layers`, `_compression_ratio`
- **Fix**: Either implement the functionality that uses these variables or remove them if truly unused
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 28. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 29. Fix unused variable: scaled_logits in sampling/gumbel.rs
- **File**: `packages/fluent-ai-candle/src/sampling/gumbel.rs`
- **Line**: 152
- **Warning**: `unused variable: 'scaled_logits'`
- **Fix**: Either implement the functionality that uses this variable or remove it if truly unused
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 30. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 31. Fix unused variable: config in sampling/mirostat.rs
- **File**: `packages/fluent-ai-candle/src/sampling/mirostat.rs`
- **Line**: 522
- **Warning**: `unused variable: 'config'`
- **Fix**: Either implement the functionality that uses this variable or remove it if truly unused
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 32. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 33. Fix unused variable: chunk in streaming/mod.rs
- **File**: `packages/fluent-ai-candle/src/streaming/mod.rs`
- **Line**: 664
- **Warning**: `unused variable: 'chunk'`
- **Fix**: Either implement the functionality that uses this variable or remove it if truly unused
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 34. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 35. Fix unnecessary mutable variable in streaming/mod.rs
- **File**: `packages/fluent-ai-candle/src/streaming/mod.rs`
- **Line**: 725
- **Warning**: `variable does not need to be mutable`
- **Fix**: Remove mut keyword if variable is never mutated
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 36. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 37. Fix unused variables in var_builder.rs (multiple)
- **File**: `packages/fluent-ai-candle/src/var_builder.rs`
- **Lines**: 1120, 1157, 1294
- **Warnings**: Multiple unused variables: `inner`, `name`, `arch`
- **Fix**: Either implement the functionality that uses these variables or remove them if truly unused
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 38. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

---

## OTHER CRATE WARNINGS

### 39. Fix static_mut_refs warning in fluent-ai-async
- **File**: `packages/fluent-ai-async/src/thread_pool.rs`
- **Line**: 74
- **Warning**: `creating a shared reference to mutable static`
- **Fix**: Replace mutable static with safer alternative (Arc<Mutex<T>>, OnceCell, etc.)
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 40. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 41. Fix unused variable: start in fluent-ai-simd
- **File**: `packages/fluent-ai-simd/src/benchmark/mod.rs`
- **Line**: 33
- **Warning**: `unused variable: 'start'`
- **Fix**: Either implement the functionality that uses this variable or remove it if truly unused
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 42. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 43. Fix unused import: warn in progresshub-client-quic
- **File**: `/Volumes/samsung_t9/progresshub/client_quic/src/client.rs`
- **Line**: 18
- **Warning**: `unused import: 'warn'`
- **Fix**: Remove unused import or implement the functionality that needs it
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 44. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 45. Fix unused imports in progresshub-client-quic chunk_manager.rs
- **File**: `/Volumes/samsung_t9/progresshub/client_quic/src/chunk_manager.rs`
- **Line**: 27-28
- **Warning**: `unused imports: 'StateError' and 'client::QuicClientError'`
- **Fix**: Remove unused imports or implement the functionality that needs them
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 46. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 47. Fix unused variable: status in progresshub-client-quic
- **File**: `/Volumes/samsung_t9/progresshub/client_quic/src/chunk_manager.rs`
- **Line**: 620
- **Warning**: `unused variable: 'status'`
- **Fix**: Either implement the functionality that uses this variable or prefix with underscore if intentionally unused
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 48. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

---

## VALIDATION TASKS

### 49. Run cargo check to verify all errors and warnings are fixed
- **Command**: `cargo check --message-format short`
- **Expected Result**: 0 errors, 0 warnings
- **Working Directory**: `/Volumes/samsung_t9/fluent-ai/`
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 50. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

### 51. Test that the code actually works by running it like an end user
- **Command**: `cargo run` or appropriate test command
- **Expected Result**: Application runs without errors
- **Working Directory**: `/Volumes/samsung_t9/fluent-ai/`
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

### 52. **Act as an Objective Rust Expert and rate the quality of the fix on a scale of 1 - 10 (complete failure through significant improvement). Provide specific feedback on any issues or truly great work (objectively without bragging).**

---

## NOTES

- **CURRENT STATUS**: 195 errors + 59 warnings in fluent_ai_candle, plus warnings in other crates
- **PRIORITY**: Fix critical blocking errors first (AsyncStream imports, const function error)
- **SUCCESS CRITERIA**: 0 errors, 0 warnings, code actually runs
- **CONSTRAINTS**: No blocking code, production quality, minimal surgical changes only