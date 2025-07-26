# fluent-ai-candle TODO: Compilation Errors + Documentation

## 🎯 CRITICAL: FINAL 2 COMPILATION ERRORS TO FIX

**BREAKTHROUGH PROGRESS**: Reduced from 159 errors to just 2 errors (98.7% reduction!)

### Remaining Critical Errors

#### 1. Missing `export` method on HistoryExporter
- **File**: `src/types/candle_chat/chat/search/manager.rs`
- **Line**: 118
- **Error**: `error[E0599]: no method named 'export' found for struct 'exporter::HistoryExporter'`
- **Issue**: `exporter.export(&options)` call fails
- **Fix Strategy**: Implement missing `export` method with zero-allocation streaming pattern
- **Architecture**: Use `AsyncStream<ExportResult>` with fluent-ai-async patterns

#### 2. Type mismatch in search result collection
- **File**: `src/types/candle_chat/chat/search/searcher.rs` 
- **Line**: 160
- **Error**: `error[E0308]: mismatched types: expected Result, found Vec<SearchResult>`
- **Issue**: `search_stream.collect()` returns `Vec` but code expects `Result`
- **Fix Strategy**: Adjust pattern matching to handle `Vec<SearchResult>` directly
- **Architecture**: Use proper error handling with streaming collection patterns

### Implementation Plan

1. **Fix HistoryExporter.export method** 
   - Add `pub fn export(&self, options: &ExportOptions) -> AsyncStream<ExportResult>`
   - Use zero-allocation streaming with `AsyncStream::with_channel`
   - Implement proper error handling without `unwrap()` or `expect()`

2. **Fix searcher type mismatch**
   - Change `if let Ok(stream_results) = search_stream.collect()` pattern
   - Handle `Vec<SearchResult>` directly or add proper error wrapping
   - Ensure blazing-fast performance with no allocations in hot path

---

## ✅ MAJOR COMPILATION FIXES COMPLETED

### Phase 1: HashMap Import Errors (COMPLETED)
- ✅ Fixed HashMap imports in processor.rs and context_impls.rs  
- ✅ Fixed HashMap::new() methods with ahash::RandomState issues
- ✅ Fixed type mismatches in external.rs (HeaderName vs &str)

### Phase 2: Type System Issues (COMPLETED)  
- ✅ Fixed generic argument issues in macros/context.rs
- ✅ Fixed iterator collection issues in macros/system.rs
- ✅ Fixed missing fields in MacroMetadata initializer
- ✅ Fixed duplicate MacroMetadata structs and field conflicts

### Phase 3: Access and Method Issues (COMPLETED)
- ✅ Fixed private field access in ActionHandlerRegistry
- ✅ Fixed missing message field and clone methods in search
- ✅ Added ConsistentCounter Clone implementation
- ✅ Resolved Arc import scope issues

---

## 🎯 OBJECTIVE: DOCUMENT EVERY SINGLE METHOD

**STATUS**: 2,375 missing documentation warnings detected with `#![deny(missing_docs)]`

## Top Files by Missing Documentation Count

| Count | File |
|-------|------|
| 129 | packages/fluent-ai-candle/src/types/candle_chat/templates/core.rs |
| 129 | packages/fluent-ai-candle/src/types/candle_chat/chat/templates/core.rs |
| 98 | packages/fluent-ai-candle/src/types/candle_chat/macros/types.rs |
| 84 | packages/fluent-ai-candle/src/types/candle_chat/commands/types/events.rs |
| 74 | packages/fluent-ai-candle/src/types/candle_chat/chat/macros/types.rs |
| 59 | packages/fluent-ai-candle/src/types/candle_chat/chat/commands/types/actions.rs |
| 45 | packages/fluent-ai-candle/src/model/fluent/kimi_k2/model.rs |
| 36 | packages/fluent-ai-candle/src/types/candle_context/provider/events.rs |
| 35 | packages/fluent-ai-candle/src/types/candle_chat/realtime/events.rs |
| 30 | packages/fluent-ai-candle/src/types/candle_context/chunk.rs |
| 30 | packages/fluent-ai-candle/src/model/cache/mod.rs |
| 29 | packages/fluent-ai-candle/src/types/candle_chat/commands/validation.rs |
| 29 | packages/fluent-ai-candle/src/types/candle_chat/chat/commands/validation.rs |
| 26 | packages/fluent-ai-candle/src/constraints/json.rs |
| 25 | packages/fluent-ai-candle/src/types/candle_chat/realtime/errors.rs |
| 25 | packages/fluent-ai-candle/src/types/candle_chat/chat/commands/types/events.rs |
| 21 | packages/fluent-ai-candle/src/types/candle_completion/error.rs |
| 21 | packages/fluent-ai-candle/src/types/candle_chat/chat/commands/types/core.rs |
| 21 | packages/fluent-ai-candle/src/model/error.rs |
| 20 | packages/fluent-ai-candle/src/types/candle_chat/chat/macros/errors.rs |

## Documentation Strategy

### Phase 1: High-Impact Files (100+ warnings)
- [x] src/types/candle_chat/templates/core.rs (129 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/chat/templates/core.rs (129 warnings) ✅ COMPLETED

### Phase 2: Major Files (50-99 warnings)  
- [x] src/types/candle_chat/macros/types.rs (98 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/commands/types/events.rs (84 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/chat/macros/types.rs (74 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/chat/commands/types/actions.rs (59 warnings) ✅ COMPLETED

### Phase 3: Significant Files (30-49 warnings)
- [x] src/model/fluent/kimi_k2/model.rs (45 warnings) ✅ COMPLETED
- [x] src/types/candle_context/provider/events.rs (36 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/realtime/events.rs (35 warnings) ✅ COMPLETED
- [x] src/types/candle_context/chunk.rs (30 warnings) ✅ COMPLETED
- [x] src/model/cache/mod.rs (30 warnings) ✅ COMPLETED

### Phase 4: Moderate Files (20-29 warnings)
- [x] src/types/candle_chat/commands/validation.rs (29 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/chat/commands/validation.rs (29 warnings) ✅ COMPLETED
- [x] src/constraints/json.rs (26 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/realtime/errors.rs (25 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/chat/commands/types/events.rs (25 warnings) ✅ COMPLETED
- [x] src/types/candle_completion/error.rs (21 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/chat/commands/types/core.rs (21 warnings) ✅ COMPLETED
- [x] src/model/error.rs (21 warnings) ✅ COMPLETED
- [x] src/types/candle_chat/chat/macros/errors.rs (20 warnings) ✅ COMPLETED

## ✅ PHASE 4 COMPLETE! All Moderate Files (20-29 warnings) DOCUMENTED

### Phase 5: Remaining Files (<20 warnings)
- [ ] All other files with <20 warnings each (~1,370 warnings remaining)

## Progress Tracking

Total warnings: 2,375
- [x] Phase 1: 258 warnings (11%) ✅ COMPLETED
- [x] Phase 2: 315 warnings (13%) ✅ COMPLETED
- [x] Phase 3: 176 warnings (7%) ✅ COMPLETED
- [x] Phase 4: 256 warnings (11%) ✅ COMPLETED
- [ ] Phase 5: Remaining ~1,370 warnings (58%)

## Documentation Standards

Each method/struct/enum/field must have:
- /// Brief description of purpose
- /// Parameters (if any) with descriptions  
- /// Return value description
- /// Examples for complex methods
- /// Errors/panics documentation where applicable
- /// Performance notes for critical paths