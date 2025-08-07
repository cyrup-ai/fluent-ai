# CHAT.md - Chat System Unification Plan

## OBJECTIVE
Merge ./src/chat/ superior features into ./src/domain/chat/ to create one unified, enterprise-grade chat system. Remove ./src/chat/ entirely upon completion.

## ARCHITECTURE NOTES
- **Target System**: ./src/domain/chat/ (live/connected to agent builders)
- **Source System**: ./src/chat/ (contains superior search/filtering features)
- **Naming Convention**: All types use CandleXxx prefixes
- **Concurrency**: Use DashMap, atomic operations, lock-free patterns
- **Streaming**: AsyncStream-only architecture, no Future patterns
- **Error Handling**: Never use unwrap() in src/, never use expect() in src/

---

## PHASE 1: PREPARATION & ANALYSIS

### 1. Audit Current Import Dependencies
**File**: Create `./migration_analysis.md`
- [ ] Scan all files importing `use crate::chat::` vs `use crate::domain::chat::`
- [ ] Document all type mappings (Message -> CandleMessage, etc.)
- [ ] Identify files that import from both systems
- [ ] Create complete dependency graph of cross-system imports

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify the import analysis is complete, accurate, and identifies all cross-dependencies without adding unnecessary items.

### 2. Create Type Migration Mapping
**File**: Update `./src/domain/chat/mod.rs` lines 1-67
- [ ] Document exact type equivalencies between systems
- [ ] Plan CandleXxx naming for all migrated types
- [ ] Verify no naming conflicts exist in domain system

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify the type mapping is complete and follows Candle naming conventions without introducing breaking changes.

---

## PHASE 2: SEARCH SYSTEM MIGRATION (HIGHEST PRIORITY)

### 3. Migrate EnhancedHistoryManager
**File**: Create `./src/domain/chat/search/manager/mod.rs` 
**Source**: Copy from `./src/chat/search/manager/mod.rs` (entire directory)
- [ ] Copy entire manager/ directory structure to domain/chat/search/
- [ ] Update all internal imports to use crate::domain::chat::
- [ ] Add Candle prefixes to all public types
- [ ] Replace HashMap with DashMap for concurrency
- [ ] Ensure all methods return proper CandleXxx types

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify EnhancedHistoryManager migration preserves all functionality while following domain system patterns.

### 4. Migrate ConversationTagger  
**File**: Create `./src/domain/chat/search/tagger/mod.rs`
**Source**: Copy from `./src/chat/search/tagger/mod.rs` (entire directory)
- [ ] Copy entire tagger/ directory structure to domain/chat/search/
- [ ] Update all internal imports to use crate::domain::chat::
- [ ] Add Candle prefixes to all public types  
- [ ] Replace atomic operations with domain system patterns
- [ ] Integrate with domain search index

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify ConversationTagger migration maintains tagging functionality and performance characteristics.

### 5. Enhance Domain ChatSearcher with Advanced Features
**File**: Update `./src/domain/chat/search/mod.rs` lines 40-127
- [ ] Add missing query operators: Not, Phrase, Proximity from original
- [ ] Implement comprehensive filtering system (date, user, session, tag, content)
- [ ] Add multiple sorting options (Relevance, DateDesc/Asc, UserDesc/Asc)
- [ ] Implement pagination support (offset, max_results)
- [ ] Add statistics tracking with query time averaging
- [ ] Integrate apply_filters(), apply_sorting(), apply_pagination() methods

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify enhanced ChatSearcher implements all advanced features without breaking existing functionality.

### 6. Migrate Search Algorithms
**File**: Update `./src/domain/chat/search/algorithms.rs`
**Source**: Copy advanced methods from `./src/chat/search/algorithms.rs`
- [ ] Add search_not_stream() for NOT operator queries
- [ ] Add search_phrase_stream() for exact phrase matching
- [ ] Add search_proximity_stream() with configurable distance
- [ ] Maintain SIMD optimizations throughout
- [ ] Ensure all methods return CandleSearchResult types

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify search algorithms maintain performance characteristics and return correct result types.

### 7. Enhance Search Index Capabilities
**File**: Update `./src/domain/chat/search/index.rs`
**Source**: Integrate methods from `./src/chat/search/index.rs`
- [ ] Add search_and_stream() method for AND operator
- [ ] Add search_not_stream() method for NOT operator  
- [ ] Add search_phrase_stream() method for phrase queries
- [ ] Add search_proximity_stream() method with distance parameter
- [ ] Implement update_statistics() method for performance tracking
- [ ] Add increment_query_counter() for usage metrics

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify search index enhancements provide all query types while maintaining zero-allocation patterns.

---

## PHASE 3: COMMAND SYSTEM ENHANCEMENT

### 8. Enhance CommandExecutionContext 
**File**: Update `./src/domain/chat/commands/types/events.rs` lines 12-108
**Source**: Integrate features from `./src/chat/commands/types/events.rs`
- [ ] Add atomic execution_counter and event_counter fields
- [ ] Implement next_execution_id() and next_event_id() methods
- [ ] Add elapsed_time() calculation method
- [ ] Maintain builder pattern while adding atomic sequencing
- [ ] Ensure thread-safe operation with atomic operations

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify CommandExecutionContext enhancements maintain thread safety and builder pattern functionality.

### 9. Add Progress Reporting to Command Execution
**File**: Update `./src/domain/chat/commands/execution.rs` lines 80-200
**Source**: Integrate progress features from `./src/chat/commands/execution.rs`
- [ ] Add detailed progress reporting (25%, 50%, 75%, 100%) to streaming methods
- [ ] Implement realistic progress simulation for export/search operations
- [ ] Add progress events to all long-running command operations
- [ ] Maintain unwrapped AsyncStream patterns throughout
- [ ] Ensure all progress events use microsecond timestamps

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify progress reporting integration maintains streaming performance and provides meaningful progress updates.

---

## PHASE 4: INTEGRATION & TESTING

### 10. Update Domain Search Module Exports
**File**: Update `./src/domain/chat/search/mod.rs` lines 1-30
- [ ] Export new CandleEnhancedHistoryManager type
- [ ] Export new CandleConversationTagger type  
- [ ] Export enhanced search statistics types
- [ ] Remove imports from original chat system
- [ ] Verify all exports use Candle prefixes

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify module exports are complete and follow naming conventions without breaking existing imports.

### 11. Update Domain Chat Module Exports
**File**: Update `./src/domain/chat/mod.rs` lines 30-67
- [ ] Add exports for enhanced search components
- [ ] Update command exports with enhanced features
- [ ] Remove any remaining references to original chat system
- [ ] Verify CandleChatLoop integration is complete

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify domain chat module exports provide complete unified interface without import conflicts.

### 12. Test Agent Builder Integration
**File**: Update `./src/main.rs` to test unified system
- [ ] Verify CandleFluentAi::agent_role() chain works with enhanced features
- [ ] Test .chat() method returns enhanced CandleMessageChunk types
- [ ] Validate all builder methods work with unified domain system
- [ ] Ensure zero compilation errors with enhanced features

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify agent builder integration works seamlessly with all enhanced features and produces correct output.

### 13. Run Comprehensive Compilation Check
**File**: Run `cargo check` from project root
- [ ] Verify zero compilation errors across entire project
- [ ] Verify zero warnings across entire project  
- [ ] Check all imports resolve correctly to domain system
- [ ] Validate all type references use correct Candle prefixes

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify compilation check passes completely with zero errors and zero warnings.

---

## PHASE 5: CLEANUP & FINALIZATION

### 14. Update All System Imports
**Files**: All files importing from `crate::chat::`
- [ ] Systematically update all imports to use `crate::domain::chat::`
- [ ] Remove any remaining dual imports
- [ ] Update lib.rs exports to reflect unified system
- [ ] Verify builders continue to function correctly

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify all import updates are correct and no references to original chat system remain.

### 15. Remove Original Chat System
**Directory**: Delete `./src/chat/` entirely
- [ ] Verify all functionality has been successfully migrated
- [ ] Confirm no imports remain pointing to deleted directory
- [ ] Run full compilation check after deletion
- [ ] Ensure agent builders still function perfectly

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify original chat system removal is complete and no functionality has been lost.

### 16. Final Validation
**File**: Test `./src/main.rs` end-to-end functionality
- [ ] Verify agent chat completion works with all enhanced features
- [ ] Test advanced search capabilities work correctly
- [ ] Validate real-time features and monitoring work
- [ ] Confirm enterprise-grade event system functions
- [ ] Run `cargo check` final verification (0 errors, 0 warnings)

**DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.**

- [ ] **QA Task**: Act as an Objective QA Rust developer and verify the unified system provides all functionality from both original systems with enhanced capabilities and enterprise-grade features.

---

## SUCCESS CRITERIA
- ✅ Single unified chat system exists only in ./src/domain/chat/
- ✅ All superior search/filtering features from original system integrated
- ✅ Agent builders work seamlessly with enhanced capabilities
- ✅ Zero compilation errors, zero warnings
- ✅ All advanced features (proximity search, filtering, pagination) functional
- ✅ Enterprise monitoring and real-time features preserved
- ✅ Original ./src/chat/ system completely removed