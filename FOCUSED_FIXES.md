# FOCUSED FIXES: Build on Existing TOOLREGISTRY Implementation

## What Already Exists ✅

The TOOLREGISTRY TYPESTATE BUILDER is **fully implemented** in `/packages/provider/src/clients/anthropic/tools.rs`:
- Complete typestate builder pattern with compile-time safety
- Zero-allocation TypedToolStorage with ArrayVec
- Full tool execution pipeline with streaming
- Integration with existing ToolRegistry
- Comprehensive error handling system

## Only Fix What's Broken 🔧

### Critical Constraint Violations Found:

1. **Fix unwrap() violations** (violates user's "never use unwrap() (period!)" rule):
   - `expression_evaluator.rs:222,252,292` - Replace with proper Result handling
   - `vertexai/streaming.rs:148` - Replace nested unwrap() with error handling

2. **Fix expect() violation in src/** (not test code):
   - `perplexity/client.rs:245` - Replace with proper Result handling

### Minimal TODO List:

#### Task 1: Fix Expression Evaluator unwrap() Violations
**File:** `./packages/provider/src/clients/anthropic/expression_evaluator.rs`
**Lines:** 222, 252, 292
**Fix:** Replace `inner_iter.next().unwrap().as_str()` with proper error handling

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

#### Task 2: Act as an Objective QA Rust developer
Rate the work performed previously on these requirements: Are all unwrap() calls removed? Does code maintain existing functionality? Does `cargo check` pass?

#### Task 3: Fix VertexAI Streaming unwrap() Violation  
**File:** `./packages/provider/src/clients/vertexai/streaming.rs`
**Line:** 148
**Fix:** Replace nested unwrap() with proper error handling

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

#### Task 4: Act as an Objective QA Rust developer
Rate the work performed previously on these requirements: Is the unwrap() call removed? Does streaming functionality work? Does `cargo check` pass?

#### Task 5: Fix Perplexity Client expect() Violation
**File:** `./packages/provider/src/clients/perplexity/client.rs`
**Line:** 245
**Fix:** Replace expect() with proper Result handling

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

#### Task 6: Act as an Objective QA Rust developer
Rate the work performed previously on these requirements: Is the expect() call removed? Does the API work correctly? Does `cargo check` pass?

#### Task 7: Validate Clean Compilation
**Command:** `cargo fmt && cargo check --message-format short --quiet`
**Fix:** Address any remaining warnings/errors

DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required.

#### Task 8: Act as an Objective QA Rust developer
Rate the work performed previously on these requirements: Does compilation pass without warnings? Are all constraint violations resolved?

## Success Criteria:
- [ ] Zero unwrap() calls in src/ code
- [ ] Zero expect() calls in src/ code (excluding tests)
- [ ] `cargo fmt && cargo check --message-format short --quiet` passes
- [ ] Existing TOOLREGISTRY functionality unchanged
- [ ] All fixes are minimal and surgical

## What NOT to do:
- ❌ Don't re-implement existing TOOLREGISTRY features
- ❌ Don't add new features not requested
- ❌ Don't refactor working code
- ❌ Don't add unnecessary tests or documentation
- ❌ Don't modify the existing typestate builder implementation

## Build on Existing:
The existing TOOLREGISTRY implementation is production-ready. These fixes will make it fully compliant with user constraints while preserving all existing functionality.
