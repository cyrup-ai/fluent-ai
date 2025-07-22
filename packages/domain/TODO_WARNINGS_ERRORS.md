# COMPREHENSIVE ERROR AND WARNING FIX LIST - UPDATED

**OBJECTIVE: 0 ERRORS, 0 WARNINGS**
**CURRENT STATE: 241 ERRORS + 5 WARNINGS = 246 TOTAL ISSUES**

## WARNINGS (5 total)

### W1. Unused Import: StreamExt (duplicate)
**File**: `packages/domain/src/agent/chat.rs:242:13`
**Issue**: `warning: unused import: futures_util::StreamExt`
**Fix Required**: Remove unused import or implement missing usage

### W2. Unused Import: StreamExt (duplicate)
**File**: `packages/domain/src/agent/chat.rs:269:13`
**Issue**: `warning: unused import: futures_util::StreamExt`
**Fix Required**: Remove unused import or implement missing usage

### W3. Unused Variable: files_context
**File**: `packages/domain/src/context/provider.rs:658:42`
**Issue**: `warning: unused variable: files_context`
**Fix Required**: Implement usage or prefix with underscore if intentionally unused

### W4. Unused Variable: directory_context
**File**: `packages/domain/src/context/provider.rs:698:46`
**Issue**: `warning: unused variable: directory_context`
**Fix Required**: Implement usage or prefix with underscore if intentionally unused

### W5. Unused Variable: github_context
**File**: `packages/domain/src/context/provider.rs:738:43`
**Issue**: `warning: unused variable: github_context`
**Fix Required**: Implement usage or prefix with underscore if intentionally unused

## ERRORS (241 total)

### E1. Multiple Definitions: StreamExt
**File**: `packages/domain/src/agent/chat.rs:242:13`
**Issue**: `the name StreamExt is defined multiple times: StreamExt reimported here`
**Fix Required**: Remove duplicate import or use qualified imports

### E2. Multiple Definitions: StreamExt (duplicate)
**File**: `packages/domain/src/agent/chat.rs:269:13`
**Issue**: `the name StreamExt is defined multiple times: StreamExt reimported here`
**Fix Required**: Remove duplicate import or use qualified imports

### E3. Cannot Find Value: memory_arc
**File**: `packages/domain/src/agent/core.rs:99:28`
**Issue**: `cannot find value memory_arc in this scope: not found in this scope`
**Fix Required**: Define memory_arc variable or fix variable name

### E4. Cannot Find Value: memory_arc (duplicate)
**File**: `packages/domain/src/agent/core.rs:148:28`
**Issue**: `cannot find value memory_arc in this scope: not found in this scope`
**Fix Required**: Define memory_arc variable or fix variable name

### E5. Cannot Find Value: memory_arc (duplicate)
**File**: `packages/domain/src/agent/core.rs:182:28`
**Issue**: `cannot find value memory_arc in this scope: not found in this scope`
**Fix Required**: Define memory_arc variable or fix variable name

### E6. Multiple Applicable Items: next
**File**: `packages/domain/src/agent/builder.rs:138:33`
**Issue**: `multiple applicable items in scope: multiple next found`
**Fix Required**: Use qualified method call or resolve ambiguity

### E7. Multiple Applicable Items: next (duplicate)
**File**: `packages/domain/src/agent/chat.rs:220:44`
**Issue**: `multiple applicable items in scope: multiple next found`
**Fix Required**: Use qualified method call or resolve ambiguity

### E8. Multiple Applicable Items: next (duplicate)
**File**: `packages/domain/src/agent/chat.rs:244:44`
**Issue**: `multiple applicable items in scope: multiple next found`
**Fix Required**: Use qualified method call or resolve ambiguity

### E9. Multiple Applicable Items: next (duplicate)
**File**: `packages/domain/src/agent/chat.rs:271:44`
**Issue**: `multiple applicable items in scope: multiple next found`
**Fix Required**: Use qualified method call or resolve ambiguity

### E10. Multiple Applicable Items: next (duplicate)
**File**: `packages/domain/src/agent/core.rs:86:35`
**Issue**: `multiple applicable items in scope: multiple next found`
**Fix Required**: Use qualified method call or resolve ambiguity

### E11. Multiple Applicable Items: next (duplicate)
**File**: `packages/domain/src/agent/core.rs:135:35`
**Issue**: `multiple applicable items in scope: multiple next found`
**Fix Required**: Use qualified method call or resolve ambiguity

### E12. Type Mismatch: Option<Arc<str>> vs Option<String>
**File**: `packages/domain/src/chat/commands/execution.rs:94:35`
**Issue**: `mismatched types: expected Option<Arc<str>>, found Option<String>`
**Fix Required**: Convert String to Arc<str> or adjust expected type

### E13. Incorrect Method Arguments
**File**: `packages/domain/src/chat/commands/execution.rs:103:23`
**Issue**: `arguments to this method are incorrect`
**Fix Required**: Fix method call arguments to match signature

### E14. Incorrect Method Arguments (duplicate)
**File**: `packages/domain/src/chat/commands/execution.rs:109:23`
**Issue**: `arguments to this method are incorrect`
**Fix Required**: Fix method call arguments to match signature

### E15. Async Block in Non-Async Context
**File**: `packages/domain/src/engine.rs:312:31`
**Issue**: `expected a FnOnce() closure, found {async block}`
**Fix Required**: Convert async block to sync closure or use proper async context

### E16. Async Block in Non-Async Context (duplicate)
**File**: `packages/domain/src/engine.rs:345:20`
**Issue**: `expected a FnOnce() closure, found {async block}`
**Fix Required**: Convert async block to sync closure or use proper async context

### E17. Type Mismatch: AsyncStream vs UnboundedReceiverStream
**File**: `packages/domain/src/memory/manager.rs:415:9`
**Issue**: `expected AsyncStream<Memory>, found UnboundedReceiverStream<Memory>`
**Fix Required**: Convert UnboundedReceiverStream to AsyncStream

### E18. Borrowed Data Escapes Method
**File**: `packages/domain/src/completion/request.rs:169:9`
**Issue**: `self escapes the method body here, argument requires that 'a must outlive 'static`
**Fix Required**: Fix lifetime annotations or use owned data

### E19. Borrowed Data Escapes Method (duplicate)
**File**: `packages/domain/src/completion/request.rs:193:9`
**Issue**: `self escapes the method body here, argument requires that 'a must outlive 'static`
**Fix Required**: Fix lifetime annotations or use owned data

### E20. Borrowed Data Escapes Method (duplicate)
**File**: `packages/domain/src/completion/request.rs:212:9`
**Issue**: `self escapes the method body here, argument requires that 'a must outlive 'static`
**Fix Required**: Fix lifetime annotations or use owned data

### E21. Lifetime Not Long Enough
**File**: `packages/domain/src/completion/response.rs:296:9`
**Issue**: `lifetime may not live long enough: returning this value requires that 'a must outlive 'static`
**Fix Required**: Fix lifetime annotations or use owned data

**NOTE: The compilation output shows 241 total errors. The above represents the first ~21 errors that are clearly visible. More errors exist and will be discovered as we fix these initial ones.**

## METHODOLOGY

1. **Fix warnings first** (5 items) - these are typically easier
2. **Fix errors systematically** starting with the most fundamental issues
3. **Add QA step after each fix** (rate 1-10, requeue if <9)
4. **Never cross off until verified by clean `cargo check`**
5. **Use Desktop Commander for all operations**
6. **Never use blocking code without explicit approval**

## PRIORITY ORDER

1. **Import conflicts** (E1, E2, W1, W2) - resolve StreamExt import issues
2. **Missing variables** (E3-E5) - fix memory_arc scope issues
3. **Method ambiguity** (E6-E11) - resolve multiple `next` method conflicts
4. **Type mismatches** (E12-E14, E17) - core type system issues
5. **Async/await violations** (E15, E16) - architectural compliance
6. **Lifetime issues** (E18-E21) - memory safety
7. **Unused variables** (W3-W5) - code cleanliness

**STARTING WITH: W1 & W2 - Remove duplicate StreamExt imports**