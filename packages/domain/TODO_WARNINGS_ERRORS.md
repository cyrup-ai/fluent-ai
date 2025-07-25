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

### W6. Missing Documentation: Variant 'Named'
**File**: `packages/domain/src/tool/core.rs:57:5`
**Issue**: `warning: missing documentation for a variant`
**Fix Required**: Add documentation for the 'Named' variant

### W7. Missing Documentation: Struct 'Prompt'
**File**: `packages/domain/src/prompt/mod.rs:6:1`
**Issue**: `warning: missing documentation for a struct`
**Fix Required**: Add documentation for the 'Prompt' struct

### W8. Missing Documentation: Struct Field 'content'
**File**: `packages/domain/src/prompt/mod.rs:7:5`
**Issue**: `warning: missing documentation for a struct field`
**Fix Required**: Add documentation for the 'content' field

### W9. Missing Documentation: Struct Field 'role'
**File**: `packages/domain/src/prompt/mod.rs:9:5`
**Issue**: `warning: missing documentation for a struct field`
**Fix Required**: Add documentation for the 'role' field

### W10. Missing Documentation: Function 'new'
**File**: `packages/domain/src/prompt/mod.rs:17:5`
**Issue**: `warning: missing documentation for an associated function`
**Fix Required**: Add documentation for the 'new' function

### W11. Missing Documentation: Method 'content'
**File**: `packages/domain/src/prompt/mod.rs:24:5`
**Issue**: `warning: missing documentation for a method`
**Fix Required**: Add documentation for the 'content' method

### W12. Missing Documentation: Function 'new' (Tool)
**File**: `packages/domain/src/tool/core.rs:44:5`
**Issue**: `warning: missing documentation for an associated function`
**Fix Required**: Add documentation for the 'new' function

### W13. Missing Documentation: Method 'push'
**File**: `packages/domain/src/tool/core.rs:48:5`
**Issue**: `warning: missing documentation for a method`
**Fix Required**: Add documentation for the 'push' method

### W14. Missing Documentation: Variant 'Typed'
**File**: `packages/domain/src/tool/core.rs:56:5`
**Issue**: `warning: missing documentation for a variant`
**Fix Required**: Add documentation for the 'Typed' variant

## ERRORS (241 total)

[Previous error entries remain unchanged...]

## METHODOLOGY

1. **Fix warnings first** (14 items) - these are typically easier
2. **Fix errors systematically** starting with the most fundamental issues
3. **Add QA step after each fix** (rate 1-10, requeue if <9)
4. **Never cross off until verified by clean `cargo check`**
5. **Use Desktop Commander for all operations**
6. **Never use blocking code without explicit approval**

## PRIORITY ORDER

1. **Documentation Warnings** (W6-W14) - Quick wins, improve code quality
2. **Import conflicts** (E1, E2, W1, W2) - resolve StreamExt import issues
3. **Missing variables** (E3-E5) - fix memory_arc scope issues
4. **Method ambiguity** (E6-E11) - resolve multiple `next` method conflicts
5. **Type mismatches** (E12-E14, E17) - core type system issues
6. **Async/await violations** (E15, E16) - architectural compliance
7. **Lifetime issues** (E18-E21) - memory safety
8. **Unused variables** (W3-W5) - code cleanliness

**STARTING WITH: W6-W14 - Documentation Warnings**