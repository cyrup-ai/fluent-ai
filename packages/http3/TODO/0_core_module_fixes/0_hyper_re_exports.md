# Fix Core Hyper Module Re-exports

## Description
Fix missing re-exports in `src/hyper/mod.rs` for `Certificate`, `Identity`, `IntoUrl`, `Proxy` types that are being imported by other modules throughout the codebase.

## Success Criteria
- All missing imports for `Certificate`, `Identity`, `IntoUrl`, `Proxy` types are resolved
- No new compilation errors introduced
- Existing functionality preserved
- Module structure remains clean and logical

## Dependencies
- None (foundational task)

## Implementation Details
**File**: `src/hyper/mod.rs`
**Lines**: 21-23 (examine existing re-export structure)

**Actions**:
1. Examine existing re-export structure in `src/hyper/mod.rs`
2. Add proper re-exports for missing types:
   - `Certificate` from tls module
   - `Identity` from tls module  
   - `IntoUrl` from into_url module
   - `Proxy` from proxy module
3. Verify module paths and type availability
4. Test compilation to ensure no circular dependencies

## Constraints
- DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA
- Make ONLY THE MINIMAL, SURGICAL CHANGES required
- Do not modify or rewrite any portion of the app outside scope
- Never use unwrap() or expect() in src/ or examples/
- Use expect() only in tests/ for clear test failure messages

## Estimated Complexity
Medium - requires understanding module structure and dependency graph