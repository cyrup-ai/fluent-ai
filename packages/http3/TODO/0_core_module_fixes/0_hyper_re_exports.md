# Hyper Re-exports and Core Module Structure

## Description
Fix the foundational hyper re-exports and core module structure that other modules depend on. This includes establishing proper module boundaries and ensuring hyper types are correctly exposed.

## Success Criteria
- All hyper types properly re-exported from appropriate modules
- Core module structure established and documented
- No circular dependencies in module graph
- All dependent modules can import required hyper types

## Dependencies
- None (foundational task)

## Estimated Complexity
**High** - This affects the entire module architecture

## Technical Details

### Issues to Resolve
1. Missing hyper re-exports causing import failures across multiple modules
2. Inconsistent module structure preventing proper type resolution
3. Core types not properly exposed from root modules

### Implementation Strategy
1. **Audit Current Re-exports**: Map all existing hyper re-exports
2. **Define Module Boundaries**: Establish clear separation between hyper, wasm, and core modules
3. **Create Consistent Export Pattern**: Ensure all modules follow same export conventions
4. **Validate Dependencies**: Ensure no circular dependencies introduced

### Files Affected
- `src/lib.rs` - Root module exports
- `src/hyper/mod.rs` - Hyper module structure
- `src/wasm/mod.rs` - WASM module structure
- Various submodules importing hyper types

### Validation Steps
1. Run `cargo check --message-format short --quiet` to verify no import errors
2. Ensure all modules can resolve their hyper type dependencies
3. Verify no circular dependency warnings
4. Test that public API surface is maintained