# Module Exports and Resolution

## Description
Fix missing module exports and internal module resolution issues that prevent proper type and function access across the codebase.

## Success Criteria
- All missing module exports properly defined
- Internal module resolution working correctly
- No "failed to resolve" errors for internal modules
- Consistent module export patterns established

## Dependencies
- Must complete after: 1_wasm_dependencies.md

## Estimated Complexity
**High** - Requires understanding complex module interdependencies

## Technical Details

### Issues to Resolve

#### Missing Module Exports
1. **proxy/core/matcher_integration.rs:11** - `super::super::matcher::Matcher_`
2. **wasm/request/builder_core.rs:9** - `super::Body`, `super::Client`
3. **wasm/request/builder_execution.rs:3** - `super::Client`, `super::Response`
4. **wasm/request/conversions.rs:8** - `super::Body`
5. **wasm/request/types.rs:9** - `super::Body`
6. **wasm/client/fetch.rs:26** - `crate::hyper::response::Response`
7. **wasm/client/fetch.rs:30** - `crate::hyper::wasm::AbortController`, `crate::hyper::wasm::AbortSignal`

#### Module Resolution Issues
8. **async_impl/request/auth.rs:26** - `hyper::util`
9. **async_impl/request/headers.rs:12** - `hyper::util`
10. **wasm/request/builder_core.rs:128,198** - `crate::util`
11. **json_path/functions/function_evaluator/core.rs:25,31** - `super::string_counting`
12. **json_path/functions/function_evaluator/core.rs:41** - `super::value_conversion`

### Implementation Strategy
1. **Map Module Dependencies**: Create dependency graph of all modules
2. **Define Missing Exports**: Add proper `pub use` statements
3. **Create Missing Modules**: Implement missing submodules
4. **Establish Export Conventions**: Consistent patterns across all modules

### Files to Create/Modify
- `src/proxy/matcher/mod.rs` - Export `Matcher_`
- `src/wasm/mod.rs` - Export `Body`, `Client`, `Response`
- `src/hyper/wasm/mod.rs` - Export `AbortController`, `AbortSignal`
- `src/util/mod.rs` - Create utility module
- `src/json_path/functions/function_evaluator/string_counting.rs` - Create module
- `src/json_path/functions/function_evaluator/value_conversion.rs` - Create module

### Validation Steps
1. Run `cargo check --message-format short --quiet`
2. Verify all "failed to resolve" errors are eliminated
3. Ensure no circular dependencies introduced
4. Test that all modules can access required types