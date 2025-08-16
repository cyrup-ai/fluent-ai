# Type and Trait Resolution Fixes

## Description
Fix type resolution issues, missing traits, and pattern matching problems that are preventing compilation.

## Success Criteria
- All type resolution errors fixed
- Missing traits properly imported or implemented
- Pattern matching issues resolved
- Type generics properly specified

## Dependencies
- None (independent fixes)

## Estimated Complexity
**Medium** - Requires understanding type system and trait bounds

## Technical Details

### Type Issues
1. **wasm/body/body_impl.rs:21** - `JsValue` type not found
2. **wasm/body/single_impl.rs:19** - `JsValue` type not found  
3. **wasm/body/single_impl.rs:23** - `Uint8Array` type not found
4. **wasm/client/fetch.rs:60** - `AsyncStream` type not found
5. **wasm/mod.rs:109** - Type alias takes 1 generic but 2 supplied
6. **wasm/client/fetch.rs:60** - Missing generics for `http::Request`

### Missing Traits
7. **async_impl/response/core/static_constructors.rs:81** - `IntoHeaderValue` trait

### Pattern Matching Issues
8. **async_impl/client/tls_setup.rs:167** - Expected tuple struct but found unit variant `TlsBackend::BuiltRustls`

### Implementation Strategy

#### WASM Type Issues
- Import `JsValue` and `Uint8Array` from `wasm-bindgen` and `js-sys`
- Define or import `AsyncStream` type for WASM context
- Fix type alias generic parameters

#### Missing Traits
- Import `IntoHeaderValue` from appropriate HTTP crate
- Implement trait if not available in dependencies

#### Pattern Matching
- Fix `TlsBackend::BuiltRustls` enum variant structure
- Ensure pattern matches enum definition

### Required Imports
```rust
// For WASM types
use wasm_bindgen::JsValue;
use js_sys::Uint8Array;

// For HTTP traits  
use http::header::IntoHeaderValue;
```

### Files Affected
- `src/wasm/body/body_impl.rs`
- `src/wasm/body/single_impl.rs`
- `src/wasm/client/fetch.rs`
- `src/wasm/mod.rs`
- `src/async_impl/response/core/static_constructors.rs`
- `src/async_impl/client/tls_setup.rs`

### Validation Steps
1. Run `cargo check --message-format short --quiet`
2. Verify all type resolution errors eliminated
3. Test pattern matching compiles correctly
4. Ensure trait implementations work as expected