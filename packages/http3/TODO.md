# HTTP3 Test Extraction and Compilation Fix Plan

## Milestone 1: Fix Critical Module Import Errors (173 compilation errors)

### Phase 1A: Fix Core Module Re-exports in src/hyper/mod.rs

- [ ] **Fix missing re-exports in src/hyper/mod.rs lines 21-23**: Add proper re-exports for `Certificate`, `Identity`, `IntoUrl`, `Proxy` types that are being imported by other modules. Examine existing re-export structure and add missing items without breaking existing functionality.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on fixing re-exports in src/hyper/mod.rs. Verify all missing imports are resolved, no new compilation errors introduced, and existing functionality preserved.

- [ ] **Fix config module re-export in src/hyper/mod.rs**: Add `pub mod config;` and proper re-exports for config types being imported by client modules. Check lines referencing `crate::hyper::config` in compilation errors.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on config module re-exports. Verify config types are properly accessible and no circular dependencies created.

### Phase 1B: Fix IntoUrl Module Structure

- [ ] **Create proper IntoUrl re-export in src/hyper/mod.rs**: Add `pub use into_url::IntoUrl;` to expose IntoUrl trait that's being imported by multiple modules. Check src/hyper/into_url.rs exists and contains IntoUrl trait.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on IntoUrl re-export. Verify IntoUrl trait is accessible from crate::hyper::IntoUrl path and resolves import errors.

### Phase 1C: Fix TLS Module Re-exports

- [ ] **Fix Certificate and Identity re-exports in src/hyper/mod.rs**: Add proper re-exports for TLS types from tls module. Check src/hyper/tls/mod.rs for Certificate and Identity types and create appropriate re-exports.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on TLS re-exports. Verify Certificate and Identity types are accessible and TLS functionality preserved.

### Phase 1D: Fix Proxy Module Re-exports

- [ ] **Fix Proxy type re-export in src/hyper/mod.rs**: Add `pub use proxy::Proxy;` to expose Proxy type being imported by client modules. Verify proxy module structure and types.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on Proxy re-export. Verify Proxy type is accessible and proxy functionality preserved.

### Phase 1E: Fix Connect Module Issues

- [ ] **Fix private re-export errors in src/hyper/connect/service/mod.rs lines 17,19**: Remove or fix private re-exports of `NativeTlsConnection` and `RustlsConnection`. Make types public if needed or remove invalid re-exports.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on connect service re-exports. Verify no private items are being re-exported and module compiles correctly.

- [ ] **Fix missing Connect and HttpConnector re-exports in src/hyper/mod.rs line 21**: Add proper re-exports for connect types or remove invalid imports. Check src/hyper/connect/mod.rs for available types.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on Connect re-exports. Verify connect functionality is properly exposed and accessible.

### Phase 1F: Fix JSON Path Error Module Issues

- [ ] **Fix missing error constructors in src/json_path/error/constructors/mod.rs**: Add proper re-exports for `invalid_expression_error`, `buffer_error`, `stream_error` functions that are being imported throughout the codebase.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on error constructor re-exports. Verify all error functions are accessible and error handling preserved.

- [ ] **Fix missing deserialization_error in src/json_path/error/mod.rs**: Add proper re-export for deserialization_error function being imported by filter/core.rs.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on deserialization_error re-export. Verify error function is accessible and deserialization functionality preserved.

### Phase 1G: Fix Response Module Issues

- [ ] **Fix missing response re-export in src/hyper/async_impl/response/conversions.rs line 25**: Fix import `crate::hyper::response` by adding proper response module re-export in src/hyper/mod.rs.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on response module re-export. Verify response functionality is accessible and conversions work correctly.

- [ ] **Fix missing error trait implementation in src/hyper/async_impl/response/core/trait_impls.rs line 9**: Implement missing `error` method for trait implementation to resolve compilation error.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on trait implementation. Verify trait is properly implemented and no missing methods remain.

### Phase 1H: Fix WASM Module Dependencies

- [ ] **Add conditional WASM dependencies to Cargo.toml**: Add wasm-bindgen, js-sys, web-sys, wasm-bindgen-futures dependencies with proper target conditions for wasm32 architecture.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on WASM dependencies. Verify WASM code compiles correctly and dependencies are properly configured.

### Phase 1I: Fix Conflicting Debug Implementation

- [ ] **Fix conflicting Debug trait implementation in src/hyper/proxy/url_handling.rs line 261**: Remove duplicate Debug implementation for `url_handling::Custom` type to resolve trait conflict.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on Debug trait conflict. Verify no conflicting implementations remain and type still has Debug capability.

## Milestone 2: Complete Surgical Test Extraction (89 remaining files)

### Phase 2A: Extract Remaining Embedded Test Modules

- [ ] **Extract tests from src/hyper/connect/tcp/dns.rs lines 56-end**: Move embedded test module to tests/hyper/connect/tcp/dns.rs with proper imports and module structure.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on dns.rs test extraction. Verify tests are properly extracted, imports work, and original functionality preserved.

- [ ] **Extract tests from src/hyper/connect/tcp/tls.rs lines 62-end**: Move embedded test module to tests/hyper/connect/tcp/tls.rs with proper imports and module structure.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on tls.rs test extraction. Verify tests are properly extracted, imports work, and TLS functionality preserved.

- [ ] **Extract tests from src/hyper/connect/types/tcp_impl.rs lines 130-end**: Move embedded test module to tests/hyper/connect/types/tcp_impl.rs with proper imports.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on tcp_impl.rs test extraction. Verify tests are properly extracted and TCP functionality preserved.

### Phase 2B: Extract JSON Path Test Modules

- [ ] **Extract tests from src/json_path/functions/function_evaluator/length.rs lines 62-end**: Move embedded test module to tests/json_path/functions/function_evaluator/length.rs with proper imports.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on length.rs test extraction. Verify tests are properly extracted and length function preserved.

- [ ] **Extract tests from src/json_path/core_evaluator/mod.rs lines 151-end**: Move embedded test module to tests/json_path/core_evaluator/mod.rs with proper imports.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on core_evaluator mod.rs test extraction. Verify tests are properly extracted and evaluator functionality preserved.

### Phase 2C: Extract Remaining Cache and Response Tests

- [ ] **Extract tests from src/common/cache/response_cache/mod.rs lines 21-end**: Move embedded test module to tests/common/cache/response_cache/mod.rs with proper imports.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on response_cache mod.rs test extraction. Verify tests are properly extracted and cache functionality preserved.

- [ ] **Extract tests from src/response/body/mod.rs lines 35-end**: Move embedded test module to tests/response/body/mod.rs with proper imports.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the work performed on response body test extraction. Verify tests are properly extracted and response body functionality preserved.

## Milestone 3: Verification and Quality Assurance

### Phase 3A: Compilation Verification

- [ ] **Run cargo fmt && cargo check --message-format short --quiet**: Verify zero compilation errors and zero warnings after all fixes.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the compilation verification. Verify codebase compiles cleanly with no errors or warnings.

### Phase 3B: Test Execution Verification

- [ ] **Run cargo nextest run**: Verify all tests pass in their new locations with proper imports and functionality.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the test execution verification. Verify all tests pass and functionality is preserved.

### Phase 3C: Final Cleanup and Verification

- [ ] **Verify no test code remains in src/ directory**: Run `find src -name "*.rs" -exec grep -l "#\[cfg(test)\]" {} \;` and confirm zero results.
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **Act as an Objective QA Rust developer**: Rate the final cleanup verification. Verify complete test extraction with no test code remaining in src/ and all tests properly located in tests/.

## Architecture Notes

**Module Structure**: Maintain existing module hierarchy in tests/ mirroring src/ structure for intuitive navigation and import resolution.

**Import Strategy**: Use absolute imports from crate root (e.g., `use fluent_ai_http3::hyper::Client`) in test files to ensure proper module resolution.

**Dependency Management**: Preserve all existing functionality while ensuring clean separation between production code (src/) and test code (tests/).

**Error Handling**: Never use unwrap() or expect() in src/ or examples/. Use expect() in tests/ for clear test failure messages. Never use unwrap() anywhere.

**WASM Compatibility**: Ensure WASM-specific code compiles correctly with proper conditional compilation and dependencies.