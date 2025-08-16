# TODO: Extract All Embedded Tests from ./src/ to ./tests/

## Phase 1: Analysis and Categorization

- [ ] **Task 1.1**: Scan and categorize all 104+ files with embedded tests into three categories:
  - Dedicated test files in src/ (e.g., src/json_path/functions/function_evaluator/value/tests/*.rs)
  - Production files with embedded test modules (e.g., src/json_path/compiler.rs, src/lib.rs)  
  - Test directories within src/ (e.g., src/json_path/core_evaluator/tests/)
  
  **Files**: All files identified by `find src -name "*.rs" -exec grep -l "#\[cfg(test)\]" {} \;`
  **Implementation**: Create comprehensive inventory with file paths and test content analysis
  **Architecture**: Maintain existing test organization structure in ./tests/
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 1.1**: Act as an Objective QA Rust developer and verify that the categorization is complete and accurate. Rate the work performed on completeness of test identification, proper categorization, and zero missed embedded tests.

- [ ] **Task 1.2**: Analyze import dependencies and module references for each test file to understand what needs to be updated when moving to ./tests/
  
  **Files**: Focus on complex test files with extensive imports from parent modules
  **Implementation**: Map all `use super::*`, `use crate::*`, and relative imports that will break
  **Architecture**: Plan import transformation strategy (super:: -> crate::module_path::)
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 1.2**: Act as an Objective QA Rust developer and verify that all import dependencies have been properly analyzed. Rate the work on completeness of dependency mapping and accuracy of planned import transformations.

## Phase 2: Extract Dedicated Test Files

- [ ] **Task 2.1**: Move all dedicated test files from src/json_path/functions/function_evaluator/value/tests/ to tests/json_path/functions/function_evaluator/value/
  
  **Files**: 
  - src/json_path/functions/function_evaluator/value/tests/current_context.rs → tests/json_path/functions/function_evaluator/value/current_context.rs
  - src/json_path/functions/function_evaluator/value/tests/property_access.rs → tests/json_path/functions/function_evaluator/value/property_access.rs
  - src/json_path/functions/function_evaluator/value/tests/edge_cases.rs → tests/json_path/functions/function_evaluator/value/edge_cases.rs
  - src/json_path/functions/function_evaluator/value/tests/argument_validation.rs → tests/json_path/functions/function_evaluator/value/argument_validation.rs
  - src/json_path/functions/function_evaluator/value/tests/literal_values.rs → tests/json_path/functions/function_evaluator/value/literal_values.rs
  - src/json_path/functions/function_evaluator/value/tests/type_conversion.rs → tests/json_path/functions/function_evaluator/value/type_conversion.rs
  
  **Implementation**: Update all imports from `use super::super::core::evaluate_value_function` to `use fluent_ai_http3::json_path::functions::function_evaluator::value::evaluate_value_function`
  **Architecture**: Maintain test module structure but update import paths for external crate access
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 2.1**: Act as an Objective QA Rust developer and verify that all dedicated test files have been moved correctly with proper imports. Rate the work on file movement accuracy, import correctness, and maintained test functionality.

- [ ] **Task 2.2**: Move all dedicated test files from src/json_path/core_evaluator/tests/ to tests/json_path/core_evaluator/
  
  **Files**:
  - src/json_path/core_evaluator/tests/filter_expressions.rs → tests/json_path/core_evaluator/filter_expressions.rs
  - src/json_path/core_evaluator/tests/array_operations.rs → tests/json_path/core_evaluator/array_operations.rs
  - src/json_path/core_evaluator/tests/recursive_descent.rs → tests/json_path/core_evaluator/recursive_descent.rs
  - src/json_path/core_evaluator/tests/basic_selectors.rs → tests/json_path/core_evaluator/basic_selectors.rs
  - src/json_path/core_evaluator/tests/rfc_compliance.rs → tests/json_path/core_evaluator/rfc_compliance.rs
  - src/json_path/core_evaluator/tests/edge_cases_debug.rs → tests/json_path/core_evaluator/edge_cases_debug.rs
  
  **Implementation**: Update imports to reference crate root instead of relative paths
  **Architecture**: Preserve test organization while enabling external crate access
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 2.2**: Act as an Objective QA Rust developer and verify that core evaluator test files have been moved correctly. Rate the work on completeness, import accuracy, and preserved test functionality.

- [ ] **Task 2.3**: Move all dedicated test files from src/hyper/proxy/tests/ to tests/hyper/proxy/
  
  **Files**:
  - src/hyper/proxy/tests/auth_tests.rs → tests/hyper/proxy/auth_tests.rs
  - src/hyper/proxy/tests/no_proxy_tests.rs → tests/hyper/proxy/no_proxy_tests.rs
  - src/hyper/proxy/tests/basic_proxy_tests.rs → tests/hyper/proxy/basic_proxy_tests.rs
  - src/hyper/proxy/tests/matcher_tests.rs → tests/hyper/proxy/matcher_tests.rs
  
  **Implementation**: Update imports for proxy types and functionality
  **Architecture**: Maintain proxy test organization in ./tests/ structure
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 2.3**: Act as an Objective QA Rust developer and verify that proxy test files have been moved correctly. Rate the work on file organization, import updates, and test functionality preservation.

- [ ] **Task 2.4**: Move all dedicated test files from src/hyper/async_impl/request/tests/ to tests/hyper/async_impl/request/
  
  **Files**:
  - src/hyper/async_impl/request/tests/builder_tests.rs → tests/hyper/async_impl/request/builder_tests.rs
  - src/hyper/async_impl/request/tests/basic_tests.rs → tests/hyper/async_impl/request/basic_tests.rs
  - src/hyper/async_impl/request/tests/auth_tests.rs → tests/hyper/async_impl/request/auth_tests.rs
  - src/hyper/async_impl/request/tests/header_tests.rs → tests/hyper/async_impl/request/header_tests.rs
  - src/hyper/async_impl/request/tests/body_tests.rs → tests/hyper/async_impl/request/body_tests.rs
  
  **Implementation**: Update imports for request builder and async implementation types
  **Architecture**: Preserve async request test structure in ./tests/
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 2.4**: Act as an Objective QA Rust developer and verify that async request test files have been moved correctly. Rate the work on import accuracy, test organization, and functionality preservation.

## Phase 3: Extract Embedded Test Modules from Production Files

- [ ] **Task 3.1**: Extract embedded test module from src/json_path/compiler.rs (lines ~150-200)
  
  **Files**: 
  - Extract from: src/json_path/compiler.rs
  - Create: tests/json_path/compiler.rs
  
  **Implementation**: Remove `#[cfg(test)] mod tests { ... }` from src/json_path/compiler.rs and create standalone test file
  **Architecture**: Update imports from `use super::*` to `use fluent_ai_http3::json_path::compiler::*`
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 3.1**: Act as an Objective QA Rust developer and verify that compiler test module has been extracted correctly. Rate the work on clean extraction, proper imports, and maintained test functionality.

- [ ] **Task 3.2**: Extract embedded test module from src/lib.rs (lines ~200-250)
  
  **Files**:
  - Extract from: src/lib.rs  
  - Create: tests/lib.rs
  
  **Implementation**: Remove embedded test module and create standalone integration test file
  **Architecture**: Update imports to reference public crate API instead of internal modules
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 3.2**: Act as an Objective QA Rust developer and verify that lib.rs test module has been extracted correctly. Rate the work on proper extraction, import updates, and integration test functionality.

- [ ] **Task 3.3**: Extract embedded test modules from all remaining production files with #[cfg(test)]
  
  **Files**: Process remaining ~90 files identified in the scan, including:
  - src/json_path/core_evaluator/evaluator.rs
  - src/hyper/response.rs
  - src/config/core/mod.rs
  - src/common/cache/response_cache/core.rs
  - And all other production files with embedded tests
  
  **Implementation**: Systematically extract each embedded test module to corresponding test file in ./tests/
  **Architecture**: Maintain test organization while updating all imports for external crate access
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 3.3**: Act as an Objective QA Rust developer and verify that all embedded test modules have been extracted correctly. Rate the work on completeness, import accuracy, and zero test functionality loss.

## Phase 4: Update Module References and Clean Up

- [ ] **Task 4.1**: Remove all test directories from src/ after successful extraction
  
  **Files**: Remove directories:
  - src/json_path/functions/function_evaluator/value/tests/
  - src/json_path/core_evaluator/tests/
  - src/hyper/proxy/tests/
  - src/hyper/async_impl/request/tests/
  - All other test directories within src/
  
  **Implementation**: Delete directories only after confirming all tests are successfully moved and working in ./tests/
  **Architecture**: Clean separation between production code in src/ and test code in tests/
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 4.1**: Act as an Objective QA Rust developer and verify that all test directories have been properly removed from src/. Rate the work on clean removal and maintained directory structure.

- [ ] **Task 4.2**: Update all mod.rs files in src/ to remove test module declarations
  
  **Files**: Update mod.rs files that previously declared test modules:
  - src/json_path/functions/function_evaluator/value/mod.rs (remove `#[cfg(test)] mod tests;`)
  - src/json_path/core_evaluator/mod.rs (remove test module references)
  - All other mod.rs files with test module declarations
  
  **Implementation**: Remove only test-related module declarations while preserving all production module declarations
  **Architecture**: Clean module organization with no test references in production code
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 4.2**: Act as an Objective QA Rust developer and verify that all mod.rs files have been properly updated. Rate the work on accurate test module removal and preserved production module structure.

## Phase 5: Verification and Testing

- [ ] **Task 5.1**: Verify no embedded tests remain in src/
  
  **Files**: Run verification command: `find src -name "*.rs" -exec grep -l "#\[cfg(test)\]" {} \;`
  **Implementation**: Command should return empty result, confirming zero embedded tests in src/
  **Architecture**: Complete separation achieved between production and test code
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 5.1**: Act as an Objective QA Rust developer and verify that zero embedded tests remain in src/. Rate the work on complete test extraction and clean production code separation.

- [ ] **Task 5.2**: Run cargo nextest to verify all tests pass in their new locations
  
  **Files**: Execute `cargo nextest run` to verify all extracted tests function correctly
  **Implementation**: All tests must pass with zero failures, confirming successful extraction
  **Architecture**: Validate that test organization in ./tests/ works correctly with cargo test runner
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 5.2**: Act as an Objective QA Rust developer and verify that all tests pass after extraction. Rate the work on test functionality preservation, import correctness, and zero test failures.

- [ ] **Task 5.3**: Verify cargo build succeeds with clean src/ without embedded tests
  
  **Files**: Execute `cargo build` to confirm production code compiles without embedded test modules
  **Implementation**: Build must succeed, confirming clean separation doesn't break production code
  **Architecture**: Validate that removing embedded tests doesn't affect production compilation
  
  DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA Task 5.3**: Act as an Objective QA Rust developer and verify that cargo build succeeds with clean src/. Rate the work on production code integrity and successful test separation.

## CONSTRAINTS APPLIED TO ALL TASKS:
- Never use unwrap() in any code
- Never use expect() in src/* or examples
- DO USE expect() in ./tests/*
- DO NOT use unwrap() in ./tests/*
- Make only minimal, surgical changes required
- Preserve all existing test functionality
- Maintain production code integrity
- Follow Rust ecosystem best practices for test organization