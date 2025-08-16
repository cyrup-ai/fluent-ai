# Test Extraction Analysis - HTTP3 Package

## Summary: 104 Files with Embedded Tests Found

### Category 1: Dedicated Test Files in src/ (Need Moving)
**Count: 28 files**

#### JSON Path Function Evaluator Value Tests (6 files)
- src/json_path/functions/function_evaluator/value/tests/current_context.rs
- src/json_path/functions/function_evaluator/value/tests/property_access.rs
- src/json_path/functions/function_evaluator/value/tests/edge_cases.rs
- src/json_path/functions/function_evaluator/value/tests/argument_validation.rs
- src/json_path/functions/function_evaluator/value/tests/literal_values.rs
- src/json_path/functions/function_evaluator/value/tests/type_conversion.rs

#### JSON Path Core Evaluator Tests (6 files)
- src/json_path/core_evaluator/tests/filter_expressions.rs
- src/json_path/core_evaluator/tests/array_operations.rs
- src/json_path/core_evaluator/tests/recursive_descent.rs
- src/json_path/core_evaluator/tests/basic_selectors.rs
- src/json_path/core_evaluator/tests/rfc_compliance.rs
- src/json_path/core_evaluator/tests/edge_cases_debug.rs

#### JSON Path Function Evaluator Integration Tests (7 files)
- src/json_path/functions/function_evaluator/integration_tests/unicode_special.rs
- src/json_path/functions/function_evaluator/integration_tests/complex_data.rs
- src/json_path/functions/function_evaluator/integration_tests/edge_cases.rs
- src/json_path/functions/function_evaluator/integration_tests/mod.rs
- src/json_path/functions/function_evaluator/integration_tests/performance.rs
- src/json_path/functions/function_evaluator/integration_tests/value_conversion.rs
- src/json_path/functions/function_evaluator/integration_tests/function_dispatch.rs
- src/json_path/functions/function_evaluator/integration_tests/error_handling.rs

#### JSON Path Function Evaluator Regex Tests (4 files)
- src/json_path/functions/function_evaluator/regex_functions/basic_tests.rs
- src/json_path/functions/function_evaluator/regex_functions/test_utils.rs
- src/json_path/functions/function_evaluator/regex_functions/advanced_tests.rs

#### Hyper Proxy Tests (4 files)
- src/hyper/proxy/tests/auth_tests.rs
- src/hyper/proxy/tests/no_proxy_tests.rs
- src/hyper/proxy/tests/basic_proxy_tests.rs
- src/hyper/proxy/tests/matcher_tests.rs

#### Hyper Async Implementation Request Tests (5 files)
- src/hyper/async_impl/request/tests/builder_tests.rs
- src/hyper/async_impl/request/tests/basic_tests.rs
- src/hyper/async_impl/request/tests/auth_tests.rs
- src/hyper/async_impl/request/tests/header_tests.rs
- src/hyper/async_impl/request/tests/body_tests.rs

#### Other Dedicated Test Files (6 files)
- src/json_path/core_evaluator/evaluator/property_operations/tests.rs
- src/json_path/safe_parsing/tests.rs
- src/json_path/null_semantics/tests.rs
- src/json_path/error/conversions/tests.rs
- src/json_path/error/constructors/tests.rs
- src/hyper/async_stream_service/tests.rs
- src/hyper/async_impl/body/tests.rs
- src/hyper/async_impl/multipart/tests.rs
- src/hyper/async_impl/client/tests.rs
- src/hyper/proxy/internal/tests.rs
- src/hyper/wasm/body/tests.rs
- src/hyper/wasm/multipart/tests.rs
- src/hyper/tls/tests.rs
- src/hyper/redirect/tests.rs

### Category 2: Production Files with Embedded Test Modules (Need Test Extraction)
**Count: 76 files**

#### Core Library Files
- src/lib.rs (main library integration tests)
- src/json_path/compiler.rs (compiler tests)
- src/json_path/mod.rs (module tests)
- src/config/core/mod.rs (configuration tests)

#### JSON Path Core Components
- src/middleware/cache/mod.rs
- src/json_path/normalized_paths/mod.rs
- src/json_path/deserializer/byte_processor/mod.rs
- src/json_path/state_machine/mod.rs
- src/json_path/state_machine/utils.rs
- src/json_path/core_evaluator/filter_support.rs
- src/json_path/core_evaluator/evaluator/descendant_operations/mod.rs
- src/json_path/core_evaluator/evaluator/property_operations/mod.rs
- src/json_path/core_evaluator/evaluator/mod.rs
- src/json_path/core_evaluator/evaluator/timeout_handler.rs
- src/json_path/core_evaluator/evaluator/evaluation_engine.rs
- src/json_path/core_evaluator/evaluator/core_types.rs
- src/json_path/core_evaluator/timeout_protection.rs
- src/json_path/core_evaluator/array_operations.rs
- src/json_path/core_evaluator/selector_engine.rs
- src/json_path/core_evaluator/mod.rs

#### JSON Path Functions
- src/json_path/functions/function_evaluator/length.rs
- src/json_path/functions/function_evaluator/value/mod.rs
- src/json_path/functions/function_evaluator/regex_functions/mod.rs
- src/json_path/functions/function_evaluator/mod.rs
- src/json_path/functions/function_evaluator/count.rs

#### JSON Path Expression System
- src/json_path/expression/core.rs
- src/json_path/expression/evaluation.rs
- src/json_path/expression/complexity.rs
- src/json_path/expression/mod.rs

#### JSON Path Support Systems
- src/json_path/safe_parsing/mod.rs
- src/json_path/json_array_stream/mod.rs
- src/json_path/null_semantics/mod.rs
- src/json_path/error/types/mod.rs
- src/json_path/error/conversions/mod.rs
- src/json_path/error/constructors/mod.rs
- src/json_path/error/mod.rs
- src/json_path/type_system/mod.rs
- src/json_path/buffer/mod.rs

#### HTTP Response and Body
- src/response/body/mod.rs
- src/common/cache/response_cache/core.rs
- src/common/cache/response_cache/eviction.rs
- src/common/cache/response_cache/mod.rs
- src/common/cache/response_cache/operations.rs

#### Builder Methods
- src/builder/methods/mod.rs

#### Hyper Implementation
- src/hyper/async_stream_service/mod.rs
- src/hyper/response.rs
- src/hyper/async_impl/h3_client/mod.rs
- src/hyper/async_impl/h3_client/connect/utilities.rs
- src/hyper/async_impl/response/mod.rs
- src/hyper/async_impl/body/mod.rs
- src/hyper/async_impl/body/streaming.rs
- src/hyper/async_impl/multipart/mod.rs
- src/hyper/async_impl/request/mod.rs
- src/hyper/async_impl/client/config/mod.rs
- src/hyper/async_impl/client/mod.rs

#### Hyper Proxy System
- src/hyper/proxy/types.rs
- src/hyper/proxy/into_proxy.rs
- src/hyper/proxy/internal/mod.rs
- src/hyper/proxy/mod.rs
- src/hyper/proxy/builder/types.rs
- src/hyper/proxy/builder/configuration.rs
- src/hyper/proxy/builder/constructors.rs
- src/hyper/proxy/builder/mod.rs

#### Hyper WASM Support
- src/hyper/wasm/body/mod.rs
- src/hyper/wasm/multipart/mod.rs

#### Hyper Utilities
- src/hyper/into_url.rs
- src/hyper/tls/mod.rs
- src/hyper/redirect/mod.rs

#### Hyper Connection System
- src/hyper/connect/types/mod.rs
- src/hyper/connect/types/tcp_impl.rs
- src/hyper/connect/types/connector.rs
- src/hyper/connect/tcp/dns.rs
- src/hyper/connect/tcp/tls.rs
- src/hyper/dns/resolve/mod.rs

## Extraction Strategy

### Phase 1: Move Dedicated Test Files (28 files)
These are complete test files that just need to be moved to ./tests/ with import updates.

### Phase 2: Extract Embedded Test Modules (76 files)  
These production files contain #[cfg(test)] modules that need extraction while preserving production code.

### Import Update Patterns
- `use super::*` → `use fluent_ai_http3::module_path::*`
- `use crate::*` → `use fluent_ai_http3::*`
- Relative imports → Absolute crate imports

### Verification
- All 104 files processed
- Zero #[cfg(test)] modules remain in src/
- All tests pass with cargo nextest
- Production code compiles cleanly