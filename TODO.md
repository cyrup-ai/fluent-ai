# TODO

## JSONPath Feature Completion  
- Implement buffer shrinking optimization in json_path/buffer.rs (BytesMut limitation workaround)
- Complete HTTP3 Builder integration with existing JSONPath functionality
- Wire JSONPath processing into HTTP response handling
- Implement JsonStreamProcessor for HTTP response chunk handling  
- Add JsonArrayStream integration with Http3Builder

## RFC 9535 Critical Compliance Gaps

### Missing ABNF Grammar Tests (RFC Appendix A)
- Add dot notation syntax validation tests (missing from current coverage)
- ✅ **COMPLETED**: Add current node identifier (@) validation tests in proper contexts - `rfc9535_current_node_identifier_tests.rs`
- ✅ **COMPLETED**: Add function expression syntax tests (function calls, arguments) - `rfc9535_function_composition_tests.rs`
- ✅ **COMPLETED**: Add logical expression tests (&&, ||, ! operators) - `rfc9535_filter_precedence_comprehensive_tests.rs`
- Add comparison operator tests (<, <=, >, >=, ==, !=)
- Add complete shorthand syntax validation tests
- Add bracket notation escape sequence tests

### Missing Core RFC Requirements  
- Add well-formedness vs validity distinction tests (RFC Section 2.1)
- ✅ **COMPLETED**: Add I-JSON integer range boundary tests [-(2^53)+1, (2^53)-1] - `rfc9535_ijson_boundary_tests.rs`
- ✅ **COMPLETED**: Add comprehensive filter selector tests with nested expressions - `rfc9535_filter_precedence_comprehensive_tests.rs`
- ✅ **COMPLETED**: Add function integration tests (functions calling other functions) - `rfc9535_function_composition_tests.rs`
- Add complete null vs missing value semantic tests (RFC Section 2.6)
- Add normalized paths canonical form enforcement tests (RFC Section 2.7)

### Missing Security and Error Handling
- ✅ **COMPLETED**: Add DoS protection tests for recursive descent patterns (RFC Section 4) - `rfc9535_dos_protection_tests.rs`
- Add parser vulnerability tests for malformed inputs
- Add comprehensive UTF-8 decode error handling tests
- Add memory exhaustion protection tests for deep nesting

### Missing Function System Tests
- Add complete function type system validation (ValueType, LogicalType, NodesType)
- Add function argument type checking tests
- Add function result type validation tests
- Add function composition and nested call tests

### Missing Advanced Features
- Add IANA media type registration tests (application/jsonpath)
- Add function extension registry validation tests
- ✅ **COMPLETED**: Add JSON Pointer compatibility tests (RFC 6901 interop) - `rfc9535_json_pointer_compatibility_tests.rs`
- Add boundary condition tests for deeply nested expressions
- Add performance regression tests for streaming behavior

## JSONPath RFC 9535 Test Compilation Failures

### **Fix RFC9535 Function Length Test Syntax Error**
- **FILE**: `packages/http3/tests/rfc9535_function_length.rs:44-45`
- **ERROR**: `error[E0308]: mismatched types` - incomplete expression
- **MODULE TO CHANGE**: Test file only
- **WORK REQUIRED**: 
  - Fix malformed line: `let mut stream = JsonArrayStream::<serde_json::Value>::new(expr);`
  - Remove extra semicolon or complete the expression properly
  - Ensure all JSON path expressions are valid

### **Fix Missing JsonPathErrorExt Import**
- **FILE**: `packages/http3/tests/json_path_error_tests.rs:6`
- **ERROR**: `error[E0432]: unresolved import JsonPathErrorExt`
- **MODULE TO CHANGE**: `packages/http3/src/json_path/error.rs` OR remove import
- **WORK REQUIRED**:
  - Export `JsonPathErrorExt` trait from error module
  - OR remove import and update test to use available error API
  - Add `with_stream_context` method if needed for tests

### **Fix StreamStateMachine Private Field Access in JSONPath Tests**
- **FILE**: `packages/http3/tests/json_path_state_machine_tests.rs:18,19,28`
- **ERROR**: `error[E0616]: field 'state' of struct 'StreamStateMachine' is private`
- **MODULE TO CHANGE**: `packages/http3/src/json_path/state_machine.rs`
- **WORK REQUIRED**:
  - Add public getter methods: `pub fn state(&self) -> &JsonStreamState`
  - Add public getter methods: `pub fn stats(&self) -> &StreamStats`
  - Update tests to use `sm.state()` instead of `sm.state`
  - Update tests to use `sm.stats()` instead of `sm.stats`

### **Fix StreamBuffer Private Field Access in JSONPath Tests**
- **FILE**: `packages/http3/tests/json_path_buffer_tests.rs:18`
- **ERROR**: `error[E0616]: field 'buffer' of struct 'StreamBuffer' is private`
- **MODULE TO CHANGE**: `packages/http3/src/json_path/buffer.rs`
- **WORK REQUIRED**:
  - Add public method: `pub fn capacity(&self) -> usize { self.buffer.capacity() }`
  - OR make buffer field `pub(crate)` for test access
  - Update test to use `buffer.capacity()` instead of `buffer.buffer.capacity()`

### **Fix State Machine Initialization Type Mismatch in JSONPath Tests**
- **FILE**: `packages/http3/tests/json_path_state_machine_tests.rs:27`
- **ERROR**: `error[E0308]: expected JsonPathExpression, found Result<JsonPathExpression, JsonPathError>`
- **MODULE TO CHANGE**: Test file only
- **WORK REQUIRED**:
  - Parse JsonPathExpression first: `let expr = JsonPathParser::compile("$.test")?;`
  - Pass unwrapped expression to `initialize()` method
  - Handle parsing errors appropriately in test setup

### **Fix AsyncStream Iterator Usage in RFC9535 Tests**
- **FILES**: Multiple RFC9535 test files
- **ERROR**: `error[E0599]: 'AsyncStream<i32>' is not an iterator`
- **MODULE TO CHANGE**: Test files - update to use proper stream API
- **WORK REQUIRED**:
  - Replace `.collect()` calls on AsyncStream with proper stream consumption
  - Use JsonArrayStream `.process_chunk()` method correctly
  - Follow streams-first architecture patterns instead of iterator patterns