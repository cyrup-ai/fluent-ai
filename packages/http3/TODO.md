# HTTP3 Tokio Cleanup - Final Remaining Items

## Milestone: Complete Remaining Tokio Dependencies

### Phase 1: Remove TokioExecutor Usage

- [ ] **Replace TokioExecutor with fluent_ai_async executor** (src/hyper/async_impl/client.rs, line 560)
  - Replace `hyper_util::rt::TokioExecutor::new()` with custom executor using fluent_ai_async spawn_task
  - Implement executor that bridges hyper's executor trait with fluent_ai_async patterns
  - Maintain existing connection pooling and TLS functionality
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA: Verify TokioExecutor replacement**
  Act as an Objective QA Rust developer and verify that TokioExecutor has been properly replaced with fluent_ai_async patterns. Confirm that HTTP operations still work correctly and connection pooling is maintained.

### Phase 2: Remove Tokio Features from Dependencies

- [ ] **Remove tokio features from hyper-util** (Cargo.toml, line 53)
  - Use `cargo remove hyper-util` then `cargo add hyper-util --features="http1,http2,client,client-legacy"`
  - Remove "tokio" from features list
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA: Verify hyper-util tokio feature removal**
  Act as an Objective QA Rust developer and verify that hyper-util tokio features have been properly removed without breaking HTTP functionality.

- [ ] **Remove tokio features from hickory-resolver** (Cargo.toml, line 66)
  - Use `cargo remove hickory-resolver` then `cargo add hickory-resolver --optional`
  - Remove "tokio" from features list, use alternative async patterns if needed
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA: Verify hickory-resolver tokio feature removal**
  Act as an Objective QA Rust developer and verify that hickory-resolver tokio features have been properly removed and DNS resolution still works.

- [ ] **Remove tokio features from quinn** (Cargo.toml, line 69)
  - Use `cargo remove quinn` then `cargo add quinn --features="rustls" --optional`
  - Remove "runtime-tokio" from features list
  - Ensure HTTP3/QUIC still works with alternative runtime patterns
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA: Verify quinn tokio feature removal**
  Act as an Objective QA Rust developer and verify that quinn tokio features have been properly removed and HTTP3/QUIC functionality is maintained.

- [ ] **Remove tokio features from dev-dependencies hyper-util** (Cargo.toml, line 77)
  - Update dev-dependencies to remove "tokio" feature from hyper-util
  - Ensure tests still compile and run correctly
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA: Verify dev-dependencies tokio feature removal**
  Act as an Objective QA Rust developer and verify that dev-dependencies tokio features have been properly removed and tests still work.

### Phase 3: Validation and Testing

- [ ] **Verify package compiles without tokio runtime**
  - Run `cargo check` to ensure no compilation errors
  - Run `cargo test` to ensure all tests pass
  - Verify HTTP requests work with existing AsyncStream patterns
  - DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.

- [ ] **QA: Verify complete tokio elimination**
  Act as an Objective QA Rust developer and verify that all tokio dependencies have been eliminated and the package works correctly with pure fluent_ai_async patterns.