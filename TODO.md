# TODO: Future-to-Stream Refactoring Plan

This document outlines the detailed plan to refactor the entire `fluent-ai` codebase from a `Future`-based architecture to a pure `AsyncStream`-based architecture. All work will adhere strictly to the user's architectural guidelines.

## Milestone 1: Establish Canonical Streaming Infrastructure

**Architecture Notes**: The foundation of this refactoring is a canonical, single-source-of-truth implementation of the streaming primitives. We will create a new, low-level crate, `fluent_ai_core`, to house these primitives. This ensures that all other crates have a stable, common base to build upon and eliminates the current issue of multiple, conflicting `AsyncStream` definitions.

- [ ] **Task 1.1**: Create a new crate named `fluent_ai_core` at `packages/core`.
    - **Implementation Notes**: Use `cargo new --lib packages/core` to create the crate. Add it to the root `Cargo.toml` workspace members.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 1.1**: Act as an Objective QA Rust developer. Verify that the `fluent_ai_core` crate has been created correctly, is part of the workspace, and contains no logic other than the default template.

- [ ] **Task 1.2**: Define the canonical `AsyncStream<T>` type alias in `packages/core/src/stream.rs`.
    - **Implementation Notes**: The definition will be `pub type AsyncStream<T> = tokio_stream::wrappers::UnboundedReceiverStream<T>;`. Add `tokio` and `tokio_stream` as dependencies to `fluent_ai_core`.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 1.2**: Act as an Objective QA Rust developer. Verify that the `AsyncStream<T>` type alias is correctly defined in `fluent_ai_core` and that all necessary dependencies are present.

- [ ] **Task 1.3**: Define the `emit!` and `handle_error!` macros in `packages/core/src/macros.rs`.
    - **Implementation Notes**: These macros will handle the unwrapped value emission and error logging for the streaming pattern. They will be simple wrappers around `sender.send()` and `log::error!`.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 1.3**: Act as an Objective QA Rust developer. Verify that the macros are correctly implemented and exported.

- [ ] **Task 1.4**: Define the `AsyncStreamExt<T>` trait in `packages/core/src/stream.rs`.
    - **Implementation Notes**: This trait will provide the `on_chunk` finalizer method. It will be implemented for any `Stream` of unwrapped values. The `on_chunk` method will consume the stream and apply the provided handler closure to each item.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 1.4**: Act as an Objective QA Rust developer. Verify that the `AsyncStreamExt` trait and its implementation are correct and adhere to the streams-only, unwrapped-value principle.

- [ ] **Task 1.5**: Refactor all existing `AsyncStream` type aliases throughout the codebase to use the canonical definition from `fluent_ai_core::stream::AsyncStream`.
    - **Implementation Notes**: This involves finding all other definitions and replacing them with a `use` statement. Add `fluent_ai_core` as a dependency where needed.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 1.5**: Act as an Objective QA Rust developer. Verify that all duplicate `AsyncStream` definitions have been removed and replaced with the canonical import.

## Milestone 2: Refactor `fluent_ai_domain`

**Architecture Notes**: The `domain` crate is the foundation of the application logic. Refactoring this crate first ensures that all higher-level crates will be building upon the new streaming architecture.

- [ ] **Task 2.1**: Refactor `CompletionProvider` trait in `packages/domain/src/completion/provider.rs`.
    - **Implementation Notes**: Convert all `async fn` methods to return `AsyncStream<T>`. For example, `stream_completion` will now return `AsyncStream<CompletionChunk>` directly, not a `Result` wrapping a stream.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 2.1**: Act as an Objective QA Rust developer. Verify that the `CompletionProvider` trait exclusively uses `AsyncStream` for all asynchronous operations.

## Milestone 3: Refactor `fluent_ai_provider`

**Architecture Notes**: This crate contains the concrete implementations of the `CompletionProvider` trait. Each client (OpenAI, Anthropic, etc.) will be refactored to use the new streaming pattern internally and expose it through the trait interface.

- [ ] **Task 3.1**: Refactor `OpenAIClient` in `packages/provider/src/clients/openai/`.
    - **Implementation Notes**: Modify all `async fn` methods, such as those for completions and embeddings, to return `AsyncStream<T>`. The internal HTTP calls will now use the streaming capabilities of `fluent_ai_http3` to produce these streams.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 3.1**: Act as an Objective QA Rust developer. Verify that the `OpenAIClient` correctly implements the streaming `CompletionProvider` trait and that all internal logic adheres to the streams-only pattern.

- [ ] **Task 3.2**: Refactor `AnthropicClient` in `packages/provider/src/clients/anthropic/`.
    - **Implementation Notes**: Apply the same refactoring pattern as with the `OpenAIClient`.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 3.2**: Act as an Objective QA Rust developer. Verify the `AnthropicClient`'s adherence to the new streaming architecture.

- [ ] **Task 3.3**: Refactor all other provider clients.
    - **Implementation Notes**: Systematically go through each remaining client in `packages/provider/src/clients/` and apply the same future-to-stream refactoring.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 3.3**: Act as an Objective QA Rust developer. Verify that all provider clients have been successfully migrated to the new architecture.

## Milestone 4: Refactor `fluent_ai_memory`

**Architecture Notes**: The memory package consumes provider services. All of its internal logic that involves fetching data or performing cognitive operations with LLMs must be converted to the streaming pattern.

- [ ] **Task 4.1**: Refactor cognitive operations in `packages/memory/src/cognitive/`.
    - **Implementation Notes**: Identify all `async fn` calls to providers and convert them to use the new streaming interface. Propagate the streaming pattern up through the memory system's public API.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 4.1**: Act as an Objective QA Rust developer. Verify that the cognitive components of the memory system are fully compliant with the streams-only architecture.

## Milestone 5: Refactor `fluent_ai` (Top-Level Crate)

**Architecture Notes**: This is the final step, where the top-level application logic and public-facing API are updated to use the new, purely stream-based architecture.

- [ ] **Task 5.1**: Refactor the `Agent` and `Engine` implementations in `packages/fluent-ai/src/`.
    - **Implementation Notes**: Update all agent and engine logic to consume the streaming APIs from the memory and provider crates. Expose streaming interfaces where appropriate.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 5.1**: Act as an Objective QA Rust developer. Verify that the top-level crate is fully migrated and that the end-user-facing API correctly reflects the new streaming architecture.

## Milestone 6: Refactor Examples and Tests

- [ ] **Task 6.1**: Update all examples in the `examples/` directory.
    - **Implementation Notes**: Change all `main` functions and other async blocks to use the new streaming API, consuming the streams with `on_chunk` or by iterating over them.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 6.1**: Act as an Objective QA Rust developer. Verify that all examples compile, run, and correctly demonstrate the new streaming-only patterns.

- [ ] **Task 6.2**: Update all integration and unit tests.
    - **Implementation Notes**: Refactor tests to work with the new `AsyncStream` return types. This may involve collecting the stream into a `Vec` to assert the final state.
    - **Warning**: DO NOT MOCK, FABRICATE, FAKE or SIMULATE ANY OPERATION or DATA. Make ONLY THE MINIMAL, SURGICAL CHANGES required. Do not modify or rewrite any portion of the app outside scope.
- [ ] **QA Task 6.2**: Act as an Objective QA Rust developer. Verify that all tests pass and provide complete coverage for the new streaming logic.