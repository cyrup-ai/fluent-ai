# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust workspace containing the core fluent AI builder pattern library and multiple standalone implementation crates. The core library provides a composable, streaming-first approach to AI operations built around trait-based design with polymorphic error handling.

### Workspace Structure
- **`fluent-ai/`** (core library) - Defines traits, builders, domain types, and closure interfaces
- **Implementation crates** - Standalone crates that implement the traits and closures provided by fluent-ai for specific providers (OpenAI, Anthropic, etc.)

## Core Architecture

### Fluent Builder Pattern
The library implements a fluent API centered around `FluentAi` as the main entry point. All operations follow the pattern:
```rust
FluentAi::operation()
    .configuration_methods()
    .on_error(|e| /* handle */)
    .terminal_operation() // -> AsyncTask<T> or AsyncStream<T>
```

### Key Design Constraints
- **NotResult Constraint**: `AsyncTask<T>` and `AsyncStream<T>` cannot contain `Result` types
- **Explicit Error Handling**: Must call `.on_error()` before terminal operations
- **Streaming-First**: Built around `AsyncStream` for real-time operations
- **Pure Traits**: Everything is trait-based with composable builders and closure interfaces

### Module Structure
- `fluent.rs` - Master `FluentAi` builder entry point
- `async_task/` - Custom async primitives (`AsyncTask`, `AsyncStream`, emission utilities)
- `domain/` - Core domain types (agents, completions, documents, memory, workflows, tools, MCP)
- `sugars/` - Utility macros and extensions
- `loaders/` - File format loaders (PDF, EPUB)

## Common Development Commands

### Workspace Commands
```bash
cargo check                    # Type check all workspace crates
cargo build                    # Build all workspace crates
cargo test                     # Run tests for all workspace crates
cargo test -p fluent-ai        # Test only the core library
```

### Core Library Commands
```bash
cargo check -p fluent-ai                    # Type check core library
cargo test -p fluent-ai architecture_api    # Run API design verification tests
```

### Implementation Crate Commands
```bash
cargo check -p fluent-ai-openai             # Check specific implementation
cargo test -p fluent-ai-anthropic           # Test specific implementation
cargo run --example agent_openai            # Run example from implementation crate
```

## Current Development Status

The project is actively in development with compilation errors present. Before making changes:

1. Check `TODO.md` for current implementation priorities
2. Review `ARCHITECTURE.md` for API design specifications
3. Run `cargo check` to see current compilation status

### Known Issues
- Multiple compilation errors (trait implementations, module conflicts)
- Duplicate module definitions in `lib.rs`
- Missing concrete implementations for many fluent builders

## Key Components

### Agent System
- `Agent` - AI agent with tools and context
- `AgentRole` - Persistent agent personas with system prompts
- Tool integration via MCP (Model Context Protocol)

### Document Processing
- `Document` - File loading and processing with polymorphic error handling
- Support for multiple file formats (PDF, EPUB, text)
- Context loading from files, GitHub, URLs, glob patterns

### Memory System
- Persistent context storage and retrieval
- Integration with agent roles for conversation history

### Workflow Engine
- Composable task pipelines
- Chainable operations with error propagation

## Development Guidelines

### Error Handling
All async operations must use the polymorphic error handling pattern:
```rust
operation()
    .on_error(|e| /* handle error */)
    .terminal_method()
```

### Streaming Operations
Prefer `AsyncStream<T>` for real-time operations:
```rust
.on_chunk(|chunk| { /* handle streaming data */ })
.stream_method() // -> AsyncStream<T>
```

### Testing
- Use `tests/architecture_api_test.rs` as reference for API design verification
- All new APIs should follow the fluent builder pattern established in the architecture

## Implementation Crate Guidelines

### Creating New Implementation Crates
When creating a new implementation crate (e.g., `fluent-ai-openai`):

1. **Crate Structure**:
   ```
   fluent-ai-provider/
   ├── Cargo.toml
   ├── src/
   │   ├── lib.rs
   │   ├── completion.rs      # Implement CompletionProvider trait
   │   ├── agent.rs           # Implement AgentProvider trait  
   │   ├── memory.rs          # Implement MemoryProvider trait
   │   ├── tools.rs           # Provider-specific tool implementations
   │   └── error.rs           # Provider-specific error types
   └── examples/
       └── basic_usage.rs
   ```

2. **Dependencies**:
   - Add `fluent-ai = { path = "../fluent-ai" }` to implement core traits
   - Include provider-specific SDK dependencies
   - Use same async runtime (`tokio`) as core library
   - **Important**: After adding dependencies, run `cargo hakari generate && cargo hakari manage-deps` to update workspace-hack

3. **Trait and Closure Implementation**:
   - Implement all required traits defined in `fluent-ai`
   - Implement closure interfaces for streaming operations, error handling, and event callbacks
   - Follow the polymorphic error handling pattern
   - Ensure `NotResult` constraint compliance in `AsyncTask<T>` and `AsyncStream<T>`

4. **Examples**:
   - Include at least one complete usage example
   - Demonstrate both streaming and non-streaming operations
   - Show error handling patterns

### Provider-Specific Considerations
- **OpenAI**: Focus on GPT models, function calling, streaming completions
- **Anthropic**: Claude models, tool use, message-based API
- **Local Models**: Ollama, LLaMA.cpp integration, local inference
- **Custom Providers**: Generic HTTP API implementations

### Testing Implementation Crates
- Create integration tests that verify trait and closure interface compliance
- Test both success and error scenarios with proper closure handling
- Include performance benchmarks for streaming operations
- Verify memory usage patterns for large context operations
- Test closure behavior for error handling, streaming chunks, and event callbacks

## Dependencies

### Dependency Management
**This workspace uses [cargo-hakari](./CARGO_HAKARI.md) for dependency management.** This is the proper way to manage all Cargo dependencies in this workspace. Key points:

- All crates depend on `workspace-hack` for unified feature resolution
- After adding/removing dependencies, always run: `cargo hakari generate && cargo hakari manage-deps`
- Never manually edit `workspace-hack/Cargo.toml` - it's auto-generated
- See [CARGO_HAKARI.md](./CARGO_HAKARI.md) for complete usage guide

### Core Dependencies
- `tokio` - Async runtime
- `serde`/`serde_json` - Serialization
- `reqwest` - HTTP client
- `futures` - Async utilities
- `async-trait` - Async trait support
- `hashbrown` - High-performance HashMap
- `parking_lot` - Synchronization primitives
- `workspace-hack` - Hakari dependency unification

## Nightly Features

The project uses Rust nightly features. Ensure you're using a compatible nightly toolchain when developing.