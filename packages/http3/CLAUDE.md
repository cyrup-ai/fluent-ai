# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# fluent_ai_http3 - Streaming-First HTTP/3 Client

A zero-allocation HTTP/3 (QUIC) client with HTTP/2 fallback designed for AI providers and streaming applications. Built on top of `fluent_ai_async` for pure streaming architecture.

## Core Architecture Principles

### Streaming-First Design
- **NO Futures**: All async operations use `fluent_ai_async::AsyncStream` patterns
- **Zero Allocation**: Lock-free, atomic operations with minimal memory footprint
- **Pure Streams**: Traditional "futures-like" behavior available via `.collect()` on streams
- **Emit-Based**: Use `emit!` macro for stream events, `AsyncStream::with_channel` for producers

### Protocol Strategy
- **HTTP/3 Prioritization**: QUIC/HTTP3 first, with automatic HTTP/2 fallback
- **QUICHE Only**: Currently migrating from Quinn to QUICHE for HTTP/3 implementation
- **Modular Protocols**: Clean separation between H2 and H3 implementations in `src/protocols/`

### Builder Pattern
- **Fluent Interface**: Ergonomic, type-safe builders reminiscent of Axum
- **State Types**: Compile-time enforcement of builder state (BodySet/BodyNotSet)
- **Zero-Copy**: Efficient header and body handling without unnecessary allocations

## Key Development Commands

```bash
# Basic compilation check
cargo check

# Run all tests (uses nextest for performance)
cargo nextest run

# Run specific test
cargo nextest run test_name

# Run with output capture
cargo nextest run --nocapture

# Quick syntax check
cargo check --message-format short --quiet

# Build examples
cargo build --examples

# Run fluent builder example
cargo run --example fluent_builder

# Run specific integration test
cargo nextest run --test dns_resolver_integration
```

## Architecture Overview

### Core Modules

**`src/lib.rs`**: Main exports and global client management
- Global client instance with connection pooling
- Type aliases for Result and Error types
- Re-exports of all public APIs

**`src/protocols/`**: Direct protocol implementations
- `h2/`: HTTP/2 connection and streaming logic
- `h3/`: HTTP/3 connection and streaming logic  
- `quiche/`: QUICHE-specific HTTP/3 implementation
- Pure `fluent_ai_async` patterns, NO middleware layers

**`src/streaming/`**: Core streaming foundation
- `chunks.rs`: HttpChunk, QuicheStreamChunk types
- `stream.rs`: HttpStream, SseStream, JsonStream implementations
- `response.rs`: HttpResponse with streaming capabilities
- Zero-allocation, lock-free streaming patterns

**`src/builder/`**: Fluent API construction
- `builder_core.rs`: Main Http3Builder with state types
- `fluent.rs`: DownloadBuilder and fluent interfaces
- `methods/`: HTTP method implementations
- Type-safe builder pattern with compile-time state enforcement

**`src/client/`**: HttpClient and connection management
- Connection pooling and reuse logic
- Client statistics and telemetry
- Configuration management

**`src/jsonpath/`**: JSONPath streaming processor
- `deserializer/`: Stream-based JSON deserialization with JSONPath
- `core_evaluator/`: JSONPath expression evaluation engine
- `stream_processor/`: Incremental JSON stream processing
- Zero-allocation path matching

### Stream Processing Patterns

```rust
// Primary usage - streaming first
let stream = client.get("https://api.example.com")
    .send_stream().await?;

// Process as stream
let mut sse_stream = stream.sse();
while let Some(event) = sse_stream.next().await {
    // Handle streaming events
}

// Traditional collection (when needed)
let response = stream.collect().await?;
```

### Protocol Abstraction

The library provides clean abstraction over HTTP versions:
- `HttpProtocol`: Version-agnostic interface
- `ConnectionManager`: Handles protocol negotiation
- `TransportConnection`: Unified connection interface
- Automatic fallback from HTTP/3 → HTTP/2 → HTTP/1.1

## Current Migration Status

**QUICHE Transition**: Currently migrating from Quinn to QUICHE for HTTP/3
- Remove all `h3_quinn` and `quinn` dependencies
- Replace with `quiche` equivalents maintaining streaming patterns
- See `TODO.md` for detailed migration tasks
- NO mocking or simulation - only surgical changes to real implementation

## Important Development Notes

### Streaming Architecture
- All HTTP operations return streams by default
- Use `.collect()` only when traditional response handling is needed
- Leverage `fluent_ai_async` patterns throughout
- Avoid any Future-based abstractions

### Error Handling
- Use `HttpError` and `HttpResult<T>` types
- Comprehensive error types with retry semantics
- No `unwrap()` or `expect()` in production code
- Graceful fallback patterns for protocol downgrades

### Testing Patterns
- Integration tests in `tests/` directory
- Builder pattern tests in `tests/builder/`
- JSONPath streaming tests in `tests/jsonpath/`
- Use `nextest` for parallel test execution

### Performance Considerations
- Connection pooling is enabled by default
- HTTP/3 provides significant performance benefits for concurrent requests
- Stream processing avoids memory accumulation for large responses
- Lock-free data structures throughout critical paths

## Code Style Guidelines

- Follow Rust 2024 edition patterns
- Use `#![deny(unsafe_code)]` - no unsafe code allowed
- Leverage clippy pedantic warnings
- Prefer explicit error handling over panics
- Use `tracing` for structured logging, not `println!`

## Integration with fluent-ai Ecosystem

This library is a foundational component of the fluent-ai ecosystem:
- Depends on `fluent_ai_async` for streaming primitives
- Used by higher-level AI provider clients
- Integrates with memory system and tool execution flows
- Supports WASM compilation for browser environments

## Critical Implementation Notes

### Current Issues (from TODO.md)
The codebase currently has compilation issues that need addressing:
- Unstable Rust features in use (`#![feature(...)]` in lib.rs) - must be removed
- Panic-based error handling in global client initialization
- API mismatches between HttpRequest builder methods
- Stubbed implementations in JSONPath deserializer
- Missing multipart encoding implementation

### Testing Approach
When running tests, use `cargo nextest` for better output and parallelization:
```bash
# Run all tests
cargo nextest run

# Run with specific timeout for recursive descent tests
cargo nextest run test_recursive_descent --nocapture

# Check compilation without running
cargo check --all-targets
```

### Key Dependencies
- `fluent_ai_async`: Custom async streaming library (NO tokio futures)
- `quiche`: HTTP/3 implementation (replacing quinn)
- `h2`: HTTP/2 protocol support
- `rustls`: TLS implementation
- `hickory-resolver`: DNS resolution
- `cyrup_sugars`: Syntactic sugar macros