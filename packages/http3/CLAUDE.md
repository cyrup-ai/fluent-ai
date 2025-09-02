# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# fluent_ai_http3 - Streaming-First HTTP/3 Client

A zero-allocation HTTP/3 (QUIC) client with HTTP/2 fallback designed for AI providers and streaming applications. Built on top of `fluent_ai_async` for pure streaming architecture.

## Project Structure

The codebase is organized as a workspace with two main packages:
- **`packages/api/`**: Public API with fluent builder pattern (`fluent_ai_http3`)
- **`packages/client/`**: Core implementation with protocol support (`fluent_ai_http3_client`)

## Core Architecture Principles

### Streaming-First Design
- **NO Futures**: All async operations use `fluent_ai_async::AsyncStream` patterns
- **Zero Allocation**: Lock-free, atomic operations with minimal memory footprint
- **Pure Streams**: Traditional "futures-like" behavior available via `.collect()` on streams
- **Emit-Based**: Use `emit!` macro for stream events, `AsyncStream::with_channel` for producers

### Protocol Strategy
- **HTTP/3 Prioritization**: QUIC/HTTP3 first, with automatic HTTP/2 fallback
- **QUICHE Implementation**: Using QUICHE for HTTP/3 protocol support
- **Modular Protocols**: Clean separation between H2 and H3 implementations in `src/protocols/`

### Builder Pattern
- **Fluent Interface**: Ergonomic, type-safe builders reminiscent of Axum
- **State Types**: Compile-time enforcement of builder state
- **Zero-Copy**: Efficient header and body handling without unnecessary allocations

## Key Development Commands

```bash
# Basic compilation check
cargo check --workspace --all-targets

# Run all tests with nextest (preferred)
cargo nextest run --workspace

# Run specific test
cargo nextest run test_name

# Run with output capture
cargo nextest run --nocapture

# Quick format and check (using justfile)
just check

# Build release
just build

# Run tests
just test

# Build examples
cargo build --examples

# Run fluent builder example
cargo run --example fluent_builder

# Check compilation for api package
cargo check -p fluent_ai_http3

# Check compilation for client package  
cargo check -p fluent_ai_http3_client
```

## Architecture Overview

### API Package (`packages/api/`)

**`src/lib.rs`**: Main public exports
- Re-exports from client implementation
- Public builder API
- Type aliases for convenience

**`src/builder/`**: Fluent API construction
- Public-facing builder methods
- Type-safe state management
- Syntactic sugar for common patterns

### Client Package (`packages/client/`)

**Core Modules:**

**`src/protocols/`**: Direct protocol implementations
- `h2/`: HTTP/2 connection and streaming logic
- `h3/`: HTTP/3 connection and streaming logic  
- `quiche/`: QUICHE-specific HTTP/3 implementation
- `http1/`: HTTP/1.1 fallback support
- Pure `fluent_ai_async` patterns, NO middleware layers

**`src/builder/`**: Core builder implementation
- `builder_core.rs`: Main Http3Builder implementation
- `fluent.rs`: DownloadBuilder and fluent interfaces
- `methods/`: HTTP method implementations
- `streaming.rs`: Stream-based response handling
- Type-safe builder pattern with compile-time state enforcement

**`src/client/`**: HttpClient and connection management
- `core.rs`: Main client implementation
- `configuration.rs`: Client configuration
- `execution.rs`: Request execution logic
- `stats.rs`: Client statistics and telemetry

**`src/jsonpath/`**: Revolutionary JSONPath streaming processor
- `deserializer/`: Stream-based JSON deserialization with JSONPath
- `core_evaluator/`: JSONPath expression evaluation engine
- `stream_processor/`: Incremental JSON stream processing
- `buffer/`: Zero-allocation buffering for stream processing
- `state_machine/`: Incremental JSON parsing state machine
- `json_array_stream/`: High-level streaming array processor
- Zero-allocation path matching with timeout protection

**`src/connect/`**: Connection establishment
- `builder/`: Connection builder with TLS configuration
- `service/`: Service layer for connection management
- `proxy/`: Proxy support implementation
- `types/`: Connection type definitions

**`src/http/`**: HTTP protocol support
- `request.rs`: Request building and serialization
- `response.rs`: Response parsing and handling
- `resolver/`: DNS and service resolution
- `headers.rs`: Header manipulation utilities

**`src/tls/`**: TLS configuration and management
- `builder/`: TLS configuration builders
- `certificate/`: Certificate handling
- `ocsp.rs`: OCSP stapling support
- `crl_cache.rs`: Certificate revocation list caching

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

## 🚀 JSONPath Streaming - Revolutionary Feature

**IMPORTANT**: This library includes a revolutionary JSONPath streaming feature that enables real-time processing of massive JSON responses without loading them into memory.

### Quick Example: Streaming OpenAI Models

```rust
use fluent_ai_http3::Http3;
use serde::Deserialize;

#[derive(Deserialize)]
struct OpenAIModel {
    id: String,
    owned_by: String,
}

// Stream models as they arrive - processing starts immediately!
Http3::json()
    .array_stream("$.data[*]")  // JSONPath expression
    .bearer_token(&api_key)
    .get("https://api.openai.com/v1/models")
    .on_chunk(|model: OpenAIModel| {
        // Called AS SOON as each model is found in the stream
        // Not after the entire response downloads!
        println!("Found: {}", model.id);
        model
    })
```

### Key Benefits
- **Zero Memory Overhead**: Process TB of JSON with constant ~8KB RAM
- **Instant Processing**: First result in milliseconds, not after full download
- **100K+ objects/second**: Blazing fast streaming throughput
- **Full JSONPath Support**: Filters, wildcards, recursive descent, slicing

### Architecture Highlights
- `jsonpath/buffer/`: Zero-allocation streaming buffer
- `jsonpath/state_machine/`: Incremental JSON parser
- `jsonpath/stream_processor/`: Real-time JSONPath evaluation
- `jsonpath/core_evaluator/`: Compiled JSONPath expressions

**📚 For complete documentation, examples, and implementation details, see [docs/JSONPATH.md](docs/JSONPATH.md)**

## Current Development Status

### Known Issues (from TODO.md)
The codebase has compilation issues that need addressing:
- **91 errors** and **227 warnings** to fix
- Import resolution errors in JSONPath and protocol modules
- Missing methods and type mismatches in various modules
- Lifetime and borrowing issues in protocol implementations

### Testing Approach
```bash
# Use nextest for better output and parallelization
cargo nextest run

# Run specific test suites
cargo nextest run --test integration_tests
cargo nextest run --lib

# Check compilation without running
cargo check --all-targets --workspace
```

## Important Development Notes

### Streaming Architecture
- All HTTP operations return streams by default
- Use `.collect()` only when traditional response handling is needed
- Leverage `fluent_ai_async` patterns throughout
- Avoid any Future-based abstractions

### Error Handling
- Use `HttpError` and `HttpResult<T>` types from error module
- Comprehensive error types with retry semantics
- No `unwrap()` or `expect()` in production code
- Graceful fallback patterns for protocol downgrades

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

## Working with Workspace Dependencies

- Dependencies are managed at the workspace level
- Use `cargo hakari` for dependency management
- Run `just hakari-regenerate` after dependency changes
- NO `workspace = true` in dependency specifications
- Always use latest versions unless explicitly documented

## Example Usage Patterns

The primary example demonstrating the fluent API is in `examples/fluent_builder.rs`:

```rust
// Stream-based usage (primary pattern)
Http3::json()
    .debug()
    .headers([("x-api-key", "abc123")])
    .body(&request)
    .post(&server_url)
    .on_chunk(|result| { /* process chunks */ });

// Collect to typed response
let response = Http3::json()
    .api_key("abc123")
    .body(&request)
    .post(&server_url)
    .collect_one::<ResponseType>();

// Form-encoded requests
Http3::form_urlencoded()
    .basic_auth([("user", "password")])
    .body(&form_data)
    .post(&server_url)
    .collect_one::<ResponseType>();
```

## Debugging and Logging

Enable debug logging with:
```rust
std::env::set_var("RUST_LOG", "fluent_ai_http3=debug,h2=debug,quiche=debug");
env_logger::init();
```

Or use the `.debug()` method on builders for request-specific debugging.