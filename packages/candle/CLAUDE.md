# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

Fluent AI Candle is a **standalone** Rust package providing native Candle ML framework integration. It depends ONLY on:
- `fluent_ai_http3` for HTTP communications  
- `./packages/async-stream` for AsyncStream patterns
- `cyrup_sugars` for utilities like ZeroOneOrMany

## Core Architecture Constraints

### Streams-Only Architecture
- **ALL async operations**: Use `AsyncStream` patterns only
- **NO Futures**: No `async fn`, no `Future` trait usage
- **NO Result wrapping**: All streams are unwrapped
- **Pattern**: `AsyncStream::with_channel(|sender| { ... })`

### HTTP3-Only Communications
- **ALL HTTP calls**: Must use `fluent_ai_http3`
- **Streaming patterns**: `.collect()`, `.collect_or_else()` 
- **Example**: `Http3::json().body(&request).post(url).collect::<T>()`

### Standalone Independence
- **FORBIDDEN dependencies**: domain, fluent-ai, provider, memory packages
- **Self-contained**: All Candle-prefixed types defined locally
- **Zero external domain coupling**

## Development Commands

Run from `/packages/candle/`:

```bash
# Essential workflow
cargo check                    # Verify compilation (run after every change)
cargo fmt --all               # Format code
cargo test                    # Run tests
cargo build --release         # Release build

# Workspace commands (from repo root)
just check                     # Workspace-wide check + clippy
just hakari-regenerate         # After dependency changes
```

## Current Status

⚠️ **222 compilation errors + 18 warnings need fixing**

See `TODO.md` for detailed breakdown focusing on:
- Generic type parameter fixes (`CandleContext<T>`, `CandleTool<T>`)
- AsyncStream pattern implementation 
- HTTP3 pattern implementation
- Import/naming error resolution

## Key Architecture

### Domain Structure (`src/domain/`)
- **Agent**: `CandleAgentRole`, `CandleAgent`, conversation types
- **Completion**: `CandleCompletionProvider`, request/response types  
- **Context**: `CandleContext<T>` for files, directories, repos
- **Tools**: `CandleTool`, `CandleMcpTool` traits
- **Chat**: `CandleChatLoop`, `CandleMessageChunk` streaming

### Builder Pattern (`src/builders/`)
- **Entry**: `CandleFluentAi::agent_role("name")`
- **Fluent API**: Method chaining as shown in `ARCHITECTURE.md`
- **Type-safe**: Zero-allocation builder patterns

### Providers (`src/providers/`)
- **KimiK2**: `CandleKimiK2Provider` for local Candle inference

## Code Patterns

### AsyncStream Usage
```rust
// Correct pattern
AsyncStream::with_channel(|sender| {
    // stream operations
})

// Forbidden patterns
async fn something() -> Result<T, E> { ... }  // ❌ NO FUTURES
Future::ready(value)                          // ❌ NO FUTURES
```

### HTTP3 Usage  
```rust
// Correct streaming collection
Http3::json().body(&request).post(url).collect::<ResponseType>()

// Error handling
Http3::json().body(&request).post(url).collect_or_else(|err| { 
    // handle error
})
```

### Type Conventions
- **Prefixes**: All types use `CandleXxx` naming
- **HashMap**: Use `hashbrown::HashMap` for consistency
- **Value**: Import `serde_json::Value` as `Value`
- **Utilities**: Use `cyrup_sugars::ZeroOneOrMany`

## Integration Points

- **async-stream**: `AsyncStream`, `AsyncTask`, `spawn_task` primitives
- **http3**: HTTP client with streaming patterns
- **cyrup_sugars**: `ZeroOneOrMany`, `OneOrMany`, `ByteSize`
- **candle-core/nn/transformers**: ML framework integration

## Testing

```bash
cargo test                     # All tests
cargo test test_name          # Specific test
cargo test -- --nocapture    # With output
```

Verify:
- Builder pattern matches `ARCHITECTURE.md` examples
- AsyncStream patterns function correctly  
- HTTP3 integration works
- Standalone compilation (no forbidden dependencies)