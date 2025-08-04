# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

Fluent AI Candle is a **standalone** Rust package providing native Candle ML framework integration. It wraps the HuggingFace Candle framework (located in `../../tmp/candle/`) while maintaining complete architectural independence from other fluent-ai packages.

**ONLY ALLOWED dependencies:**
- `fluent_ai_http3` for HTTP communications  
- `../async-stream` for AsyncStream patterns
- `cyrup_sugars` for utilities like ZeroOneOrMany
- Direct Candle framework crates (candle-core, candle-nn, candle-transformers)

## Core Architecture Constraints

### Streams-Only Architecture
- **ALL async operations**: Use `AsyncStream` patterns only - NO exceptions
- **NO Futures**: Absolutely no `async fn` or `Future` trait usage anywhere
- **NO Result wrapping**: All streams are unwrapped
- **Pattern**: `AsyncStream::with_channel(|sender| { ... })`
- **Integration**: Must work with fluent-ai-async primitives

### HTTP3-Only Communications
- **ALL HTTP calls**: Must use `fluent_ai_http3` exclusively
- **Streaming patterns**: Use `.collect()`, `.collect_or_else()` methods
- **Example**: `Http3::json().body(&request).post(url).collect::<T>()`
- **No reqwest/hyper**: Only HTTP3 client allowed

### Standalone Independence
- **FORBIDDEN dependencies**: Cannot depend on `domain`, `fluent-ai`, `provider`, `memory` packages
- **Self-contained**: All Candle-prefixed types defined locally
- **Zero external domain coupling**: Complete architectural isolation
- **Candle prefix**: All types use `CandleXxx` naming convention

## Development Commands

Run from `/packages/candle/`:

```bash
# Essential workflow - run after every change  
cargo check              # Verify compilation (CRITICAL - must pass)
cargo fmt --all         # Format code
cargo test              # Run tests  
cargo build --release   # Release build

# Workspace commands (from repo root)
just check              # Workspace-wide check + clippy
just hakari-regenerate  # After dependency changes ONLY
```

## Current Status & Critical Tasks

⚠️ **COMPILATION ERRORS**: Package currently has compilation issues that must be resolved

### Priority 1: Generic Type Parameter Fixes
- **CandleContext<T>**: Missing generic parameters in multiple files
- **CandleTool<T>**: Missing generic parameters in tool implementations  
- **CandleClient<T>**: Missing generic parameters in client types
- See `TODO.md` for detailed breakdown of 240+ issues

### Priority 2: AsyncStream Pattern Implementation
- Convert all async/await patterns to AsyncStream
- Remove all Future trait usage
- Implement streaming completion providers
- Integrate with HTTP3 streaming patterns

### Priority 3: Import Resolution
- Fix missing CandlePrompt, CandleCompletionParams imports
- Resolve module structure issues
- Ensure all Candle-prefixed types are properly defined

## Key Architecture Components

### Domain Structure (`src/domain/`)
- **Agent**: `CandleAgentRole`, `CandleAgent` with conversation management
- **Completion**: `CandleCompletionProvider` with streaming responses
- **Context**: `CandleContext<T>` for files, directories, repositories
- **Tools**: `CandleTool`, `CandleMcpTool` trait implementations
- **Chat**: `CandleChatLoop`, `CandleMessageChunk` streaming

### Builder Pattern (`src/builders/`)
- **Entry Point**: `CandleFluentAi::agent_role("name")` 
- **Fluent API**: Method chaining as shown in `ARCHITECTURE.md`
- **Type Safety**: Zero-allocation builder patterns with compile-time guarantees
- **Integration**: Works with all domain types seamlessly

### Providers (`src/providers/`)  
- **KimiK2**: `CandleKimiK2Provider` for local Candle inference
- **Tokenizer**: `CandleTokenizer` for text processing
- **Integration**: Direct integration with HuggingFace Candle framework

### Candle Framework Integration

The package integrates with the sophisticated HuggingFace Candle ML framework:

- **Location**: Framework source in `../../tmp/candle/`
- **Models**: Supports LLaMA, Mistral, Phi, Gemma, Whisper, Stable Diffusion, YOLO, and 50+ other models
- **Performance**: SIMD optimizations, Metal/CUDA GPU acceleration, zero-allocation patterns
- **Features**: Advanced sampling (Mirostat, nucleus, typical), KV caching, memory-mapped loading

## Code Patterns & Conventions

### AsyncStream Usage
```rust
// ✅ CORRECT - AsyncStream pattern
AsyncStream::with_channel(|sender| {
    // All streaming operations here
    sender.send(chunk).unwrap();
})

// ❌ FORBIDDEN - Future patterns  
async fn something() -> Result<T, E> { ... }
Future::ready(value)
```

### HTTP3 Integration
```rust
// ✅ CORRECT - HTTP3 streaming
Http3::json().body(&request).post(url).collect::<ResponseType>()

// Error handling with streaming
Http3::json().body(&request).post(url).collect_or_else(|err| { 
    // Handle streaming errors
})
```

### Type Conventions
- **Prefixes**: ALL types use `CandleXxx` naming (no exceptions)
- **HashMap**: Use `hashbrown::HashMap` for consistency with domain
- **Value**: Import `serde_json::Value` as `Value`
- **ZeroOneOrMany**: Use `cyrup_sugars::ZeroOneOrMany` directly

### Builder Pattern Implementation
```rust
// Example from ARCHITECTURE.md - must compile exactly as shown
let stream = CandleFluentAi::agent_role("rusty-squire")
    .completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))
    .temperature(1.0)
    .max_tokens(8000)
    .system_prompt("You are a helpful assistant")
    .context(
        CandleContext::<CandleFile>::of("/path/to/file.pdf"),
        CandleContext::<CandleFiles>::glob("/path/**/*.{md,txt}")
    )
    .tools(
        CandleTool::<CandlePerplexity>::new([("citations", "true")]),
        CandleTool::named("cargo").bin("~/.cargo/bin")
    )
    .into_agent()
    .chat("Hello") // Returns AsyncStream<CandleMessageChunk>
    .collect();
```

## Quality Standards

### Code Quality Requirements
- **Zero allocation** where possible using `Cow<T>` patterns
- **Blazing performance** with SIMD optimizations and memory alignment
- **Memory safety** - absolutely no unsafe code
- **Lock-free** concurrent data structures (`crossbeam-skiplist`, `dashmap`)
- **Elegant APIs** with builder patterns and fluent interfaces
- **Never unwrap()** - all errors handled with `Result<T, E>`
- **Never expect()** in src/ - only in tests/
- **Production-ready** with comprehensive error handling

### Performance Optimizations
- Inline critical functions with `#[inline(always)]`
- Use `#[cold]` for error handling paths
- Memory layout optimization with `#[repr(C)]`
- SIMD operations through explicit vectorization
- Memory pool reuse to minimize allocations
- Zero-copy operations wherever possible

## Integration Points

### With fluent-ai-async
- `AsyncStream`, `AsyncTask`, `spawn_task` primitives
- Streaming patterns throughout codebase
- Task spawning for concurrent operations

### With fluent-ai-http3  
- HTTP client with streaming collection patterns
- Error handling with stream continuation
- Request/response streaming

### With cyrup_sugars
- `ZeroOneOrMany`, `OneOrMany` for variadic arguments
- `ByteSize` for memory management
- Utility types and macros

### With Candle Framework
- `candle-core` for tensor operations and Device abstraction
- `candle-nn` for neural network building blocks
- `candle-transformers` for model implementations and utilities
- Memory-mapped model loading and SIMD-optimized operations

## Testing & Verification

```bash
# Test specific functionality
cargo test --lib                    # Library tests
cargo test --bin candle-chat        # Binary tests  
cargo test test_name                # Specific test
cargo test -- --nocapture          # With output

# Architecture verification
cargo run --example candle_agent_role_builder  # Test ARCHITECTURE.md patterns
```

**Verification Requirements:**
- Builder pattern exactly matches `ARCHITECTURE.md` examples
- AsyncStream patterns function correctly without blocking
- HTTP3 integration works with streaming responses  
- Standalone compilation succeeds (no forbidden dependencies)
- All generic type parameters resolve correctly

## Common Issues & Solutions

### Compilation Errors
1. **Missing generics**: Add `<T>` parameters to CandleContext, CandleTool, etc.
2. **Import errors**: Ensure all Candle-prefixed types are defined and exported
3. **Trait confusion**: Use concrete types, not trait objects in builder patterns

### AsyncStream Integration
1. **No blocking**: Replace all `async/await` with AsyncStream patterns
2. **Stream chaining**: Use `.collect()` and `.collect_or_else()` for composition
3. **Error propagation**: Handle errors within stream context

### HTTP3 Integration  
1. **Streaming responses**: Use HTTP3 streaming patterns, not one-shot requests
2. **Error handling**: Implement stream-aware error recovery
3. **Request building**: Use HTTP3 builder patterns consistently

## Architecture Validation

Before committing changes, verify:

1. **Compilation**: `cargo check` passes with 0 errors and 0 warnings
2. **Architecture**: No forbidden dependencies (run `cargo tree` to verify)
3. **Patterns**: AsyncStream usage throughout, no Future patterns
4. **Integration**: HTTP3 streaming works correctly
5. **Examples**: ARCHITECTURE.md examples compile and run
6. **Performance**: Zero-allocation patterns where specified
7. **Error Handling**: Comprehensive Result<T, E> usage without unwrap()

## Documentation Standards

- **Module docs**: All public modules require documentation
- **Type docs**: All public types require documentation  
- **Method docs**: All public methods require documentation
- **Examples**: Include usage examples in documentation
- **Architecture**: Document integration points and patterns
- **Performance**: Document zero-allocation and SIMD usage