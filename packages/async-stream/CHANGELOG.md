# Changelog

All notable changes to the AsyncStream library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of AsyncStream library
- Zero-allocation streaming primitives for fluent-ai ecosystem
- AsyncStreamBuilder pattern for custom error handling
- Support for const-generic capacities
- Crossbeam-based lock-free concurrency
- MessageChunk trait for proper error handling
- Comprehensive example suite demonstrating real-world patterns

### Features

#### Core Streaming
- `AsyncStream<T, CAP>` - Primary streaming type with const-generic capacity
- `AsyncStreamBuilder<T, CAP>` - Builder pattern for stream creation
- `AsyncTask<T>` - Single-value async computation result
- `AsyncStreamSender<T, CAP>` - Send-side of stream channel

#### Builder Pattern
- `.builder()` - Create AsyncStreamBuilder instance
- `.on_chunk()` - Custom error handling with Result<T, E> → T transformation
- `.with_channel()` - Producer function setup
- `.from_receiver()` - Create stream from mpsc::Receiver

#### Task Spawning
- `spawn_task()` - Spawn single-value background computation
- `spawn_stream()` - Spawn streaming background computation
- `spawn_stream_with_capacity()` - Spawn with custom capacity

#### Channel Operations
- `channel()` - Create sender/receiver pair with default capacity
- `channel_with_capacity()` - Create with custom capacity
- `unbounded_channel()` - Alias for default capacity channel

#### Consumption Methods
- `.collect()` - Collect all values (blocking)
- `.collect_or_else()` - Collect with error recovery
- `.into_iter()` - Convert to iterator
- `.try_next()` - Non-blocking pop
- `.next().await` - Async pop (requires async context)

#### Macros
- `emit!()` - Ergonomic streaming with automatic error handling
- `on_chunk_pattern!()` - Pattern matching for chunk processing
- `handle_error!()` - Error handling utilities

### Architecture

#### Streams-Only Design
- No `Result<T, E>` types inside streams
- Error handling via `on_chunk` patterns
- Zero-allocation streaming with crossbeam primitives
- No external async runtime dependencies

#### Performance Optimizations
- Lock-free `ArrayQueue` for stream storage
- Const-generic capacities for compile-time optimization
- Minimal allocations with pre-sized buffers
- Zero-copy streaming where possible

#### Type Safety
- `MessageChunk` trait ensures proper error handling
- No `async_trait` usage - native async/await only
- Compile-time capacity checking
- Strong type safety with generic constraints

### Examples

#### Pattern Examples
- `on_chunk_pattern.rs` - Builder pattern with custom error handling
- `spawn_task_pattern.rs` - Task spawning patterns (spawn_task/spawn_stream)
- `with_channel_pattern.rs` - Producer/consumer setup with with_channel
- `emit_macro_pattern.rs` - Ergonomic streaming with emit! macro
- `collect_or_else_pattern.rs` - Error recovery patterns

#### Real-World Usage
- HTTP response streaming patterns
- Log file processing with error recovery
- System event monitoring
- JSON parsing with graceful error handling
- Background task coordination

### Documentation
- Comprehensive README with builder pattern emphasis
- API documentation with examples
- Contributing guidelines
- Dual MIT/Apache-2.0 licensing
- Real-world usage statistics (987+ usage sites in fluent-ai)

### Dependencies
- `crossbeam` - Lock-free data structures
- `cyrup-sugars` - MessageChunk trait and error handling
- `log` - Logging support

### Compatibility
- Rust edition 2024
- No external async runtime dependencies
- Compatible with fluent-ai ecosystem
- Zero-allocation streaming architecture

## [0.1.0] - 2024-08-14

### Added
- Initial implementation of AsyncStream library
- Core streaming primitives and builder patterns
- Comprehensive example suite
- Documentation and contributing guidelines
- Dual MIT/Apache-2.0 licensing

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this changelog and the project.

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
