# AsyncStream - Zero-Allocation Streaming for fluent-ai

[![Crates.io](https://img.shields.io/crates/v/fluent_ai_async.svg)](https://crates.io/crates/fluent_ai_async)
[![Documentation](https://docs.rs/fluent_ai_async/badge.svg)](https://docs.rs/fluent_ai_async)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Zero-allocation streaming primitives for the fluent-ai ecosystem. Provides core streaming types and utilities that enforce the streams-only architecture with crossbeam-based concurrency.

## 🏗️ Builder Pattern (Primary Usage)

The **AsyncStreamBuilder** is the primary interface for creating streams with custom error handling and chunk processing. This is the recommended approach for all production use cases.

### Key Builder Methods

```rust
use fluent_ai_async::prelude::*;

// 1. Basic builder with default error handling
let stream = AsyncStream::<MyType>::builder()
    .with_channel(|sender| {
        emit!(sender, my_data);
    });

// 2. Builder with custom error handling
let stream = AsyncStream::<MyType>::builder()
    .on_chunk(|result| match result {
        Ok(chunk) => chunk,
        Err(e) => MyType::bad_chunk(e),
    })
    .with_channel(|sender| {
        emit!(sender, my_data);
    });

// 3. Builder with custom capacity
let stream = AsyncStream::<MyType, 2048>::builder()
    .with_channel(|sender| {
        emit!(sender, my_data);
    });
```

### Builder Architecture Benefits

- **Zero-allocation streaming** with const-generic capacity
- **Custom error handling** via `on_chunk` patterns
- **Type-safe chunk processing** with `MessageChunk` trait
- **No async_trait or boxed futures** - pure native async
- **Crossbeam-based concurrency** for maximum performance

## 📋 Pattern Examples

### 1. Builder Pattern
**Primary usage pattern with custom error handling**
```bash
cargo run --example on_chunk_pattern
```
[View Example](examples/on_chunk_pattern.rs)

### 2. Task Spawning Patterns
**Background computation with `spawn_task` and `spawn_stream`**
```bash
cargo run --example spawn_task_pattern
```
[View Example](examples/spawn_task_pattern.rs)

### 3. Channel Pattern
**Producer/consumer setup with `with_channel`**
```bash
cargo run --example with_channel_pattern
```
[View Example](examples/with_channel_pattern.rs)

### 4. Emit Macro Pattern
**Ergonomic streaming with the `emit!` macro**
```bash
cargo run --example emit_macro_pattern
```
[View Example](examples/emit_macro_pattern.rs)

### 5. Error Recovery Pattern
**Graceful error handling with `collect_or_else`**
```bash
cargo run --example collect_or_else_pattern
```
[View Example](examples/collect_or_else_pattern.rs)

## 🚀 Quick Start

Add to your `Cargo.toml`:
```toml
[dependencies]
fluent_ai_async = "0.1.0"
```

### Basic Usage

```rust
use fluent_ai_async::prelude::*;

#[derive(Debug, Clone, Default)]
struct MyData {
    value: String,
    processed: bool,
}

impl MessageChunk for MyData {
    fn bad_chunk(error: String) -> Self {
        Self {
            value: format!("ERROR: {}", error),
            processed: false,
        }
    }

    fn is_error(&self) -> bool {
        !self.processed
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some(&self.value)
        } else {
            None
        }
    }
}

fn main() {
    // Create stream with builder pattern
    let stream = AsyncStream::<MyData>::builder()
        .on_chunk(|result| match result {
            Ok(data) => data,
            Err(e) => MyData::bad_chunk(e),
        })
        .with_channel(|sender| {
            let data = MyData {
                value: "Hello, World!".to_string(),
                processed: true,
            };
            emit!(sender, data);
        });

    // Consume the stream
    let results: Vec<MyData> = stream.collect();
    println!("Received {} items", results.len());
}
```

## 🏛️ Architecture Principles

### Streams-Only Architecture
- **No `Result<T, E>` in streams** - errors handled via `on_chunk` patterns
- **Zero-allocation streaming** with const-generic capacities
- **Crossbeam primitives** for lock-free concurrency
- **No external async runtimes** - pure fluent-ai ecosystem

### Type Safety
- **`MessageChunk` trait** ensures proper error handling
- **Const-generic capacities** for compile-time optimization
- **No `async_trait`** - native async/await only

### Performance
- **Lock-free queues** with `crossbeam::ArrayQueue`
- **Zero-copy streaming** where possible
- **Minimal allocations** with pre-sized buffers

## 📚 Core Types

### `AsyncStream<T, CAP>`
The primary streaming type with const-generic capacity.

### `AsyncStreamBuilder<T, CAP>`
Builder for creating streams with custom error handling.

### `AsyncTask<T>`
Single-value async computation result.

### `AsyncStreamSender<T, CAP>`
Send-side of a stream channel.

## 🔧 Key Functions

### Stream Creation
- `AsyncStream::builder()` - Create builder (recommended)
- `AsyncStream::with_channel()` - Direct channel creation
- `spawn_stream()` - Spawn streaming background task
- `spawn_task()` - Spawn single-value background task

### Consumption
- `.collect()` - Collect all values (blocking)
- `.into_iter()` - Convert to iterator
- `.try_next()` - Non-blocking pop
- `.next().await` - Async pop (requires async context)

## 🎯 Real-World Usage

This library is used extensively throughout the fluent-ai ecosystem:

- **Provider clients** (OpenAI, Anthropic, Mistral) - 200+ usage sites
- **Memory SDK** - Cognitive memory operations - 150+ usage sites
- **Workflow engine** - Step chaining and parallel execution - 300+ usage sites
- **Embedding providers** - Local and remote inference - 100+ usage sites
- **HTTP3 client** - Streaming response processing - 200+ usage sites

**Total: 987+ real-world usage sites across the fluent-ai codebase**

## 🔍 Advanced Patterns

### Chained Processing
```rust
let processed_stream = AsyncStream::<ProcessedData>::builder()
    .with_channel(move |sender| {
        for item in input_stream.into_iter() {
            let processed = process_item(item);
            emit!(sender, processed);
        }
    });
```

### Error Recovery
```rust
let safe_stream = AsyncStream::<MyData>::builder()
    .on_chunk(|result| match result {
        Ok(data) => data,
        Err(e) => {
            log::warn!("Processing error: {}", e);
            MyData::bad_chunk(e)
        }
    })
    .with_channel(|sender| {
        // Producer logic that might fail
    });
```

### Custom Capacities
```rust
// High-throughput streaming
let stream = AsyncStream::<Data, 4096>::builder()
    .with_channel(|sender| {
        // High-volume producer
    });
```

## 📖 Documentation

- [API Documentation](https://docs.rs/fluent_ai_async)
- [Examples](examples/)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## 📄 License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
