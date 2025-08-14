# Contributing to AsyncStream

We welcome contributions to the AsyncStream library! This document provides guidelines for contributing to the project.

## 🏗️ Development Setup

### Prerequisites

- Rust nightly (edition 2024)
- `cargo-nextest` for testing
- `cargo-edit` for dependency management

### Getting Started

```bash
# Clone the repository
git clone https://github.com/fluent-ai/async-stream
cd async-stream

# Install dependencies
cargo build

# Run tests
cargo nextest run

# Format and check code
cargo fmt && cargo check --message-format short --quiet
```

## 📋 Code Standards

### Rust Style Guidelines

- **Edition**: Always use Rust edition 2024
- **Formatting**: Use `cargo fmt` for consistent formatting
- **Linting**: All code must pass `cargo check --message-format short --quiet` without warnings
- **Testing**: Use `nextest` for all test execution
- **Line Limit**: No single file should exceed 300 lines - decompose into modules

### Architecture Requirements

#### ❌ Prohibited Patterns
- **NEVER use `async_trait`** or `#[async_trait]` annotations
- **NO `Box<dyn Future>`** or dynamic async dispatch
- **NO external async runtimes** (tokio, async-std, etc.)
- **NO `Result<T, E>` inside streams** - use `on_chunk` patterns instead
- **NO suppression of compiler warnings** with `#[allow()]` or underscore prefixes

#### ✅ Required Patterns
- **Use `fluent_ai_async` ecosystem** exclusively for async operations
- **Zero-allocation streaming** with const-generic capacities
- **Crossbeam primitives** for concurrency
- **Native async/await** without trait objects
- **MessageChunk trait** for proper error handling

### Error Handling

```rust
// ❌ Wrong - Result in stream
AsyncStream<Result<Data, Error>>

// ✅ Correct - MessageChunk with error handling
AsyncStream<Data> // where Data: MessageChunk
```

### Testing

- Tests go in `tests/` directory, not co-located with source
- Use `nextest` for all test execution
- Test async code properly without blocking patterns
- Focus on key elements that prove real-world functionality

## 🔧 Development Workflow

### Before Starting Work

1. **Always run cleanup first**:
   ```bash
   cargo fmt && cargo check --message-format short --quiet
   ```

2. **Fix all warnings and errors** before proceeding
3. **Get documentation** if working with new libraries:
   - Check `./docs/` for existing documentation
   - Use MCP tools (`context7`, `github-mcp-server`) for additional docs
   - Save new documentation in `docs/` with semantic tags

### Making Changes

1. **Write minimal code** needed to implement the feature
2. **Do NOT add features** that aren't requested
3. **Focus on interface first** - make it drop-in easy for users
4. **Test like an end-user** - run `cargo run` for binaries
5. **Ask questions** before making assumptions

### Code Review Checklist

- [ ] Passes `cargo fmt && cargo check --message-format short --quiet`
- [ ] No warnings or errors
- [ ] Uses only fluent-ai async ecosystem
- [ ] No `async_trait` usage
- [ ] Proper `MessageChunk` implementations
- [ ] Tests in `tests/` directory using `nextest`
- [ ] Documentation updated if needed
- [ ] Examples work with `cargo run --example`

## 📚 Documentation

### Code Documentation

- Use clear, concise doc comments
- Include examples in doc comments where helpful
- Document public APIs thoroughly
- Explain architectural decisions in module-level docs

### Examples

- All examples must be runnable with `cargo run --example <name>`
- Examples should demonstrate real-world usage patterns
- Include proper error handling with `MessageChunk`
- Show the builder pattern as the primary approach

## 🧪 Testing

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fluent_ai_async::prelude::*;

    #[test]
    fn test_feature() {
        // Test implementation
    }
}
```

### Running Tests

```bash
# Run all tests
cargo nextest run

# Run specific test
cargo nextest run test_name

# Run with output
cargo nextest run -- --nocapture
```

## 🚀 Submitting Changes

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the guidelines above
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all checks pass**:
   ```bash
   cargo fmt && cargo check --message-format short --quiet
   cargo nextest run
   ```
6. **Submit a pull request** with a clear description

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add builder pattern for custom error handling
fix: resolve synchronization issue in with_channel pattern
docs: update README with builder pattern examples
test: add comprehensive tests for spawn_stream functionality
```

### Pull Request Description

Include:
- **What** you changed and **why**
- **How** to test the changes
- **Any breaking changes** or migration notes
- **Related issues** or discussions

## 🏛️ Architecture Principles

### Streams-Only Architecture

The AsyncStream library enforces a streams-only architecture:

- **All async operations** return `AsyncStream<T>` of unwrapped values
- **Error handling** via `on_chunk` patterns, not `Result<T, E>` in streams
- **Zero-allocation** streaming with crossbeam primitives
- **No external dependencies** on async runtimes

### Performance First

- **Lock-free concurrency** with crossbeam
- **Const-generic capacities** for compile-time optimization
- **Minimal allocations** with pre-sized buffers
- **Zero-copy patterns** where possible

## 🤝 Community

### Getting Help

- **Documentation**: Check the [API docs](https://docs.rs/fluent_ai_async)
- **Examples**: Review the [examples](examples/) directory
- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## 📄 License

By contributing to AsyncStream, you agree that your contributions will be licensed under both the MIT and Apache-2.0 licenses, as described in the main README.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual licensed as above, without any additional terms or conditions.
