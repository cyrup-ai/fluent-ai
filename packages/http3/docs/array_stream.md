# Array Stream Architecture Specification

**Version**: 1.0  
**Status**: Design Phase  
**Target**: HTTP3 JSONPath Streaming Integration  
**RFC Compliance**: RFC 9535 JSONPath Query Expressions  

## Overview

The Array Stream architecture provides zero-allocation, blazing-fast streaming of individual JSON array elements from HTTP responses using JSONPath expressions. This enables elegant consumption of API responses like OpenAI's `{"data": [...]}` format without loading entire arrays into memory.

## High-Level Feature Goal

```rust
Http3::json()
    .array_stream("$.data[*]")  // JSONPath to stream array elements
    .get("https://api.openai.com/v1/models")
    .on_chunk(|model| {         // Each model streams individually
        Ok => model.into(),      // Clean syntax, no macro exposure
        Err(e) => BadChunk::from_err(e)
    })
```

## Architecture Principles

### 1. Streams-First, No Futures
- **Foundation**: Built on `fluent_ai_async::AsyncStream<T>` primitives
- **Constraint**: ALL asynchronous work uses unwrapped Stream, no Result-wrapped architecture
- **Performance**: Zero-allocation hot paths, lock-free concurrent processing

### 2. RFC 9535 Compliance
- **Specification**: Full JSONPath specification support
- **Selectors**: Name, wildcard, index, slice, filter expressions
- **Segments**: Child `[...]` and descendant `..[...]` navigation
- **Functions**: Built-in extensions (length, count, match, search, value)

### 3. Elegant Builder Pattern
- **Type Safety**: State-typed builders prevent invalid configurations
- **Fluent API**: Beautiful, ergonomic syntax without macro exposure
- **Error Handling**: Pattern matching syntax, no Result types in streams

## Core Components

### 1. Http3Builder Integration

```rust
pub struct Http3Builder<S> {
    client: HttpClient,
    request: HttpRequest,
    state: PhantomData<S>,
    jsonpath_config: Option<JsonPathStreaming>,
}

impl Http3Builder<JsonConfigured> {
    /// Transform HTTP response into streaming JSON array elements
    pub fn array_stream(self, jsonpath: &str) -> Http3Builder<JsonPathStreaming>
}
```

### 2. JsonArrayStream Processor

```rust
pub struct JsonArrayStream<T> {
    path_expression: JsonPathExpression,    // Compiled JSONPath
    buffer: StreamBuffer,                   // Zero-allocation buffer
    state: StreamStateMachine,              // Parsing state tracker
    _phantom: PhantomData<T>,               // Target type marker
}

impl<T: DeserializeOwned> JsonArrayStream<T> {
    /// Process HTTP chunks and yield individual array elements
    pub fn process_chunk(&mut self, chunk: Bytes) -> impl Iterator<Item = JsonPathResult<T>>
}
```

### 3. JSONPath Expression Parser

```rust
pub struct JsonPathExpression {
    selectors: Vec<JsonSelector>,           // Optimized selector chain
    original: String,                       // Original expression
    is_array_stream: bool,                  // Array streaming detection
}

pub enum JsonSelector {
    Root,                                   // $
    Child { name: String, exact_match: bool }, // .property, ['property']
    RecursiveDescent,                       // ..
    Index { index: i64, from_end: bool },   // [0], [-1]
    Slice { start: Option<i64>, end: Option<i64>, step: Option<i64> }, // [1:3:2]
    Wildcard,                              // *
    Filter { expression: FilterExpression }, // ?(@.price < 10)
    Union { selectors: Vec<JsonSelector> }, // [0,1,2]
}
```

### 4. Streaming State Machine

```rust
pub enum JsonStreamState {
    WaitingForRoot,                         // Awaiting JSON start
    Navigating { depth: usize, remaining_selectors: Vec<JsonSelector>, current_value: JsonValue },
    StreamingArray { target_depth: usize, current_index: usize, in_element: bool, element_depth: usize },
    ProcessingObject { depth: usize, brace_depth: usize, in_string: bool, escaped: bool },
    Complete,                               // Parsing finished
    Error { message: String, recoverable: bool },
}
```

## API Flow Architecture

### 1. Request Configuration
```rust
Http3::json()                           // Configure for JSON content
    .array_stream("$.data[*]")          // Set JSONPath expression
    .bearer_auth(&api_key)              // Add authentication
    .get("https://api.openai.com/v1/models") // Execute HTTP request
```

### 2. Stream Processing
```rust
.on_chunk(|model: Model| {              // Process each array element
    Ok => model.into(),                 // Success case - clean syntax
    Err(e) => BadChunk::from_err(e)     // Error case - explicit handling
})
```

### 3. Result Collection
- **Individual Processing**: Each array element streams independently
- **Type Safety**: Generic `T: DeserializeOwned` ensures compile-time safety
- **Error Recovery**: Malformed elements don't break entire stream

## Performance Characteristics

### Zero-Allocation Hot Paths
- **Buffer Management**: Pre-allocated, reusable stream buffers
- **JSONPath Compilation**: One-time compilation at builder construction
- **Deserialization**: Direct streaming without intermediate collections

### Blazing-Fast Execution
- **Lock-Free Architecture**: Concurrent processing without synchronization
- **Incremental Parsing**: Process JSON as bytes arrive
- **Optimized Selectors**: Compiled JSONPath expressions for runtime efficiency

### Memory Efficiency
- **Streaming Model**: No full JSON loading into memory
- **Bounded Buffers**: Configurable buffer sizes prevent memory bloat
- **Immediate Processing**: Array elements processed and released immediately

## Error Handling Strategy

### Graceful Degradation
```rust
pub enum JsonPathError {
    InvalidExpression { message: String, position: Option<usize> },
    JsonParseError { message: String, offset: usize, context: String },
    DeserializationError { message: String, json_fragment: String, target_type: &'static str },
    StreamError { message: String, state: String, recoverable: bool },
}
```

### Recovery Patterns
- **Element-Level**: Skip malformed array elements, continue streaming
- **Stream-Level**: Recover from transient network/parsing errors
- **Expression-Level**: Validate JSONPath expressions at compile time

## Integration Points

### 1. OpenAI API Compatibility
```json
{
  "object": "list",
  "data": [
    {"id": "gpt-4o", "object": "model", "created": 1714509474},
    {"id": "gpt-4", "object": "model", "created": 1678604602}
  ]
}
```

**JSONPath**: `$.data[*]` → Stream individual model objects

### 2. Generic API Support
- **Nested Arrays**: `$.results.items[*]`
- **Filtered Streaming**: `$.data[?(@.active == true)]`
- **Complex Navigation**: `$..products[?(@.price < 100)]`

## Testing Strategy

### RFC 9535 Compliance Matrix
- **Core Syntax**: Root identifier, segments, selectors *(100% coverage)*
- **Selector Types**: Name, wildcard, index, slice, filter *(100% coverage)*
- **Function Extensions**: length(), count(), match(), search(), value() *(100% coverage)*
- **Edge Cases**: Unicode, escaping, malformed input *(100% coverage)*

### Performance Benchmarks
- **Throughput**: Array elements processed per second
- **Memory Usage**: Peak memory consumption during streaming
- **Latency**: Time from HTTP chunk to deserialized object

### Integration Tests
- **OpenAI API**: Real API response streaming
- **Error Scenarios**: Network failures, malformed JSON
- **Concurrent Usage**: Multiple simultaneous streams

## Implementation Constraints

### Architectural Requirements
- **Zero Allocation**: Hot paths must not allocate
- **No Unsafe Code**: Memory safety without unsafe blocks
- **No Locking**: Lock-free concurrent architecture
- **Blazing Fast**: Optimized for maximum throughput
- **Never unwrap()**: Explicit error handling throughout

### Code Quality Standards
- **Elegant Ergonomics**: Beautiful, intuitive API design
- **Type Safety**: Compile-time guarantees via builder states
- **Documentation**: Comprehensive examples and error messages
- **Testing**: Exhaustive test coverage with failing-first TDD

## Success Criteria

### 1. Functional Requirements
- ✅ Stream individual JSON array elements from HTTP responses
- ✅ Support full RFC 9535 JSONPath specification
- ✅ Integrate seamlessly with Http3Builder pattern
- ✅ Provide elegant error handling without macro exposure

### 2. Performance Requirements
- 🎯 Zero-allocation processing in hot paths
- 🎯 Lock-free concurrent execution
- 🎯 Sub-millisecond latency per array element
- 🎯 Gigabyte-scale throughput capabilities

### 3. Quality Requirements
- 📋 100% RFC 9535 compliance test coverage
- 📋 Comprehensive error handling and recovery
- 📋 Beautiful, ergonomic API without implementation details
- 📋 Production-ready reliability and monitoring

---

**Next Steps**: Fix compilation errors → Implement missing methods → Verify test suite execution → Performance optimization