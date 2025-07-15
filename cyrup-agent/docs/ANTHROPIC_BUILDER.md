# Anthropic Provider Template Analysis

## Overview

The Anthropic provider serves as the **gold standard** for provider implementation in this codebase. This document extracts the key patterns and architectural decisions that should be applied to all other providers.

## File Structure Template

```
/providers/anthropic/
├── mod.rs                      # Exports, constants, documentation
├── client.rs                   # Typestate builder pattern
├── completion.rs               # Core CompletionModel implementation  
├── streaming.rs                # Streaming implementation
├── completions_builder_ext.rs  # Provider-specific extensions
└── decoders/                   # Provider-specific decoders
    ├── mod.rs
    ├── sse.rs                  # Server-sent events decoder
    ├── line.rs                 # Line decoder utility
    └── jsonl.rs                # JSON Lines decoder
```

## Key Architectural Patterns

### 1. Typestate Builder Pattern

**Location**: `client.rs`

**Pattern**: Uses typestate to enforce compile-time guarantees about builder state.

```rust
// Typestate markers
pub struct NeedsPrompt;
pub struct HasPrompt;

pub struct AnthropicCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // ... fields
    _state: std::marker::PhantomData<S>,
}
```

**Key Features**:
- Builder starts in `NeedsPrompt` state
- Calling `.prompt()` transitions to `HasPrompt` state  
- Only `HasPrompt` builders can call `.send()` or `.stream()`
- Compile-time prevention of incomplete requests

### 2. Synchronous Public API with AsyncTask Returns

**Pattern**: All public methods are synchronous and return `AsyncTask<Result<...>>`

```rust
// ❌ NEVER in public API
async fn completion(&self, req: CompletionRequest) -> Result<...> 

// ✅ ALWAYS in public API  
fn completion(&self, req: CompletionRequest) -> AsyncTask<Result<...>> {
    rt::spawn_async(async move { 
        self.clone().perform_completion(req).await 
    })
}
```

**Benefits**:
- Hides async complexity from users
- No need for `Box<dyn Future>` or `Pin<Box<dyn Future>>`
- Clean, ergonomic API surface
- Consistent with "async FTW w/ Hidden Box/Pin" convention

### 3. Default Configuration Factory

**Pattern**: Provides sensible defaults via `default_for_chat()`

```rust
impl<'a> AnthropicCompletionBuilder<'a, NeedsPrompt> {
    pub fn default_for_chat(client: &'a Client) -> AnthropicCompletionBuilder<'a, HasPrompt> {
        Self::new(client, CLAUDE_4_SONNET)
            .temperature(0.0)
            .extended_thinking(true)
            .cache_control(CacheControl::MaxAgeSecs(86_400))
            .prompt_enhancement("prompt-tools-generate")
            .prompt(Message::user("")) // dummy; replaced in .chat(..)
    }
}
```

**Features**:
- Pre-configured with sensible defaults
- Returns builder in `HasPrompt` state (ready to use)
- Model-specific optimizations applied

### 4. Streaming Implementation Pattern

**Location**: `streaming.rs`

**Pattern**: Public sync facade returning AsyncTask, internal async implementation

```rust
impl CompletionModel {
    // Public sync facade
    pub fn stream(&self, req: CompletionRequest) 
        -> AsyncTask<Result<RigStreaming<StreamingCompletionResponse>, CompletionError>> 
    {
        rt::spawn_async(self.clone().drive_stream(req))
    }

    // Internal async driver (NOT public)
    async fn drive_stream(self, req: CompletionRequest) 
        -> Result<RigStreaming<StreamingCompletionResponse>, CompletionError> 
    {
        // ... implementation
    }
}
```

**Key Elements**:
- Uses `async_stream!` macro for stream generation
- Proper SSE decoding with custom decoder
- Error handling with graceful degradation
- Token usage tracking and reporting

### 5. Error Handling Pattern

**Pattern**: Provider-specific errors with consistent structure

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiResponse<T> {
    Message(T),
    Error(ApiErrorResponse),
}

// Consistent error conversion
match serde_json::from_str::<ApiResponse<CompletionResponse>>(&body)? {
    ApiResponse::Message(msg) => msg.try_into(),
    ApiResponse::Error(err) => Err(CompletionError::ProviderError(err.message)),
}
```

### 6. JSON Request Building Pattern

**Pattern**: In-place JSON merging to avoid allocations

```rust
let mut payload = json!({
    "model": self.model,
    "messages": history,
    "max_tokens": max_tokens,
    "system": completion_request.preamble.unwrap_or_default(),
});

// Conditional field addition via in-place merge
if let Some(t) = completion_request.temperature {
    json_utils::merge_inplace(&mut payload, json!({ "temperature": t }));
}

if !completion_request.tools.is_empty() {
    json_utils::merge_inplace(&mut payload, json!({
        "tools": completion_request.tools.into_iter().map(|t| ToolDefinition {
            name: t.name,
            description: Some(t.description),
            input_schema: t.parameters,
        }).collect::<Vec<_>>(),
        "tool_choice": ToolChoice::Auto,
    }));
}
```

## Implementation Checklist

When implementing any provider, ensure:

### File Structure
- [ ] Directory structure with separation of concerns
- [ ] `mod.rs` with clean exports and constants
- [ ] `client.rs` with typestate builder
- [ ] `completion.rs` with AsyncTask implementation
- [ ] `streaming.rs` with async facade pattern

### API Consistency
- [ ] No `async fn` in public trait implementations
- [ ] All public methods return `AsyncTask<Result<...>>`
- [ ] Typestate builder pattern implemented
- [ ] `default_for_chat()` factory method
- [ ] Proper error handling with provider-specific types

### Code Quality
- [ ] All warnings resolved (no suppressions)
- [ ] Proper tracing/logging integration  
- [ ] Comprehensive error handling
- [ ] Memory-efficient JSON building
- [ ] Clean separation of sync/async boundaries

### Testing
- [ ] Basic completion test
- [ ] Streaming test (if supported)
- [ ] Builder pattern functionality test
- [ ] Default configuration test
- [ ] Error handling test

## Reusable Patterns

### Typestate Builder Template

```rust
pub struct NeedsPrompt;
pub struct HasPrompt;

pub struct ProviderCompletionBuilder<'a, S> {
    client: &'a Client,
    model_name: &'a str,
    // ... provider-specific fields
    prompt: Option<Message>,
    _state: std::marker::PhantomData<S>,
}

impl<'a> ProviderCompletionBuilder<'a, NeedsPrompt> {
    pub fn new(client: &'a Client, model_name: &'a str) -> Self { /*...*/ }
    
    pub fn default_for_chat(client: &'a Client) -> ProviderCompletionBuilder<'a, HasPrompt> {
        // Provider-specific defaults
    }
    
    pub fn prompt(self, p: impl Into<Message>) -> ProviderCompletionBuilder<'a, HasPrompt> {
        // State transition
    }
}

impl<'a, S> ProviderCompletionBuilder<'a, S> {
    // Fluent setters that preserve state
    pub fn temperature(mut self, t: f64) -> Self { /*...*/ }
    pub fn max_tokens(mut self, n: u64) -> Self { /*...*/ }
}

impl<'a> ProviderCompletionBuilder<'a, HasPrompt> {
    pub fn send(self) -> AsyncTask<Result<CompletionResponse, CompletionError>> {
        rt::spawn_async(async move { /*...*/ })
    }
    
    pub fn stream(self) -> AsyncTask<Result<StreamingResponse, CompletionError>> {
        rt::spawn_async(async move { /*...*/ })
    }
}
```

### AsyncTask Pattern Template

```rust
impl completion::CompletionModel for CompletionModel {
    type Response = ProviderCompletionResponse;
    type StreamingResponse = ProviderStreamingResponse;

    fn completion(&self, req: CompletionRequest) 
        -> AsyncTask<Result<completion::CompletionResponse<Self::Response>, CompletionError>> 
    {
        let this = self.clone();
        rt::spawn_async(async move { this.perform_completion(req).await })
    }

    fn stream(&self, req: CompletionRequest) 
        -> AsyncTask<Result<RigStreaming<Self::StreamingResponse>, CompletionError>> 
    {
        let this = self.clone();
        rt::spawn_async(async move { this.perform_stream(req).await })
    }
}
```

## Migration Priority

1. **High Priority**: Large single-file providers (Azure, Groq)
2. **Medium Priority**: Directory-based providers needing pattern updates
3. **Low Priority**: Small single-file providers

Each migration should follow this template exactly to ensure consistency across the entire provider ecosystem.