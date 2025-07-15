# StreamingCompletionResponse

| Property | Type | Example |
|----------|------|---------|
| `inner` | `StreamingResult<R>` | `StreamingResult::new(response_stream)` |
| `text` | `String` | `"Hello, how can I help you today?"` |
| `tool_calls` | `Vec<[ToolCall](../tool-call/)>` | `vec![ToolCall { id: "call_123", function: ToolFunction { name: "calculator", arguments: json!({"op": "add"}) } }]` |
| `choice` | `[OneOrMany](../one-or-many/)<[AssistantContent](../assistant-content/)>` | `OneOrMany::one(AssistantContent::Text("Accumulated response text"))` |
| `response` | `Option<R>` | `Some(OpenAIStreamingResponse { id: "chatcmpl-123", model: "gpt-4", ... })` |