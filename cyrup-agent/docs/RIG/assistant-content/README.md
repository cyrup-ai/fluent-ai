# AssistantContent

| Property | Type | Example |
|----------|------|---------|
| `Text` | `String` | `AssistantContent::Text("Hello, how can I help you?")` |
| `ToolCall` | `[ToolCall](../tool-call/)` | `AssistantContent::ToolCall(ToolCall { id: "call_123", function: tool_fn })` |