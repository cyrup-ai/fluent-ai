# CompletionResponse

| Property | Type | Example |
|----------|------|---------|
| `choice` | `[OneOrMany](../one-or-many/)<[AssistantContent](../assistant-content/)>` | `OneOrMany::one(AssistantContent::Text("The answer is 42"))` |
| `raw_response` | `T` | `OpenAIResponse { id: "chatcmpl-123", object: "chat.completion", ... }` |