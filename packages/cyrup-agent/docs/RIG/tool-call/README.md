# ToolCall

| Property | Type | Example |
|----------|------|---------|
| `id` | `String` | `"call_abc123"` |
| `function` | `[ToolFunction](../tool-function/)` | `ToolFunction { name: "calculator", arguments: json!({"operation": "add", "a": 5, "b": 3}) }` |