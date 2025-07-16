# ToolResult

| Property | Type | Example |
|----------|------|---------|
| `id` | `String` | `"call_abc123"` |
| `content` | `[OneOrMany](../one-or-many/)<[ToolResultContent](../tool-result-content/)>` | `OneOrMany::one(ToolResultContent::Text(Text { text: "42" }))` |