# ToolDefinition

| Property | Type | Example |
|----------|------|---------|
| `name` | `String` | `"calculator"` |
| `description` | `String` | `"Performs basic arithmetic operations"` |
| `parameters` | `serde_json::Value` | `json!({"type": "object", "properties": {"operation": {"type": "string"}, "a": {"type": "number"}, "b": {"type": "number"}}, "required": ["operation", "a", "b"]})` |