# CompletionRequestBuilder

| Property | Type | Example |
|----------|------|---------|
| `model` | `M: CompletionModel` | `openai_client.completion_model("gpt-4")` |
| `prompt` | `[Message](../message/)` | `Message::user("What is the weather like?")` |
| `preamble` | `String` | `"You are a helpful weather assistant"` |
| `chat_history` | `Vec<[Message](../message/)>` | `vec![Message::user("Hello"), Message::assistant("Hi there!")]` |
| `documents` | `Vec<[Document](../document/)>` | `vec![Document { id: "weather_doc", text: "Weather data for today", additional_props: HashMap::new() }]` |
| `tools` | `Vec<[ToolDefinition](../tool-definition/)>` | `vec![ToolDefinition { name: "get_weather", description: "Get current weather", parameters: json!({"type": "object"}) }]` |
| `temperature` | `Option<f64>` | `Some(0.7)` |
| `max_tokens` | `Option<u64>` | `Some(1000)` |
| `additional_params` | `Option<serde_json::Value>` | `Some(json!({"top_p": 0.9, "presence_penalty": 0.0}))` |