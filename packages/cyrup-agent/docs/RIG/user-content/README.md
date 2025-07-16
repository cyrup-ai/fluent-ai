# UserContent

| Property | Type | Example |
|----------|------|---------|
| `Text` | `[Text](../text/)` | `UserContent::Text(Text { text: "Hello world" })` |
| `ToolResult` | `[ToolResult](../tool-result/)` | `UserContent::ToolResult(ToolResult { id: "call_123", content: result })` |
| `Image` | `[Image](../image/)` | `UserContent::Image(Image { data: "base64...", format: Some(Base64), .. })` |
| `Audio` | `[Audio](../audio/)` | `UserContent::Audio(Audio { data: "base64...", media_type: Some(MP3), .. })` |
| `Document` | `[Document](../document/)` | `UserContent::Document(Document { data: "base64...", media_type: Some(PDF), .. })` |