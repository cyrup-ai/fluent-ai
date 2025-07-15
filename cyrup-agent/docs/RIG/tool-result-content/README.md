# ToolResultContent

| Property | Type | Example |
|----------|------|---------|
| `Text` | `[Text](../text/)` | `ToolResultContent::Text(Text { text: "The calculation result is 42" })` |
| `Image` | `[Image](../image/)` | `ToolResultContent::Image(Image { data: "iVBORw0KGgoAAAANSU...", format: Some(ContentFormat::Base64), media_type: Some(ImageMediaType::PNG), detail: Some(ImageDetail::High) })` |