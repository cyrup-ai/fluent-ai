# CompletionError

| Property | Type | Example |
|----------|------|---------|
| `HttpError` | `reqwest::Error` | `CompletionError::HttpError(reqwest::Error::from(std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout")))` |
| `JsonError` | `serde_json::Error` | `CompletionError::JsonError(serde_json::Error::syntax(serde_json::error::ErrorCode::TrailingComma, 1, 2))` |
| `RequestError` | `Box<dyn std::error::Error + Send + Sync>` | `CompletionError::RequestError(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid request")))` |
| `ResponseError` | `String` | `CompletionError::ResponseError("Invalid response format".to_string())` |
| `ProviderError` | `String` | `CompletionError::ProviderError("OpenAI API key invalid".to_string())` |