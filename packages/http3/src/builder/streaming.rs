//! JSONPath streaming functionality for HTTP responses
//!
//! Pure streams-first architecture - NO Futures, NO Result wrapping
//! Transforms HTTP byte streams into individual JSON objects via JSONPath

use fluent_ai_async::AsyncStream;
use serde::de::DeserializeOwned;

use crate::{HttpChunk, HttpStream};

/// Pure AsyncStream of JSON objects via JSONPath - NO Result wrapping
pub struct JsonPathStream<T> {
    inner: AsyncStream<T>,
}

impl<T> JsonPathStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Create JSONPath stream from HTTP response - pure streams architecture
    pub fn new(http_stream: HttpStream, jsonpath_expr: String) -> Self {
        Self {
            inner: AsyncStream::with_channel(move |sender| {
                std::thread::spawn(move || {
                    let mut buffer = Vec::new();
                    let mut stream = http_stream;

                    // Collect all HTTP chunks
                    while let Some(chunk) = stream.poll_next() {
                        match chunk {
                            HttpChunk::Body(bytes) => {
                                buffer.extend_from_slice(&bytes);
                            }
                            HttpChunk::Error(_) => break,
                            _ => continue,
                        }
                    }

                    // Simple JSONPath processing - deserialize full response first
                    if !buffer.is_empty() {
                        if let Ok(json_value) = serde_json::from_slice::<serde_json::Value>(&buffer)
                        {
                            // Basic JSONPath evaluation - support simple array patterns
                            let objects =
                                JsonPathStream::<T>::extract_objects(&json_value, &jsonpath_expr);

                            for obj_value in objects {
                                if let Ok(typed_obj) = serde_json::from_value::<T>(obj_value) {
                                    if sender.send(typed_obj).is_err() {
                                        break; // Stream closed
                                    }
                                }
                            }
                        }
                    }
                });
            }),
        }
    }

    /// Get next item from stream
    pub fn poll_next(&mut self) -> Option<T> {
        self.inner.poll_next()
    }

    /// Extract objects matching JSONPath expression - simplified implementation
    fn extract_objects(
        json_value: &serde_json::Value,
        jsonpath_expr: &str,
    ) -> Vec<serde_json::Value> {
        // Simple JSONPath support for array streaming patterns
        match jsonpath_expr {
            "$" => vec![json_value.clone()],
            expr if expr.ends_with("[*]") => {
                // Extract array field name from expressions like "$.items[*]" or "$.users[*]"
                let field_path = expr
                    .strip_prefix("$.")
                    .unwrap_or(expr)
                    .strip_suffix("[*]")
                    .unwrap_or(expr);

                if field_path.is_empty() {
                    // Root array: $[*]
                    if let serde_json::Value::Array(arr) = json_value {
                        arr.clone()
                    } else {
                        vec![]
                    }
                } else {
                    // Nested field array: $.field[*]
                    if let Some(serde_json::Value::Array(arr)) = json_value.get(field_path) {
                        arr.clone()
                    } else {
                        vec![]
                    }
                }
            }
            _ => {
                // Fallback: return the whole value for other expressions
                vec![json_value.clone()]
            }
        }
    }

    /// Process each object - pure streaming pattern
    pub fn on_chunk<F, U>(mut self, mut handler: F) -> AsyncStream<U>
    where
        F: FnMut(T) -> U + Send + 'static,
        U: Send + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                while let Some(item) = self.poll_next() {
                    let processed = handler(item);
                    if sender.send(processed).is_err() {
                        break;
                    }
                }
            });
        })
    }
}
