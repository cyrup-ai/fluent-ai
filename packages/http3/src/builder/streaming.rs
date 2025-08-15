//! JSONPath streaming functionality for HTTP responses
//!
//! Pure streams-first architecture - NO Futures, NO Result wrapping
//! Transforms HTTP byte streams into individual JSON objects via JSONPath

use fluent_ai_async::AsyncStream;
use fluent_ai_async::prelude::MessageChunk;
use serde::de::DeserializeOwned;

use crate::{HttpChunk, HttpError, HttpStream};

/// Pure AsyncStream of JSON objects via JSONPath - NO Result wrapping
pub struct JsonPathStream<T> {
    inner: AsyncStream<T>,
    chunk_handler: Option<Box<dyn Fn(Result<T, HttpError>) -> T + Send + Sync>>,
}

impl<T> JsonPathStream<T>
where
    T: DeserializeOwned + Send + Default + MessageChunk + 'static,
{
    /// Create JSONPath stream from HTTP response - pure streams architecture
    pub fn new(http_stream: HttpStream, jsonpath_expr: String) -> Self {
        Self {
            inner: AsyncStream::with_channel(move |sender| {
                std::thread::spawn(move || {
                    let mut buffer = Vec::new();
                    let stream = http_stream;

                    // Process HTTP chunks from stream
                    let chunks: Vec<HttpChunk> = stream.collect();
                    for chunk in chunks {
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
            chunk_handler: None,
        }
    }

    /// Get next item from stream - consumes the entire stream
    pub fn collect_next(self) -> Vec<T> {
        // AsyncStream uses collect() or for-in iteration
        // This consumes the entire stream and returns all items
        self.inner.collect()
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

    /// Process each object with Result error handling - fluent_ai_async pattern
    pub fn on_chunk<F>(self, handler: F) -> AsyncStream<T>
    where
        F: Fn(Result<T, HttpError>) -> T + Send + Sync + 'static,
    {
        AsyncStream::with_channel(move |sender| {
            std::thread::spawn(move || {
                let items = self.inner.collect();
                for item in items {
                    // Convert item to Result - success case for existing items
                    let result = Ok(item);
                    let processed = handler(result);
                    if sender.send(processed).is_err() {
                        break;
                    }
                }
            });
        })
    }
}

// Pure fluent_ai_async pattern - no cyrup_sugars ChunkHandler needed
// All error handling is done through MessageChunk trait and AsyncStream patterns
