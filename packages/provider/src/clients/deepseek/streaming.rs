// ============================================================================
// File: src/providers/deepseek/streaming.rs
// ----------------------------------------------------------------------------
// DeepSeek streaming implementation using HTTP3 client (OpenAI-compatible SSE format)
// ============================================================================

use fluent_ai_domain::completion::CompletionCoreError;
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest as Http3Request};
use futures_util::StreamExt;
// Note: runtime module doesn't exist - using tokio equivalents
use tokio::{self as runtime, task::JoinHandle as AsyncTask};
use fluent_ai_http3::HttpClient;
use fluent_ai_http3::HttpRequest;

// Re-export OpenAI streaming response type since DeepSeek uses the same format
pub use crate::clients::openai::StreamingCompletionResponse;

/// Send a streaming request to DeepSeek using HTTP3 client and return an AsyncStream
pub fn send_deepseek_streaming_request(
    http3_request: Http3Request,
) -> AsyncTask<
    Result<
        StreamingCompletionResponse, // Simplified - no nested streaming wrapper
        CompletionCoreError,
    >,
> {
    crate::runtime::spawn_async(async move {
        // Create the AsyncStream channel
        let (tx, stream) =
            runtime::async_stream::<Result<StreamingCompletionResponse, CompletionError>>(512);

        // Create HTTP3 client with streaming configuration
        let client = HttpClient::with_config(HttpConfig::streaming_optimized()).map_err(|e| {
            CompletionError::RequestError(format!("Failed to create HTTP3 client: {}", e))
        })?;

        // Spawn the streaming task
        runtime::spawn_async(async move {
            // Send streaming request
            let stream_response = match client.send_stream(http3_request).await {
                Ok(stream) => stream,
                Err(e) => {
                    let _ = tx.try_send(Err(CompletionError::RequestError(e.to_string())));
                    return Ok::<(), CompletionError>(());
                }
            };

            // Process SSE events
            let mut sse_stream = stream_response.sse();

            while let Some(event) = sse_stream.next().await {
                match event {
                    Ok(event) => {
                        if event.is_done() {
                            tracing::debug!(target: "rig", "DeepSeek streaming completed");
                            break;
                        }

                        let data = event.data_string();
                        if data.trim().is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<StreamingCompletionResponse>(&data) {
                            Ok(response) => {
                                if tx.try_send(Ok(response)).is_err() {
                                    tracing::warn!(target: "rig", "DeepSeek streaming receiver dropped");
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::error!(target: "rig", "Failed to parse DeepSeek streaming response: {}", e);
                                let _ = tx.try_send(Err(CompletionError::DeserializationError(
                                    e.to_string(),
                                )));
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(target: "rig", "DeepSeek streaming error: {}", e);
                        let _ = tx.try_send(Err(CompletionError::RequestError(e.to_string())));
                        break;
                    }
                }
            }

            Ok::<(), CompletionError>(())
        });

        Ok(crate::streaming::StreamingCompletionResponse::new(
            Box::pin(stream),
        ))
    })
}
