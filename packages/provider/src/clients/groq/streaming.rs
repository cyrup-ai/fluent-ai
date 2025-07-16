// ============================================================================
// File: src/providers/groq/streaming.rs
// ----------------------------------------------------------------------------
// Groq streaming implementation (OpenAI-compatible SSE format)
// ============================================================================

use futures::StreamExt;
use serde_json::Value;

use crate::{
    completion::CompletionError,
    http::{HttpClient, HttpRequest},
    clients::openai::{CompletionResponse, StreamingChoice, StreamingMessage},
    runtime::{self, AsyncStream},
};

// Re-export OpenAI streaming response type since Groq uses the same format
pub use crate::clients::openai::StreamingCompletionResponse;

/// Send a streaming request to Groq and return an AsyncStream
pub fn send_groq_streaming_request(
    client: HttpClient,
    request: HttpRequest,
) -> crate::runtime::AsyncTask<
    Result<crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>,
> {
    crate::runtime::spawn_async(async move {
        // Create the AsyncStream channel
        let (tx, stream) =
            runtime::async_stream::<Result<StreamingCompletionResponse, CompletionError>>(512);

        // Spawn the streaming task
        runtime::spawn_async(async move {
            let response = match client.send(request).await {
                Ok(response) => response,
                Err(e) => {
                    let _ = tx.try_send(Err(CompletionError::RequestError(e.to_string())));
                    return;
                }
            };

            if !response.status().is_success() {
                let error_body = String::from_utf8_lossy(response.body());
                let _ = tx.try_send(Err(CompletionError::ProviderError(error_body.to_string())));
                return;
            }

            // Get SSE stream from HTTP3 response
            let mut sse_stream = response.sse();
            tracing::debug!(target: "rig", "Groq streaming connection opened");

            while let Some(event) = sse_stream.next().await {
                match event {
                    Ok(sse_event) => {
                        if let Some(data) = sse_event.data {
                            if data == "[DONE]" {
                                break;
                            }

                            match serde_json::from_str::<StreamingCompletionResponse>(&data) {
                                Ok(response) => {
                                    if tx.try_send(Ok(response)).is_err() {
                                        tracing::warn!(target: "rig", "Groq streaming receiver dropped");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    tracing::error!(target: "rig", "Failed to parse Groq streaming response: {}", e);
                                    let _ = tx.try_send(Err(CompletionError::DeserializationError(
                                        e.to_string(),
                                    )));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(target: "rig", "Groq streaming error: {}", e);
                        let _ = tx.try_send(Err(CompletionError::RequestError(e.to_string())));
                        break;
                    }
                }
            }
        });

        Ok(crate::streaming::StreamingCompletionResponse::new(Box::pin(stream)))
    })
}
