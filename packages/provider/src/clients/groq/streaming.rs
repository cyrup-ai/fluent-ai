// ============================================================================
// File: src/providers/groq/streaming.rs
// ----------------------------------------------------------------------------
// Groq streaming implementation (OpenAI-compatible SSE format)
// ============================================================================

use fluent_ai_domain::AsyncTask;
use futures_util::Stream;
use futures_util::StreamExt;
use serde_json::Value;
use tokio_stream;
use fluent_ai_http3::HttpClient;
use fluent_ai_http3::HttpRequest;

// Re-export OpenAI streaming response type since Groq uses the same format
pub use crate::clients::openai::StreamingCompletionResponse;
use crate::{
    clients::openai::{CompletionResponse, StreamingChoice, StreamingMessage},
    completion::CompletionError,
    http::{HttpClient, HttpRequest}};

/// Send a streaming request to Groq and return an AsyncStream
pub fn send_groq_streaming_request(
    client: HttpClient,
    request: HttpRequest,
) -> AsyncTask<
    Result<
        crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
        CompletionError,
    >,
> {
    let (task_tx, rx) = tokio::sync::oneshot::channel();
    let result_task = tokio::spawn(async move {
        // Create the AsyncStream channel using tokio mpsc
        let (stream_tx, stream_rx) =
            tokio::sync::mpsc::channel::<Result<StreamingCompletionResponse, CompletionError>>(512);

        // Spawn the streaming task
        tokio::spawn(async move {
            let response = match client.send(request).await {
                Ok(response) => response,
                Err(e) => {
                    let _ = stream_tx
                        .send(Err(CompletionError::InvalidRequest(e.to_string())))
                        .await;
                    return;
                }
            };

            if !response.status().is_success() {
                let error_body = String::from_utf8_lossy(response.body());
                let _ = stream_tx.send(Err(CompletionError::ProviderUnavailable(
                    error_body.to_string(),
                )));
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
                                    let _ =
                                        tx.try_send(Err(CompletionError::Internal(e.to_string())));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(target: "rig", "Groq streaming error: {}", e);
                        let _ = stream_tx
                            .send(Err(CompletionError::InvalidRequest(e.to_string())))
                            .await;
                        break;
                    }
                }
            }
        });

        // Convert the receiver into a stream
        let stream = tokio_stream::wrappers::ReceiverStream::new(stream_rx);

        let result = Ok(crate::streaming::StreamingCompletionResponse::new(
            Box::pin(stream),
        ));
        let _ = task_tx.send(result);
    });

    AsyncTask::from_receiver(rx)
}
