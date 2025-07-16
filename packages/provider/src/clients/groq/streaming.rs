// ============================================================================
// File: src/providers/groq/streaming.rs
// ----------------------------------------------------------------------------
// Groq streaming implementation (OpenAI-compatible SSE format)
// ============================================================================

use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde_json::Value;

use crate::{
    completion::CompletionError,
    providers::openai::{CompletionResponse, StreamingChoice, StreamingMessage},
    runtime::{self, AsyncStream},
};

// Re-export OpenAI streaming response type since Groq uses the same format
pub use crate::providers::openai::StreamingCompletionResponse;

/// Send a streaming request to Groq and return an AsyncStream
pub fn send_groq_streaming_request(
    builder: reqwest::RequestBuilder,
) -> crate::runtime::AsyncTask<
    Result<crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>,
> {
    crate::runtime::spawn_async(async move {
        // Create the AsyncStream channel
        let (tx, stream) =
            runtime::async_stream::<Result<StreamingCompletionResponse, CompletionError>>(512);

        // Clone the request for the async task
        let request = builder
            .try_clone()
            .ok_or_else(|| CompletionError::RequestError("Failed to clone request".to_string()))?;

        // Spawn the streaming task
        runtime::spawn_async(async move {
            let mut event_source = EventSource::new(request)
                .map_err(|e| CompletionError::RequestError(e.to_string()))?;

            while let Some(event) = event_source.next().await {
                match event {
                    Ok(Event::Open) => {
                        tracing::debug!(target: "rig", "Groq streaming connection opened");
                    }
                    Ok(Event::Message(message)) => {
                        if message.data == "[DONE]" {
                            break;
                        }

                        match serde_json::from_str::<StreamingCompletionResponse>(&message.data) {
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
                    Err(e) => {
                        tracing::error!(target: "rig", "Groq streaming error: {}", e);
                        let _ = tx.try_send(Err(CompletionError::RequestError(e.to_string())));
                        break;
                    }
                }
            }

            Ok::<(), CompletionError>(())
        });

        Ok(crate::streaming::StreamingCompletionResponse::new(Box::pin(stream)))
    })
}
