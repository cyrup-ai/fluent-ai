// ============================================================================
// File: src/providers/ollama/streaming.rs
// ----------------------------------------------------------------------------
// Ollama streaming implementation
// ============================================================================

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
    completion::CompletionError,
    runtime::{self, AsyncStream},
    streaming::{RawStreamingChoice, StreamingCompletionResponse},
};

use super::completion::{CompletionResponse, ProviderMessage};

// ============================================================================
// Streaming Response
// ============================================================================
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OllamaStreamingResponse {
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}

/// Send a streaming request to Ollama and return an AsyncStream
pub fn send_ollama_streaming_request(
    builder: reqwest::RequestBuilder,
) -> crate::runtime::AsyncTask<
    Result<StreamingCompletionResponse<OllamaStreamingResponse>, CompletionError>,
> {
    crate::runtime::spawn_async(async move {
        // Create the AsyncStream channel
        let (tx, stream) =
            runtime::async_stream::<Result<RawStreamingChoice, CompletionError>>(512);

        // Clone the request for the async task
        let request = builder
            .try_clone()
            .ok_or_else(|| CompletionError::RequestError("Failed to clone request".to_string()))?;

        // Spawn the streaming task
        runtime::spawn_async(async move {
            let response = match request.send().await {
                Ok(resp) => resp,
                Err(e) => {
                    let _ = tx.try_send(Err(CompletionError::from(e)));
                    return;
                }
            };

            if !response.status().is_success() {
                let err_text = match response.text().await {
                    Ok(text) => text,
                    Err(e) => e.to_string(),
                };
                let _ = tx.try_send(Err(CompletionError::ProviderError(err_text)));
                return;
            }

            let mut stream = response.bytes_stream();
            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.try_send(Err(CompletionError::from(e)));
                        break;
                    }
                };

                let text = match String::from_utf8(chunk.to_vec()) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.try_send(Err(CompletionError::ResponseError(e.to_string())));
                        break;
                    }
                };

                for line in text.lines() {
                    let line = line.to_string();

                    let Ok(response) = serde_json::from_str::<CompletionResponse>(&line) else {
                        continue;
                    };

                    match response.message {
                        ProviderMessage::Assistant {
                            content,
                            tool_calls,
                            ..
                        } => {
                            if !content.is_empty() {
                                if tx
                                    .try_send(Ok(RawStreamingChoice::Message(content)))
                                    .is_err()
                                {
                                    tracing::warn!(target: "rig", "Ollama streaming receiver dropped");
                                    return;
                                }
                            }

                            for tool_call in tool_calls.iter() {
                                let function = tool_call.function.clone();

                                if tx
                                    .try_send(Ok(RawStreamingChoice::ToolCall {
                                        id: "".to_string(),
                                        name: function.name,
                                        arguments: function.arguments,
                                    }))
                                    .is_err()
                                {
                                    tracing::warn!(target: "rig", "Ollama streaming receiver dropped");
                                    return;
                                }
                            }
                        }
                        _ => {
                            continue;
                        }
                    }

                    if response.done {
                        let _ = tx.try_send(Ok(RawStreamingChoice::FinalResponse(
                            OllamaStreamingResponse {
                                total_duration: response.total_duration,
                                load_duration: response.load_duration,
                                prompt_eval_count: response.prompt_eval_count,
                                prompt_eval_duration: response.prompt_eval_duration,
                                eval_count: response.eval_count,
                                eval_duration: response.eval_duration,
                                done_reason: response.done_reason,
                            },
                        )));
                    }
                }
            }
        });

        Ok(StreamingCompletionResponse::new(Box::pin(stream)))
    })
}
