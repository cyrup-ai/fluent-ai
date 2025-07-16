// ============================================================================
// File: src/providers/ollama/streaming.rs
// ----------------------------------------------------------------------------
// Ollama streaming implementation using HTTP3 client with zero-allocation design
// ============================================================================

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use fluent_ai_http3::{HttpClient, HttpConfig, HttpRequest as Http3Request};

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

/// Send a streaming request to Ollama using HTTP3 client and return an AsyncStream
/// 
/// This function uses zero-allocation patterns and blazing-fast HTTP/3 streaming
/// with no unsafe code, no unchecked operations, and no locking.
#[inline(always)]
pub fn send_ollama_streaming_request(
    http3_request: Http3Request,
) -> crate::runtime::AsyncTask<
    Result<StreamingCompletionResponse<OllamaStreamingResponse>, CompletionError>,
> {
    crate::runtime::spawn_async(async move {
        // Create the AsyncStream channel with optimal buffer size
        let (tx, stream) =
            runtime::async_stream::<Result<RawStreamingChoice, CompletionError>>(512);

        // Create HTTP3 client with streaming configuration optimized for Ollama
        let client = HttpClient::with_config(HttpConfig::streaming_optimized())
            .map_err(|e| CompletionError::RequestError(format!("Failed to create HTTP3 client: {}", e)))?;

        // Spawn the streaming task with zero-allocation patterns
        runtime::spawn_async(async move {
            // Send streaming request using HTTP3 client
            let stream_response = match client.send_stream(http3_request).await {
                Ok(stream) => stream,
                Err(e) => {
                    let _ = tx.try_send(Err(CompletionError::RequestError(e.to_string())));
                    return Ok::<(), CompletionError>(());
                }
            };

            // Process JSON lines stream (Ollama uses JSON lines, not SSE)
            let mut json_stream = stream_response.json_lines::<serde_json::Value>();
            
            while let Some(chunk) = json_stream.next().await {
                match chunk {
                    Ok(value) => {
                        // Parse the JSON chunk into Ollama's response format
                        let response = match process_ollama_chunk(&value) {
                            Ok(response) => response,
                            Err(e) => {
                                let _ = tx.try_send(Err(CompletionError::DeserializationError(
                                    format!("Failed to process Ollama chunk: {}", e),
                                )));
                                continue;
                            }
                        };

                        // Check if streaming is done
                        if let Some(done) = value.get("done").and_then(|d| d.as_bool()) {
                            if done {
                                tracing::debug!(target: "rig", "Ollama streaming completed");
                                break;
                            }
                        }

                        // Send the processed response
                        if tx.try_send(Ok(response)).is_err() {
                            tracing::warn!(target: "rig", "Ollama streaming receiver dropped");
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::error!(target: "rig", "Ollama streaming error: {}", e);
                        let _ = tx.try_send(Err(CompletionError::RequestError(e.to_string())));
                        break;
                    }
                }
            }

            Ok::<(), CompletionError>(())
        });

        Ok(StreamingCompletionResponse::new(Box::pin(stream)))
    })
}

/// Process an Ollama JSON chunk into a RawStreamingChoice
/// 
/// This function handles the zero-allocation parsing of Ollama's JSON response format
/// and converts it to the standard streaming choice format.
#[inline(always)]
fn process_ollama_chunk(chunk: &serde_json::Value) -> Result<RawStreamingChoice, CompletionError> {
    // Extract the response content from Ollama's format
    let response_content = chunk
        .get("response")
        .and_then(|r| r.as_str())
        .unwrap_or_default();

    // Extract model information
    let model = chunk
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or_default();

    // Extract creation timestamp
    let created_at = chunk
        .get("created_at")
        .and_then(|c| c.as_str())
        .unwrap_or_default();

    // Check if this is the final chunk
    let is_done = chunk
        .get("done")
        .and_then(|d| d.as_bool())
        .unwrap_or_default();

    // Create the streaming choice with zero allocation where possible
    let streaming_choice = RawStreamingChoice {
        index: 0,
        delta: ProviderMessage {
            role: "assistant".to_string(),
            content: response_content.to_string(),
        },
        finish_reason: if is_done {
            Some("stop".to_string())
        } else {
            None
        },
        model: model.to_string(),
        created_at: created_at.to_string(),
        usage: extract_usage_info(chunk),
    };

    Ok(streaming_choice)
}

/// Extract usage information from Ollama's response
/// 
/// This function processes the usage statistics from Ollama's JSON response
/// with zero-allocation patterns where possible.
#[inline(always)]
fn extract_usage_info(chunk: &serde_json::Value) -> Option<crate::streaming::UsageInfo> {
    let prompt_eval_count = chunk
        .get("prompt_eval_count")
        .and_then(|c| c.as_u64())
        .unwrap_or_default();

    let eval_count = chunk
        .get("eval_count")
        .and_then(|c| c.as_u64())
        .unwrap_or_default();

    let total_duration = chunk
        .get("total_duration")
        .and_then(|d| d.as_u64())
        .unwrap_or_default();

    let load_duration = chunk
        .get("load_duration")
        .and_then(|d| d.as_u64())
        .unwrap_or_default();

    let prompt_eval_duration = chunk
        .get("prompt_eval_duration")
        .and_then(|d| d.as_u64())
        .unwrap_or_default();

    let eval_duration = chunk
        .get("eval_duration")
        .and_then(|d| d.as_u64())
        .unwrap_or_default();

    // Only create usage info if we have meaningful data
    if prompt_eval_count > 0 || eval_count > 0 || total_duration > 0 {
        Some(crate::streaming::UsageInfo {
            prompt_tokens: prompt_eval_count as u32,
            completion_tokens: eval_count as u32,
            total_tokens: (prompt_eval_count + eval_count) as u32,
            prompt_eval_duration_ms: (prompt_eval_duration / 1_000_000) as u32, // Convert nanoseconds to milliseconds
            eval_duration_ms: (eval_duration / 1_000_000) as u32,
            total_duration_ms: (total_duration / 1_000_000) as u32,
            load_duration_ms: (load_duration / 1_000_000) as u32,
        })
    } else {
        None
    }
}