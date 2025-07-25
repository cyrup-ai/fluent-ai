use async_stream::stream;
use futures_util::StreamExt;
use serde::Deserialize;

use super::completion::{CompletionModel, create_request_body, gemini_api_types::ContentCandidate};
use crate::{
    completion::{CompletionError, CompletionRequest},
    streaming::{self}};

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PartialUsage {
    pub total_token_count: i32}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamGenerateContentResponse {
    /// Candidate responses from the model.
    pub candidates: Vec<ContentCandidate>,
    pub model_version: Option<String>,
    pub usage_metadata: Option<PartialUsage>}

#[derive(Clone, Debug)]
pub struct StreamingCompletionResponse {
    pub usage_metadata: PartialUsage}

impl CompletionModel {
    pub(crate) async fn stream_internal(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request = create_request_body(completion_request)?;

        let response = self
            .client
            .post_sse(&format!(
                "/v1beta/models/{}:streamGenerateContent",
                self.model
            ))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(format!(
                "{}: {}",
                response.status(),
                // Domain uses HTTP3, provider delegates to domain
                todo!("Delegate to domain layer for HTTP3 streaming")
            )));
        }

        let stream = Box::pin(stream! {
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(CompletionError::from(e));
                        break;
                    }
                };

                let text = match String::from_utf8(chunk.to_vec()) {
                    Ok(t) => t,
                    Err(e) => {
                        yield Err(CompletionError::ResponseError(e.to_string()));
                        break;
                    }
                };


                for line in text.lines() {
                    let Some(line) = line.strip_prefix("data: ") else { continue; };

                    let Ok(data) = serde_json::from_str::<StreamGenerateContentResponse>(line) else {
                        continue;
                    };

                    // Handle missing candidates gracefully instead of panicking
                    let Some(choice) = data.candidates.first() else {
                        yield Err(CompletionError::ResponseError(
                            "Streaming response missing candidates".to_string()
                        ));
                        continue;
                    };

                    match choice.content.parts.first() {
                        Some(super::completion::gemini_api_types::Part::Text(text)) =>
                            yield Ok(streaming::RawStreamingChoice::Message(text)),
                        Some(super::completion::gemini_api_types::Part::FunctionCall(function_call)) =>
                            yield Ok(streaming::RawStreamingChoice::ToolCall {
                                name: function_call.name,
                                id: "".to_string(),
                                arguments: function_call.args
                            }),
                        Some(_) => {
                            // Handle unsupported response types gracefully
                            yield Err(CompletionError::ResponseError(
                                "Unsupported response type in streaming".to_string()
                            ));
                            continue;
                        }
                        None => {
                            // Handle missing parts gracefully
                            yield Err(CompletionError::ResponseError(
                                "Streaming response missing content parts".to_string()
                            ));
                            continue;
                        }
                    };

                    if choice.finish_reason.is_some() {
                        yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                            usage_metadata: PartialUsage {
                                total_token_count: data.usage_metadata.as_ref()
                                    .ok_or_else(|| CompletionError::ServerError("Missing usage metadata in streaming response".to_string()))?
                                    .total_token_count}
                        }))
                    }
                }
            }
        });

        Ok(streaming::StreamingCompletionResponse::new(stream))
    }
}
