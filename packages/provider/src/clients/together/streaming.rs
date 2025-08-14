use fluent_ai_domain::completion::{CompletionCoreError as CompletionError, CompletionRequest};
use serde_json::json;

use super::completion::TogetherCompletionModel;
use crate::clients::openai;
use crate::clients::openai::send_compatible_streaming_request;
use crate::clients::xai::streaming::merge;
use crate::streaming::DefaultStreamingResponse;

impl TogetherCompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<DefaultStreamingResponse<openai::StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(request, json!({"stream_tokens": true}));

        let request_body = serde_json::to_vec(&request).map_err(|e| {
            CompletionError::ProviderError(format!("Failed to serialize request: {}", e))
        })?;

        let http_request = self
            .client
            .post("/v1/chat/completions", request_body)
            .map_err(|e| {
                CompletionError::ProviderError(format!("Failed to create request: {}", e))
            })?;

        openai::send_compatible_streaming_request(self.client.http_client.clone(), http_request)
            .await
    }
}
