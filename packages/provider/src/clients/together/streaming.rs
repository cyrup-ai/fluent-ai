use serde_json::json;

use super::completion::CompletionModel;
use crate::clients::openai;
use crate::clients::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use fluent_ai_domain::completion::{CompletionCoreError as CompletionError, CompletionRequest};

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<openai::StreamingCompletionResponse>, CompletionError>
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
