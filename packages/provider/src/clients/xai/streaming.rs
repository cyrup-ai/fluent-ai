use crate::domain::completion::CompletionRequest;
use serde_json::json;
use super::completion::XaiCompletionModel;
use crate::clients::openai;
use crate::completion_provider::CompletionError;
use crate::streaming::DefaultStreamingResponse;
use fluent_ai_http3::{Http3, header};
/// Helper function to merge two JSON values
pub fn merge(mut base: serde_json::Value, other: serde_json::Value) -> serde_json::Value {
    if let (serde_json::Value::Object(ref mut base_map), serde_json::Value::Object(other_map)) =
        (&mut base, other)
    {
        base_map.extend(other_map);
        base
    } else {
        other
    }
}

impl XaiCompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<DefaultStreamingResponse<openai::StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(request, json!({"stream": true}));

        let url = format!("{}/v1/chat/completions", self.client.base_url());
        
        let stream = Http3::json()
            .bearer_auth(&self.client.api_key())
            .body(&request)
            .post(&url);
        
        // XAI uses OpenAI-compatible streaming, so we can use similar parsing
        // TODO: Implement proper XAI streaming response parsing
        Err(CompletionError::ProviderError("XAI streaming not yet implemented".to_string()))
    }
}
