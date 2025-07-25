use fluent_ai_domain::completion::CompletionRequest;
use serde_json::json;
use super::completion::CompletionModel;
use crate::clients::openai;
use crate::completion_provider::CompletionError;
use crate::streaming::StreamingCompletionResponse;
use fluent_ai_http3::{Http3, header};
/// Helper function to merge two JSON values
fn merge(mut base: serde_json::Value, other: serde_json::Value) -> serde_json::Value {
    if let (serde_json::Value::Object(ref mut base_map), serde_json::Value::Object(other_map)) =
        (&mut base, other)
    {
        base_map.extend(other_map);
        base
    } else {
        other
    }
}
use crate::streaming::StreamingCompletionResponse;

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<openai::StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(request, json!({"stream": true}));

        let url = format!("{}/v1/chat/completions", self.client.base_url());
        
        let stream = Http3::json()
            .bearer_auth(&self.client.api_key())
            .body(&request)
            .post(&url);
        
        openai::process_openai_compatible_streaming_response(stream).await
    }
}
