use fluent_ai_domain::completion::CompletionRequest;
use serde_json::json;

use super::completion::CompletionModel;
use crate::clients::openai;
use crate::clients::openai::send_compatible_streaming_request;
use crate::completion_provider::CompletionError;
/// Helper function to merge two JSON values
fn merge(mut base: serde_json::Value, other: serde_json::Value) -> serde_json::Value {
    if let (serde_json::Value::Object(ref mut base_map), serde_json::Value::Object(other_map)) = (&mut base, other) {
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

        let builder = self.client.post("/v1/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}
