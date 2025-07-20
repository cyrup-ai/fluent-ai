use fluent_ai_domain::completion::CompletionRequest;
use serde_json::json;

use super::completion::CompletionModel;
use crate::clients::openai;
use crate::clients::openai::send_compatible_streaming_request;
use crate::completion_provider::CompletionError;
use crate::json_util::merge;
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
