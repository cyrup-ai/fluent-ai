use std::convert::Infallible;
use std::str::FromStr;

use fluent_ai_domain::completion::{CompletionCoreError as CompletionError, CompletionRequest};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::completion::CompletionModel;
use crate::clients::openai::{StreamingCompletionResponse, send_compatible_streaming_request};
/// Helper function to merge JSON values in-place
fn merge_inplace(base: &mut serde_json::Value, other: serde_json::Value) {
    if let (serde_json::Value::Object(ref mut base_map), serde_json::Value::Object(other_map)) =
        (base, other)
    {
        base_map.extend(other_map);
    } else {
        *base = other;
    }
}
use crate::streaming;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase", tag = "type")]
/// Represents the content sent back in the StreamDelta for an Assistant
enum AssistantContent {
    Text { text: String }}

// Ensure that string contents can be serialized correctly
impl FromStr for AssistantContent {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent::Text {
            text: s.to_string()})
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase", tag = "role")]
enum StreamDelta {
    Assistant {
        #[serde(deserialize_with = "json_util::string_or_vec")]
        content: Vec<AssistantContent>}}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct StreamingChoice {
    index: usize,
    delta: StreamDelta,
    logprobs: Value,
    finish_reason: Option<String>}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct CompletionChunk {
    id: String,
    created: i32,
    model: String,
    #[serde(default)]
    system_fingerprint: String,
    choices: Vec<StreamingChoice>}

impl CompletionModel {
    pub(crate) async fn stream_internal(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_request_body(&completion_request)?;

        // Enable streaming
        merge_inplace(
            &mut request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        if let Some(ref params) = completion_request.additional_params {
            merge_inplace(&mut request, params.clone());
        }

        // HF Inference API uses the model in the path even though its specified in the request body
        let path = self.client.sub_provider.completion_endpoint(&self.model);

        let builder = self.client.post(&path).json(&request);

        send_compatible_streaming_request(builder).await
    }
}
