use crate::clients::openai::OpenAIClient;

/// Type alias for unified client interface
type UnifiedClient = OpenAIClient;

/// Factory function to create unified client
pub fn create_unified_client(api_key: String) -> UnifiedClient {
    OpenAIClient::new(api_key).expect("Failed to create OpenAI client")
}

async fn example_http3(request: &CompletionRequest, api_key: &str) -> Result<CompletionResponse, HttpError> {
    Http3::json()
        .api_key(api_key)
        .body(request)
        .post("https://api.openai.com/v1/chat/completions")
        .collect::<CompletionResponse>()
        .await
}
