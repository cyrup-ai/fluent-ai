//! Provider-specific HTTP3 builders
//!
//! Pre-configured builders for different AI providers with optimized settings

use crate::builder::{Http3Builder, ContentType};
use crate::HttpClient;
use http::{HeaderName, HeaderValue};

/// Factory for provider-specific HTTP3 builders
pub struct Http3Builders;

impl Http3Builders {
    /// OpenAI-optimized HTTP3 builder
    pub fn openai() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0 (OpenAI)")
            )
    }

    /// Anthropic-optimized HTTP3 builder
    pub fn anthropic() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0 (Anthropic)")
            )
            .header(
                HeaderName::from_static("anthropic-version"),
                HeaderValue::from_static("2023-06-01")
            )
    }

    /// XAI-optimized HTTP3 builder
    pub fn xai() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0 (XAI)")
            )
    }

    /// Together AI-optimized HTTP3 builder
    pub fn together() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0 (Together)")
            )
    }

    /// OpenRouter-optimized HTTP3 builder
    pub fn openrouter() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0 (OpenRouter)")
            )
    }

    /// Perplexity-optimized HTTP3 builder
    pub fn perplexity() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0 (Perplexity)")
            )
    }

    /// Gemini-optimized HTTP3 builder
    pub fn gemini() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0 (Gemini)")
            )
    }

    /// Generic provider HTTP3 builder
    pub fn generic() -> Http3Builder {
        let client = HttpClient::default();
        Http3Builder::new(&client)
            .content_type(ContentType::ApplicationJson)
            .header(
                HeaderName::from_static("user-agent"),
                HeaderValue::from_static("fluent-ai/1.0")
            )
    }
}