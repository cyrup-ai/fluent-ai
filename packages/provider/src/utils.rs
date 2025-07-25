//! HTTP utility functions for provider clients
//!
//! Contains validation and helper functions used across provider implementations

use thiserror::Error;

/// Provider enumeration for different AI services
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    Anthropic,
    XAI,
    AI21,
    Perplexity,
    Gemini,
    OpenRouter,
    Together,
}

/// HTTP utility error types
#[derive(Debug, Error)]
pub enum HttpUtilError {
    #[error("Invalid max_tokens value: {0}")]
    InvalidMaxTokens(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// HTTP utility functions for validation and processing
pub struct HttpUtils;

impl HttpUtils {
    /// Validate max_tokens parameter for a given provider
    pub fn validate_max_tokens(max_tokens: u32, provider: Provider) -> Result<u32, HttpUtilError> {
        // Define provider-specific max token limits
        let max_limit = match provider {
            Provider::OpenAI => 4096,       // GPT-4 context window
            Provider::Anthropic => 100000,  // Claude-3 context window
            Provider::XAI => 131072,        // Grok context window
            Provider::AI21 => 8192,         // Jurassic context window
            Provider::Perplexity => 4096,   // Perplexity context window
            Provider::Gemini => 32768,      // Gemini Pro context window
            Provider::OpenRouter => 4096,   // Default limit
            Provider::Together => 8192,     // Together AI context window
        };

        if max_tokens == 0 {
            return Err(HttpUtilError::InvalidMaxTokens(
                "max_tokens must be greater than 0".to_string()
            ));
        }

        if max_tokens > max_limit {
            return Err(HttpUtilError::InvalidMaxTokens(
                format!("max_tokens {} exceeds limit {} for provider {:?}", 
                    max_tokens, max_limit, provider)
            ));
        }

        Ok(max_tokens)
    }

    /// Validate temperature parameter
    pub fn validate_temperature(temperature: f32) -> Result<f32, HttpUtilError> {
        if !temperature.is_finite() {
            return Err(HttpUtilError::ValidationError(
                "Temperature must be a finite number".to_string()
            ));
        }

        if temperature < 0.0 || temperature > 2.0 {
            return Err(HttpUtilError::ValidationError(
                "Temperature must be between 0.0 and 2.0".to_string()
            ));
        }

        Ok(temperature)
    }

    /// Validate top_p parameter
    pub fn validate_top_p(top_p: f32) -> Result<f32, HttpUtilError> {
        if !top_p.is_finite() {
            return Err(HttpUtilError::ValidationError(
                "top_p must be a finite number".to_string()
            ));
        }

        if top_p < 0.0 || top_p > 1.0 {
            return Err(HttpUtilError::ValidationError(
                "top_p must be between 0.0 and 1.0".to_string()
            ));
        }

        Ok(top_p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_max_tokens_success() {
        assert_eq!(HttpUtils::validate_max_tokens(100, Provider::OpenAI).unwrap(), 100);
        assert_eq!(HttpUtils::validate_max_tokens(4096, Provider::OpenAI).unwrap(), 4096);
    }

    #[test]
    fn test_validate_max_tokens_zero_error() {
        assert!(HttpUtils::validate_max_tokens(0, Provider::OpenAI).is_err());
    }

    #[test]
    fn test_validate_max_tokens_exceeds_limit() {
        assert!(HttpUtils::validate_max_tokens(10000, Provider::OpenAI).is_err());
    }

    #[test]
    fn test_validate_temperature_success() {
        assert_eq!(HttpUtils::validate_temperature(0.7).unwrap(), 0.7);
        assert_eq!(HttpUtils::validate_temperature(0.0).unwrap(), 0.0);
        assert_eq!(HttpUtils::validate_temperature(2.0).unwrap(), 2.0);
    }

    #[test]
    fn test_validate_temperature_error() {
        assert!(HttpUtils::validate_temperature(-0.1).is_err());
        assert!(HttpUtils::validate_temperature(2.1).is_err());
        assert!(HttpUtils::validate_temperature(f32::NAN).is_err());
    }
}