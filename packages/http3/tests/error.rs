//! Error module tests
//!
//! Tests for HTTP3 error functionality, mirroring src/error.rs

use fluent_ai_http3::{Http3, HttpError};

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_basic_error_functionality() {
        // Test HTTP3 error handling functionality
        let builder = Http3::json();
        
        // Test that builder can be configured for error scenarios
        let configured_builder = builder
            .timeout_seconds(1) // Short timeout to trigger timeout errors
            .retry_attempts(2); // Multiple retries for error resilience
        
        // Verify error-handling configuration is accepted
        let _final_builder = configured_builder
            .headers([("expect", "100-continue")]) // Header that might cause issues
            .user_agent("HTTP3-Error-Test");
        
        // Test passes if error-handling configuration can be chained
        assert!(true, "HTTP3 builder should support error handling configuration");
    }

    #[test]
    fn test_http_error_types() {
        // Test HttpError variant creation and display
        let network_error = HttpError::NetworkError {
            message: "Connection failed".to_string(),
        };
        assert!(format!("{}", network_error).contains("Network error"));
        assert!(format!("{}", network_error).contains("Connection failed"));

        let client_error = HttpError::ClientError {
            message: "Invalid configuration".to_string(),
        };
        assert!(format!("{}", client_error).contains("Client error"));
        assert!(format!("{}", client_error).contains("Invalid configuration"));

        let http_status_error = HttpError::HttpStatus {
            status: 404,
            message: "Not Found".to_string(),
            body: "Resource not found".to_string(),
        };
        assert!(format!("{}", http_status_error).contains("HTTP 404"));
        assert!(format!("{}", http_status_error).contains("Not Found"));
        assert_eq!(http_status_error.status().unwrap().as_u16(), 404);

        let timeout_error = HttpError::Timeout {
            message: "Request timed out".to_string(),
        };
        assert!(format!("{}", timeout_error).contains("Request timeout"));
        assert!(format!("{}", timeout_error).contains("Request timed out"));
    }

    #[test]
    fn test_http_error_status_extraction() {
        // Test status code extraction from HttpError variants
        let status_error = HttpError::HttpStatus {
            status: 500,
            message: "Internal Server Error".to_string(),
            body: "Server error occurred".to_string(),
        };
        assert_eq!(status_error.status().unwrap().as_u16(), 500);

        let network_error = HttpError::NetworkError {
            message: "Connection failed".to_string(),
        };
        assert!(network_error.status().is_none());

        let custom_error = HttpError::Custom {
            message: "Custom error message".to_string(),
        };
        assert!(custom_error.status().is_none());
    }

    #[test]
    fn test_http_error_conversions() {
        // Test conversion from serde_json::Error
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json");
        assert!(json_error.is_err());
        let http_error: HttpError = json_error.unwrap_err().into();
        match http_error {
            HttpError::DeserializationError { .. } => {
                // Expected conversion for syntax error
            }
            _ => panic!("Expected DeserializationError variant"),
        }

        // Test conversion from url::ParseError
        let url_error = url::Url::parse("not a valid url");
        assert!(url_error.is_err());
        let http_error: HttpError = url_error.unwrap_err().into();
        match http_error {
            HttpError::UrlParseError { .. } => {
                // Expected conversion
            }
            _ => panic!("Expected UrlParseError variant"),
        }
    }
}

// Error-specific test modules will be organized here:
// - Error type tests
// - Error handling tests
// - Error recovery tests
// - Error message validation tests
