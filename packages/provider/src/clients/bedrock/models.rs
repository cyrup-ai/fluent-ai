//! Bedrock model validation utilities
//!
//! Simple capability validation without duplicating build.rs model enumeration.
//! Build.rs handles ALL model discovery and enum generation.

use arrayvec::ArrayString;

/// Basic model capability validation
///
/// Most models support these capabilities, this is just basic validation
pub fn validate_model_capability(model_id: &str, capability: &str) -> Result<(), ArrayString<128>> {
    // Basic capability check - most Bedrock models support these features
    match capability {
        "tools" => {
            // Most Bedrock models support tools except embedding models
            if model_id.contains("embed") {
                let mut error = ArrayString::new();
                let _ = error.try_push_str(&format!(
                    "embedding model {} does not support tools",
                    model_id
                ));
                Err(error)
            } else {
                Ok(())
            }
        }
        "vision" => {
            // Claude and Nova models support vision
            if model_id.contains("claude") || model_id.contains("nova") {
                Ok(())
            } else {
                let mut error = ArrayString::new();
                let _ = error.try_push_str(&format!("model {} does not support vision", model_id));
                Err(error)
            }
        }
        "streaming" => {
            // All text models support streaming except embeddings
            if model_id.contains("embed") {
                let mut error = ArrayString::new();
                let _ = error.try_push_str(&format!(
                    "embedding model {} does not support streaming",
                    model_id
                ));
                Err(error)
            } else {
                Ok(())
            }
        }
        _ => {
            let mut error = ArrayString::new();
            let _ = error.try_push_str("unknown capability");
            Err(error)
        }
    }
}
