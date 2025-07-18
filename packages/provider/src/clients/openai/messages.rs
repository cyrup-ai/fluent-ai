//! Zero-allocation OpenAI message handling with comprehensive content type support
//!
//! Provides complete message conversion and handling for OpenAI's chat completion API
//! with support for text, images, audio, tool calls, and function calls.

use crate::domain::{Message as DomainMessage, MessageRole, ToolCall as DomainToolCall, ToolFunction};
use super::{OpenAIError, OpenAIResult};
use crate::ZeroOneOrMany;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use base64::Engine;

/// OpenAI message structure for chat completions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<OpenAIFunctionCall>,
}

/// OpenAI content types (text, image, audio)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    Text(String),
    Array(Vec<OpenAIContentPart>),
}

/// Individual content part for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<OpenAIImageUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<OpenAIAudioContent>,
}

/// Image URL content for vision models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "low", "high", "auto"
}

/// Audio content for speech models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIAudioContent {
    pub data: String, // base64 encoded audio
    pub format: String, // "mp3", "wav", "flac", etc.
}

/// OpenAI tool call structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

/// OpenAI function call structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Tool call result for assistant responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolResult {
    pub tool_call_id: String,
    pub output: String,
}

impl OpenAIMessage {
    /// Create system message with zero allocations
    #[inline(always)]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(OpenAIContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    /// Create user message with text content
    #[inline(always)]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(OpenAIContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    /// Create assistant message with text content
    #[inline(always)]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(OpenAIContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    /// Create user message with image URL (for vision models)
    #[inline(always)]
    pub fn user_with_image(text: impl Into<String>, image_url: impl Into<String>, detail: Option<String>) -> Self {
        let content_parts = vec![
            OpenAIContentPart {
                content_type: "text".to_string(),
                text: Some(text.into()),
                image_url: None,
                audio: None,
            },
            OpenAIContentPart {
                content_type: "image_url".to_string(),
                text: None,
                image_url: Some(OpenAIImageUrl {
                    url: image_url.into(),
                    detail,
                }),
                audio: None,
            },
        ];

        Self {
            role: "user".to_string(),
            content: Some(OpenAIContent::Array(content_parts)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    /// Create user message with base64 image data
    #[inline(always)]
    pub fn user_with_image_data(
        text: impl Into<String>, 
        image_data: &[u8], 
        mime_type: impl Into<String>,
        detail: Option<String>
    ) -> Self {
        let base64_data = base64::engine::general_purpose::STANDARD.encode(image_data);
        let data_url = format!("data:{};base64,{}", mime_type.into(), base64_data);
        
        Self::user_with_image(text, data_url, detail)
    }

    /// Create user message with audio content
    #[inline(always)]
    pub fn user_with_audio(text: impl Into<String>, audio_data: &[u8], format: impl Into<String>) -> Self {
        let base64_audio = base64::engine::general_purpose::STANDARD.encode(audio_data);
        
        let content_parts = vec![
            OpenAIContentPart {
                content_type: "text".to_string(),
                text: Some(text.into()),
                image_url: None,
                audio: None,
            },
            OpenAIContentPart {
                content_type: "audio".to_string(),
                text: None,
                image_url: None,
                audio: Some(OpenAIAudioContent {
                    data: base64_audio,
                    format: format.into(),
                }),
            },
        ];

        Self {
            role: "user".to_string(),
            content: Some(OpenAIContent::Array(content_parts)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            function_call: None,
        }
    }

    /// Create assistant message with tool calls
    #[inline(always)]
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<OpenAIToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.map(OpenAIContent::Text),
            name: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            function_call: None,
        }
    }

    /// Create tool result message
    #[inline(always)]
    pub fn tool_result(tool_call_id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(OpenAIContent::Text(result.into())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            function_call: None,
        }
    }

    /// Create message with custom name (for user identification)
    #[inline(always)]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Check if message contains images
    #[inline(always)]
    pub fn has_images(&self) -> bool {
        match &self.content {
            Some(OpenAIContent::Array(parts)) => {
                parts.iter().any(|part| part.content_type == "image_url")
            }
            _ => false,
        }
    }

    /// Check if message contains audio
    #[inline(always)]
    pub fn has_audio(&self) -> bool {
        match &self.content {
            Some(OpenAIContent::Array(parts)) => {
                parts.iter().any(|part| part.content_type == "audio")
            }
            _ => false,
        }
    }

    /// Get text content from message
    #[inline(always)]
    pub fn get_text_content(&self) -> Option<String> {
        match &self.content {
            Some(OpenAIContent::Text(text)) => Some(text.clone()),
            Some(OpenAIContent::Array(parts)) => {
                let text_parts: Vec<String> = parts.iter()
                    .filter_map(|part| part.text.as_ref())
                    .cloned()
                    .collect();
                if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join(" "))
                }
            }
            None => None,
        }
    }

    /// Extract tool calls from message
    #[inline(always)]
    pub fn get_tool_calls(&self) -> Vec<OpenAIToolCall> {
        self.tool_calls.clone().unwrap_or_default()
    }
}

/// Convert domain ToolCall to OpenAI format with zero allocations
#[inline(always)]
fn convert_tool_call(tool_call: &DomainToolCall) -> OpenAIResult<OpenAIToolCall> {
    Ok(OpenAIToolCall {
        id: tool_call.id.clone(),
        call_type: "function".to_string(), // OpenAI uses "function" as the type
        function: OpenAIFunctionCall {
            name: tool_call.function.name.clone(),
            arguments: tool_call.function.parameters.to_string(),
        },
    })
}

/// Extract tool calls from message content if present in structured format
#[inline(always)]
fn extract_tool_calls_from_content(content: &str) -> Option<Vec<DomainToolCall>> {
    // Attempt to parse content as JSON that might contain tool calls
    // This handles cases where tool calls are embedded in the message content
    if let Ok(parsed) = serde_json::from_str::<Value>(content) {
        if let Some(tool_calls_array) = parsed.get("tool_calls").and_then(|v| v.as_array()) {
            let mut tool_calls = Vec::new();
            for tool_call_value in tool_calls_array {
                if let Ok(tool_call) = serde_json::from_value::<DomainToolCall>(tool_call_value.clone()) {
                    tool_calls.push(tool_call);
                }
            }
            if !tool_calls.is_empty() {
                return Some(tool_calls);
            }
        }
    }
    None
}

/// Convert fluent-ai Message to OpenAI format with comprehensive tool call support
#[inline(always)]
pub fn convert_message(message: &Message) -> OpenAIResult<OpenAIMessage> {
    let role = match message.role {
        MessageRole::System => "system",
        MessageRole::User => "user", 
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };

    // Extract tool calls from content if present
    let (content, tool_calls) = if message.content.is_empty() {
        (None, None)
    } else {
        // Check if content contains structured tool calls
        if let Some(domain_tool_calls) = extract_tool_calls_from_content(&message.content) {
            // Convert domain tool calls to OpenAI format
            let openai_tool_calls: Result<Vec<_>, _> = domain_tool_calls
                .iter()
                .map(convert_tool_call)
                .collect();
            
            match openai_tool_calls {
                Ok(calls) if !calls.is_empty() => {
                    // If we have tool calls, content might be empty or contain additional text
                    let content = if message.content.trim().starts_with('{') && message.content.trim().ends_with('}') {
                        // Content is pure JSON with tool calls, no additional text content
                        None
                    } else {
                        // Content has additional text besides tool calls
                        Some(OpenAIContent::Text(message.content.clone()))
                    };
                    (content, Some(calls))
                }
                _ => {
                    // Failed to convert tool calls, treat as regular text content
                    (Some(OpenAIContent::Text(message.content.clone())), None)
                }
            }
        } else {
            // No tool calls found, treat as regular text content
            (Some(OpenAIContent::Text(message.content.clone())), None)
        }
    };

    Ok(OpenAIMessage {
        role: role.to_string(),
        content,
        name: message.name.clone(),
        tool_calls,
        tool_call_id: None,
        function_call: None,
    })
}

/// Convert multiple messages with batch optimization
#[inline(always)]
pub fn convert_messages(messages: &ZeroOneOrMany<Message>) -> OpenAIResult<Vec<OpenAIMessage>> {
    match messages {
        ZeroOneOrMany::None => Ok(Vec::new()),
        ZeroOneOrMany::One(message) => {
            Ok(vec![convert_message(message)?])
        }
        ZeroOneOrMany::Many(messages) => {
            let mut result = Vec::with_capacity(messages.len());
            for message in messages {
                result.push(convert_message(message)?);
            }
            Ok(result)
        }
    }
}

/// Extract text content from OpenAI response message
#[inline(always)]
pub fn extract_text_from_response(message: &OpenAIMessage) -> String {
    message.get_text_content().unwrap_or_default()
}

/// Extract tool calls from OpenAI response message
#[inline(always)]
pub fn extract_tool_calls_from_response(message: &OpenAIMessage) -> Vec<OpenAIToolCall> {
    message.get_tool_calls()
}

/// Validate message for model compatibility
#[inline(always)]
pub fn validate_message_for_model(message: &OpenAIMessage, model: &str) -> OpenAIResult<()> {
    // Check if model supports vision features
    if message.has_images() && !is_vision_model(model) {
        return Err(OpenAIError::FeatureNotSupported {
            feature: "vision".to_string(),
            model: model.to_string(),
        });
    }

    // Check if model supports audio features
    if message.has_audio() && !is_audio_model(model) {
        return Err(OpenAIError::FeatureNotSupported {
            feature: "audio".to_string(),
            model: model.to_string(),
        });
    }

    // Check if model supports tool calls
    if !message.get_tool_calls().is_empty() && !is_tool_calling_model(model) {
        return Err(OpenAIError::FeatureNotSupported {
            feature: "tool_calling".to_string(),
            model: model.to_string(),
        });
    }

    Ok(())
}

/// Check if model supports vision features
#[inline(always)]
pub fn is_vision_model(model: &str) -> bool {
    matches!(model,
        "gpt-4o" | "gpt-4o-mini" | "gpt-4-vision-preview" | "gpt-4-turbo" |
        "gpt-4-turbo-2024-04-09" | "gpt-4-1106-vision-preview" |
        "chatgpt-4o-latest" | "gpt-4o-search-preview" | "gpt-4o-mini-search-preview"
    )
}

/// Check if model supports audio features  
#[inline(always)]
pub fn is_audio_model(model: &str) -> bool {
    matches!(model,
        "gpt-4o" | "gpt-4o-mini" | "chatgpt-4o-latest" |
        "whisper-1" | "tts-1" | "tts-1-hd"
    )
}

/// Check if model supports tool/function calling
#[inline(always)]
pub fn is_tool_calling_model(model: &str) -> bool {
    matches!(model,
        "gpt-4" | "gpt-4-0613" | "gpt-4-1106-preview" | "gpt-4-turbo" |
        "gpt-4-turbo-2024-04-09" | "gpt-4o" | "gpt-4o-mini" |
        "gpt-3.5-turbo" | "gpt-3.5-turbo-0613" | "gpt-3.5-turbo-1106" |
        "chatgpt-4o-latest" | "o3" | "o3-mini" | "o4-mini" |
        "gpt-4-1" | "gpt-4-1-mini" | "gpt-4-1-nano"
    )
}

/// Compatibility aliases for message types
pub type Message = OpenAIMessage;
pub type AssistantContent = OpenAIContent;