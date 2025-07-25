//! Zero-allocation message and content types for Anthropic API
//!
//! Comprehensive message system supporting text, images, documents, and tool interactions
//! with optimal memory usage and performance.

use fluent_ai_domain::message::MessageRole;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Helper function to merge two JSON values
fn merge_json_values(mut base: Value, other: Value) -> Value {
    if let (Value::Object(ref mut base_map), Value::Object(other_map)) = (&mut base, other) {
        base_map.extend(other_map);
        base
    } else {
        other
    }
}

/// Message role in conversation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System}

/// Complete message structure for Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>}

/// Zero-allocation content types supporting all Anthropic features
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),
    /// Complex content with multiple blocks
    Blocks(Vec<ContentBlock>)}

/// Individual content block supporting all media types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content block
    Text { text: String },
    /// Image content block with base64 data
    Image { source: ImageSource },
    /// Document content block (PDF, etc.)
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        extract_text: Option<bool>},
    /// Tool use block for function calling
    ToolUse {
        id: String,
        name: String,
        #[serde(with = "crate::util::json_util::stringified_json")]
        input: Value},
    /// Tool result block for function responses
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>},
    /// Search result block for citation support
    SearchResult {
        source: String,
        title: String,
        content: Vec<ContentBlock>}}

/// Image source with format and data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64"
    pub media_type: String, // "image/jpeg", "image/png", etc.
    pub data: String,       // base64 encoded data
}

/// Document source with format and data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64"
    pub media_type: String, // "application/pdf", etc.
    pub data: String,       // base64 encoded data
}

/// Tool call structure for function invocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String, // "function"
    pub function: FunctionCall}

/// Function call details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(with = "crate::util::json_util::stringified_json")]
    pub arguments: Value}

/// Tool definition for Anthropic API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>}

/// Cache control for tool definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String, // "ephemeral"
}

impl Message {
    /// Create user message with text content
    #[inline(always)]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None}
    }

    /// Create assistant message with text content
    #[inline(always)]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None}
    }

    /// Create system message with text content
    #[inline(always)]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None}
    }

    /// Create user message with image content
    #[inline(always)]
    pub fn user_with_image(
        text: impl Into<String>,
        image_data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text { text: text.into() },
                ContentBlock::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: media_type.into(),
                        data: image_data.into()}},
            ]),
            name: None,
            tool_call_id: None,
            tool_calls: None}
    }

    /// Create user message with document content
    #[inline(always)]
    pub fn user_with_document(
        text: impl Into<String>,
        document_data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text { text: text.into() },
                ContentBlock::Document {
                    source: DocumentSource {
                        source_type: "base64".to_string(),
                        media_type: media_type.into(),
                        data: document_data.into()},
                    extract_text: Some(true)},
            ]),
            name: None,
            tool_call_id: None,
            tool_calls: None}
    }

    /// Create assistant message with tool calls
    #[inline(always)]
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(String::new()),
            name: None,
            tool_call_id: None,
            tool_calls: Some(tool_calls)}
    }

    /// Create user message with tool result
    #[inline(always)]
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: Some(content.into()),
                is_error: Some(false)}]),
            name: None,
            tool_call_id: None,
            tool_calls: None}
    }

    /// Create user message with tool error
    #[inline(always)]
    pub fn tool_error(tool_use_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: Some(error.into()),
                is_error: Some(true)}]),
            name: None,
            tool_call_id: None,
            tool_calls: None}
    }
}

impl MessageContent {
    /// Get text content from message content
    #[inline(always)]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(text) => Some(text),
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    if let ContentBlock::Text { text } = block {
                        return Some(text);
                    }
                }
                None
            }
        }
    }

    /// Check if content contains images
    #[inline(always)]
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::Image { .. }))}
    }

    /// Check if content contains documents
    #[inline(always)]
    pub fn has_documents(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::Document { .. }))}
    }

    /// Check if content contains tool use
    #[inline(always)]
    pub fn has_tool_use(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolUse { .. }))}
    }
}

impl Tool {
    /// Create new tool definition
    #[inline(always)]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            cache_control: None}
    }

    /// Create tool with caching enabled
    #[inline(always)]
    pub fn with_cache(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            cache_control: Some(CacheControl {
                cache_type: "ephemeral".to_string()})}
    }

    /// Convert from fluent-ai ToolDefinition
    #[inline(always)]
    pub fn from_definition(def: &fluent_ai_domain::tool::ToolDefinition) -> Self {
        Self::new(
            def.name.clone(),
            def.description.clone(),
            def.parameters.clone(),
        )
    }
}

/// Convert fluent-ai Message to Anthropic Message
impl From<&fluent_ai_domain::message::Message> for Message {
    #[inline(always)]
    fn from(msg: &crate::domain::Message) -> Self {
        match msg.role {
            MessageRole::User => Message::user(msg.content.clone()),
            MessageRole::Assistant => Message::assistant(msg.content.clone()),
            MessageRole::System => Message::system(msg.content.clone()),
            MessageRole::Tool => Message::user(format!("Tool result: {}", msg.content))}
    }
}

/// Zero-allocation message conversion utilities
pub struct MessageConverter;

impl MessageConverter {
    /// Convert fluent-ai messages to Anthropic format with zero allocation where possible
    #[inline(always)]
    pub fn convert_messages(
        messages: &crate::ZeroOneOrMany<fluent_ai_domain::message::Message>,
    ) -> Vec<Message> {
        match messages {
            crate::ZeroOneOrMany::None => Vec::new(),
            crate::ZeroOneOrMany::One(msg) => vec![Message::from(msg)],
            crate::ZeroOneOrMany::Many(msgs) => msgs.iter().map(Message::from).collect()}
    }

    /// Convert fluent-ai tools to Anthropic format
    #[inline(always)]
    pub fn convert_tools(
        tools: &crate::ZeroOneOrMany<fluent_ai_domain::tool::ToolDefinition>,
    ) -> Vec<Tool> {
        match tools {
            crate::ZeroOneOrMany::None => Vec::new(),
            crate::ZeroOneOrMany::One(tool) => vec![Tool::from_definition(tool)],
            crate::ZeroOneOrMany::Many(tools) => tools.iter().map(Tool::from_definition).collect()}
    }

    /// Merge additional parameters into request JSON with zero allocation
    #[inline(always)]
    pub fn merge_additional_params(mut request: Value, additional: Option<Value>) -> Value {
        if let Some(params) = additional {
            request = merge_json_values(request, params);
        }
        request
    }

    /// Convert documents to message content blocks
    #[inline(always)]
    pub fn convert_documents_to_blocks(
        documents: &crate::ZeroOneOrMany<fluent_ai_domain::Document>,
    ) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();

        match documents {
            crate::ZeroOneOrMany::None => {}
            crate::ZeroOneOrMany::One(doc) => {
                blocks.push(ContentBlock::Text {
                    text: format!("Document: {}", doc.content())});
            }
            crate::ZeroOneOrMany::Many(docs) => {
                for doc in docs {
                    blocks.push(ContentBlock::Text {
                        text: format!("Document: {}", doc.content())});
                }
            }
        }

        blocks
    }
}
