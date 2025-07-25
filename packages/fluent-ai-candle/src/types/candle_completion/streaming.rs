//! Streaming response types for completion operations
//!
//! Contains types for handling streaming completion responses.

use serde::{Deserialize, Serialize};

/// Streaming response chunk from completion operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleStreamingResponse {
    /// Unique identifier for this completion
    pub id: String,

    /// Object type (always "chat.completion.chunk")
    pub object: String,

    /// Unix timestamp of when the completion was created
    pub created: u64,

    /// Model used for the completion
    pub model: String,

    /// List of completion choices
    pub choices: Vec<CandleStreamingChoice>,

    /// Token usage information (only present in final chunk)
    pub usage: Option<crate::types::CandleUsage>,

    /// System fingerprint for reproducibility
    pub system_fingerprint: Option<String>}

/// Individual choice in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleStreamingChoice {
    /// Index of this choice
    pub index: u32,

    /// Delta containing the incremental content
    pub delta: CandleStreamingDelta,

    /// Reason why the completion finished (if finished)
    pub finish_reason: Option<CandleFinishReason>,

    /// Log probabilities for the tokens (if requested)
    pub logprobs: Option<CandleLogProbs>}

/// Delta containing incremental content in streaming response
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleStreamingDelta {
    /// Role of the message (usually only in first chunk)
    pub role: Option<String>,

    /// Incremental content
    pub content: Option<String>,

    /// Tool calls (if any)
    pub tool_calls: Option<Vec<CandleToolCallDelta>>,

    /// Function call (deprecated, use tool_calls)
    pub function_call: Option<CandleFunctionCallDelta>}

/// Tool call delta in streaming response
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleToolCallDelta {
    /// Index of this tool call
    pub index: u32,

    /// Unique identifier for this tool call
    pub id: Option<String>,

    /// Type of tool call (usually "function")
    #[serde(rename = "type")]
    pub tool_type: Option<String>,

    /// Function call details
    pub function: Option<CandleFunctionCallDelta>}

/// Function call delta in streaming response
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleFunctionCallDelta {
    /// Name of the function (usually only in first chunk)
    pub name: Option<String>,

    /// Incremental arguments as JSON string
    pub arguments: Option<String>}

/// Reason why the completion finished
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CandleFinishReason {
    /// Natural completion
    Stop,
    /// Maximum length reached
    Length,
    /// Tool/function call requested
    ToolCalls,
    /// Function call requested (deprecated)
    FunctionCall,
    /// Content filtered
    ContentFilter,
    /// Model error
    Error}

/// Log probabilities for tokens
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleLogProbs {
    /// Content log probabilities
    pub content: Option<Vec<CandleTokenLogProb>>}

/// Log probability for a single token
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleTokenLogProb {
    /// The token
    pub token: String,

    /// Log probability of the token
    pub logprob: f64,

    /// Raw bytes of the token (if available)
    pub bytes: Option<Vec<u8>>,

    /// Top alternative tokens with their log probabilities
    pub top_logprobs: Vec<CandleTopLogProb>}

/// Top alternative token with log probability
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandleTopLogProb {
    /// The token
    pub token: String,

    /// Log probability of the token
    pub logprob: f64,

    /// Raw bytes of the token (if available)
    pub bytes: Option<Vec<u8>>}

impl CandleStreamingResponse {
    /// Create a new streaming response
    pub fn new(id: String, model: String, created: u64) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created,
            model,
            choices: Vec::new(),
            usage: None,
            system_fingerprint: None}
    }

    /// Add a choice to the response
    pub fn add_choice(&mut self, choice: CandleStreamingChoice) {
        self.choices.push(choice);
    }

    /// Check if this is the final chunk (has usage information)
    pub fn is_final(&self) -> bool {
        self.usage.is_some() || self.choices.iter().any(|c| c.finish_reason.is_some())
    }

    /// Get the content from the first choice, if any
    pub fn content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }

    /// Check if this chunk contains a tool call
    pub fn has_tool_calls(&self) -> bool {
        self.choices.iter().any(|c| c.delta.tool_calls.is_some())
    }
}

impl Default for CandleStreamingResponse {
    fn default() -> Self {
        Self {
            id: String::new(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: String::new(),
            choices: Vec::new(),
            usage: None,
            system_fingerprint: None}
    }
}

impl CandleStreamingChoice {
    /// Create a new streaming choice with content
    pub fn with_content(index: u32, content: String) -> Self {
        Self {
            index,
            delta: CandleStreamingDelta {
                role: None,
                content: Some(content),
                tool_calls: None,
                function_call: None},
            finish_reason: None,
            logprobs: None}
    }

    /// Create a new streaming choice with role
    pub fn with_role(index: u32, role: String) -> Self {
        Self {
            index,
            delta: CandleStreamingDelta {
                role: Some(role),
                content: None,
                tool_calls: None,
                function_call: None},
            finish_reason: None,
            logprobs: None}
    }

    /// Set the finish reason
    pub fn with_finish_reason(mut self, reason: CandleFinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }
}

impl Default for CandleStreamingDelta {
    fn default() -> Self {
        Self {
            role: None,
            content: None,
            tool_calls: None,
            function_call: None}
    }
}
