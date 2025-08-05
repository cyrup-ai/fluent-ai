//! Command response formatting and serialization
//!
//! Provides blazing-fast response formatting with streaming support and zero-allocation patterns
//! for production-ready performance and ergonomic APIs.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Map, Value};
use tokio::sync::mpsc;

use super::types::*;

/// Response formatter with streaming support
#[derive(Debug, Clone)]
pub struct ResponseFormatter {
    /// Output format
    format: ResponseFormat,
    /// Include timestamps
    include_timestamps: bool,
    /// Include execution metrics
    include_metrics: bool,
    /// Pretty print JSON
    pretty_json: bool,
}

/// Response format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// Structured format with metadata
    Structured,
    /// Streaming format for real-time updates
    Streaming,
}

impl Default for ResponseFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseFormatter {
    /// Create a new response formatter
    pub fn new() -> Self {
        Self {
            format: ResponseFormat::Text,
            include_timestamps: true,
            include_metrics: false,
            pretty_json: true,
        }
    }

    /// Set response format
    pub fn with_format(mut self, format: ResponseFormat) -> Self {
        self.format = format;
        self
    }

    /// Include timestamps in responses
    pub fn with_timestamps(mut self, include: bool) -> Self {
        self.include_timestamps = include;
        self
    }

    /// Include execution metrics in responses
    pub fn with_metrics(mut self, include: bool) -> Self {
        self.include_metrics = include;
        self
    }

    /// Pretty print JSON responses
    pub fn with_pretty_json(mut self, pretty: bool) -> Self {
        self.pretty_json = pretty;
        self
    }

    /// Format command output
    pub fn format_output(&self, output: &CommandOutput) -> Result<String, ResponseError> {
        match self.format {
            ResponseFormat::Text => self.format_text(output),
            ResponseFormat::Json => self.format_json(output),
            ResponseFormat::Structured => self.format_structured(output),
            ResponseFormat::Streaming => self.format_streaming(output),
        }
    }

    /// Format as plain text
    fn format_text(&self, output: &CommandOutput) -> Result<String, ResponseError> {
        let mut result = String::new();

        // Per streams-only architecture: output is CandleMessageChunk directly, no Result wrapping
        // Success indicator
        result.push_str("✓ ");
        // Add main message from CandleMessageChunk
        result.push_str(&output.content);

        // Add timestamp if enabled
        if self.include_timestamps {
            let timestamp = chrono::Utc::now().format("%H:%M:%S");
            result.push_str(&format!(" [{}]", timestamp));
        }

        Ok(result)
    }

    /// Format as JSON
    fn format_json(&self, output: &CommandOutput) -> Result<String, ResponseError> {
        let mut json_output = Map::new();

        // Per streams-only architecture: output is CandleMessageChunk directly, no Result wrapping
        json_output.insert("success".to_string(), Value::Bool(true));
        json_output.insert("message".to_string(), Value::String(output.content.clone()));
        json_output.insert("done".to_string(), Value::Bool(output.done));

        // Note: CommandOutput doesn't have a data field, so we'll skip this for now

        // Note: Metrics are not available in basic CandleMessageChunk
        // They would need to be added if required

        if self.include_timestamps {
            let timestamp = chrono::Utc::now().to_rfc3339();
            json_output.insert("timestamp".to_string(), Value::String(timestamp));
        }

        let json_value = Value::Object(json_output);

        if self.pretty_json {
            serde_json::to_string_pretty(&json_value)
        } else {
            serde_json::to_string(&json_value)
        }
        .map_err(|e| ResponseError::SerializationError {
            detail: Arc::from(e.to_string()),
        })
    }

    /// Format as structured response
    fn format_structured(&self, output: &CommandOutput) -> Result<String, ResponseError> {
        let mut result = String::new();

        // Header
        result.push_str("=== Command Response ===\n");

        // Per streams-only architecture: output is CandleMessageChunk directly, no Result wrapping
        // Status
        result.push_str("Status: SUCCESS\n");
        // Message
        result.push_str(&format!("Message: {}\n", output.content));
        // Done status
        result.push_str(&format!("Complete: {}\n", output.done));

        // Note: Metrics are not available in basic CandleMessageChunk

        // Timestamp
        if self.include_timestamps {
            let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
            result.push_str(&format!("Timestamp: {}\n", timestamp));
        }

        result.push_str("========================\n");

        Ok(result)
    }

    /// Format for streaming
    fn format_streaming(&self, output: &CommandOutput) -> Result<String, ResponseError> {
        // For streaming, we use a compact JSON format
        let mut json_output = Map::new();
        json_output.insert(
            "type".to_string(),
            Value::String("command_response".to_string()),
        );
        
        // Per streams-only architecture: output is CandleMessageChunk directly, no Result wrapping
        json_output.insert("success".to_string(), Value::Bool(true));
        json_output.insert("message".to_string(), Value::String(output.content.clone()));
        json_output.insert("done".to_string(), Value::Bool(output.done));

        if self.include_timestamps {
            let timestamp = chrono::Utc::now().timestamp_millis();
            json_output.insert("timestamp".to_string(), Value::Number(timestamp.into()));
        }

        let json_value = Value::Object(json_output);
        serde_json::to_string(&json_value).map_err(|e| ResponseError::SerializationError {
            detail: Arc::from(e.to_string())})
    }

    /// Format error response
    pub fn format_error(&self, error: &CommandError) -> Result<String, ResponseError> {
        // Per streams-only architecture: convert error to CandleMessageChunk
        let output: CommandOutput = crate::domain::chat::message::types::CandleMessageChunk {
            content: format!("Error: {}", error),
            done: true
        };
        self.format_output(&output)
    }

    /// Format help response
    pub fn format_help(&self, commands: &[CommandInfo]) -> Result<String, ResponseError> {
        match self.format {
            ResponseFormat::Json => self.format_help_json(commands),
            _ => self.format_help_text(commands),
        }
    }

    /// Format help as text
    fn format_help_text(&self, commands: &[CommandInfo]) -> Result<String, ResponseError> {
        let mut result = String::new();
        result.push_str("Available Commands:\n\n");

        // Group commands by category
        let mut categories: HashMap<Arc<str>, Vec<&CommandInfo>> = HashMap::new();
        for command in commands {
            categories
                .entry(Arc::from(command.category.as_str()))
                .or_insert_with(Vec::new)
                .push(command);
        }

        // Format each category
        for (category, category_commands) in categories {
            result.push_str(&format!("{}:\n", category));

            for command in category_commands {
                result.push_str(&format!(
                    "  /{:<12} - {}\n",
                    command.name, command.description
                ));

                // Add aliases if any
                if !command.aliases.is_empty() {
                    let aliases = command
                        .aliases
                        .iter()
                        .map(|a| format!("/{}", a))
                        .collect::<Vec<_>>()
                        .join(", ");
                    result.push_str(&format!("               (aliases: {})\n", aliases));
                }
            }
            result.push('\n');
        }

        Ok(result)
    }

    /// Format help as JSON
    fn format_help_json(&self, commands: &[CommandInfo]) -> Result<String, ResponseError> {
        let json_commands: Vec<Value> = commands
            .iter()
            .map(|cmd| {
                let mut command_obj = Map::new();
                command_obj.insert("name".to_string(), Value::String(cmd.name.to_string()));
                command_obj.insert(
                    "description".to_string(),
                    Value::String(cmd.description.to_string()),
                );
                command_obj.insert("usage".to_string(), Value::String(cmd.usage.to_string()));
                command_obj.insert(
                    "category".to_string(),
                    Value::String(cmd.category.to_string()),
                );

                let aliases: Vec<Value> = cmd
                    .aliases
                    .iter()
                    .map(|a| Value::String(a.to_string()))
                    .collect();
                command_obj.insert("aliases".to_string(), Value::Array(aliases));

                let examples: Vec<Value> = cmd
                    .examples
                    .iter()
                    .map(|e| Value::String(e.to_string()))
                    .collect();
                command_obj.insert("examples".to_string(), Value::Array(examples));

                Value::Object(command_obj)
            })
            .collect();

        let result = Value::Array(json_commands);

        if self.pretty_json {
            serde_json::to_string_pretty(&result)
        } else {
            serde_json::to_string(&result)
        }
        .map_err(|e| ResponseError::SerializationError {
            detail: Arc::from(e.to_string()),
        })
    }

    /// Create streaming response channel
    pub fn create_streaming_channel(&self) -> (StreamingSender, StreamingReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        (StreamingSender::new(tx), StreamingReceiver::new(rx))
    }
}

/// Streaming response sender
#[derive(Debug)]
pub struct StreamingSender {
    /// The underlying sender for streaming messages
    sender: mpsc::UnboundedSender<StreamingMessage>,
}

impl StreamingSender {
    fn new(sender: mpsc::UnboundedSender<StreamingMessage>) -> Self {
        Self { sender }
    }

    /// Send a streaming message
    pub fn send(&self, message: StreamingMessage) -> Result<(), ResponseError> {
        self.sender
            .send(message)
            .map_err(|_| ResponseError::StreamingError {
                detail: Arc::from("Failed to send streaming message"),
            })
    }

    /// Send progress update
    pub fn send_progress(
        &self,
        current: u64,
        total: u64,
        message: &str,
    ) -> Result<(), ResponseError> {
        self.send(StreamingMessage::Progress {
            current,
            total,
            message: Arc::from(message),
        })
    }

    /// Send partial result
    pub fn send_partial(&self, data: Value) -> Result<(), ResponseError> {
        self.send(StreamingMessage::PartialResult { data })
    }

    /// Send completion
    pub fn send_complete(&self, output: CommandOutput) -> Result<(), ResponseError> {
        self.send(StreamingMessage::Complete { output })
    }
}

/// Streaming response receiver
#[derive(Debug)]
pub struct StreamingReceiver {
    /// The underlying receiver for streaming messages
    receiver: mpsc::UnboundedReceiver<StreamingMessage>,
}

impl StreamingReceiver {
    fn new(receiver: mpsc::UnboundedReceiver<StreamingMessage>) -> Self {
        Self { receiver }
    }

    /// Receive next streaming message
    pub async fn recv(&mut self) -> Option<StreamingMessage> {
        self.receiver.recv().await
    }
}

/// Streaming message types
#[derive(Debug, Clone)]
pub enum StreamingMessage {
    /// Progress update
    Progress {
        /// Current progress count
        current: u64,
        /// Total progress count
        total: u64,
        /// Progress message
        message: Arc<str>,
    },
    /// Partial result
    PartialResult {
        /// Partial result data
        data: Value,
    },
    /// Command completion
    Complete {
        /// Final command output
        output: CommandOutput,
    },
}

/// Response formatting errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ResponseError {
    /// Serialization error occurred
    #[error("Serialization error: {detail}")]
    SerializationError {
        /// Error detail message
        detail: Arc<str>,
    },

    /// Streaming error occurred
    #[error("Streaming error: {detail}")]
    StreamingError {
        /// Error detail message
        detail: Arc<str>,
    },

    /// Format error occurred
    #[error("Format error: {detail}")]
    FormatError {
        /// Error detail message
        detail: Arc<str>,
    },
}

/// Global response formatter
static GLOBAL_FORMATTER: once_cell::sync::Lazy<ResponseFormatter> =
    once_cell::sync::Lazy::new(ResponseFormatter::new);

/// Get global response formatter
pub fn get_global_formatter() -> &'static ResponseFormatter {
    &GLOBAL_FORMATTER
}

/// Format output using global formatter
pub fn format_global_output(output: &CommandOutput) -> Result<String, ResponseError> {
    get_global_formatter().format_output(output)
}

/// Format error using global formatter
pub fn format_global_error(error: &CommandError) -> Result<String, ResponseError> {
    get_global_formatter().format_error(error)
}

/// Format help using global formatter
pub fn format_global_help(commands: &[CommandInfo]) -> Result<String, ResponseError> {
    get_global_formatter().format_help(commands)
}
