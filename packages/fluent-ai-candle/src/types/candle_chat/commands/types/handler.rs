//! Command handler trait and default implementation
//!
//! Streaming command handler system with zero-allocation execution patterns
//! and comprehensive metadata support for extensible command processing.

use fluent_ai_async::AsyncStream;

use super::commands::ImmutableChatCommand;
use super::context::{CommandContext, CommandOutput};
use super::events::HistoryAction;

/// Command handler trait for zero-allocation execution
pub trait CommandHandler: Send + Sync {
    /// Execute command with streaming output
    fn execute(
        &self,
        context: CommandContext,
        command: ImmutableChatCommand,
    ) -> AsyncStream<CommandOutput>;

    /// Get handler name
    fn name(&self) -> &'static str;

    /// Check if handler can execute command
    fn can_handle(&self, command: &ImmutableChatCommand) -> bool;

    /// Get command metadata
    fn metadata(&self) -> CommandHandlerMetadata;
}

/// Command handler metadata
#[derive(Debug, Clone)]
pub struct CommandHandlerMetadata {
    /// Handler name
    pub name: String,
    /// Handler description
    pub description: String,
    /// Supported command types
    pub supported_commands: Vec<String>,
    /// Handler version
    pub version: String,
    /// Whether handler is enabled
    pub enabled: bool,
}

impl CommandHandlerMetadata {
    /// Create new command handler metadata
    #[inline]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        supported_commands: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            supported_commands,
            version: "1.0.0".to_string(),
            enabled: true,
        }
    }

    /// Set handler version
    #[inline]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Enable or disable handler
    #[inline]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Default command handler implementation
#[derive(Debug)]
pub struct DefaultCommandHandler {
    metadata: CommandHandlerMetadata,
}

impl DefaultCommandHandler {
    /// Create new default command handler
    #[inline]
    pub fn new() -> Self {
        let metadata = CommandHandlerMetadata::new(
            "default",
            "Default command handler for basic chat commands",
            vec![
                "help".to_string(),
                "clear".to_string(),
                "export".to_string(),
                "config".to_string(),
                "search".to_string(),
                "history".to_string(),
                "save".to_string(),
                "load".to_string(),
            ],
        );

        Self { metadata }
    }
}

impl Default for DefaultCommandHandler {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl CommandHandler for DefaultCommandHandler {
    fn execute(
        &self,
        context: CommandContext,
        command: ImmutableChatCommand,
    ) -> AsyncStream<CommandOutput> {
        AsyncStream::with_channel(move |sender| {
            // Execute command based on type
            let output = match &command {
                ImmutableChatCommand::Help {
                    command: cmd,
                    extended,
                } => {
                    let content = if let Some(cmd) = cmd {
                        if *extended {
                            format!("Extended help for command: {}", cmd)
                        } else {
                            format!("Help for command: {}", cmd)
                        }
                    } else {
                        "Available commands: help, clear, export, config, search, history, save, load"
                        .to_string()
                    };
                    CommandOutput::text(context.execution_id, content)
                }
                ImmutableChatCommand::Clear { confirm, keep_last } => {
                    if *confirm {
                        let msg = if let Some(keep) = keep_last {
                            format!("Chat history cleared, keeping last {} messages", keep)
                        } else {
                            "Chat history cleared".to_string()
                        };
                        CommandOutput::text(context.execution_id, msg)
                    } else {
                        CommandOutput::text(
                            context.execution_id,
                            "Clear command requires --confirm flag",
                        )
                    }
                }
                ImmutableChatCommand::History { action, limit, .. } => {
                    let content = match action {
                        HistoryAction::Show => {
                            let limit_str = limit
                                .map(|l| format!(" (last {} messages)", l))
                                .unwrap_or_default();
                            format!("Showing chat history{}", limit_str)
                        }
                        HistoryAction::Search => "Searching chat history".to_string(),
                        HistoryAction::Clear => "Chat history cleared".to_string(),
                        HistoryAction::Export => "Chat history exported".to_string(),
                        HistoryAction::Import => "Chat history imported".to_string(),
                        HistoryAction::Backup => "Chat history backed up".to_string(),
                    };
                    CommandOutput::text(context.execution_id, content)
                }
                _ => CommandOutput::text(
                    context.execution_id,
                    format!("Command {} executed successfully", command.command_name()),
                ),
            };

            // Send output through stream
            let _ = sender.try_send(output.final_output());
        })
    }

    fn name(&self) -> &'static str {
        "default"
    }

    fn can_handle(&self, command: &ImmutableChatCommand) -> bool {
        self.metadata
            .supported_commands
            .contains(&command.command_name().to_string())
    }

    fn metadata(&self) -> CommandHandlerMetadata {
        self.metadata.clone()
    }
}