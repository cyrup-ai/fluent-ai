//! Unit tests for command types and functionality
//!
//! Comprehensive test suite for command parsing, validation, and execution
//! with focus on zero-allocation patterns and performance.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::candle_chat::commands::types::{
        CommandParser, ImmutableChatCommand, SearchScope, StreamingCommandExecutor};

    #[test]
    fn test_command_parsing() {
        let cmd = CommandParser::parse_command("/help").unwrap();
        assert_eq!(cmd.command_name(), "help");

        let cmd = CommandParser::parse_command("clear --confirm").unwrap();
        assert_eq!(cmd.command_name(), "clear");

        let cmd = CommandParser::parse_command("export json --output test.json").unwrap();
        assert_eq!(cmd.command_name(), "export");
    }

    #[test]
    fn test_command_validation() {
        let cmd = ImmutableChatCommand::Search {
            query: "test".to_string(),
            scope: SearchScope::All,
            limit: None,
            include_context: false};
        assert!(cmd.validate().is_ok());

        let cmd = ImmutableChatCommand::Search {
            query: "".to_string(),
            scope: SearchScope::All,
            limit: None,
            include_context: false};
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_executor_stats() {
        let executor = StreamingCommandExecutor::new();
        let stats = executor.stats();
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }
}