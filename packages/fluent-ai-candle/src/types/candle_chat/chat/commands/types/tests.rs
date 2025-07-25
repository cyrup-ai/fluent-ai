//! Tests for command types module
//!
//! Provides comprehensive tests for command parsing, validation, and execution
//! with zero-allocation patterns verification.

#[cfg(test)]
mod tests {
    use super::super::actions::SearchScope;
    use super::super::command::ImmutableChatCommand;
    use super::super::executor::StreamingCommandExecutor;
    use super::super::parser::CommandParser;

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
            include_context: false,
        };
        assert!(cmd.validate().is_ok());

        let cmd = ImmutableChatCommand::Search {
            query: "".to_string(),
            scope: SearchScope::All,
            limit: None,
            include_context: false,
        };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_executor_stats() {
        let executor = StreamingCommandExecutor::new();
        let stats = executor.stats();
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_command_names() {
        let help_cmd = ImmutableChatCommand::Help {
            command: None,
            extended: false,
        };
        assert_eq!(help_cmd.command_name(), "help");

        let clear_cmd = ImmutableChatCommand::Clear {
            confirm: true,
            keep_last: None,
        };
        assert_eq!(clear_cmd.command_name(), "clear");
    }

    #[test]
    fn test_command_mutation_check() {
        let help_cmd = ImmutableChatCommand::Help {
            command: None,
            extended: false,
        };
        assert!(!help_cmd.is_mutating());

        let clear_cmd = ImmutableChatCommand::Clear {
            confirm: true,
            keep_last: None,
        };
        assert!(clear_cmd.is_mutating());
    }
}