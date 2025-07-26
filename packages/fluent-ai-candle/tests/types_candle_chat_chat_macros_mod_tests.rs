use fluent_ai_candle::types::candle_chat::chat::macros::mod::*;
use fluent_ai_candle::types::candle_chat::chat::macros::*;

use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn test_macro_validation() {
        let valid_macro = StoredMacro {
            metadata: MacroMetadata {
                id: uuid::Uuid::new_v4(),
                name: Arc::from("Test Macro"),
                description: None,
                tags: vec![],
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                modified_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                author: None,
                version: 1},
            actions: Arc::from([
                MacroAction::SendMessage {
                    content: Arc::from("Hello"),
                    message_type: Arc::from("text"),
                    timestamp: Duration::from_millis(0)}
            ]),
            variables: HashMap::new()};

        assert!(validate_macro(&valid_macro).is_ok());
    }

    #[test]
    fn test_environment_creation() {
        let (engine, handlers, parser) = create_default_environment();
        
        // Verify components are properly initialized
        assert!(engine.get_statistics().active_sessions == 0);
        assert!(!handlers.list_handlers().is_empty());
        // Parser doesn't have a simple validation method, but creation success is good
    }

    #[test]
    fn test_version_info() {
        let version = version_info();
        assert_eq!(version.version, "1.0.0");
        assert!(!version.features.is_empty());
    }
