use fluent_ai_candle::types::candle_context::provider::mod::*;
use fluent_ai_candle::types::candle_context::provider::*;

use std::path::PathBuf;

    #[test]
    fn test_context_creation() {
        let context = Context::<File>::of("test.txt");
        assert!(matches!(context.source(), ContextSourceType::File(_)));
    }

    #[test]
    fn test_files_context() {
        let context = Context::<Files>::glob("**/*.rs");
        assert!(matches!(context.source(), ContextSourceType::Files(_)));
    }

    #[test]
    fn test_directory_context() {
        let context = Context::<Directory>::of(PathBuf::from("src"));
        assert!(matches!(context.source(), ContextSourceType::Directory(_)));
    }

    #[test]
    fn test_github_context() {
        let context = Context::<Github>::glob("/repo/**/*.rs");
        assert!(matches!(context.source(), ContextSourceType::Github(_)));
    }

    #[test]
    fn test_memory_node() {
        let node = MemoryNode::new("test".to_string(), "content".to_string())
            .with_metadata("key".to_string(), "value".to_string());
        assert_eq!(node.id, "test");
        assert_eq!(node.content, "content");
        assert!(node.metadata.contains_key("key"));
    }
