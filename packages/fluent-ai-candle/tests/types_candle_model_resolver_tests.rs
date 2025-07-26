use fluent_ai_candle::types::candle_model::resolver::*;
use fluent_ai_candle::types::candle_model::*;

use crate::types::candle_model::info::ModelInfoBuilder;

    struct TestModel {
        info: &'static ModelInfo}

    impl crate::types::Model for TestModel {
        fn info(&self) -> &'static ModelInfo {
            self.info
        }
    }

    #[test]
    fn test_pattern_matching() {
        // Test exact matching
        let exact = ModelPattern::Exact("gpt-4".to_string());
        assert!(exact.matches("gpt-4"));
        assert!(!exact.matches("gpt-3"));

        // Test glob pattern matching
        let glob = ModelPattern::Pattern("gpt-*".to_string());
        assert!(glob.matches("gpt-4"));
        assert!(glob.matches("gpt-3.5"));
        assert!(!glob.matches("claude-2"));

        // Test regex pattern matching
        let regex = ModelPattern::Regex(r"^gpt-\d+$".to_string());
        assert!(regex.matches("gpt-4"));
        assert!(regex.matches("gpt-35"));
        assert!(!regex.matches("gpt-3.5"));
        assert!(!regex.matches("claude-2"));
    }

    #[test]
    fn test_resolution_rules() {
        let mut resolver = ModelResolver::new();

        // Add a resolution rule
        resolver.add_rule(ModelResolutionRule {
            pattern: ModelPattern::Pattern("gpt-*".to_string()),
            provider: "openai".to_string(),
            target: "gpt-3.5-turbo".to_string(),
            priority: 10,
            condition: None});

        // Add an alias
        resolver.add_alias("chat", "openai", "gpt-3.5-turbo");

        // Test resolution with a rule
        let resolution_stream = resolver.resolve::<TestModel>("gpt-4", None);
        let resolution = resolution_stream.collect().into_iter().next().unwrap();
        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");

        // Test resolution with an alias
        let resolution_stream = resolver.resolve::<TestModel>("chat", None);
        let resolution = resolution_stream.collect().into_iter().next().unwrap();
        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");
    }

    #[test]
    fn test_fuzzy_matching() {
        let mut registry = ModelRegistry::new();

        // Register some test models
        static INFO1: Lazy<ModelInfo> = Lazy::new(|| {
            ModelInfo::builder()
                .provider_name("openai")
                .name("gpt-3.5-turbo")
                .build()
                .unwrap()
        });

        static INFO2: Lazy<ModelInfo> = Lazy::new(|| {
            ModelInfo::builder()
                .provider_name("anthropic")
                .name("claude-2")
                .build()
                .unwrap()
        });

        let model1 = TestModel { info: &*INFO1 };
        let model2 = TestModel { info: &*INFO2 };

        registry.register("openai", model1).unwrap();
        registry.register("anthropic", model2).unwrap();

        let resolver = ModelResolver::new();

        // Test fuzzy matching with a typo
        let resolution = resolver
            .resolve_with_registry::<TestModel>(&registry, "gpt-3.5-turbo-16k", None)
            .unwrap();

        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");
        assert!(resolution.score > 0.7);

        // Test fuzzy matching with a different casing
        let resolution = resolver
            .resolve_with_registry::<TestModel>(&registry, "GPT-3.5-TURBO", None)
            .unwrap();

        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");
        assert!(resolution.score > 0.7);
    }
