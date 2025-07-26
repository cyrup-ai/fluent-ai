use fluent_ai_candle::processing::processors::top_k::builder::*;
use fluent_ai_candle::processing::processors::top_k::*;

#[test]
    fn test_builder_new() {
        let builder = TopKBuilder::new();
        let processor = builder.build().unwrap();
        assert_eq!(processor.k(), 0); // Default to disabled
        assert!(processor.is_identity());
    }

    #[test]
    fn test_builder_k() {
        let processor = TopKBuilder::new().k(75).build().unwrap();
        assert_eq!(processor.k(), 75);
        assert!(!processor.is_identity());
    }

    #[test]
    fn test_builder_presets() {
        let small = TopKBuilder::new().small().build().unwrap();
        assert_eq!(small.k(), 20);
        
        let medium = TopKBuilder::new().medium().build().unwrap();
        assert_eq!(medium.k(), 50);
        
        let large = TopKBuilder::new().large().build().unwrap();
        assert_eq!(large.k(), 100);
    }

    #[test]
    fn test_builder_disabled() {
        let processor = TopKBuilder::new().disabled().build().unwrap();
        assert_eq!(processor.k(), 0);
        assert!(processor.is_identity());
    }

    #[test]
    fn test_builder_chaining() {
        // Later calls should override earlier ones
        let processor = TopKBuilder::new()
            .small()   // k=20
            .medium()  // k=50 (overrides)
            .k(30)     // k=30 (overrides)
            .build()
            .unwrap();
        assert_eq!(processor.k(), 30);
    }

    #[test]
    fn test_builder_validation() {
        let result = TopKBuilder::new()
            .k(fluent_ai_candle::fluent_ai_candle::core::MAX_TOP_K + 1)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_default() {
        let builder = TopKBuilder::default();
        let processor = builder.build().unwrap();
        assert_eq!(processor.k(), 0);
    }
