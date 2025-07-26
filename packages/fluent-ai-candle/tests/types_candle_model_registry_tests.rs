use fluent_ai_candle::types::candle_model::registry::*;
use fluent_ai_candle::types::candle_model::*;

use crate::types::candle_model::info::ModelInfoBuilder;

    struct TestModel {
        info: &'static ModelInfo}

    impl Model for TestModel {
        fn info(&self) -> &'static ModelInfo {
            self.info
        }
    }

    #[test]
    fn test_register_and_get() {
        let registry = ModelRegistry::new();

        let info = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let model = TestModel { info: &info };

        // Register the model
        let registered = registry.register("test-provider", model).unwrap();

        // Retrieve the model
        let retrieved = registry
            .get_required::<TestModel>("test-provider", "test-model")
            .unwrap();

        assert_eq!(registered.info().name(), retrieved.info().name());
        assert_eq!(registered.info().provider(), retrieved.info().provider());
    }

    #[test]
    fn test_duplicate_registration() {
        let registry = ModelRegistry::new();

        let info1 = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let info2 = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let model1 = TestModel { info: &info1 };
        let model2 = TestModel { info: &info2 };

        // First registration should succeed
        registry.register("test-provider", model1).unwrap();

        // Second registration should fail
        let result = registry.register("test-provider", model2);
        assert!(matches!(
            result,
            Err(ModelError::ModelAlreadyExists {
                provider: "test-provider",
                name: "test-model"})
        ));
    }

    #[test]
    fn test_find_all() {
        let registry = ModelRegistry::new();

        let info1 = ModelInfo::builder()
            .provider_name("test1")
            .name("model1")
            .build()
            .unwrap();

        let info2 = ModelInfo::builder()
            .provider_name("test2")
            .name("model2")
            .build()
            .unwrap();

        let model1 = TestModel { info: &info1 };
        let model2 = TestModel { info: &info2 };

        // Register both models
        registry.register("test-provider1", model1).unwrap();
        registry.register("test-provider2", model2).unwrap();

        // Find all models of type TestModel
        let models = registry.find_all::<TestModel>();

        assert_eq!(models.len(), 2);

        let model_names: Vec<_> = models.iter().map(|m| m.info().name()).collect();

        assert!(model_names.contains(&"model1"));
        assert!(model_names.contains(&"model2"));
    }

    #[test]
    fn test_model_builder() {
        let info = ModelInfo::builder()
            .provider_name("test")
            .name("test-model")
            .build()
            .unwrap();

        let model = TestModel { info: &info };

        // Create a model builder
        let builder = ModelBuilder::new("test-provider", model);

        // Register the model using the builder
        let registered = builder.register().unwrap();

        // Verify the model was registered
        let retrieved = ModelRegistry::new()
            .get_required::<TestModel>("test-provider", "test-model")
            .unwrap();

        assert_eq!(registered.info().name(), retrieved.info().name());
    }
