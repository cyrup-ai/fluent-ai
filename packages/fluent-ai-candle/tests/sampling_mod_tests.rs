use fluent_ai_candle::sampling::mod::*;
use fluent_ai_candle::sampling::*;

#[test]
    fn test_sampling_configuration_validation() {
        // Valid configuration
        let config = SamplingConfig::new()
            .temperature(0.8)
            .expect("valid temperature")
            .top_p(0.9)
            .expect("valid top-p")
            .top_k(50)
            .expect("valid top-k");

        assert!((config.temperature - 0.8).abs() < f64::EPSILON);
        assert!((config.top_p.unwrap() - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.top_k.unwrap(), 50);
    }

    #[test]
    fn test_invalid_parameters() {
        // Invalid temperature
        assert!(matches!(
            SamplingConfig::new().temperature(0.0),
            Err(SamplingError::InvalidTemperature(0.0))
        ));

        // Invalid top-p
        assert!(matches!(
            SamplingConfig::new().top_p(1.5),
            Err(SamplingError::InvalidTopP(1.5))
        ));

        // Invalid top-k
        assert!(matches!(
            SamplingConfig::new().top_k(0),
            Err(SamplingError::InvalidTopK(0))
        ));
    }

    #[test]
    fn test_builder_pattern() {
        let builder = LogitsProcessorBuilder::new();
        let _processor = builder
            .temperature(0.8)
            .expect("valid temperature")
            .top_k(40)
            .expect("valid top-k")
            .build();
    }

    #[test]
    fn test_canonical_sampling_integration() {
        let config = SamplingConfig::new()
            .temperature(0.8)
            .expect("valid temperature")
            .top_k(40)
            .expect("valid top-k");

        let sampling = config.build_sampling();
        match sampling {
            Sampling::TopK { k, temperature } => {
                assert_eq!(k, 40);
                assert!((temperature - 0.8).abs() < f64::EPSILON);
            }
            _ => panic!("Expected TopK sampling")}

        let _processor = config.build_processor();
    }

    #[test]
    fn test_utils_argsort() {
        let values = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let sorted_indices = utils::argsort_descending(&values);

        // Should be sorted in descending order: 0.9, 0.8, 0.3, 0.2, 0.1
        assert_eq!(sorted_indices, vec![3, 1, 2, 4, 0]);
    }
