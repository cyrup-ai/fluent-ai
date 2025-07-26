use fluent_ai_candle::*;
use fluent_ai_candle::*;

#[test]
    fn test_sampling_config_validation() {
        let config = SamplingConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = SamplingConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_legacy_to_unified_conversion() {
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ..Default::default()
        };

        let unified_processor = config.to_unified_processor();
        assert!(unified_processor.is_ok());
    }

    #[test]
    fn test_logits_sampler_creation() {
        let config = SamplingConfig::default();
        let sampler = LogitsSampler::new(1000, config);
        assert!(sampler.is_ok());
    }
