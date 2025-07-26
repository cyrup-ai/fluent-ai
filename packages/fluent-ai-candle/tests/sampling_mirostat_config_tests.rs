use fluent_ai_candle::sampling::mirostat::config::*;
use fluent_ai_candle::sampling::mirostat::*;

#[test]
    fn test_config_creation() {
        let v1 = MirostatConfig::v1(5.0, 0.1).unwrap();
        assert_eq!(v1.tau(), 5.0);
        assert_eq!(v1.variant_name(), "Mirostat v1");

        let v2 = MirostatConfig::v2(5.0, 0.1).unwrap();
        assert_eq!(v2.tau(), 5.0);
        assert_eq!(v2.variant_name(), "Mirostat v2");
    }

    #[test]
    fn test_config_validation() {
        assert!(MirostatConfig::v1(-1.0, 0.1).is_err());
        assert!(MirostatConfig::v1(5.0, -1.0).is_err());
        assert!(MirostatConfig::v2(-1.0, 0.1).is_err());
        assert!(MirostatConfig::v2(5.0, -1.0).is_err());
    }

    #[test]
    fn test_default_config() {
        let config = MirostatConfig::default();
        assert_eq!(config.tau(), 5.0);
        assert_eq!(config.variant_name(), "Mirostat v1");
    }
