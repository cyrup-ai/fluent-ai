use fluent_ai_candle::model::loading::compatibility::*;
use fluent_ai_candle::model::loading::*;

use candle_core::Device;
    
    #[test]
    fn test_check_system_requirements() {
        let mut metadata = ModelMetadata::default();
        let device = Device::Cpu;
        
        // Test with no requirements
        assert!(check_system_requirements(&metadata, &device).is_ok());
        
        // Test with device requirement
        metadata.config.insert("device_type".to_string(), "cpu".to_string());
        assert!(check_system_requirements(&metadata, &device).is_ok());
        
        // Test with incompatible device
        metadata.config.insert("device_type".to_string(), "cuda".to_string());
        assert!(check_system_requirements(&metadata, &device).is_err());
    }
