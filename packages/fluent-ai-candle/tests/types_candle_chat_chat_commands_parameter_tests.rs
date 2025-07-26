use fluent_ai_candle::types::candle_chat::chat::commands::parameter::*;
use fluent_ai_candle::types::candle_chat::chat::commands::*;

#[test]
    fn test_parameter_type_as_str() {
        assert_eq!(ParameterType::String.as_str(), "string");
        assert_eq!(ParameterType::Integer.as_str(), "integer");
        assert_eq!(ParameterType::Boolean.as_str(), "boolean");
    }

    #[test]
    fn test_parameter_validation() {
        let int_param = ParameterInfo::required("test", ParameterType::Integer, "Test integer")
            .with_range(Some(1.0), Some(10.0));
        
        assert!(int_param.validate("5").is_ok());
        assert!(int_param.validate("0").is_err());
        assert!(int_param.validate("11").is_err());
        assert!(int_param.validate("not_a_number").is_err());
    }

    #[test]
    fn test_enum_parameter() {
        let enum_param = ParameterInfo::enum_param(
            "mode",
            "Operation mode",
            vec!["fast".to_string(), "slow".to_string(), "auto".to_string()],
            true,
        );
        
        assert!(enum_param.validate("fast").is_ok());
        assert!(enum_param.validate("invalid").is_err());
    }
