use fluent_ai_candle::processing::error::classification::*;
use fluent_ai_candle::processing::error::*;

#[test]
    fn test_error_category_names() {
        assert_eq!(ErrorCategory::Configuration.name(), "configuration");
        assert_eq!(ErrorCategory::Context.name(), "context");
        assert_eq!(ErrorCategory::Numerical.name(), "numerical");
        assert_eq!(ErrorCategory::Resource.name(), "resource");
        assert_eq!(ErrorCategory::External.name(), "external");
        assert_eq!(ErrorCategory::ProcessorChain.name(), "processor_chain");
        assert_eq!(ErrorCategory::Validation.name(), "validation");
        assert_eq!(ErrorCategory::Internal.name(), "internal");
    }

    #[test]
    fn test_error_severity_levels() {
        assert_eq!(ErrorSeverity::Low.level(), 1);
        assert_eq!(ErrorSeverity::Medium.level(), 2);
        assert_eq!(ErrorSeverity::High.level(), 3);
        assert_eq!(ErrorSeverity::Critical.level(), 4);
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Low < ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium < ErrorSeverity::High);
        assert!(ErrorSeverity::High < ErrorSeverity::Critical);
    }

    #[test]
    fn test_category_display() {
        assert_eq!(format!("{}", ErrorCategory::Configuration), "configuration");
        assert_eq!(format!("{}", ErrorCategory::Validation), "validation");
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", ErrorSeverity::Low), "low");
        assert_eq!(format!("{}", ErrorSeverity::Critical), "critical");
    }
