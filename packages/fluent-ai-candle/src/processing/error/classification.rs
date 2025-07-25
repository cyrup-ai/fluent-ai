//! Error Classification Module
//!
//! Provides error categorization and severity levels for monitoring, metrics, and error handling.
//! Zero-allocation enum-based classification with comprehensive Display implementations.

/// Error category for monitoring and metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Configuration,
    Context,
    Numerical,
    Resource,
    External,
    ProcessorChain,
    Validation,
    Internal}

impl ErrorCategory {
    /// Get category name as string
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Configuration => "configuration",
            Self::Context => "context",
            Self::Numerical => "numerical",
            Self::Resource => "resource",
            Self::External => "external",
            Self::ProcessorChain => "processor_chain",
            Self::Validation => "validation",
            Self::Internal => "internal"}
    }
}

impl std::fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Error severity levels for prioritization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4}

impl ErrorSeverity {
    /// Get severity name as string
    #[inline(always)]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical"}
    }

    /// Get severity level as number
    #[inline(always)]
    pub fn level(&self) -> u8 {
        *self as u8
    }
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}