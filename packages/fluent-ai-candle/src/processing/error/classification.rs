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
