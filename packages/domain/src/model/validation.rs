//! Model validation and data integrity checking
//!
//! This module provides comprehensive validation for model configurations,
//! data integrity checks, and production-readiness verification.

use serde::{Deserialize, Serialize};

/// Result type for validation operations
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validation error types for detailed error reporting
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationError {
    /// Missing required field
    MissingField { field: String, model: String },

    /// Invalid value range
    InvalidRange {
        field: String,
        value: String,
        expected: String,
    },

    /// Inconsistent data between fields
    InconsistentData { description: String },

    /// Provider name format error
    InvalidProvider { provider: String },

    /// Model name format error
    InvalidModelName { name: String },

    /// Pricing validation error
    InvalidPricing { description: String },

    /// Capability configuration error
    InvalidCapability { description: String },

    /// Configuration safety error
    UnsafeConfiguration { description: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingField { field, model } => {
                write!(
                    f,
                    "Missing required field '{}' for model '{}'",
                    field, model
                )
            }
            ValidationError::InvalidRange {
                field,
                value,
                expected,
            } => {
                write!(
                    f,
                    "Invalid value '{}' for field '{}', expected {}",
                    value, field, expected
                )
            }
            ValidationError::InconsistentData { description } => {
                write!(f, "Inconsistent data: {}", description)
            }
            ValidationError::InvalidProvider { provider } => {
                write!(f, "Invalid provider name format: '{}'", provider)
            }
            ValidationError::InvalidModelName { name } => {
                write!(f, "Invalid model name format: '{}'", name)
            }
            ValidationError::InvalidPricing { description } => {
                write!(f, "Invalid pricing configuration: {}", description)
            }
            ValidationError::InvalidCapability { description } => {
                write!(f, "Capability validation error: {}", description)
            }
            ValidationError::UnsafeConfiguration { description } => {
                write!(f, "Unsafe configuration: {}", description)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Information only
    Info,

    /// Warning - may indicate potential issues
    Warning,

    /// Error - should be fixed before production
    Error,

    /// Critical error - must be fixed
    Critical,
}

/// Validation result with severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Description of the issue
    pub message: String,

    /// Severity level
    pub severity: ValidationSeverity,

    /// Related field (if applicable)
    pub field: Option<String>,

    /// Suggested fix (if any)
    pub suggestion: Option<String>,
}

/// Comprehensive validation report
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    /// All validation issues found
    issues: Vec<ValidationIssue>,

    /// Overall readiness score (0.0 to 1.0)
    readiness_score: f32,

    /// Whether the model is production-ready
    is_production_ready: bool,
}

impl ValidationReport {
    /// Create a new empty validation report
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
            readiness_score: 1.0,
            is_production_ready: true,
        }
    }

    /// Add a validation issue to the report
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        if issue.severity >= ValidationSeverity::Error {
            self.is_production_ready = false;
        }
        self.issues.push(issue);
    }

    /// Check if there are any critical errors
    pub fn has_critical_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == ValidationSeverity::Critical)
    }

    /// Check if there are any errors (Error or Critical)
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity >= ValidationSeverity::Error)
    }

    /// Get issues by severity level
    pub fn get_issues_by_severity(&self, severity: ValidationSeverity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == severity)
            .collect()
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let critical = self
            .get_issues_by_severity(ValidationSeverity::Critical)
            .len();
        let errors = self.get_issues_by_severity(ValidationSeverity::Error).len();
        let warnings = self
            .get_issues_by_severity(ValidationSeverity::Warning)
            .len();
        let info = self.get_issues_by_severity(ValidationSeverity::Info).len();

        format!(
            "Validation Report: {} issues ({} critical, {} errors, {} warnings, {} info). Readiness: {:.1}%{}",
            self.issues.len(),
            critical,
            errors,
            warnings,
            info,
            self.readiness_score * 100.0,
            if self.is_production_ready {
                ""
            } else {
                " - NOT PRODUCTION READY"
            }
        )
    }

    /// Update the readiness score
    pub fn update_readiness_score(&mut self, score: f32) {
        self.readiness_score = score.max(0.0).min(1.0);
        if self.readiness_score < 1.0 {
            self.is_production_ready = false;
        }
    }

    /// Check if the model is production-ready
    pub fn is_production_ready(&self) -> bool {
        self.is_production_ready
    }

    /// Get all issues
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }
}