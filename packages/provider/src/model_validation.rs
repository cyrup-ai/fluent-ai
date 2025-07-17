//! Model validation and data integrity checking
//! 
//! This module provides comprehensive validation for model configurations,
//! data integrity checks, and production-readiness verification.

use crate::model_capabilities::{ModelInfoData, Capability};
use crate::completion_provider::ModelConfig;
use std::collections::HashSet;

/// Result type for validation operations
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validation error types for detailed error reporting
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Missing required field
    MissingField { field: String, model: String },
    
    /// Invalid value range
    InvalidRange { field: String, value: String, expected: String },
    
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
                write!(f, "Missing required field '{}' for model '{}'", field, model)
            }
            ValidationError::InvalidRange { field, value, expected } => {
                write!(f, "Invalid value '{}' for field '{}', expected {}", value, field, expected)
            }
            ValidationError::InconsistentData { description } => {
                write!(f, "Data inconsistency: {}", description)
            }
            ValidationError::InvalidProvider { provider } => {
                write!(f, "Invalid provider name: '{}'", provider)
            }
            ValidationError::InvalidModelName { name } => {
                write!(f, "Invalid model name format: '{}'", name)
            }
            ValidationError::InvalidPricing { description } => {
                write!(f, "Pricing validation error: {}", description)
            }
            ValidationError::InvalidCapability { description } => {
                write!(f, "Capability validation error: {}", description)
            }
            ValidationError::UnsafeConfiguration { description } => {
                write!(f, "Unsafe configuration detected: {}", description)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Informational message
    Info,
    
    /// Warning that should be addressed
    Warning,
    
    /// Error that must be fixed
    Error,
    
    /// Critical error that prevents operation
    Critical,
}

/// Validation result with severity
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue
    pub severity: ValidationSeverity,
    
    /// Error details
    pub error: ValidationError,
    
    /// Suggested fix for the issue
    pub suggested_fix: Option<String>,
}

/// Comprehensive validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Model name being validated
    pub model_name: String,
    
    /// All validation issues found
    pub issues: Vec<ValidationIssue>,
    
    /// Overall validation status
    pub is_valid: bool,
    
    /// Production readiness score (0.0 to 1.0)
    pub readiness_score: f32,
}

impl ValidationReport {
    /// Check if there are any critical errors
    pub fn has_critical_errors(&self) -> bool {
        self.issues.iter().any(|issue| issue.severity == ValidationSeverity::Critical)
    }
    
    /// Check if there are any errors (Error or Critical)
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|issue| matches!(issue.severity, ValidationSeverity::Error | ValidationSeverity::Critical))
    }
    
    /// Get issues by severity level
    pub fn get_issues_by_severity(&self, severity: ValidationSeverity) -> Vec<&ValidationIssue> {
        self.issues.iter().filter(|issue| issue.severity == severity).collect()
    }
    
    /// Generate a summary report
    pub fn summary(&self) -> String {
        let critical_count = self.get_issues_by_severity(ValidationSeverity::Critical).len();
        let error_count = self.get_issues_by_severity(ValidationSeverity::Error).len();
        let warning_count = self.get_issues_by_severity(ValidationSeverity::Warning).len();
        let info_count = self.get_issues_by_severity(ValidationSeverity::Info).len();
        
        format!(
            "Model '{}' validation: {} critical, {} errors, {} warnings, {} info. Readiness: {:.1}%",
            self.model_name,
            critical_count,
            error_count,
            warning_count,
            info_count,
            self.readiness_score * 100.0
        )
    }
}

/// Validate ModelInfoData for production readiness
/// 
/// # Arguments
/// * `model_info` - Model information to validate
/// 
/// # Returns
/// * ValidationReport with all issues found
pub fn validate_model_info(model_info: &ModelInfoData) -> ValidationReport {
    let mut issues = Vec::new();
    
    // Validate required fields
    validate_required_fields(model_info, &mut issues);
    
    // Validate field ranges and formats
    validate_field_ranges(model_info, &mut issues);
    
    // Validate data consistency
    validate_data_consistency(model_info, &mut issues);
    
    // Validate pricing information
    validate_pricing(model_info, &mut issues);
    
    // Validate capabilities
    validate_capabilities(model_info, &mut issues);
    
    // Calculate overall validation status
    let has_critical_or_error = issues.iter().any(|issue| {
        matches!(issue.severity, ValidationSeverity::Critical | ValidationSeverity::Error)
    });
    
    let readiness_score = calculate_readiness_score(&issues);
    
    ValidationReport {
        model_name: model_info.name.clone(),
        issues,
        is_valid: !has_critical_or_error,
        readiness_score,
    }
}

/// Validate ModelConfig for production safety
/// 
/// # Arguments
/// * `config` - Model configuration to validate
/// 
/// # Returns
/// * ValidationReport with configuration issues
pub fn validate_model_config(config: &ModelConfig) -> ValidationReport {
    let mut issues = Vec::new();
    
    // Validate parameter ranges
    validate_config_parameters(config, &mut issues);
    
    // Validate production safety
    validate_production_safety(config, &mut issues);
    
    // Validate provider configuration
    validate_provider_config(config, &mut issues);
    
    let has_critical_or_error = issues.iter().any(|issue| {
        matches!(issue.severity, ValidationSeverity::Critical | ValidationSeverity::Error)
    });
    
    let readiness_score = calculate_readiness_score(&issues);
    
    ValidationReport {
        model_name: config.model_name.to_string(),
        issues,
        is_valid: !has_critical_or_error,
        readiness_score,
    }
}

/// Validate a collection of models for consistency
/// 
/// # Arguments
/// * `models` - Collection of model info data
/// 
/// # Returns
/// * ValidationReport for the entire collection
pub fn validate_model_collection(models: &[ModelInfoData]) -> ValidationReport {
    let mut issues = Vec::new();
    
    // Check for duplicate model names
    validate_unique_names(models, &mut issues);
    
    // Check for provider consistency
    validate_provider_consistency(models, &mut issues);
    
    // Check for pricing consistency within providers
    validate_pricing_consistency(models, &mut issues);
    
    // Check for capability patterns
    validate_capability_patterns(models, &mut issues);
    
    let has_critical_or_error = issues.iter().any(|issue| {
        matches!(issue.severity, ValidationSeverity::Critical | ValidationSeverity::Error)
    });
    
    let readiness_score = calculate_readiness_score(&issues);
    
    ValidationReport {
        model_name: format!("Collection ({} models)", models.len()),
        issues,
        is_valid: !has_critical_or_error,
        readiness_score,
    }
}

// Private validation functions

fn validate_required_fields(model_info: &ModelInfoData, issues: &mut Vec<ValidationIssue>) {
    if model_info.provider_name.is_empty() {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Critical,
            error: ValidationError::MissingField {
                field: "provider_name".to_string(),
                model: model_info.name.clone(),
            },
            suggested_fix: Some("Set a valid provider name (e.g., 'openai', 'anthropic')".to_string()),
        });
    }
    
    if model_info.name.is_empty() {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Critical,
            error: ValidationError::MissingField {
                field: "name".to_string(),
                model: "unknown".to_string(),
            },
            suggested_fix: Some("Set a valid model name".to_string()),
        });
    }
}

fn validate_field_ranges(model_info: &ModelInfoData, issues: &mut Vec<ValidationIssue>) {
    // Validate token limits
    if let Some(max_input) = model_info.max_input_tokens {
        if max_input == 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                error: ValidationError::InvalidRange {
                    field: "max_input_tokens".to_string(),
                    value: "0".to_string(),
                    expected: "> 0".to_string(),
                },
                suggested_fix: Some("Set a positive value for maximum input tokens".to_string()),
            });
        } else if max_input > 10_000_000 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                error: ValidationError::InvalidRange {
                    field: "max_input_tokens".to_string(),
                    value: max_input.to_string(),
                    expected: "<= 10,000,000".to_string(),
                },
                suggested_fix: Some("Verify this extremely large context length is correct".to_string()),
            });
        }
    }
    
    if let Some(max_output) = model_info.max_output_tokens {
        if max_output == 0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                error: ValidationError::InvalidRange {
                    field: "max_output_tokens".to_string(),
                    value: "0".to_string(),
                    expected: "> 0".to_string(),
                },
                suggested_fix: Some("Set a positive value for maximum output tokens".to_string()),
            });
        }
    }
    
    // Validate pricing
    if let Some(input_price) = model_info.input_price {
        if input_price < 0.0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                error: ValidationError::InvalidPricing {
                    description: format!("Negative input price: {}", input_price),
                },
                suggested_fix: Some("Set a non-negative price value".to_string()),
            });
        } else if input_price > 1000.0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                error: ValidationError::InvalidPricing {
                    description: format!("Extremely high input price: ${}", input_price),
                },
                suggested_fix: Some("Verify this price is correct (per 1M tokens)".to_string()),
            });
        }
    }
    
    if let Some(output_price) = model_info.output_price {
        if output_price < 0.0 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                error: ValidationError::InvalidPricing {
                    description: format!("Negative output price: {}", output_price),
                },
                suggested_fix: Some("Set a non-negative price value".to_string()),
            });
        }
    }
}

fn validate_data_consistency(model_info: &ModelInfoData, issues: &mut Vec<ValidationIssue>) {
    // Check if output price is typically higher than input price
    if let (Some(input_price), Some(output_price)) = (model_info.input_price, model_info.output_price) {
        if output_price < input_price && input_price > 0.1 {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                error: ValidationError::InconsistentData {
                    description: format!(
                        "Output price (${}) is lower than input price (${}), which is unusual",
                        output_price, input_price
                    ),
                },
                suggested_fix: Some("Verify pricing is correct for this model".to_string()),
            });
        }
    }
    
    // Check if thinking models have appropriate budgets
    if model_info.supports_thinking == Some(true) {
        if let Some(budget) = model_info.optimal_thinking_budget {
            if budget < 100 {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    error: ValidationError::InconsistentData {
                        description: "Thinking model has very low optimal thinking budget".to_string(),
                    },
                    suggested_fix: Some("Consider increasing thinking budget for better performance".to_string()),
                });
            }
        } else {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                error: ValidationError::MissingField {
                    field: "optimal_thinking_budget".to_string(),
                    model: model_info.name.clone(),
                },
                suggested_fix: Some("Set optimal thinking budget for thinking-capable model".to_string()),
            });
        }
    }
}

fn validate_pricing(model_info: &ModelInfoData, issues: &mut Vec<ValidationIssue>) {
    match (model_info.input_price, model_info.output_price) {
        (None, None) => {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Info,
                error: ValidationError::MissingField {
                    field: "pricing".to_string(),
                    model: model_info.name.clone(),
                },
                suggested_fix: Some("Add pricing information if available".to_string()),
            });
        }
        (Some(_), None) => {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                error: ValidationError::MissingField {
                    field: "output_price".to_string(),
                    model: model_info.name.clone(),
                },
                suggested_fix: Some("Add output pricing to complete cost calculation".to_string()),
            });
        }
        (None, Some(_)) => {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                error: ValidationError::MissingField {
                    field: "input_price".to_string(),
                    model: model_info.name.clone(),
                },
                suggested_fix: Some("Add input pricing to complete cost calculation".to_string()),
            });
        }
        (Some(_), Some(_)) => {
            // Both prices present, already validated in validate_field_ranges
        }
    }
}

fn validate_capabilities(model_info: &ModelInfoData, issues: &mut Vec<ValidationIssue>) {
    // Check for capability consistency
    if model_info.supports_vision == Some(true) && model_info.max_input_tokens.unwrap_or(0) < 50000 {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Warning,
            error: ValidationError::InvalidCapability {
                description: "Vision-capable model has small context window".to_string(),
            },
            suggested_fix: Some("Verify context length is sufficient for vision tasks".to_string()),
        });
    }
    
    if model_info.supports_function_calling == Some(true) && model_info.max_output_tokens.unwrap_or(0) < 1000 {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Warning,
            error: ValidationError::InvalidCapability {
                description: "Function calling model has very limited output tokens".to_string(),
            },
            suggested_fix: Some("Consider increasing output token limit for function calling".to_string()),
        });
    }
}

fn validate_config_parameters(config: &ModelConfig, issues: &mut Vec<ValidationIssue>) {
    if config.temperature < 0.0 || config.temperature > 2.0 {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Error,
            error: ValidationError::InvalidRange {
                field: "temperature".to_string(),
                value: config.temperature.to_string(),
                expected: "0.0 to 2.0".to_string(),
            },
            suggested_fix: Some("Set temperature between 0.0 and 2.0".to_string()),
        });
    }
    
    if config.top_p <= 0.0 || config.top_p > 1.0 {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Error,
            error: ValidationError::InvalidRange {
                field: "top_p".to_string(),
                value: config.top_p.to_string(),
                expected: "> 0.0 and <= 1.0".to_string(),
            },
            suggested_fix: Some("Set top_p between 0.0 and 1.0".to_string()),
        });
    }
    
    if config.max_tokens == 0 {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Critical,
            error: ValidationError::InvalidRange {
                field: "max_tokens".to_string(),
                value: "0".to_string(),
                expected: "> 0".to_string(),
            },
            suggested_fix: Some("Set positive max_tokens value".to_string()),
        });
    }
}

fn validate_production_safety(config: &ModelConfig, issues: &mut Vec<ValidationIssue>) {
    if config.temperature > 1.5 {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Warning,
            error: ValidationError::UnsafeConfiguration {
                description: "High temperature may cause unpredictable outputs in production".to_string(),
            },
            suggested_fix: Some("Consider lowering temperature for production use".to_string()),
        });
    }
    
    if config.max_tokens > config.context_length / 2 {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Warning,
            error: ValidationError::UnsafeConfiguration {
                description: "Max tokens is more than half the context length".to_string(),
            },
            suggested_fix: Some("Ensure sufficient context for input when generating long outputs".to_string()),
        });
    }
}

fn validate_provider_config(config: &ModelConfig, issues: &mut Vec<ValidationIssue>) {
    if config.provider == "unknown" {
        issues.push(ValidationIssue {
            severity: ValidationSeverity::Error,
            error: ValidationError::InvalidProvider {
                provider: config.provider.to_string(),
            },
            suggested_fix: Some("Set a valid provider name".to_string()),
        });
    }
}

fn validate_unique_names(models: &[ModelInfoData], issues: &mut Vec<ValidationIssue>) {
    let mut seen_names = HashSet::new();
    
    for model in models {
        if !seen_names.insert(&model.name) {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                error: ValidationError::InconsistentData {
                    description: format!("Duplicate model name: '{}'", model.name),
                },
                suggested_fix: Some("Ensure all model names are unique".to_string()),
            });
        }
    }
}

fn validate_provider_consistency(models: &[ModelInfoData], issues: &mut Vec<ValidationIssue>) {
    // Check that provider names are consistent in format
    let providers: HashSet<_> = models.iter().map(|m| &m.provider_name).collect();
    
    for provider in providers {
        if provider.contains(' ') || provider.chars().any(|c| c.is_uppercase()) {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                error: ValidationError::InvalidProvider {
                    provider: provider.clone(),
                },
                suggested_fix: Some("Use lowercase, no-space provider names for consistency".to_string()),
            });
        }
    }
}

fn validate_pricing_consistency(_models: &[ModelInfoData], _issues: &mut Vec<ValidationIssue>) {
    // Could implement checks for pricing consistency within providers
    // For now, this is a placeholder for future enhancements
}

fn validate_capability_patterns(_models: &[ModelInfoData], _issues: &mut Vec<ValidationIssue>) {
    // Could implement checks for capability patterns across model families
    // For now, this is a placeholder for future enhancements  
}

fn calculate_readiness_score(issues: &[ValidationIssue]) -> f32 {
    let critical_count = issues.iter().filter(|i| i.severity == ValidationSeverity::Critical).count();
    let error_count = issues.iter().filter(|i| i.severity == ValidationSeverity::Error).count();
    let warning_count = issues.iter().filter(|i| i.severity == ValidationSeverity::Warning).count();
    
    // Start with perfect score
    let mut score = 1.0;
    
    // Deduct heavily for critical issues
    score -= critical_count as f32 * 0.4;
    
    // Deduct moderately for errors
    score -= error_count as f32 * 0.2;
    
    // Deduct lightly for warnings
    score -= warning_count as f32 * 0.05;
    
    // Ensure score stays in valid range
    score.max(0.0).min(1.0)
}