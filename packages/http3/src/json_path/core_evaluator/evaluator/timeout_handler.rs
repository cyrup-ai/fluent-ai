//! Timeout protection for JSONPath evaluation
//!
//! Provides thread-based timeout handling to prevent excessive processing time
//! on pathological JSONPath expressions or deeply nested JSON structures.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

use super::core_types::{CoreJsonPathEvaluator, JsonPathResult};

/// Configuration for timeout handling
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Maximum evaluation time before timeout
    pub timeout_duration: Duration,
    /// Whether to log timeout events
    pub log_timeouts: bool,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            timeout_duration: Duration::from_millis(1500), // 1.5 second timeout
            log_timeouts: true,
        }
    }
}

/// Timeout handler for JSONPath evaluation
pub struct TimeoutHandler;

impl TimeoutHandler {
    /// Evaluate JSONPath expression with timeout protection
    pub fn evaluate_with_timeout(
        evaluator: &CoreJsonPathEvaluator,
        json: &Value,
        config: Option<TimeoutConfig>,
    ) -> JsonPathResult<Vec<Value>> {
        let config = config.unwrap_or_default();
        let start_time = Instant::now();

        let (tx, rx) = mpsc::channel();
        let expression = evaluator.expression().to_string();
        let json_clone = json.clone();

        // Spawn evaluation in separate thread
        let handle = thread::spawn(move || {
            if config.log_timeouts {
                log::debug!("Starting JSONPath evaluation in timeout thread");
            }
            let result = Self::evaluate_internal(&expression, &json_clone);
            if config.log_timeouts {
                log::debug!("JSONPath evaluation completed in thread");
            }
            let _ = tx.send(result); // Ignore send errors if receiver dropped
        });

        // Wait for completion or timeout
        match rx.recv_timeout(config.timeout_duration) {
            Ok(result) => {
                let elapsed = start_time.elapsed();
                if config.log_timeouts {
                    log::debug!(
                        "JSONPath evaluation completed successfully in {:?}",
                        elapsed
                    );
                }
                result
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let elapsed = start_time.elapsed();
                if config.log_timeouts {
                    log::warn!(
                        "JSONPath evaluation timed out after {:?} - likely deep nesting issue",
                        elapsed
                    );
                }

                // Clean up thread - it will continue running but we ignore result
                drop(handle);

                // Return empty results for timeout - prevents hanging
                Ok(Vec::new())
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let elapsed = start_time.elapsed();
                if config.log_timeouts {
                    log::error!(
                        "JSONPath evaluation thread disconnected after {:?}",
                        elapsed
                    );
                }
                Err(crate::json_path::error::invalid_expression_error(
                    evaluator.expression(),
                    "evaluation thread disconnected unexpectedly",
                    None,
                ))
            }
        }
    }

    /// Internal evaluation method (static to avoid self reference in thread)
    fn evaluate_internal(expression: &str, json: &Value) -> JsonPathResult<Vec<Value>> {
        use super::evaluation_engine::EvaluationEngine;

        // Create temporary evaluator instance for method calls
        let temp_evaluator = CoreJsonPathEvaluator::create_temp_evaluator(expression)?;

        // Delegate to evaluation engine
        EvaluationEngine::evaluate_expression(&temp_evaluator, json)
    }

    /// Check if an expression is likely to cause timeout
    pub fn is_potentially_slow(expression: &str) -> bool {
        // Patterns that are known to be expensive
        expression.contains("..") ||  // Recursive descent
        expression.contains("*") ||   // Wildcard
        expression.contains("[?") ||  // Filters
        expression.contains("[:") ||  // Slices
        expression.matches('[').count() > 3 // Deep nesting
    }

    /// Estimate evaluation complexity
    pub fn estimate_complexity(expression: &str) -> u32 {
        let mut complexity = 1;

        // Add complexity for expensive operations
        complexity += expression.matches("..").count() as u32 * 50; // Recursive descent
        complexity += expression.matches("*").count() as u32 * 10; // Wildcard
        complexity += expression.matches("[?").count() as u32 * 20; // Filters
        complexity += expression.matches("[:").count() as u32 * 5; // Slices
        complexity += expression.matches('[').count() as u32 * 2; // Array access

        complexity
    }

    /// Get recommended timeout for expression
    pub fn recommended_timeout(expression: &str) -> Duration {
        let complexity = Self::estimate_complexity(expression);

        match complexity {
            0..=10 => Duration::from_millis(100),
            11..=50 => Duration::from_millis(500),
            51..=100 => Duration::from_millis(1000),
            _ => Duration::from_millis(2000),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_timeout_config_default() {
        let config = TimeoutConfig::default();
        assert_eq!(config.timeout_duration, Duration::from_millis(1500));
        assert_eq!(config.log_timeouts, true);
    }

    #[test]
    fn test_timeout_config_custom() {
        let config = TimeoutConfig {
            timeout_duration: Duration::from_millis(500),
            log_timeouts: false,
        };
        assert_eq!(config.timeout_duration, Duration::from_millis(500));
        assert_eq!(config.log_timeouts, false);
    }

    #[test]
    fn test_is_potentially_slow() {
        assert!(TimeoutHandler::is_potentially_slow("$..book"));
        assert!(TimeoutHandler::is_potentially_slow("$.store.*"));
        assert!(TimeoutHandler::is_potentially_slow(
            "$.store.book[?(@.price < 10)]"
        ));
        assert!(TimeoutHandler::is_potentially_slow("$.store.book[1:3]"));
        assert!(TimeoutHandler::is_potentially_slow("$.a[0].b[1].c[2].d[3]"));

        assert!(!TimeoutHandler::is_potentially_slow("$.store.book"));
        assert!(!TimeoutHandler::is_potentially_slow("$.simple"));
    }

    #[test]
    fn test_estimate_complexity() {
        assert_eq!(TimeoutHandler::estimate_complexity("$.simple"), 1);
        assert_eq!(TimeoutHandler::estimate_complexity("$..book"), 51); // 1 + 50
        assert_eq!(TimeoutHandler::estimate_complexity("$.store.*"), 11); // 1 + 10
        assert_eq!(
            TimeoutHandler::estimate_complexity("$.book[?(@.price)]"),
            23
        ); // 1 + 20 + 2
        assert_eq!(TimeoutHandler::estimate_complexity("$.book[1:3]"), 8); // 1 + 5 + 2
    }

    #[test]
    fn test_recommended_timeout() {
        assert_eq!(
            TimeoutHandler::recommended_timeout("$.simple"),
            Duration::from_millis(100)
        );
        assert_eq!(
            TimeoutHandler::recommended_timeout("$.store.*"),
            Duration::from_millis(500)
        );
        assert_eq!(
            TimeoutHandler::recommended_timeout("$..book"),
            Duration::from_millis(1000)
        );
        assert_eq!(
            TimeoutHandler::recommended_timeout("$..*[?(@.price > 10)]"),
            Duration::from_millis(2000)
        );
    }

    #[test]
    fn test_timeout_handler_simple_evaluation() {
        let evaluator = CoreJsonPathEvaluator::new("$.test").unwrap();
        let json = json!({"test": "value"});

        let result = TimeoutHandler::evaluate_with_timeout(&evaluator, &json, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_timeout_handler_with_custom_config() {
        let evaluator = CoreJsonPathEvaluator::new("$.test").unwrap();
        let json = json!({"test": "value"});

        let config = TimeoutConfig {
            timeout_duration: Duration::from_millis(100),
            log_timeouts: false,
        };

        let result = TimeoutHandler::evaluate_with_timeout(&evaluator, &json, Some(config));
        assert!(result.is_ok());
    }

    #[test]
    fn test_timeout_config_clone() {
        let config = TimeoutConfig::default();
        let cloned = config.clone();
        assert_eq!(config.timeout_duration, cloned.timeout_duration);
        assert_eq!(config.log_timeouts, cloned.log_timeouts);
    }

    #[test]
    fn test_timeout_config_debug() {
        let config = TimeoutConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("TimeoutConfig"));
        assert!(debug_str.contains("1500"));
    }
}
