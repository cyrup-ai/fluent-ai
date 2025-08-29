//! Complexity analysis for JsonPathExpression
//!
//! Sophisticated complexity metrics and scoring algorithms for performance
//! optimization and query planning with zero-allocation patterns.

use super::core::JsonPathExpression;
use crate::jsonpath::ast::{ComplexityMetrics, JsonSelector};

impl JsonPathExpression {
    /// Generate comprehensive complexity metrics for performance analysis
    ///
    /// Provides detailed breakdown of complexity factors for performance optimization guidance.
    /// All metrics are computed with zero allocations for optimal performance.
    #[inline]
    pub fn complexity_metrics(&self) -> ComplexityMetrics {
        let mut metrics = ComplexityMetrics::new();

        for selector in self.selectors() {
            metrics.total_selector_count = metrics.total_selector_count.saturating_add(1);

            match selector {
                JsonSelector::RecursiveDescent => {
                    metrics.recursive_descent_depth =
                        metrics.recursive_descent_depth.saturating_add(1);
                }
                JsonSelector::Filter { expression } => {
                    metrics.filter_complexity_sum = metrics
                        .filter_complexity_sum
                        .saturating_add(expression.complexity_score());
                }
                JsonSelector::Slice { start, end, .. } => {
                    if let (Some(s), Some(e)) = (start, end) {
                        let range = e.saturating_sub(*s).unsigned_abs() as u32;
                        metrics.max_slice_range = metrics.max_slice_range.max(range);
                    } else if start.is_some() || end.is_some() {
                        // Open-ended slices are considered moderate complexity
                        metrics.max_slice_range = metrics.max_slice_range.max(100);
                    }
                }
                JsonSelector::Union { selectors } => {
                    metrics.union_selector_count = metrics
                        .union_selector_count
                        .saturating_add(selectors.len() as u32);
                }
                _ => {
                    // Other selectors don't affect specific metrics
                }
            }
        }

        metrics
    }

    /// Calculate sophisticated complexity score for performance optimization
    ///
    /// Uses advanced algorithm considering multiple complexity factors:
    /// - Base cost: minimum cost for any JSONPath expression
    /// - Depth penalty: exponential cost increase for recursive descent
    /// - Filter complexity: sum of all filter expression complexities  
    /// - Selector multiplier: interaction effects between selectors
    /// - Union penalty: cost of multiple selectors in union operations
    /// - Slice penalty: cost based on slice range sizes
    ///
    /// Returns a complexity score where higher values indicate more expensive evaluation.
    ///
    /// # Performance
    ///
    /// All calculations use saturating arithmetic to prevent overflow and are optimized
    /// for zero allocations in hot paths.
    #[inline]
    pub fn complexity_score(&self) -> u32 {
        let metrics = self.complexity_metrics();

        // Base cost: minimum cost for any JSONPath expression
        let base_cost = 5u32;

        // Depth penalty: exponential increase for recursive descent operations
        let depth_penalty = if metrics.recursive_descent_depth > 0 {
            // Cap at reasonable value to prevent overflow
            let capped_depth = metrics.recursive_descent_depth.min(10);
            2u32.saturating_pow(capped_depth)
        } else {
            0
        };

        // Selector multiplier: interaction effects between multiple selectors
        let selector_multiplier = match metrics.total_selector_count {
            0..=2 => 1,  // Straightforward expressions
            3..=5 => 2,  // Medium complexity
            6..=10 => 4, // High complexity
            _ => 8,      // Very complex expressions
        };

        // Union penalty: multiple selectors in union operations increase complexity
        let union_penalty = metrics.union_selector_count.saturating_mul(3);

        // Slice range penalty: larger ranges are more expensive to evaluate
        let slice_penalty = metrics.max_slice_range.saturating_div(10);

        // Combine all factors with saturating arithmetic
        base_cost
            .saturating_add(depth_penalty)
            .saturating_add(metrics.filter_complexity_sum)
            .saturating_mul(selector_multiplier)
            .saturating_add(union_penalty)
            .saturating_add(slice_penalty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_expression() -> JsonPathExpression {
        JsonPathExpression::new(
            vec![JsonSelector::Root, JsonSelector::Wildcard],
            "$.*".to_string(),
            false,
        )
    }

    fn create_complex_expression() -> JsonPathExpression {
        JsonPathExpression::new(
            vec![
                JsonSelector::Root,
                JsonSelector::RecursiveDescent,
                JsonSelector::Wildcard,
            ],
            "$..*".to_string(),
            false,
        )
    }

    #[test]
    fn test_simple_complexity_metrics() {
        let expr = create_simple_expression();
        let metrics = expr.complexity_metrics();

        assert_eq!(metrics.total_selector_count, 2);
        assert_eq!(metrics.recursive_descent_depth, 0);
        assert_eq!(metrics.filter_complexity_sum, 0);
        assert_eq!(metrics.union_selector_count, 0);
    }

    #[test]
    fn test_complex_complexity_metrics() {
        let expr = create_complex_expression();
        let metrics = expr.complexity_metrics();

        assert_eq!(metrics.total_selector_count, 3);
        assert_eq!(metrics.recursive_descent_depth, 1);
    }

    #[test]
    fn test_complexity_score_simple() {
        let expr = create_simple_expression();
        let score = expr.complexity_score();

        // Simple expression should have low complexity
        assert!(score < 20);
    }

    #[test]
    fn test_complexity_score_complex() {
        let expr = create_complex_expression();
        let score = expr.complexity_score();

        // Complex expression with recursive descent should have higher complexity
        assert!(score > 10);
    }

    #[test]
    fn test_complexity_score_overflow_protection() {
        // Create expression with many recursive descents to test overflow protection
        let selectors = vec![JsonSelector::RecursiveDescent; 15];
        let expr = JsonPathExpression::new(selectors, "$..".repeat(15), false);

        // Should not panic due to overflow
        let score = expr.complexity_score();
        assert!(score > 0);
    }
}
