//! JsonPathExpression implementation with complexity analysis
//!
//! Provides the main JSONPath expression structure with sophisticated complexity
//! metrics and depth evaluation for performance optimization.

use crate::json_path::ast::{ComplexityMetrics, JsonSelector};

/// Compiled JSONPath expression optimized for streaming evaluation
#[derive(Debug, Clone)]
pub struct JsonPathExpression {
    /// Optimized selector chain for runtime execution
    selectors: Vec<JsonSelector>,
    /// Original expression string for debugging
    original: String,
    /// Whether this expression targets an array for streaming
    is_array_stream: bool,
}

impl JsonPathExpression {
    /// Create new JsonPathExpression
    #[inline]
    pub fn new(selectors: Vec<JsonSelector>, original: String, is_array_stream: bool) -> Self {
        Self {
            selectors,
            original,
            is_array_stream,
        }
    }

    /// Get original JSONPath expression string
    #[inline]
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Check if this expression targets array elements for streaming
    #[inline]
    pub fn is_array_stream(&self) -> bool {
        self.is_array_stream
    }

    /// Get compiled selector chain
    #[inline]
    pub fn selectors(&self) -> &[JsonSelector] {
        &self.selectors
    }

    /// Check if expression has recursive descent
    #[inline]
    pub fn has_recursive_descent(&self) -> bool {
        self.selectors
            .iter()
            .any(|s| matches!(s, JsonSelector::RecursiveDescent))
    }

    /// Get the starting position of recursive descent in selector chain
    #[inline]
    pub fn recursive_descent_start(&self) -> Option<usize> {
        self.selectors
            .iter()
            .position(|s| matches!(s, JsonSelector::RecursiveDescent))
    }

    /// Generate comprehensive complexity metrics for performance analysis
    ///
    /// Provides detailed breakdown of complexity factors for performance optimization guidance.
    /// All metrics are computed with zero allocations for optimal performance.
    #[inline]
    pub fn complexity_metrics(&self) -> ComplexityMetrics {
        let mut metrics = ComplexityMetrics::new();

        for selector in &self.selectors {
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
            0..=2 => 1,  // Simple expressions
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

    /// Get root selector (first non-root selector in expression)
    ///
    /// Returns the first meaningful selector after the root ($) identifier.
    /// This is commonly used to determine the root navigation behavior.
    #[inline]
    pub fn root_selector(&self) -> Option<&JsonSelector> {
        self.selectors
            .iter()
            .find(|selector| !matches!(selector, JsonSelector::Root))
    }

    /// Check if expression matches at specified JSON depth
    ///
    /// Used during streaming to determine if current parsing position
    /// matches the JSONPath expression navigation requirements.
    ///
    /// # Arguments
    ///
    /// * `depth` - Current JSON nesting depth (0 = root level)
    ///
    /// # Returns
    ///
    /// `true` if the current depth matches the expression's navigation pattern.
    #[inline]
    pub fn matches_at_depth(&self, depth: usize) -> bool {
        self.evaluate_selectors_at_depth(depth, 0).is_some()
    }

    /// Evaluate selector chain recursively to determine if current depth matches
    ///
    /// Handles recursive descent (..) by exploring all possible paths through the JSON structure.
    /// Returns the next selector index to continue evaluation, or None if no match.
    ///
    /// # Arguments
    ///
    /// * `current_depth` - Current JSON nesting depth
    /// * `selector_index` - Index in the selector chain being evaluated
    ///
    /// # Performance
    ///
    /// Uses early termination and efficient recursive evaluation for optimal performance.
    #[inline]
    fn evaluate_selectors_at_depth(
        &self,
        current_depth: usize,
        selector_index: usize,
    ) -> Option<usize> {
        // Base case: reached end of selectors
        if selector_index >= self.selectors.len() {
            return Some(selector_index);
        }

        // Base case: depth 0 should only match root selector
        if current_depth == 0 && selector_index == 0 {
            return if matches!(self.selectors[0], JsonSelector::Root) {
                self.evaluate_selectors_at_depth(current_depth, 1)
            } else {
                None
            };
        }

        let selector = &self.selectors[selector_index];

        match selector {
            JsonSelector::Root => {
                // Root can only match at the beginning
                if selector_index == 0 {
                    self.evaluate_selectors_at_depth(current_depth, selector_index + 1)
                } else {
                    None
                }
            }

            JsonSelector::RecursiveDescent => {
                // Recursive descent matches at any depth
                // Try to match the next selector at current depth or any deeper depth
                let next_selector_index = selector_index + 1;

                if next_selector_index >= self.selectors.len() {
                    // Recursive descent at end matches everything
                    return Some(next_selector_index);
                }

                // Try to match next selector at current depth
                if let Some(result) =
                    self.evaluate_selectors_at_depth(current_depth, next_selector_index)
                {
                    return Some(result);
                }

                // Try to match recursive descent at deeper levels (simulated)
                // In streaming context, this means we stay in recursive descent mode
                // until we find a matching structure
                if current_depth < 20 {
                    // Reasonable depth limit
                    self.evaluate_selectors_at_depth(current_depth + 1, selector_index)
                } else {
                    None
                }
            }

            JsonSelector::Child { .. } => {
                // Child selectors require exact depth progression
                if current_depth > 0 {
                    self.evaluate_selectors_at_depth(current_depth, selector_index + 1)
                } else {
                    None
                }
            }

            JsonSelector::Index { .. } | JsonSelector::Slice { .. } | JsonSelector::Wildcard => {
                // Array selectors require being inside an array
                if current_depth > 0 {
                    self.evaluate_selectors_at_depth(current_depth, selector_index + 1)
                } else {
                    None
                }
            }

            JsonSelector::Filter { .. } => {
                // Filter expressions require context evaluation (handled at runtime)
                if current_depth > 0 {
                    self.evaluate_selectors_at_depth(current_depth, selector_index + 1)
                } else {
                    None
                }
            }

            JsonSelector::Union { selectors } => {
                // Union matches if any selector matches at this depth
                for union_selector in selectors {
                    if self.evaluate_single_selector_at_depth(union_selector, current_depth) {
                        return self.evaluate_selectors_at_depth(current_depth, selector_index + 1);
                    }
                }
                None
            }
        }
    }

    /// Evaluate a single selector at specific depth
    #[inline]
    fn evaluate_single_selector_at_depth(&self, selector: &JsonSelector, depth: usize) -> bool {
        match selector {
            JsonSelector::Root => depth == 0,
            JsonSelector::RecursiveDescent => true, // Always matches
            JsonSelector::Child { .. }
            | JsonSelector::Index { .. }
            | JsonSelector::Slice { .. }
            | JsonSelector::Wildcard
            | JsonSelector::Filter { .. } => depth > 0,
            JsonSelector::Union { selectors } => selectors
                .iter()
                .any(|s| self.evaluate_single_selector_at_depth(s, depth)),
        }
    }
}
