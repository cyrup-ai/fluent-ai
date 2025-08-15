//! Core data structures and types for JSONPath expression compilation
//! 
//! Provides the fundamental types used throughout the JSONPath parsing and compilation process.

use crate::json_path::error::{JsonPathError, JsonPathResult};

/// Compiled JSONPath expression optimized for streaming evaluation
#[derive(Debug, Clone)]
pub struct JsonPathExpression {
    /// Optimized selector chain for runtime execution
    pub(super) selectors: Vec<JsonSelector>,
    /// Original expression string for debugging
    pub(super) original: String,
    /// Whether this expression targets an array for streaming
    pub(super) is_array_stream: bool,
}

/// Comprehensive complexity metrics for JSONPath expression analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComplexityMetrics {
    /// Count of recursive descent (..) selectors and their effective nesting depth
    pub recursive_descent_depth: u32,
    /// Total number of selectors in the expression chain
    pub total_selector_count: u32,
    /// Sum of all filter expression complexity scores
    pub filter_complexity_sum: u32,
    /// Largest slice range for slice operations (end - start)
    pub max_slice_range: u32,
    /// Number of selectors within union operations
    pub union_selector_count: u32,
}

impl ComplexityMetrics {
    /// Create new complexity metrics with zero values
    #[inline]
    pub const fn new() -> Self {
        Self {
            recursive_descent_depth: 0,
            total_selector_count: 0,
            filter_complexity_sum: 0,
            max_slice_range: 0,
            union_selector_count: 0,
        }
    }    
    /// Add metrics from another ComplexityMetrics instance
    #[inline]
    pub fn add(&mut self, other: &Self) {
        self.recursive_descent_depth = self.recursive_descent_depth.saturating_add(other.recursive_descent_depth);
        self.total_selector_count = self.total_selector_count.saturating_add(other.total_selector_count);
        self.filter_complexity_sum = self.filter_complexity_sum.saturating_add(other.filter_complexity_sum);
        self.max_slice_range = self.max_slice_range.max(other.max_slice_range);
        self.union_selector_count = self.union_selector_count.saturating_add(other.union_selector_count);
    }
    
    /// Calculate overall complexity score
    #[inline]
    pub fn complexity_score(&self) -> u32 {
        // Weighted scoring algorithm that heavily penalizes exponential operations
        self.recursive_descent_depth.saturating_mul(5)
            .saturating_add(self.filter_complexity_sum.saturating_mul(3))
            .saturating_add(self.union_selector_count.saturating_mul(2))
            .saturating_add(self.max_slice_range)
            .saturating_add(self.total_selector_count)
    }
}

/// Individual JSONPath selector for runtime evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum JsonSelector {
    /// Root selector ($)
    Root,
    /// Property access (.property or ['property'])
    Property {
        name: String,
    },
    /// Child property access (.property or ['property'])
    Child { 
        name: String,
        exact_match: bool,
    },
    /// Array index access ([0], [-1])
    Index {
        index: i32,
        from_end: bool,
    },
    /// Array slice ([start:end:step])
    Slice {
        start: Option<i32>,
        end: Option<i32>,
        step: Option<i32>,
    },
    /// Wildcard selector (*, [*])
    Wildcard,
    /// Recursive descent (..)
    RecursiveDescent,
    /// Filter expression ([?@.price > 10])
    Filter {
        expression: FilterExpression,
    },
    /// Union of multiple selectors ([0, 2, 4])
    Union {
        selectors: Vec<JsonSelector>,
    },
}/// Filter expression components for JSONPath predicates
#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpression {
    /// Current node reference (@)
    Current,
    /// Property access (@.property, @.nested.prop)
    Property {
        path: Vec<String>,
    },
    /// Literal value (string, number, boolean, null)
    Literal {
        value: FilterValue,
    },
    /// Comparison operation (@.price > 10)
    Comparison {
        left: Box<FilterExpression>,
        operator: ComparisonOp,
        right: Box<FilterExpression>,
    },
    /// Logical AND (&&)
    LogicalAnd {
        left: Box<FilterExpression>,
        right: Box<FilterExpression>,
    },
    /// Logical OR (||)
    LogicalOr {
        left: Box<FilterExpression>,
        right: Box<FilterExpression>,
    },
    /// Logical operations (&&, ||)
    Logical {
        left: Box<FilterExpression>,
        operator: LogicalOp,
        right: Box<FilterExpression>,
    },
    
    /// Function call (length(@.tags), match(@.name, 'pattern'))
    Function {
        name: String,
        args: Vec<FilterExpression>,
    },
}/// Logical operators for filter expressions
#[derive(Debug, Clone, Copy)]
pub enum LogicalOp {
    And, // &&
    Or,  // ||
}

impl FilterExpression {
    /// Calculate complexity score for filter expressions
    #[inline]
    pub fn complexity_score(&self) -> u32 {
        match self {
            FilterExpression::Current => 1,
            FilterExpression::Property { path } => path.len() as u32,
            FilterExpression::Literal { .. } => 1,
            FilterExpression::Comparison { left, right, .. } => {
                2 + left.complexity_score() + right.complexity_score()
            }
            FilterExpression::Logical { left, right, .. } => {
                3 + left.complexity_score() + right.complexity_score()
            }
            FilterExpression::Function { args, .. } => {
                5 + args.iter().map(|arg| arg.complexity_score()).sum::<u32>()
            }
            FilterExpression::LogicalAnd { left, right } => {
                3 + left.complexity_score() + right.complexity_score()
            }
            FilterExpression::LogicalOr { left, right } => {
                3 + left.complexity_score() + right.complexity_score()
            }
        }
    }
}

/// Comparison operators for filter expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Equal,      // ==
    NotEqual,   // !=
    Less,       // <
    LessEq,     // <=
    Greater,    // >
    GreaterEq,  // >=
}

/// Filter value types for comparisons and function results
#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    String(String),
    Integer(i64),
    Number(f64),
    Boolean(bool),
    Null,
}impl JsonPathExpression {
    /// Get the optimized selector chain
    #[inline]
    pub fn selectors(&self) -> &[JsonSelector] {
        &self.selectors
    }
    
    /// Get the original JSONPath expression string
    #[inline]
    pub fn original(&self) -> &str {
        &self.original
    }
    
    /// Check if this expression is optimized for array streaming
    #[inline]
    pub fn is_array_stream(&self) -> bool {
        self.is_array_stream
    }
    
    /// Calculate complexity score for performance estimation
    #[inline]
    pub fn complexity_score(&self) -> u32 {
        self.calculate_complexity_metrics().complexity_score()
    }
    
    /// Check if expression has recursive descent
    #[inline]
    pub fn has_recursive_descent(&self) -> bool {
        self.selectors.iter().any(|s| matches!(s, JsonSelector::RecursiveDescent))
    }
    
    /// Get root selector (first non-root selector)
    #[inline]
    pub fn root_selector(&self) -> Option<&JsonSelector> {
        self.selectors.iter().find(|s| !matches!(s, JsonSelector::Root))
    }
    
    /// Get recursive descent start position
    #[inline]
    pub fn recursive_descent_start(&self) -> Option<usize> {
        self.selectors.iter().position(|s| matches!(s, JsonSelector::RecursiveDescent))
    }
    
    /// Calculate detailed complexity metrics
    pub fn calculate_complexity_metrics(&self) -> ComplexityMetrics {
        let mut metrics = ComplexityMetrics::new();
        
        for selector in &self.selectors {
            metrics.total_selector_count = metrics.total_selector_count.saturating_add(1);
            
            match selector {
                JsonSelector::RecursiveDescent => {
                    metrics.recursive_descent_depth = metrics.recursive_descent_depth.saturating_add(1);
                }
                JsonSelector::Filter { expression } => {
                    metrics.filter_complexity_sum = metrics.filter_complexity_sum
                        .saturating_add(expression.complexity_score());
                }
                JsonSelector::Slice { start, end, .. } => {
                    if let (Some(s), Some(e)) = (start, end) {
                        let range = (e - s).abs() as u32;
                        metrics.max_slice_range = metrics.max_slice_range.max(range);
                    }
                }
                JsonSelector::Union { selectors } => {
                    metrics.union_selector_count = metrics.union_selector_count
                        .saturating_add(selectors.len() as u32);
                }
                _ => {}
            }
        }
        
        metrics
    }
}