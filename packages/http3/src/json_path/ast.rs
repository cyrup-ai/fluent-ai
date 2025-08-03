//! JSONPath Abstract Syntax Tree (AST) definitions
//!
//! Core type definitions for representing JSONPath expressions as structured data.
//! Provides zero-allocation AST nodes optimized for streaming evaluation.

/// Individual JSONPath selector component
#[derive(Debug, Clone)]
pub enum JsonSelector {
    /// Root selector ($)
    Root,

    /// Child property access (.property or ['property'])
    Child {
        name: String,
        /// Whether to use exact string matching (true) or case-insensitive (false)
        exact_match: bool,
    },

    /// Recursive descent (..)
    RecursiveDescent,

    /// Array index access ([0], [-1], etc.)
    Index {
        index: i64,
        /// For negative indices, whether to count from end
        from_end: bool,
    },

    /// Array slice ([start:end], [start:], [:end])
    Slice {
        start: Option<i64>,
        end: Option<i64>,
        step: Option<i64>,
    },

    /// Wildcard selector ([*] or .*)
    Wildcard,

    /// Filter expression ([?(@.property > value)])
    Filter {
        /// Filter expression AST
        expression: FilterExpression,
    },

    /// Multiple selectors (union operator)
    Union { selectors: Vec<JsonSelector> },
}

/// Filter expression AST for JSONPath predicates
#[derive(Debug, Clone)]
pub enum FilterExpression {
    /// Current node reference (@)
    Current,

    /// Property access (@.property)
    Property { path: Vec<String> },

    /// Complex JSONPath expressions (@.items[*], @.data[0:5], etc.)
    JsonPath { selectors: Vec<JsonSelector> },

    /// Literal values (strings, numbers, booleans)
    Literal { value: FilterValue },

    /// Comparison operations
    Comparison {
        left: Box<FilterExpression>,
        operator: ComparisonOp,
        right: Box<FilterExpression>,
    },

    /// Logical operations (&&, ||)
    Logical {
        left: Box<FilterExpression>,
        operator: LogicalOp,
        right: Box<FilterExpression>,
    },

    /// Regular expression matching
    Regex {
        target: Box<FilterExpression>,
        pattern: String,
    },

    /// Function calls (length, type, etc.)
    Function {
        name: String,
        args: Vec<FilterExpression>,
    },
}

/// Filter expression literal values
#[derive(Debug, Clone)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Integer(i64),
    Boolean(bool),
    Null,
}

/// Comparison operators for filter expressions
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Equal,      // ==
    NotEqual,   // !=
    Less,       // <
    LessEq,     // <=
    Greater,    // >
    GreaterEq,  // >=
    In,         // in
    NotIn,      // not in
    Contains,   // contains
    StartsWith, // starts with
    EndsWith,   // ends with
    Match,      // =~
    NotMatch,   // !~
}

/// Logical operators for filter expressions
#[derive(Debug, Clone, Copy)]
pub enum LogicalOp {
    And, // &&
    Or,  // ||
}

/// Comprehensive complexity metrics for JSONPath expression analysis
///
/// Provides detailed breakdown of complexity factors for performance optimization guidance.
/// All metrics are computed at compile time for zero runtime overhead.
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
    pub fn add(&mut self, other: &ComplexityMetrics) {
        self.recursive_descent_depth = self
            .recursive_descent_depth
            .saturating_add(other.recursive_descent_depth);
        self.total_selector_count = self
            .total_selector_count
            .saturating_add(other.total_selector_count);
        self.filter_complexity_sum = self
            .filter_complexity_sum
            .saturating_add(other.filter_complexity_sum);
        self.max_slice_range = self.max_slice_range.max(other.max_slice_range);
        self.union_selector_count = self
            .union_selector_count
            .saturating_add(other.union_selector_count);
    }
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
            FilterExpression::Regex { target, .. } => {
                5 + target.complexity_score() // Regex operations are more expensive
            }
            FilterExpression::Function { args, .. } => {
                5 + args.iter().map(|arg| arg.complexity_score()).sum::<u32>()
            }
            FilterExpression::JsonPath { selectors } => {
                selectors.len() as u32 * 2 // Complex JSONPath expressions are more expensive
            }
        }
    }
}
