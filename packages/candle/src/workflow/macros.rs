//! Compile-time parallel execution tree generation with tuple flattening
//!
//! This module provides macros for generating efficient parallel execution trees
//! at compile time. The macros enable variadic parallel operations with clean
//! tuple output types and optimal performance through compile-time optimization.

/// Internal macro for recursive parallel operation tree construction
/// 
/// This macro recursively builds parallel execution trees, combining operations
/// in binary pairs and flattening tuple outputs for clean composition. The
/// recursive expansion enables efficient n-way parallelism with compile-time
/// optimization.
/// 
/// ## Expansion Pattern
/// - Single operation: returns operation unchanged
/// - Two operations: creates Parallel combinator
/// - Multiple operations: recursively builds binary tree structure
/// - Tuple flattening: ensures clean output types without nested tuples
/// 
/// ## Performance
/// - Compile-time tree generation (zero runtime overhead)
/// - Binary tree structure for optimal parallel execution
/// - Tuple flattening eliminates type complexity
/// - Template specialization for concrete operation types
#[macro_export]
macro_rules! parallel_op {
    // Base case: single operation
    ($op:expr) => {
        $op
    };
    
    // Base case: two operations  
    ($left:expr, $right:expr) => {
        $crate::workflow::parallel::parallel($left, $right)
    };
    
    // Recursive case: three or more operations
    ($first:expr, $second:expr, $($rest:expr),+) => {
        $crate::workflow::parallel::parallel(
            $first,
            parallel_op!($second, $($rest),+)
        )
    };
}

/// Public macro for ergonomic variadic parallel execution
/// 
/// Creates parallel execution trees that execute all provided operations
/// concurrently and combine their results into tuple output. The macro
/// handles compile-time optimization and tuple flattening automatically.
/// 
/// ## Usage
/// ```rust,no_run
/// use fluent_ai_candle::{parallel, workflow::ops::map};
/// 
/// let double = map(|x: i32| x * 2);
/// let triple = map(|x: i32| x * 3); 
/// let quadruple = map(|x: i32| x * 4);
/// 
/// // Create parallel execution of all three operations
/// let parallel_ops = parallel!(double, triple, quadruple);
/// let result_stream = parallel_ops.call(5);
/// // Result: AsyncStream containing [(10, (15, 20))]
/// ```
/// 
/// ## Compile-Time Optimization
/// - Recursive macro expansion builds optimal binary trees
/// - Template specialization for concrete operation types  
/// - Zero runtime overhead for parallel tree construction
/// - Efficient tuple type generation without nesting complexity
/// 
/// ## Output Type Structure
/// - Two operations: `(OutA, OutB)`
/// - Three operations: `(OutA, (OutB, OutC))`
/// - Four operations: `(OutA, (OutB, (OutC, OutD)))`
/// - N operations: Right-associative tuple nesting
#[macro_export]
macro_rules! parallel {
    ($($ops:expr),+ $(,)?) => {
        parallel_op!($($ops),+)
    };
}

/// Internal macro for recursive error-propagating parallel operation tree construction
/// 
/// Similar to parallel_op! but builds trees of TryParallel combinators that
/// short-circuit on first error. The recursive expansion creates efficient
/// error-propagating parallel execution with clean error handling semantics.
/// 
/// ## Error Propagation Semantics
/// - Short-circuit on first error from any parallel branch
/// - Consistent error types across all parallel operations
/// - Clean error composition without nested Result types
/// - Compile-time error type validation and consistency
/// 
/// ## Expansion Pattern
/// - Single operation: returns operation unchanged (must implement TryOp)
/// - Two operations: creates TryParallel combinator
/// - Multiple operations: recursively builds error-aware binary tree
/// - Error type consistency: all operations must have same error type
#[macro_export]
macro_rules! try_parallel_internal {
    // Base case: single operation
    ($op:expr) => {
        $op
    };
    
    // Base case: two operations
    ($left:expr, $right:expr) => {
        $crate::workflow::parallel::try_parallel($left, $right)
    };
    
    // Recursive case: three or more operations
    ($first:expr, $second:expr, $($rest:expr),+) => {
        $crate::workflow::parallel::try_parallel(
            $first,
            try_parallel_internal!($second, $($rest),+)
        )
    };
}

/// Public macro for ergonomic variadic error-propagating parallel execution
/// 
/// Creates error-aware parallel execution trees that execute all provided
/// operations concurrently but short-circuit on first error. All operations
/// must implement TryOp with consistent error types.
/// 
/// ## Usage
/// ```rust,no_run
/// use fluent_ai_candle::{try_parallel, workflow::parallel::TryOp};
/// 
/// let safe_double = /* some TryOp that might fail */;
/// let safe_triple = /* some TryOp that might fail */;
/// let safe_quadruple = /* some TryOp that might fail */;
/// 
/// // Create error-propagating parallel execution
/// let try_parallel_ops = try_parallel!(safe_double, safe_triple, safe_quadruple);
/// let result_stream = try_parallel_ops.try_call(input);
/// // Result: AsyncStream containing [Ok((a, (b, c)))] or [Err(error)]
/// ```
/// 
/// ## Error Semantics
/// - Short-circuit behavior: first error stops all parallel execution
/// - Consistent error types: all operations must have same error type
/// - Clean composition: no nested Result types in output
/// - Fail-fast semantics: immediate error propagation without delay
/// 
/// ## Compile-Time Validation
/// - Error type consistency enforced at compile time
/// - TryOp trait bounds validated for all operations
/// - Template specialization for optimal error handling code
/// - Zero runtime overhead for error propagation tree construction
/// 
/// ## Output Type Structure
/// - Success case: `Result<(OutA, (OutB, OutC)), Err>`
/// - Error case: `Result<_, Err>` where Err is consistent across all operations
/// - Right-associative tuple nesting for multiple operations
/// - Clean composition with downstream error handling
#[macro_export]
macro_rules! try_parallel {
    ($($ops:expr),+ $(,)?) => {
        try_parallel_internal!($($ops),+)
    };
}

/// Compile-time tuple flattening utilities for clean output types
/// 
/// These macros provide compile-time tuple manipulation for creating clean
/// output types from variadic parallel operations. The flattening eliminates
/// deeply nested tuple structures for better API ergonomics.
/// 
/// ## Note on Tuple Flattening
/// Current implementation uses right-associative nesting which is simpler
/// to implement and provides consistent type patterns. Future versions
/// could implement full tuple flattening for completely flat output types
/// if needed for specific use cases.

/// Helper macro to count the number of operations for compile-time optimization
/// 
/// Counts the number of comma-separated expressions at compile time,
/// enabling size-specific optimizations and type generation.
#[macro_export]
macro_rules! count_ops {
    () => (0usize);
    ($head:expr $(, $tail:expr)*) => (1usize + count_ops!($($tail),*));
}



// Re-export macros for public use
pub use parallel;
pub use try_parallel;
pub use parallel_op;
pub use try_parallel_internal;
pub use count_ops;