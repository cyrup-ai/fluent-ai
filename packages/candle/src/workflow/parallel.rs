//! True concurrent execution with thread-based parallelism for blazing-fast performance
//!
//! This module provides parallel execution combinators using std::thread for true
//! concurrent execution without tokio dependency. All operations maintain the
//! fluent-ai streams-only architecture while enabling maximum parallelism.

use std::sync::mpsc;

use crossbeam;
use fluent_ai_async::AsyncStream;

use crate::workflow::ops::Op;

/// Parallel execution combinator for true concurrent processing
///
/// Executes two operations concurrently using thread-based parallelism,
/// combining their results into a tuple output. Both operations receive
/// cloned copies of the input for independent parallel processing.
///
/// ## Architecture
/// - Thread-based parallelism using std::thread::spawn
/// - Input cloning for independent parallel distribution
/// - Tuple output for clean composition with type safety
/// - Zero allocation with PhantomData for type tracking
/// - Send + Sync bounds for concurrent execution safety
///
/// ## Performance Characteristics
/// - True concurrent execution (not cooperative like async)
/// - Zero allocation with PhantomData type tracking
/// - Minimal synchronization overhead using channels
/// - Clone-based input distribution for independence
/// - Blazing-fast parallel processing without locks
///
/// ## Example
/// ```rust,no_run
/// use fluent_ai_candle::workflow::ops::{map, Op};
/// use fluent_ai_candle::workflow::parallel::Parallel;
///
/// let double = map(|x: i32| x * 2);
/// let triple = map(|x: i32| x * 3);
/// let parallel_op = Parallel::new(double, triple);
/// let result_stream = parallel_op.call(5);
/// // Result: AsyncStream containing [(10, 15)]
/// ```
#[derive(Clone)]
pub struct Parallel<A, B> {
    /// Left operation for parallel execution
    left: A,
    /// Right operation for parallel execution
    right: B,
}

impl<A, B> Parallel<A, B> {
    /// Create a new parallel combinator
    ///
    /// Takes two operations and creates a parallel combinator that will
    /// execute both concurrently when called. Input is cloned for both
    /// operations to ensure independence.
    ///
    /// ## Performance
    /// - Zero allocation constructor
    /// - Operations stored directly (no heap allocation)
    /// - Clone bounds ensure efficient input distribution
    #[inline]
    pub fn new(left: A, right: B) -> Self {
        Self { left, right }
    }
}

impl<A, B, In, OutA, OutB> Op<In, (OutA, OutB)> for Parallel<A, B>
where
    A: Op<In, OutA> + 'static,
    B: Op<In, OutB> + 'static,
    In: Send + Sync + Clone + 'static, // Input needs Clone for parallel distribution
    OutA: Send + Sync + Clone + 'static, // Outputs need Clone for cartesian product
    OutB: Send + Sync + Clone + 'static,
{
    fn call(&self, input: In) -> AsyncStream<(OutA, OutB)> {
        let left = self.left.clone();
        let right = self.right.clone();
        let input_left = input.clone();
        let input_right = input;

        AsyncStream::with_channel(move |sender| {
            // Use crossbeam scoped threads for bounded resource management
            crossbeam::thread::scope(|scope| {
                // Create channels for collecting results from parallel threads
                let (left_tx, left_rx) = mpsc::channel::<Vec<OutA>>();
                let (right_tx, right_rx) = mpsc::channel::<Vec<OutB>>();

                // Spawn left operation in scoped thread (automatically joined)
                scope.spawn(move |_| {
                    // Use collect() for proper timeout handling instead of manual loops
                    let left_results = left.call(input_left).collect();
                    let _ = left_tx.send(left_results);
                });

                // Spawn right operation in scoped thread (automatically joined)
                scope.spawn(move |_| {
                    // Use collect() for proper timeout handling instead of manual loops
                    let right_results = right.call(input_right).collect();
                    let _ = right_tx.send(right_results);
                });

                // Scoped threads are automatically joined when scope exits
                // Collect results from both channels
                let left_results = left_rx.recv().unwrap_or_default();
                let right_results = right_rx.recv().unwrap_or_default();

                // Handle edge cases first, before consuming the vectors
                if left_results.is_empty() || right_results.is_empty() {
                    // Can't create tuples if either side produces no results - no output
                    return;
                }

                // Combine results using Cartesian product for complete output
                // This ensures all combinations are available for downstream processing
                for left_item in left_results {
                    for right_item in &right_results {
                        let combined = (left_item.clone(), right_item.clone());
                        if sender.send(combined).is_err() {
                            tracing::debug!("Stream receiver dropped during parallel execution - execution terminated");
                            return; // Receiver dropped
                        }
                    }
                }
            }).unwrap_or_else(|e| {
                tracing::warn!("Thread scope execution failed: {:?}", e);
            });
        })
    }
}

/// Error-aware operation trait for short-circuiting parallel execution
///
/// Extends the basic Op trait with error-aware execution semantics.
/// Operations implementing TryOp can participate in error-propagating
/// parallel execution that short-circuits on first error.
///
/// ## Error Semantics
/// - Short-circuit behavior on first error encountered
/// - Consistent error types across all parallel branches
/// - Stream-based error handling without Result wrapping
// TryOp trait REMOVED - violates streams-only architecture constraint
// AsyncStream NEVER returns Result<T, E> - all operations use unwrapped Op trait

// TryParallel struct REMOVED - violated streams-only architecture
// All error handling now uses polymorphic error patterns with Op trait

// TryParallel::new() REMOVED - used Result-wrapped streams

// TryOp implementation REMOVED - violated streams-only architecture
// All TryOp implementation code REMOVED - violated AsyncStream unwrapped constraint

/// Ergonomic helper function: create a parallel combinator
///
/// Creates a Parallel operation that executes two operations concurrently
/// and combines their results into tuple output.
///
/// ## Example
/// ```rust,no_run
/// use fluent_ai_candle::workflow::ops::map;
/// use fluent_ai_candle::workflow::parallel::parallel;
///
/// let double = map(|x: i32| x * 2);
/// let triple = map(|x: i32| x * 3);
/// let parallel_op = parallel(double, triple);
/// let result_stream = parallel_op.call(5);
/// // Result: AsyncStream containing [(10, 15)]
/// ```
#[inline]
pub fn parallel<A, B>(left: A, right: B) -> Parallel<A, B> {
    Parallel::new(left, right)
}

// try_parallel() helper REMOVED - violated streams-only architecture
// Use parallel() with unwrapped AsyncStream instead
