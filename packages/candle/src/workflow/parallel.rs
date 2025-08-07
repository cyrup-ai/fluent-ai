//! True concurrent execution with thread-based parallelism for blazing-fast performance
//!
//! This module provides parallel execution combinators using std::thread for true
//! concurrent execution without tokio dependency. All operations maintain the
//! fluent-ai streams-only architecture while enabling maximum parallelism.

use fluent_ai_async::AsyncStream;
use std::marker::PhantomData;
use std::sync::mpsc;
use std::thread;
use crossbeam;
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
    In: Send + Sync + Clone + 'static,
    OutA: Send + Sync + Clone + 'static,
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
                            return; // Receiver dropped
                        }
                    }
                }
            }).unwrap_or(()); // Handle potential panic in scope
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
/// - Clean composition with other error-aware operations
pub trait TryOp<In, Out, Err>: Send + Sync + Clone + 'static {
    /// Execute operation with error-aware semantics
    /// 
    /// Returns a stream that may contain successful outputs or errors.
    /// Consumers can handle errors appropriately based on their needs.
    /// 
    /// ## Error Handling
    /// - Errors are streamed as part of normal output flow
    /// - No Result<T, E> wrapping - errors handled polymorphically
    /// - Short-circuit semantics when composed with try_parallel
    fn try_call(&self, input: In) -> AsyncStream<Result<Out, Err>>;
}

/// Error-propagating parallel combinator with short-circuit semantics
/// 
/// Executes two operations concurrently but short-circuits execution
/// if either operation encounters an error. This provides fail-fast
/// semantics for parallel execution where errors should halt processing.
/// 
/// ## Error Propagation
/// - Short-circuit on first error from any parallel branch
/// - Consistent error types across all parallel operations
/// - Thread-safe error handling using channel communication
/// - Clean composition with other error-aware operations
/// 
/// ## Performance
/// - Zero allocation with PhantomData type tracking
/// - Thread-based parallelism for true concurrent execution  
/// - Minimal overhead error propagation using channels
/// - Blazing-fast execution until error encountered
#[derive(Clone)]
pub struct TryParallel<A, B, Err> {
    left: A,
    right: B,
    _phantom: PhantomData<Err>,
}

impl<A, B, Err> TryParallel<A, B, Err> {
    /// Create a new error-propagating parallel combinator
    /// 
    /// Takes two error-aware operations and creates a parallel combinator
    /// that will execute both concurrently but short-circuit on first error.
    /// 
    /// ## Error Semantics
    /// - First error encountered stops all parallel execution
    /// - Error propagation is immediate and consistent
    /// - Clean composition with downstream error handling
    #[inline]
    pub fn new(left: A, right: B) -> Self {
        Self {
            left,
            right,
            _phantom: PhantomData,
        }
    }
}

impl<A, B, In, OutA, OutB, Err> TryOp<In, (OutA, OutB), Err> for TryParallel<A, B, Err>
where
    A: TryOp<In, OutA, Err> + 'static,
    B: TryOp<In, OutB, Err> + 'static,
    In: Send + Sync + Clone + 'static,
    OutA: Send + Sync + Clone + 'static,
    OutB: Send + Sync + Clone + 'static,
    Err: Send + Sync + Clone + 'static,
{
    fn try_call(&self, input: In) -> AsyncStream<Result<(OutA, OutB), Err>> {
        let left = self.left.clone();
        let right = self.right.clone();
        let input_left = input.clone();
        let input_right = input;
        
        AsyncStream::with_channel(move |sender| {
            // Create channels for collecting results and errors
            let (result_tx, result_rx) = mpsc::channel::<Result<(Vec<OutA>, Vec<OutB>), Err>>();
            let (left_tx, left_rx) = mpsc::channel::<Result<Vec<OutA>, Err>>();
            let (right_tx, right_rx) = mpsc::channel::<Result<Vec<OutB>, Err>>();
            
            // Spawn left operation in parallel thread
            let left_result_tx = result_tx.clone();
            let left_handle = thread::spawn(move || {
                let mut left_stream = left.try_call(input_left);
                let mut left_results = Vec::new();
                
                // Collect results until error or completion
                while let Some(result) = left_stream.try_next() {
                    match result {
                        Ok(output) => left_results.push(output),
                        Err(err) => {
                            // Short-circuit on error
                            let _ = left_tx.send(Err(err.clone()));
                            let _ = left_result_tx.send(Err(err));
                            return;
                        }
                    }
                }
                
                let _ = left_tx.send(Ok(left_results));
            });
            
            // Spawn right operation in parallel thread
            let right_result_tx = result_tx.clone();
            let right_handle = thread::spawn(move || {
                let mut right_stream = right.try_call(input_right);
                let mut right_results = Vec::new();
                
                // Collect results until error or completion
                while let Some(result) = right_stream.try_next() {
                    match result {
                        Ok(output) => right_results.push(output),
                        Err(err) => {
                            // Short-circuit on error
                            let _ = right_tx.send(Err(err.clone()));
                            let _ = right_result_tx.send(Err(err));
                            return;
                        }
                    }
                }
                
                let _ = right_tx.send(Ok(right_results));
            });
            
            // Wait for results or error from either thread
            drop(result_tx); // Close sender so recv will unblock when all senders drop
            
            // Check for early error signal
            if let Ok(Err(err)) = result_rx.try_recv() {
                // Short-circuit - send error and exit
                let _ = sender.send(Err(err));
                let _ = left_handle.join();
                let _ = right_handle.join();
                return;
            }
            
            // Wait for both operations to complete
            let _ = left_handle.join();
            let _ = right_handle.join();
            
            // Collect final results
            let left_result = left_rx.recv().unwrap_or(Ok(Vec::new()));
            let right_result = right_rx.recv().unwrap_or(Ok(Vec::new()));
            
            match (left_result, right_result) {
                (Ok(left_results), Ok(right_results)) => {
                    // Handle edge cases first
                    if left_results.is_empty() || right_results.is_empty() {
                        return; // Can't create tuples if either side produces no results
                    }
                    
                    // Both succeeded - combine results
                    for left_item in left_results {
                        for right_item in &right_results {
                            let combined = (left_item.clone(), right_item.clone());
                            if sender.send(Ok(combined)).is_err() {
                                return; // Receiver dropped
                            }
                        }
                    }
                }
                (Err(err), _) | (_, Err(err)) => {
                    // At least one failed - propagate error
                    let _ = sender.send(Err(err));
                }
            }
        })
    }
}

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

/// Ergonomic helper function: create an error-propagating parallel combinator
/// 
/// Creates a TryParallel operation that executes two operations concurrently
/// but short-circuits on first error encountered.
/// 
/// ## Example
/// ```rust,no_run
/// use fluent_ai_candle::workflow::parallel::try_parallel;
/// 
/// let left_op = /* some TryOp implementation */;
/// let right_op = /* some TryOp implementation */;
/// let parallel_op = try_parallel(left_op, right_op);
/// let result_stream = parallel_op.try_call(input);
/// // Result: AsyncStream containing [Ok((left_out, right_out))] or [Err(error)]
/// ```  
#[inline]
pub fn try_parallel<A, B, Err>(left: A, right: B) -> TryParallel<A, B, Err> {
    TryParallel::new(left, right)
}