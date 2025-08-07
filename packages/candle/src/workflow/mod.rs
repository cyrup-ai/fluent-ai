//! Real workflow execution system - streams-only architecture
//!
//! This module provides a complete workflow execution system built on fluent-ai
//! streams-only architecture. All operations use AsyncStream without Future/Result
//! wrapping in execution paths.
//!
//! ## Core Components
//! - **core**: WorkflowStep trait and Workflow execution struct  
//! - **ops**: Zero-cost operation combinators and transformations
//! - **parallel**: Thread-based parallel execution combinators
//! - **macros**: Compile-time variadic parallel execution macros
//!
//! ## Architecture Principles
//! - Zero-allocation with PhantomData for type safety
//! - Streams-only execution (no Future/Result wrapping)
//! - Thread-based concurrency (no tokio dependency)
//! - Extensive inlining for blazing-fast performance
//! - Lock-free design for maximum throughput

pub mod core;
pub mod ops;
pub mod parallel;
pub mod macros;

// Re-export core types for ergonomic imports
pub use core::{WorkflowStep, Workflow};
pub use ops::{Op, map, passthrough, then};
pub use parallel::{Parallel, TryParallel, TryOp, parallel, try_parallel};

// Re-export macros for public use
pub use macros::{parallel, try_parallel, parallel_op, try_parallel_internal, count_ops};