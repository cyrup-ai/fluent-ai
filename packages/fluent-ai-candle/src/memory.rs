//! Zero-allocation memory tracking and management for high-performance ML inference
//!
//! This module provides lightweight, lock-free memory tracking capabilities designed
//! for high-performance ML inference workloads. It tracks global memory usage, peak
//! usage, and allocation counts using atomic operations for thread-safe monitoring
//! without performance overhead.
//!
//! # Features
//!
//! - **Lock-free tracking**: All operations use atomic primitives for thread safety
//! - **Zero overhead**: Tracking adds minimal CPU overhead to allocation paths
//! - **Peak monitoring**: Automatically tracks peak memory usage across application lifetime
//! - **Statistics collection**: Maintains allocation counts for profiling and debugging
//! - **Reset capability**: Allows statistics reset for testing and benchmarking
//!
//! # Usage
//!
//! ```rust
//! use fluent_ai_candle::memory;
//!
//! // Track an allocation
//! memory::track_allocation(1024);
//! 
//! // Check current usage
//! let current = memory::current_usage();
//! println!("Current memory usage: {} bytes", current);
//!
//! // Track deallocation
//! memory::track_deallocation(1024);
//!
//! // Check peak usage
//! let peak = memory::peak_usage();
//! println!("Peak memory usage: {} bytes", peak);
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global memory usage counter tracking current total allocated bytes
///
/// This atomic counter maintains the current total memory usage across
/// all tracked allocations. Uses relaxed ordering for optimal performance
/// in high-frequency allocation scenarios.
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

/// Peak memory usage counter tracking maximum allocated bytes
///
/// Automatically updated whenever current usage exceeds the previous peak.
/// Uses compare-and-swap operations to handle concurrent updates correctly.
static PEAK_USAGE: AtomicUsize = AtomicUsize::new(0);

/// Total allocation count for profiling and debugging
///
/// Tracks the total number of allocation events, useful for analyzing
/// allocation patterns and frequency in ML workloads.
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Track memory allocation atomically with peak usage monitoring
///
/// Records a new memory allocation by updating both current usage and
/// allocation count atomically. Automatically updates peak usage if
/// the new total exceeds the previous maximum.
///
/// # Arguments
///
/// * `size` - Size of the allocation in bytes
///
/// # Performance
///
/// Uses relaxed atomic ordering for optimal performance. The peak usage
/// update uses compare-and-swap in a loop to handle concurrent updates
/// correctly without blocking.
///
/// # Thread Safety
///
/// Safe to call from multiple threads concurrently. All operations are
/// atomic and lock-free.
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::memory;
///
/// // Track a 1KB allocation
/// memory::track_allocation(1024);
///
/// // Track a larger allocation
/// memory::track_allocation(1024 * 1024); // 1MB
/// ```
#[inline(always)]
pub fn track_allocation(size: usize) {
    let current = TOTAL_ALLOCATED.fetch_add(size, Ordering::Relaxed) + size;
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);

    // Update peak usage atomically
    let mut peak = PEAK_USAGE.load(Ordering::Relaxed);
    while current > peak {
        match PEAK_USAGE.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(x) => peak = x}
    }
}

/// Track memory deallocation atomically with current usage update
///
/// Records a memory deallocation by decreasing the current usage counter.
/// Does not affect peak usage (which tracks historical maximum) or allocation
/// count (which tracks total allocation events).
///
/// # Arguments
///
/// * `size` - Size of the deallocation in bytes
///
/// # Performance
///
/// Uses relaxed atomic ordering for optimal performance. Single atomic
/// subtraction operation with minimal overhead.
///
/// # Thread Safety
///
/// Safe to call from multiple threads concurrently. Uses atomic operations
/// for thread-safe updates.
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::memory;
///
/// // Track allocation then deallocation
/// memory::track_allocation(1024);
/// memory::track_deallocation(1024);
/// assert_eq!(memory::current_usage(), 0);
/// ```
#[inline(always)]
pub fn track_deallocation(size: usize) {
    TOTAL_ALLOCATED.fetch_sub(size, Ordering::Relaxed);
}

/// Get current total memory usage in bytes
///
/// Returns the current total memory usage across all tracked allocations.
/// This represents the net allocated memory (allocations minus deallocations).
///
/// # Returns
///
/// Current total allocated memory in bytes
///
/// # Performance
///
/// Single atomic load operation with relaxed ordering for optimal performance.
///
/// # Thread Safety
///
/// Safe to call from multiple threads. Returns a consistent snapshot of
/// current usage at the time of the call.
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::memory;
///
/// memory::track_allocation(2048);
/// let usage = memory::current_usage();
/// assert_eq!(usage, 2048);
/// ```
#[inline(always)]
pub fn current_usage() -> usize {
    TOTAL_ALLOCATED.load(Ordering::Relaxed)
}

/// Get peak memory usage in bytes since program start or last reset
///
/// Returns the maximum memory usage that has been reached at any point
/// since program start or the last call to `reset_stats()`. This value
/// only increases and is never decreased by deallocations.
///
/// # Returns
///
/// Peak memory usage in bytes
///
/// # Performance
///
/// Single atomic load operation with relaxed ordering for optimal performance.
///
/// # Thread Safety
///
/// Safe to call from multiple threads. Returns a consistent snapshot of
/// peak usage at the time of the call.
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::memory;
///
/// memory::track_allocation(1024);
/// memory::track_allocation(2048);
/// memory::track_deallocation(1024);
/// 
/// // Peak is still 3072 despite deallocation
/// assert_eq!(memory::peak_usage(), 3072);
/// ```
#[inline(always)]
pub fn peak_usage() -> usize {
    PEAK_USAGE.load(Ordering::Relaxed)
}

/// Get total allocation count since program start or last reset
///
/// Returns the total number of allocation events tracked since program
/// start or the last call to `reset_stats()`. This count only increases
/// and is not affected by deallocations.
///
/// # Returns
///
/// Total number of allocation events
///
/// # Performance
///
/// Single atomic load operation with relaxed ordering for optimal performance.
///
/// # Thread Safety
///
/// Safe to call from multiple threads. Returns a consistent snapshot of
/// allocation count at the time of the call.
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::memory;
///
/// memory::track_allocation(1024);
/// memory::track_allocation(2048);
/// memory::track_deallocation(1024);
/// 
/// // Count is 2 despite one deallocation
/// assert_eq!(memory::allocation_count(), 2);
/// ```
#[inline(always)]
pub fn allocation_count() -> usize {
    ALLOCATION_COUNT.load(Ordering::Relaxed)
}

/// Reset all memory tracking statistics to zero
///
/// Resets all memory tracking counters to zero:
/// - Current usage
/// - Peak usage
/// - Allocation count
///
/// This is useful for benchmarking, testing, or resetting statistics
/// at specific points in program execution.
///
/// # Performance
///
/// Three atomic store operations with relaxed ordering for optimal performance.
///
/// # Thread Safety
///
/// Safe to call from multiple threads, but may create temporary inconsistencies
/// if called concurrently with tracking operations. For most accurate results,
/// call during quiescent periods.
///
/// # Examples
///
/// ```rust
/// use fluent_ai_candle::memory;
///
/// memory::track_allocation(1024);
/// assert_ne!(memory::current_usage(), 0);
/// 
/// memory::reset_stats();
/// assert_eq!(memory::current_usage(), 0);
/// assert_eq!(memory::peak_usage(), 0);
/// assert_eq!(memory::allocation_count(), 0);
/// ```
#[inline(always)]
pub fn reset_stats() {
    TOTAL_ALLOCATED.store(0, Ordering::Relaxed);
    PEAK_USAGE.store(0, Ordering::Relaxed);
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
}
