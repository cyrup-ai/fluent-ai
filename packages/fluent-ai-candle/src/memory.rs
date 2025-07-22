//! Zero-allocation memory tracking and management for high-performance ML inference

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global memory usage tracking
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_USAGE: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Track memory allocation atomically
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
            Err(x) => peak = x,
        }
    }
}

/// Track memory deallocation atomically
#[inline(always)]
pub fn track_deallocation(size: usize) {
    TOTAL_ALLOCATED.fetch_sub(size, Ordering::Relaxed);
}

/// Get current memory usage
#[inline(always)]
pub fn current_usage() -> usize {
    TOTAL_ALLOCATED.load(Ordering::Relaxed)
}

/// Get peak memory usage
#[inline(always)]
pub fn peak_usage() -> usize {
    PEAK_USAGE.load(Ordering::Relaxed)
}

/// Get allocation count
#[inline(always)]
pub fn allocation_count() -> usize {
    ALLOCATION_COUNT.load(Ordering::Relaxed)
}

/// Reset memory statistics
#[inline(always)]
pub fn reset_stats() {
    TOTAL_ALLOCATED.store(0, Ordering::Relaxed);
    PEAK_USAGE.store(0, Ordering::Relaxed);
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
}
