use std::sync::atomic::{AtomicU64, Ordering};

/// High-performance lock-free counter using relaxed memory ordering
/// Zero-allocation operations with atomic guarantees
#[derive(Debug, Default)]
pub struct RelaxedCounter {
    value: AtomicU64,
}

impl RelaxedCounter {
    /// Create a new counter with initial value
    #[inline]
    pub const fn new(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
        }
    }

    /// Create a new counter starting at zero
    #[inline]
    pub const fn default() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Create a new counter with initial value (alias for new)
    #[inline]
    pub const fn with_value(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
        }
    }

    /// Get current counter value with zero allocation
    /// Uses relaxed memory ordering for maximum performance
    #[inline]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Atomic increment returning previous value
    /// Zero-allocation, lock-free operation
    #[inline]
    pub fn inc(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    /// Atomic increment by N returning previous value
    #[inline]
    pub fn inc_by(&self, n: u64) -> u64 {
        self.value.fetch_add(n, Ordering::Relaxed)
    }

    /// Atomic fetch_add operation (matches AtomicU64 interface)
    /// Returns previous value before addition
    #[inline]
    pub fn fetch_add(&self, val: u64, _ordering: std::sync::atomic::Ordering) -> u64 {
        self.value.fetch_add(val, Ordering::Relaxed)
    }

    /// Atomic decrement with underflow protection
    /// Returns previous value, saturates at 0
    #[inline]
    pub fn sub(&self) -> u64 {
        self.value
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            })
            .unwrap_or(0)
    }

    /// Atomic decrement by N with underflow protection
    #[inline]
    pub fn sub_by(&self, n: u64) -> u64 {
        self.value
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(n)
            })
            .unwrap_or(0)
    }

    /// Reset counter to zero, returns previous value
    #[inline]
    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }

    /// Compare and swap operation
    #[inline]
    pub fn compare_and_swap(&self, current: u64, new: u64) -> Result<u64, u64> {
        self.value
            .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
    }

    /// Set counter to specific value, returns previous value
    #[inline]
    pub fn set(&self, value: u64) -> u64 {
        self.value.swap(value, Ordering::Relaxed)
    }

    /// Store counter value (compatible with AtomicU64 interface)
    #[inline]
    pub fn store(&self, value: u64, _ordering: std::sync::atomic::Ordering) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment counter and return new value
    #[inline]
    pub fn inc_and_get(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrement counter and return new value (with underflow protection)
    #[inline]
    pub fn sub_and_get(&self) -> u64 {
        self.value
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            })
            .map(|prev| prev.saturating_sub(1))
            .unwrap_or(0)
    }
}

impl Clone for RelaxedCounter {
    fn clone(&self) -> Self {
        Self::with_value(self.get())
    }
}

impl PartialEq for RelaxedCounter {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for RelaxedCounter {}

impl std::fmt::Display for RelaxedCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl From<u64> for RelaxedCounter {
    fn from(value: u64) -> Self {
        Self::with_value(value)
    }
}

impl From<&RelaxedCounter> for u64 {
    fn from(counter: &RelaxedCounter) -> Self {
        counter.get()
    }
}
