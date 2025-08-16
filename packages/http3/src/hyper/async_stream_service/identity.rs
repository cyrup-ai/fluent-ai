//! Identity layer implementation for AsyncStream services
//!
//! Provides a no-op passthrough layer equivalent to tower::layer::util::Identity
//! for use in AsyncStream service composition.

use super::core::AsyncStreamLayer;

/// Identity layer - AsyncStream equivalent of tower::layer::util::Identity
#[derive(Clone, Debug)]
pub struct AsyncStreamIdentityLayer;

impl AsyncStreamIdentityLayer {
    /// Create a new identity layer (no-op passthrough)
    pub fn new() -> Self {
        Self
    }
}

impl<S> AsyncStreamLayer<S> for AsyncStreamIdentityLayer {
    type Service = S;

    fn layer(&self, service: S) -> Self::Service {
        service
    }
}
