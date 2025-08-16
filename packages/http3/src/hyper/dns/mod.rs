//! DNS resolution

pub use resolve::{Addrs, DynResolver, GaiResolver, Name, Resolve, Resolving};

pub(crate) mod gai;
#[cfg(feature = "hickory-dns")]
pub(crate) mod hickory;
pub(crate) mod resolve;

// Type alias for compatibility
pub type DnsResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
