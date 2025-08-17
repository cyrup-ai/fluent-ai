//! DNS resolution

pub use resolve::{Addrs, DynResolver, GaiResolver, Name, Resolve, Resolving};

pub(crate) mod gai;
pub(crate) mod hickory;
pub(crate) mod resolve;

// Type alias for compatibility
pub type DnsResult<T> = Result<T, crate::error::HttpError>;
