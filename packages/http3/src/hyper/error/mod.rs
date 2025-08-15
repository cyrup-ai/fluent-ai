pub mod types;
pub mod constructors;
pub mod classification;
pub mod helpers;

// Re-export main types and functions for backward compatibility
pub use types::{Error, Result, Kind, Inner};
pub use constructors::*;
pub use helpers::{TimedOut, BadScheme, ConnectionClosed, OperationCanceled, IncompleteMessage, UnexpectedMessage};

// Re-export internal types needed by other modules
pub(crate) type BoxError = Box<dyn std::error::Error + Send + Sync>;

// Re-export classification methods through the Error type
// (these are implemented as inherent methods on Error in classification.rs)
