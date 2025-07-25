//! Immutable message formatting with streaming operations
//!
//! This module provides a complete formatting system with zero-allocation, lock-free
//! message formatting and streaming operations. The module is organized into focused 
//! submodules for optimal maintainability and performance.
//!
//! # Architecture
//!
//! - Zero-allocation patterns with owned strings and borrowed data
//! - Lock-free atomic operations for thread safety
//! - Streaming-first design with AsyncStream patterns
//! - Immutable data structures for consistency
//! - Comprehensive validation and error handling

pub mod content;
pub mod error;
pub mod options;
pub mod streaming;
pub mod styles;
pub mod themes;

#[cfg(test)]
mod tests;

// Re-export all public types for convenience
pub use content::ImmutableMessageContent;
pub use error::{FormatError, FormatResult};
pub use options::ImmutableFormatOptions;
pub use themes::{ImmutableColorScheme, OutputFormat, SyntaxTheme};
pub use streaming::{FormattingEvent, FormatterStats, StreamingMessageFormatter};
pub use styles::{FormatStyle, ImmutableCustomFormatRule, StyleType};

// Legacy compatibility re-exports removed - use Immutable* versions instead

// MessageFormatter deprecated - use StreamingMessageFormatter instead