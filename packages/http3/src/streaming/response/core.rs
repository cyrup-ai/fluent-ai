//! Core HTTP response types and functionality
//!
//! This module provides the main HttpResponse type and related functionality.

use crate::streaming::chunks::HttpChunk;
use crate::types::{HttpVersion, RequestMetadata, TimeoutConfig};

// Re-export HttpChunk as HttpResponse for backward compatibility
pub type HttpResponse = HttpChunk;
