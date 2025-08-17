//! Core HTTP response types and functionality
//!
//! This module provides the main HttpResponse type and related functionality.

use crate::types::HttpResponseChunk;

// Re-export HttpResponseChunk as HttpResponse for backward compatibility
pub type HttpResponse = HttpResponseChunk;
