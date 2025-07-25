//! Core client functionality modules
//!
//! This module contains the decomposed core client implementation including
//! the main struct definition, initialization, and streaming functionality.

pub mod client;
pub mod initialization; 
pub mod streaming;

// Re-export main client type for public API
pub use client::CandleCompletionClient;