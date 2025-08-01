//! Context module
//!
//! This module provides comprehensive context management functionality including
//! application context, sampling context, subscriptions, and global context
//! management with zero allocation patterns and blazing-fast performance.

// Import types from sibling modules
pub mod logger;
pub mod memory_adapter;
pub mod rpc;

// Import core context functionality
pub mod core;

// Re-export key types/functions from submodules
pub use logger::ConsoleLogger;
pub use memory_adapter::MemoryContextAdapter;
pub use rpc::{
    ContextChangedNotification, ContextContent, ContextItem, GetContextRequest, GetContextResult,
    SubscribeContextRequest, SubscribeContextResult, context_get, context_subscribe,
};

// Re-export core context types and functions
pub use core::{
    APPLICATION_CONTEXT, ApplicationContext, CONTEXT_SUBSCRIPTIONS, ContextStats,
    ContextSubscription, ContextSubscriptionManager, GlobalContextManager, GlobalContextStats,
    SAMPLING_CONTEXT, SamplingContext, SamplingStats, SubscriptionStats, context_access,
    initialize_global_context,
};

// Convenience functions for creating contexts and subscriptions
pub use core::{
    app_context, sampling_context, subscription, try_app_context, try_sampling_context,
};
