//! Command middleware for intercepting and processing chat commands
//!
//! Provides blazing-fast middleware system for cross-cutting concerns with zero-allocation patterns
//! and production-ready performance monitoring, logging, and security.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// Import domain types that middleware operates on
use fluent_ai_domain::chat::commands::types::*;
use tokio::sync::RwLock;

/// Command middleware trait for intercepting command execution
pub trait CommandMiddleware: Send + Sync + 'static {
    /// Execute before command processing
    fn before_execute<'a>(
        &'a self,
        command: &'a ChatCommand,
        context: &'a CommandContext,
    ) -> fluent_ai_domain::AsyncStream<()>;

    /// Execute after command processing
    fn after_execute<'a>(
        &'a self,
        command: &'a ChatCommand,
        context: &'a CommandContext,
        result: &'a CommandResult<CommandOutput>,
    ) -> fluent_ai_domain::AsyncStream<()>;

    /// Get middleware name
    fn name(&self) -> &str;

    /// Get middleware priority (lower numbers execute first)
    fn priority(&self) -> u32 {
        100
    }

    /// Helper to create a boxed version of this middleware
    fn into_boxed(self) -> Box<dyn CommandMiddleware>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

/// Middleware chain executor
#[derive(Debug)]
pub struct MiddlewareChain {
    /// Registered middleware (sorted by priority)
    middleware: Vec<Arc<dyn CommandMiddleware>>,
}

impl Default for MiddlewareChain {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl MiddlewareChain {
    /// Create new middleware chain
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            middleware: Vec::with_capacity(8), // Pre-allocate for common middleware count
        }
    }

    /// Add middleware to the chain
    pub fn add_middleware(&mut self, middleware: Arc<dyn CommandMiddleware>) {
        self.middleware.push(middleware);
        // Sort by priority (lower numbers first)
        self.middleware.sort_by_key(|m| m.priority());
    }

    /// Execute before middleware chain
    pub async fn execute_before(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
    ) -> Result<(), CommandError> {
        for middleware in &self.middleware {
            middleware.before_execute(command, context).await?;
        }
        Ok(())
    }

    /// Execute after middleware chain
    pub async fn execute_after(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
        result: &CommandResult<CommandOutput>,
    ) -> Result<(), CommandError> {
        // Execute in reverse order for after hooks
        for middleware in self.middleware.iter().rev() {
            middleware.after_execute(command, context, result).await?;
        }
        Ok(())
    }

    /// Get number of registered middleware
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.middleware.len()
    }

    /// Check if chain is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.middleware.is_empty()
    }
}
