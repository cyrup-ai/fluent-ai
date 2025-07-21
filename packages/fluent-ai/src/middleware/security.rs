//! Security middleware for authorization and access control
//!
//! Provides blazing-fast security checks with zero-allocation patterns
//! and production-ready authorization mechanisms.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_utils::CachePadded;
use fluent_ai_domain::chat::commands::types::*;

use super::command::CommandMiddleware;

/// Security policy for command execution
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Allowed command types
    pub allowed_commands: HashSet<String>,
    /// Rate limit per user (commands per minute)
    pub rate_limit: u32,
    /// Maximum execution time in seconds
    pub max_execution_time: u32,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            allowed_commands: HashSet::new(),
            rate_limit: 60,          // 60 commands per minute
            max_execution_time: 300, // 5 minutes
        }
    }
}

/// Rate limiting tracker
#[derive(Debug, Default)]
pub struct RateLimiter {
    /// Request count in current window
    request_count: CachePadded<AtomicU64>,
    /// Window start time (Unix timestamp)
    window_start: CachePadded<AtomicU64>,
}

impl RateLimiter {
    /// Create new rate limiter
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if request is allowed under rate limit
    pub fn is_allowed(&self, limit: u32) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let window_start = self.window_start.load(Ordering::Relaxed);

        // Reset window if more than 60 seconds have passed
        if now - window_start >= 60 {
            self.window_start.store(now, Ordering::Relaxed);
            self.request_count.store(0, Ordering::Relaxed);
        }

        let current_count = self.request_count.fetch_add(1, Ordering::Relaxed);
        current_count < limit as u64
    }
}

/// Security middleware for authorization and rate limiting
#[derive(Debug)]
pub struct SecurityMiddleware {
    /// Security policy
    policy: SecurityPolicy,
    /// Rate limiter
    rate_limiter: Arc<RateLimiter>,
    /// Middleware name
    name: String,
}

impl SecurityMiddleware {
    /// Create new security middleware
    pub fn new(policy: SecurityPolicy) -> Self {
        Self {
            policy,
            rate_limiter: Arc::new(RateLimiter::new()),
            name: "security".to_string(),
        }
    }

    /// Create security middleware with default policy
    pub fn with_defaults() -> Self {
        Self::new(SecurityPolicy::default())
    }

    /// Check if command is authorized
    fn is_command_authorized(&self, command: &ChatCommand) -> bool {
        if self.policy.allowed_commands.is_empty() {
            return true; // Allow all if no restrictions
        }

        let command_type = match command {
            ChatCommand::Help { .. } => "help",
            ChatCommand::Clear { .. } => "clear",
            ChatCommand::History { .. } => "history",
            ChatCommand::Save { .. } => "save",
            ChatCommand::Load { .. } => "load",
            ChatCommand::Export { .. } => "export",
            ChatCommand::Import { .. } => "import",
            ChatCommand::Settings { .. } => "settings",
            ChatCommand::Debug { .. } => "debug",
            ChatCommand::Custom { name, .. } => name,
            ChatCommand::Config { .. } => "config",
            ChatCommand::Template { .. } => "template",
            ChatCommand::Macro { .. } => "macro",
            ChatCommand::Search { .. } => "search",
            ChatCommand::Branch { .. } => "branch",
            ChatCommand::Session { .. } => "session",
            ChatCommand::Tool { .. } => "tool",
            ChatCommand::Stats { .. } => "stats",
            ChatCommand::Theme { .. } => "theme",
        };

        self.policy.allowed_commands.contains(command_type)
    }
}

impl Default for SecurityMiddleware {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl CommandMiddleware for SecurityMiddleware {
    fn before_execute<'a>(
        &'a self,
        command: &'a ChatCommand,
        _context: &'a CommandContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), CommandError>> + Send + 'a>>
    {
        Box::pin(async move {
            // Check rate limiting
            if !self.rate_limiter.is_allowed(self.policy.rate_limit) {
                return Err(CommandError::RateLimitExceeded);
            }

            // Check command authorization
            if !self.is_command_authorized(command) {
                return Err(CommandError::Unauthorized);
            }

            Ok(())
        })
    }

    fn after_execute<'a>(
        &'a self,
        _command: &'a ChatCommand,
        _context: &'a CommandContext,
        _result: &'a CommandResult<CommandOutput>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), CommandError>> + Send + 'a>>
    {
        Box::pin(async move {
            // Security post-processing if needed
            Ok(())
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u32 {
        1 // Highest priority for security checks
    }
}
