//! Command middleware and interceptors
//!
//! Provides blazing-fast middleware system for cross-cutting concerns with zero-allocation patterns
//! and production-ready performance monitoring, logging, and security.

use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use tokio::sync::RwLock;

use super::types::*;

/// Command middleware trait for intercepting command execution
#[async_trait::async_trait]
pub trait CommandMiddleware: Send + Sync {
    /// Execute before command processing
    async fn before_execute(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
    ) -> Result<(), CommandError>;

    /// Execute after command processing
    async fn after_execute(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
        result: &CommandResult<CommandOutput>,
    ) -> Result<(), CommandError>;

    /// Get middleware name
    fn name(&self) -> &str;

    /// Get middleware priority (lower numbers execute first)
    fn priority(&self) -> u32 {
        100
    }
}

/// Middleware chain executor
#[derive(Debug)]
pub struct MiddlewareChain {
    /// Registered middleware (sorted by priority)
    middleware: Vec<Arc<dyn CommandMiddleware>>,
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

impl MiddlewareChain {
    /// Create a new middleware chain
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
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

    /// Execute after middleware chain (in reverse order)
    pub async fn execute_after(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
        result: &CommandResult<CommandOutput>,
    ) -> Result<(), CommandError> {
        for middleware in self.middleware.iter().rev() {
            middleware.after_execute(command, context, result).await?;
        }
        Ok(())
    }

    /// Get middleware count
    pub fn count(&self) -> usize {
        self.middleware.len()
    }

    /// List middleware names
    pub fn list_middleware(&self) -> Vec<String> {
        self.middleware.iter()
            .map(|m| m.name().to_string())
            .collect()
    }
}

/// Logging middleware for command execution
#[derive(Debug)]
pub struct LoggingMiddleware {
    /// Log level
    level: LogLevel,
    /// Include command details
    include_details: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LoggingMiddleware {
    /// Create new logging middleware
    pub fn new(level: LogLevel, include_details: bool) -> Self {
        Self { level, include_details }
    }
}

#[async_trait::async_trait]
impl CommandMiddleware for LoggingMiddleware {
    async fn before_execute(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
    ) -> Result<(), CommandError> {
        let command_name = match command {
            ChatCommand::Help { .. } => "help",
            ChatCommand::Clear { .. } => "clear",
            ChatCommand::Export { .. } => "export",
            ChatCommand::Config { .. } => "config",
            ChatCommand::Search { .. } => "search",
            ChatCommand::Template { .. } => "template",
            ChatCommand::Macro { .. } => "macro",
            ChatCommand::Branch { .. } => "branch",
            ChatCommand::Session { .. } => "session",
            ChatCommand::Tool { .. } => "tool",
            ChatCommand::Stats { .. } => "stats",
            ChatCommand::Theme { .. } => "theme",
            ChatCommand::Debug { .. } => "debug",
        };

        if self.include_details {
            tracing::info!(
                command = command_name,
                user_id = %context.user_id,
                session_id = %context.session_id,
                "Executing command"
            );
        } else {
            tracing::info!(command = command_name, "Executing command");
        }

        Ok(())
    }

    async fn after_execute(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
        result: &CommandResult<CommandOutput>,
    ) -> Result<(), CommandError> {
        let command_name = match command {
            ChatCommand::Help { .. } => "help",
            ChatCommand::Clear { .. } => "clear",
            ChatCommand::Export { .. } => "export",
            ChatCommand::Config { .. } => "config",
            ChatCommand::Search { .. } => "search",
            ChatCommand::Template { .. } => "template",
            ChatCommand::Macro { .. } => "macro",
            ChatCommand::Branch { .. } => "branch",
            ChatCommand::Session { .. } => "session",
            ChatCommand::Tool { .. } => "tool",
            ChatCommand::Stats { .. } => "stats",
            ChatCommand::Theme { .. } => "theme",
            ChatCommand::Debug { .. } => "debug",
        };

        match result {
            Ok(output) => {
                if self.include_details {
                    tracing::info!(
                        command = command_name,
                        user_id = %context.user_id,
                        session_id = %context.session_id,
                        execution_time_us = output.execution_time,
                        success = output.success,
                        "Command completed"
                    );
                } else {
                    tracing::info!(
                        command = command_name,
                        success = output.success,
                        "Command completed"
                    );
                }
            }
            Err(error) => {
                tracing::error!(
                    command = command_name,
                    error = %error,
                    "Command failed"
                );
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "logging"
    }

    fn priority(&self) -> u32 {
        10 // High priority (execute early)
    }
}

/// Performance monitoring middleware
#[derive(Debug)]
pub struct PerformanceMiddleware {
    /// Performance metrics storage
    metrics: Arc<RwLock<HashMap<String, PerformanceMetrics>>>,
    /// Slow command threshold in microseconds
    slow_threshold_us: u64,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_executions: u64,
    pub total_time_us: u64,
    pub average_time_us: u64,
    pub min_time_us: u64,
    pub max_time_us: u64,
    pub slow_executions: u64,
}

impl PerformanceMiddleware {
    /// Create new performance middleware
    pub fn new(slow_threshold_us: u64) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            slow_threshold_us,
        }
    }

    /// Get performance metrics
    pub async fn get_metrics(&self) -> HashMap<String, PerformanceMetrics> {
        self.metrics.read().await.clone()
    }
}

#[async_trait::async_trait]
impl CommandMiddleware for PerformanceMiddleware {
    async fn before_execute(
        &self,
        _command: &ChatCommand,
        _context: &CommandContext,
    ) -> Result<(), CommandError> {
        // Performance tracking is handled in after_execute
        Ok(())
    }

    async fn after_execute(
        &self,
        command: &ChatCommand,
        _context: &CommandContext,
        result: &CommandResult<CommandOutput>,
    ) -> Result<(), CommandError> {
        if let Ok(output) = result {
            let command_name = match command {
                ChatCommand::Help { .. } => "help",
                ChatCommand::Clear { .. } => "clear",
                ChatCommand::Export { .. } => "export",
                ChatCommand::Config { .. } => "config",
                ChatCommand::Search { .. } => "search",
                ChatCommand::Template { .. } => "template",
                ChatCommand::Macro { .. } => "macro",
                ChatCommand::Branch { .. } => "branch",
                ChatCommand::Session { .. } => "session",
                ChatCommand::Tool { .. } => "tool",
                ChatCommand::Stats { .. } => "stats",
                ChatCommand::Theme { .. } => "theme",
                ChatCommand::Debug { .. } => "debug",
            }.to_string();

            let execution_time = output.execution_time;
            let mut metrics = self.metrics.write().await;
            
            let entry = metrics.entry(command_name.clone()).or_insert(PerformanceMetrics {
                total_executions: 0,
                total_time_us: 0,
                average_time_us: 0,
                min_time_us: u64::MAX,
                max_time_us: 0,
                slow_executions: 0,
            });

            entry.total_executions += 1;
            entry.total_time_us += execution_time;
            entry.average_time_us = entry.total_time_us / entry.total_executions;
            entry.min_time_us = entry.min_time_us.min(execution_time);
            entry.max_time_us = entry.max_time_us.max(execution_time);

            if execution_time > self.slow_threshold_us {
                entry.slow_executions += 1;
                tracing::warn!(
                    command = command_name,
                    execution_time_us = execution_time,
                    threshold_us = self.slow_threshold_us,
                    "Slow command execution detected"
                );
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "performance"
    }

    fn priority(&self) -> u32 {
        20 // Medium priority
    }
}

/// Security middleware for command validation
#[derive(Debug)]
pub struct SecurityMiddleware {
    /// Allowed commands per user role
    role_permissions: HashMap<String, Vec<String>>,
    /// Rate limiting settings
    rate_limits: HashMap<String, RateLimit>,
}

#[derive(Debug, Clone)]
pub struct RateLimit {
    pub max_requests: u32,
    pub window_seconds: u64,
    pub current_count: u32,
    pub window_start: Instant,
}

impl SecurityMiddleware {
    /// Create new security middleware
    pub fn new() -> Self {
        let mut role_permissions = HashMap::new();
        role_permissions.insert("admin".to_string(), vec![
            "help".to_string(), "clear".to_string(), "export".to_string(),
            "config".to_string(), "search".to_string(), "template".to_string(),
            "macro".to_string(), "branch".to_string(), "session".to_string(),
            "tool".to_string(), "stats".to_string(), "theme".to_string(),
            "debug".to_string(),
        ]);
        role_permissions.insert("user".to_string(), vec![
            "help".to_string(), "clear".to_string(), "export".to_string(),
            "search".to_string(), "template".to_string(), "macro".to_string(),
            "branch".to_string(), "session".to_string(), "theme".to_string(),
        ]);
        role_permissions.insert("guest".to_string(), vec![
            "help".to_string(), "search".to_string(),
        ]);

        Self {
            role_permissions,
            rate_limits: HashMap::new(),
        }
    }

    /// Check if user has permission for command
    fn has_permission(&self, user_role: &str, command_name: &str) -> bool {
        self.role_permissions
            .get(user_role)
            .map(|commands| commands.contains(&command_name.to_string()))
            .unwrap_or(false)
    }
}

#[async_trait::async_trait]
impl CommandMiddleware for SecurityMiddleware {
    async fn before_execute(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
    ) -> Result<(), CommandError> {
        let command_name = match command {
            ChatCommand::Help { .. } => "help",
            ChatCommand::Clear { .. } => "clear",
            ChatCommand::Export { .. } => "export",
            ChatCommand::Config { .. } => "config",
            ChatCommand::Search { .. } => "search",
            ChatCommand::Template { .. } => "template",
            ChatCommand::Macro { .. } => "macro",
            ChatCommand::Branch { .. } => "branch",
            ChatCommand::Session { .. } => "session",
            ChatCommand::Tool { .. } => "tool",
            ChatCommand::Stats { .. } => "stats",
            ChatCommand::Theme { .. } => "theme",
            ChatCommand::Debug { .. } => "debug",
        };

        // Check permissions (simplified - in real implementation would check user roles)
        let user_role = context.permissions.get(0)
            .map(|p| p.as_ref())
            .unwrap_or("guest");

        if !self.has_permission(user_role, command_name) {
            return Err(CommandError::PermissionDenied {
                command: Arc::from(command_name),
            });
        }

        Ok(())
    }

    async fn after_execute(
        &self,
        _command: &ChatCommand,
        _context: &CommandContext,
        _result: &CommandResult<CommandOutput>,
    ) -> Result<(), CommandError> {
        // Security logging could be added here
        Ok(())
    }

    fn name(&self) -> &str {
        "security"
    }

    fn priority(&self) -> u32 {
        5 // Very high priority (execute first)
    }
}

/// Caching middleware for command results
#[derive(Debug)]
pub struct CachingMiddleware {
    /// Cache storage
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Cache TTL in seconds
    ttl_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub output: CommandOutput,
    pub created_at: Instant,
}

impl CachingMiddleware {
    /// Create new caching middleware
    pub fn new(ttl_seconds: u64) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl_seconds,
        }
    }

    /// Generate cache key for command
    fn generate_cache_key(&self, command: &ChatCommand, context: &CommandContext) -> String {
        // Simplified cache key generation
        format!("{}:{}:{:?}", context.user_id, context.session_id, command)
    }

    /// Check if cache entry is valid
    fn is_cache_valid(&self, entry: &CacheEntry) -> bool {
        entry.created_at.elapsed().as_secs() < self.ttl_seconds
    }
}

#[async_trait::async_trait]
impl CommandMiddleware for CachingMiddleware {
    async fn before_execute(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
    ) -> Result<(), CommandError> {
        // Check cache for cacheable commands
        match command {
            ChatCommand::Help { .. } | ChatCommand::Stats { .. } => {
                let cache_key = self.generate_cache_key(command, context);
                let cache = self.cache.read().await;
                
                if let Some(entry) = cache.get(&cache_key) {
                    if self.is_cache_valid(entry) {
                        tracing::debug!(cache_key = cache_key, "Cache hit");
                        // In a real implementation, we would return the cached result
                        // This would require modifying the middleware trait
                    }
                }
            }
            _ => {} // Other commands are not cached
        }

        Ok(())
    }

    async fn after_execute(
        &self,
        command: &ChatCommand,
        context: &CommandContext,
        result: &CommandResult<CommandOutput>,
    ) -> Result<(), CommandError> {
        // Cache successful results for cacheable commands
        if let Ok(output) = result {
            match command {
                ChatCommand::Help { .. } | ChatCommand::Stats { .. } => {
                    let cache_key = self.generate_cache_key(command, context);
                    let mut cache = self.cache.write().await;
                    
                    cache.insert(cache_key.clone(), CacheEntry {
                        output: output.clone(),
                        created_at: Instant::now(),
                    });
                    
                    tracing::debug!(cache_key = cache_key, "Result cached");
                }
                _ => {} // Other commands are not cached
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "caching"
    }

    fn priority(&self) -> u32 {
        90 // Low priority (execute late)
    }
}

/// Create default middleware chain
pub fn create_default_middleware_chain() -> MiddlewareChain {
    let mut chain = MiddlewareChain::new();
    
    // Add default middleware in priority order
    chain.add_middleware(Arc::new(SecurityMiddleware::new()));
    chain.add_middleware(Arc::new(LoggingMiddleware::new(LogLevel::Info, true)));
    chain.add_middleware(Arc::new(PerformanceMiddleware::new(1_000_000))); // 1 second threshold
    chain.add_middleware(Arc::new(CachingMiddleware::new(300))); // 5 minute TTL
    
    chain
}

/// Global middleware chain
static GLOBAL_MIDDLEWARE: once_cell::sync::Lazy<Arc<RwLock<MiddlewareChain>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(create_default_middleware_chain())));

/// Get global middleware chain
pub async fn get_global_middleware() -> Arc<RwLock<MiddlewareChain>> {
    GLOBAL_MIDDLEWARE.clone()
}

/// Add middleware to global chain
pub async fn add_global_middleware(middleware: Arc<dyn CommandMiddleware>) {
    let chain = get_global_middleware().await;
    let mut chain_guard = chain.write().await;
    chain_guard.add_middleware(middleware);
}
