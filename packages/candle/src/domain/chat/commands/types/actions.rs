//! Action type definitions for command variants with zero allocation patterns
//!
//! Provides blazing-fast action enumeration dispatch with owned strings for
//! maximum performance. No Arc usage, no locking, pure enum-based dispatch.

use serde::{Deserialize, Serialize};

/// Template-related action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateAction {
    /// List available templates
    List,
    /// Create a new template
    Create, 
    /// Delete a template
    Delete,
    /// Edit an existing template
    Edit,
    /// Use a template
    Use,
    /// Show template details
    Show,
    /// Validate template syntax
    Validate,
    /// Export template to file
    Export,
    /// Import template from file
    Import,
    /// Copy template to new name
    Copy,
    /// Rename existing template
    Rename,
}

impl TemplateAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Create => "create",
            Self::Delete => "delete", 
            Self::Edit => "edit",
            Self::Use => "use",
            Self::Show => "show",
            Self::Validate => "validate",
            Self::Export => "export",
            Self::Import => "import",
            Self::Copy => "copy",
            Self::Rename => "rename",
        }
    }
    
    /// Check if action requires template name parameter
    #[inline]
    pub const fn requires_name(&self) -> bool {
        matches!(self, 
            Self::Create | Self::Delete | Self::Edit | Self::Use | 
            Self::Show | Self::Validate | Self::Export | Self::Copy | Self::Rename
        )
    }
    
    /// Check if action modifies templates
    #[inline]
    pub const fn is_mutating(&self) -> bool {
        matches!(self, 
            Self::Create | Self::Delete | Self::Edit | Self::Import | 
            Self::Copy | Self::Rename
        )
    }
}

/// Macro-related action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MacroAction {
    /// List available macros
    List,
    /// Create a new macro
    Create,
    /// Delete a macro
    Delete,
    /// Edit an existing macro
    Edit,
    /// Execute a macro
    Execute,
    /// Show macro details
    Show, 
    /// Record new macro from commands
    Record,
    /// Stop macro recording
    Stop,
    /// Pause macro execution
    Pause,
    /// Resume paused macro
    Resume,
    /// Export macro to file
    Export,
    /// Import macro from file
    Import,
}

impl MacroAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Create => "create",
            Self::Delete => "delete",
            Self::Edit => "edit",
            Self::Execute => "execute",
            Self::Show => "show",
            Self::Record => "record",
            Self::Stop => "stop",
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::Export => "export",
            Self::Import => "import",
        }
    }
    
    /// Check if action requires macro name parameter
    #[inline]
    pub const fn requires_name(&self) -> bool {
        matches!(self, 
            Self::Create | Self::Delete | Self::Edit | Self::Execute | 
            Self::Show | Self::Export | Self::Record
        )
    }
    
    /// Check if action modifies macros
    #[inline]
    pub const fn is_mutating(&self) -> bool {
        matches!(self, 
            Self::Create | Self::Delete | Self::Edit | Self::Record | 
            Self::Stop | Self::Import
        )
    }
    
    /// Check if action affects running macros
    #[inline]
    pub const fn affects_execution(&self) -> bool {
        matches!(self, 
            Self::Execute | Self::Pause | Self::Resume | Self::Stop
        )
    }
}

/// Branch-related action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BranchAction {
    /// List available branches
    List,
    /// Create a new branch
    Create,
    /// Switch to a different branch
    Switch,
    /// Merge branches
    Merge,
    /// Delete a branch
    Delete,
    /// Show branch details
    Show,
    /// Compare branches
    Compare,
    /// Rename branch
    Rename,
    /// Archive branch
    Archive,
    /// Restore archived branch
    Restore,
}

impl BranchAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Create => "create",
            Self::Switch => "switch",
            Self::Merge => "merge",
            Self::Delete => "delete",
            Self::Show => "show",
            Self::Compare => "compare",
            Self::Rename => "rename",
            Self::Archive => "archive",
            Self::Restore => "restore",
        }
    }
    
    /// Check if action requires branch name parameter
    #[inline]
    pub const fn requires_name(&self) -> bool {
        matches!(self, 
            Self::Create | Self::Switch | Self::Delete | Self::Show | 
            Self::Rename | Self::Archive | Self::Restore
        )
    }
    
    /// Check if action modifies branches
    #[inline]
    pub const fn is_mutating(&self) -> bool {
        matches!(self, 
            Self::Create | Self::Switch | Self::Merge | Self::Delete | 
            Self::Rename | Self::Archive | Self::Restore
        )
    }
}

/// Session-related action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SessionAction {
    /// List available sessions
    List,
    /// Create a new session
    New,
    /// Switch to a different session
    Switch,
    /// Delete a session
    Delete,
    /// Export session data
    Export,
    /// Import session data
    Import,
    /// Show session details
    Show,
    /// Archive session
    Archive,
    /// Backup session
    Backup,
    /// Restore session from backup
    Restore,
    /// Clone session with new name
    Clone,
    /// Rename session
    Rename,
}

impl SessionAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::List => "list",
            Self::New => "new",
            Self::Switch => "switch",
            Self::Delete => "delete",
            Self::Export => "export",
            Self::Import => "import",
            Self::Show => "show",
            Self::Archive => "archive",
            Self::Backup => "backup",
            Self::Restore => "restore",
            Self::Clone => "clone",
            Self::Rename => "rename",
        }
    }
    
    /// Check if action requires session name parameter
    #[inline]
    pub const fn requires_name(&self) -> bool {
        matches!(self, 
            Self::New | Self::Switch | Self::Delete | Self::Export | 
            Self::Show | Self::Archive | Self::Backup | Self::Clone | Self::Rename
        )
    }
    
    /// Check if action modifies sessions
    #[inline]
    pub const fn is_mutating(&self) -> bool {
        matches!(self, 
            Self::New | Self::Switch | Self::Delete | Self::Import | 
            Self::Archive | Self::Restore | Self::Clone | Self::Rename
        )
    }
}

/// Tool-related action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolAction {
    /// List available tools
    List,
    /// Install a new tool
    Install,
    /// Remove a tool
    Remove,
    /// Configure tool settings
    Configure,
    /// Update a tool
    Update,
    /// Execute a tool
    Execute,
    /// Show tool details
    Show,
    /// Enable tool
    Enable,
    /// Disable tool
    Disable,
    /// Reset tool configuration
    Reset,
    /// Test tool functionality
    Test,
    /// Get tool help
    Help,
}

impl ToolAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::List => "list",
            Self::Install => "install",
            Self::Remove => "remove",
            Self::Configure => "configure",
            Self::Update => "update",
            Self::Execute => "execute",
            Self::Show => "show",
            Self::Enable => "enable",
            Self::Disable => "disable",
            Self::Reset => "reset",
            Self::Test => "test",
            Self::Help => "help",
        }
    }
    
    /// Check if action requires tool name parameter
    #[inline]
    pub const fn requires_name(&self) -> bool {
        matches!(self, 
            Self::Install | Self::Remove | Self::Configure | Self::Update | 
            Self::Execute | Self::Show | Self::Enable | Self::Disable | 
            Self::Reset | Self::Test | Self::Help
        )
    }
    
    /// Check if action modifies tools
    #[inline]
    pub const fn is_mutating(&self) -> bool {
        matches!(self, 
            Self::Install | Self::Remove | Self::Configure | Self::Update | 
            Self::Enable | Self::Disable | Self::Reset
        )
    }
}

/// Search scope enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchScope {
    /// Search all content
    All,
    /// Search current session only
    Session,
    /// Search user messages only
    User,
    /// Search assistant messages only
    Assistant,
    /// Search system messages only
    System,
    /// Search current branch only
    Branch,
    /// Search specific time range
    TimeRange,
    /// Search by message type
    MessageType,
    /// Search by tags
    Tags,
    /// Search archived content
    Archived,
}

impl SearchScope {
    /// Get scope name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Session => "session",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Branch => "branch",
            Self::TimeRange => "time_range",
            Self::MessageType => "message_type",
            Self::Tags => "tags",
            Self::Archived => "archived",
        }
    }
    
    /// Check if scope requires additional parameters
    #[inline]
    pub const fn requires_parameters(&self) -> bool {
        matches!(self, 
            Self::Session | Self::Branch | Self::TimeRange | 
            Self::MessageType | Self::Tags
        )
    }
}

/// Stats type enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatsType {
    /// Usage statistics
    Usage,
    /// Performance metrics
    Performance,
    /// History logs
    History,
    /// Token usage statistics
    Tokens,
    /// Cost statistics
    Costs,
    /// Error statistics
    Errors,
    /// Memory usage statistics
    Memory,
    /// Network statistics
    Network,
    /// Cache statistics
    Cache,
    /// Command execution statistics
    Commands,
}

impl StatsType {
    /// Get type name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Usage => "usage",
            Self::Performance => "performance",
            Self::History => "history",
            Self::Tokens => "tokens",
            Self::Costs => "costs",
            Self::Errors => "errors",
            Self::Memory => "memory",
            Self::Network => "network",
            Self::Cache => "cache",
            Self::Commands => "commands",
        }
    }
    
    /// Check if stats type requires time range
    #[inline]
    pub const fn supports_time_range(&self) -> bool {
        matches!(self, 
            Self::Usage | Self::Performance | Self::History | 
            Self::Tokens | Self::Costs | Self::Errors | Self::Commands
        )
    }
}

/// Theme action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThemeAction {
    /// Set active theme
    Set,
    /// List available themes
    List,
    /// Create a new theme
    Create,
    /// Export theme data
    Export,
    /// Import theme data
    Import,
    /// Edit existing theme
    Edit,
    /// Delete theme
    Delete,
    /// Reset to default theme
    Reset,
    /// Preview theme changes
    Preview,
    /// Clone existing theme
    Clone,
}

impl ThemeAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Set => "set",
            Self::List => "list",
            Self::Create => "create",
            Self::Export => "export",
            Self::Import => "import",
            Self::Edit => "edit",
            Self::Delete => "delete",
            Self::Reset => "reset",
            Self::Preview => "preview",
            Self::Clone => "clone",
        }
    }
    
    /// Check if action requires theme name parameter
    #[inline]
    pub const fn requires_name(&self) -> bool {
        matches!(self, 
            Self::Set | Self::Create | Self::Export | Self::Edit | 
            Self::Delete | Self::Preview | Self::Clone
        )
    }
    
    /// Check if action modifies themes
    #[inline]
    pub const fn is_mutating(&self) -> bool {
        matches!(self, 
            Self::Set | Self::Create | Self::Import | Self::Edit | 
            Self::Delete | Self::Reset | Self::Clone
        )
    }
}

/// Debug action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DebugAction {
    /// Show debug information
    Info,
    /// Display system logs
    Logs,
    /// Show performance metrics
    Performance,
    /// Display memory usage
    Memory,
    /// Show network statistics
    Network,
    /// Display cache statistics
    Cache,
    /// Show thread information
    Threads,
    /// Display environment variables
    Environment,
    /// Show configuration details
    Config,
    /// Display trace information
    Trace,
    /// Show profiling data
    Profile,
    /// Run system diagnostics
    Diagnostics,
}

impl DebugAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Logs => "logs",
            Self::Performance => "performance",
            Self::Memory => "memory",
            Self::Network => "network",
            Self::Cache => "cache",
            Self::Threads => "threads",
            Self::Environment => "environment",
            Self::Config => "config",
            Self::Trace => "trace",
            Self::Profile => "profile",
            Self::Diagnostics => "diagnostics",
        }
    }
    
    /// Check if action supports filtering parameters
    #[inline]
    pub const fn supports_filtering(&self) -> bool {
        matches!(self, 
            Self::Logs | Self::Performance | Self::Memory | Self::Network | 
            Self::Trace | Self::Profile
        )
    }
}

/// History action enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HistoryAction {
    /// Show history entries
    Show,
    /// Search through history
    Search,
    /// Clear history
    Clear,
    /// Export history data
    Export,
    /// Import history data
    Import,
    /// Backup history
    Backup,
    /// Restore from backup
    Restore,
    /// Compact history storage
    Compact,
    /// Analyze history patterns
    Analyze,
    /// Show history statistics
    Stats,
}

impl HistoryAction {
    /// Get action name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Show => "show",
            Self::Search => "search",
            Self::Clear => "clear",
            Self::Export => "export",
            Self::Import => "import",
            Self::Backup => "backup",
            Self::Restore => "restore",
            Self::Compact => "compact",
            Self::Analyze => "analyze",
            Self::Stats => "stats",
        }
    }
    
    /// Check if action modifies history
    #[inline]
    pub const fn is_mutating(&self) -> bool {
        matches!(self, 
            Self::Clear | Self::Import | Self::Restore | Self::Compact
        )
    }
    
    /// Check if action supports time range filtering
    #[inline]
    pub const fn supports_time_range(&self) -> bool {
        matches!(self, 
            Self::Show | Self::Search | Self::Export | Self::Backup | 
            Self::Analyze | Self::Stats
        )
    }
}

/// Import type enumeration for zero allocation dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImportType {
    /// Import chat data
    Chat,
    /// Import configuration
    Config,
    /// Import templates
    Templates,
    /// Import macros
    Macros,
    /// Import themes
    Themes,
    /// Import history
    History,
    /// Import sessions
    Sessions,
    /// Import tools
    Tools,
    /// Import all data
    All,
}

impl ImportType {
    /// Get type name as static string for zero allocation
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Config => "config",
            Self::Templates => "templates",
            Self::Macros => "macros",
            Self::Themes => "themes",
            Self::History => "history",
            Self::Sessions => "sessions",
            Self::Tools => "tools",
            Self::All => "all",
        }
    }
    
    /// Check if import type requires file path
    #[inline]
    pub const fn requires_file_path(&self) -> bool {
        true // All import types require a source file
    }
    
    /// Check if import type supports selective import
    #[inline]
    pub const fn supports_selective_import(&self) -> bool {
        matches!(self, 
            Self::Templates | Self::Macros | Self::Themes | 
            Self::Sessions | Self::Tools
        )
    }
}