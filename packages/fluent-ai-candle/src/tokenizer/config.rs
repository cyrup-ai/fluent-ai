//! Tokenizer Configuration Types
//!
//! Provides comprehensive configuration types for tokenizer behavior including
//! padding, truncation, special tokens, and model-specific settings.

/// Tokenizer configuration for production usage
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Add BOS (Beginning of Sequence) token
    pub add_bos_token: bool,
    /// Add EOS (End of Sequence) token  
    pub add_eos_token: bool,
    /// Maximum sequence length for truncation
    pub max_length: Option<usize>,
    /// Padding configuration
    pub padding: PaddingConfig,
    /// Truncation configuration
    pub truncation: TruncationConfig}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            add_bos_token: false,
            add_eos_token: false,
            max_length: Some(2048),
            padding: PaddingConfig::default(),
            truncation: TruncationConfig::default()}
    }
}

/// Padding configuration
#[derive(Debug, Clone)]
pub struct PaddingConfig {
    /// Enable padding
    pub enabled: bool,
    /// Padding token
    pub token: String,
    /// Padding length
    pub length: Option<usize>}

impl Default for PaddingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            token: "<pad>".to_string(),
            length: None}
    }
}

/// Truncation configuration
#[derive(Debug, Clone)]
pub struct TruncationConfig {
    /// Enable truncation
    pub enabled: bool,
    /// Maximum length for truncation
    pub max_length: usize,
    /// Truncation strategy
    pub strategy: TruncationStrategy}

impl Default for TruncationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_length: 2048,
            strategy: TruncationStrategy::LongestFirst}
    }
}

/// Truncation strategies for sequence processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Truncate from the beginning
    OnlyFirst,
    /// Truncate from the end  
    OnlySecond,
    /// Truncate the longest sequence first
    LongestFirst,
    /// Do not truncate
    DoNotTruncate}

/// Builder for tokenizer configuration
pub struct TokenizerConfigBuilder {
    config: TokenizerConfig}

impl TokenizerConfigBuilder {
    /// Create new configuration builder
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default()}
    }

    /// Enable/disable BOS token addition
    #[inline(always)]
    pub fn add_bos_token(mut self, add: bool) -> Self {
        self.config.add_bos_token = add;
        self
    }

    /// Enable/disable EOS token addition
    #[inline(always)]
    pub fn add_eos_token(mut self, add: bool) -> Self {
        self.config.add_eos_token = add;
        self
    }

    /// Set maximum sequence length
    #[inline(always)]
    pub fn max_length(mut self, length: Option<usize>) -> Self {
        self.config.max_length = length;
        self
    }

    /// Configure padding
    #[inline(always)]
    pub fn padding(mut self, config: PaddingConfig) -> Self {
        self.config.padding = config;
        self
    }

    /// Configure truncation
    #[inline(always)]
    pub fn truncation(mut self, config: TruncationConfig) -> Self {
        self.config.truncation = config;
        self
    }

    /// Build the configuration
    #[inline(always)]
    pub fn build(self) -> TokenizerConfig {
        self.config
    }
}

impl Default for TokenizerConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}