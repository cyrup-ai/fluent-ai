//! Configuration types for SIMD-accelerated operations

/// Configuration for logits processing operations
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    /// Controls randomness (1.0 = no change, < 1.0 = less random, > 1.0 = more random)
    pub temperature: f32,

    /// Number of highest probability tokens to keep (None = keep all)
    pub top_k: Option<usize>,

    /// Nucleus sampling parameter (None = disabled, 0.0 < top_p <= 1.0)
    pub top_p: Option<f32>,

    /// Penalty for repeated tokens (1.0 = no penalty, > 1.0 = more penalty)
    pub repetition_penalty: f32,

    /// Penalty based on token frequency (0.0 = no penalty, > 0.0 = more penalty)
    pub frequency_penalty: f32,

    /// Penalty for tokens present in context (0.0 = no penalty, > 0.0 = more penalty)
    pub presence_penalty: f32,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Error type for configuration validation
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Invalid temperature value - must be positive
    #[error("Invalid temperature value: {0}. Must be positive")]
    InvalidTemperature(f32),

    /// Invalid top_p value - must be in range (0.0, 1.0]
    #[error("Invalid top_p value: {0}. Must be in range (0.0, 1.0]")]
    InvalidTopP(f32),

    /// Invalid repetition penalty - must be >= 1.0
    #[error("Invalid repetition penalty: {0}. Must be >= 1.0")]
    InvalidRepetitionPenalty(f32),

    /// Invalid frequency penalty - must be >= 0.0
    #[error("Invalid frequency penalty: {0}. Must be >= 0.0")]
    InvalidFrequencyPenalty(f32),

    /// Invalid presence penalty - must be >= 0.0
    #[error("Invalid presence penalty: {0}. Must be >= 0.0")]
    InvalidPresencePenalty(f32),
}

impl ProcessorConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.temperature <= 0.0 {
            return Err(ConfigError::InvalidTemperature(self.temperature));
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(ConfigError::InvalidTopP(top_p));
            }
        }

        if self.repetition_penalty < 1.0 {
            return Err(ConfigError::InvalidRepetitionPenalty(
                self.repetition_penalty,
            ));
        }

        if self.frequency_penalty < 0.0 {
            return Err(ConfigError::InvalidFrequencyPenalty(self.frequency_penalty));
        }

        if self.presence_penalty < 0.0 {
            return Err(ConfigError::InvalidPresencePenalty(self.presence_penalty));
        }

        Ok(())
    }

    /// Set temperature parameter
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k parameter
    pub fn with_top_k(mut self, top_k: Option<usize>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set top-p parameter
    pub fn with_top_p(mut self, top_p: Option<f32>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set frequency penalty
    pub fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty
    pub fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProcessorConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, None);
        assert_eq!(config.top_p, None);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.frequency_penalty, 0.0);
        assert_eq!(config.presence_penalty, 0.0);
    }

    #[test]
    fn test_validation() {
        let mut config = ProcessorConfig::default()
            .with_temperature(0.5)
            .with_top_p(Some(0.9))
            .with_repetition_penalty(1.2)
            .with_frequency_penalty(0.1)
            .with_presence_penalty(0.1);

        assert!(config.validate().is_ok());

        // Test invalid temperature
        config.temperature = -1.0;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidTemperature(_))
        ));
        config.temperature = 1.0;

        // Test invalid top_p
        config.top_p = Some(1.1);
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidTopP(_))
        ));
        config.top_p = Some(0.9);

        // Test invalid repetition penalty
        config.repetition_penalty = 0.5;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidRepetitionPenalty(_))
        ));
        config.repetition_penalty = 1.2;

        // Test invalid frequency penalty
        config.frequency_penalty = -0.1;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidFrequencyPenalty(_))
        ));
        config.frequency_penalty = 0.1;

        // Test invalid presence penalty
        config.presence_penalty = -0.1;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidPresencePenalty(_))
        ));
    }
}
