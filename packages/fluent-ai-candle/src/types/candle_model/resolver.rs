//! Model resolution and lookup utilities

// Removed unused import: std::borrow::Cow
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

// Removed unused import: std::sync::Arc
use ahash::RandomState;
use dashmap::DashMap;
use fluent_ai_async::AsyncStream;
#[cfg(test)]
use once_cell::sync::Lazy;
// Removed unused import: once_cell::sync::Lazy
use regex::Regex;
use serde::{Deserialize, Serialize};
use strsim;

use crate::types::CandleCompletionResult as Result;
use crate::types::candle_model::error::ModelError;
use crate::types::candle_model::registry::{ModelRegistry, RegisteredModel};
use crate::types::{CandleModel, CandleModelInfo as ModelInfo};
// Removed unused import: strsim::jaro_winkler

/// A pattern that can be used to match model names
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum ModelPattern {
    /// Match exact model name
    Exact(String),

    /// Match model name with a glob pattern
    Pattern(String),

    /// Match model name with a regular expression
    Regex(String)}

impl ModelPattern {
    /// Check if the pattern matches the given model name
    pub fn matches(&self, model_name: &str) -> bool {
        match self {
            ModelPattern::Exact(pattern) => pattern == model_name,
            ModelPattern::Pattern(pattern) => {
                // Convert glob pattern to regex
                let mut regex = String::with_capacity(pattern.len() * 2);
                regex.push('^');

                for c in pattern.chars() {
                    match c {
                        '*' => regex.push_str(".*"),
                        '?' => regex.push('.'),
                        '.' | '^' | '$' | '|' | '(' | ')' | '[' | ']' | '{' | '}' | '+' => {
                            regex.push('\\');
                            regex.push(c);
                        }
                        '\\' => {
                            regex.push('\\');
                            regex.push(c);
                        }
                        _ => regex.push(c)}
                }

                regex.push('$');

                // Compile the regex and check for a match
                Regex::new(&regex)
                    .map(|re| re.is_match(model_name))
                    .unwrap_or(false)
            }
            ModelPattern::Regex(pattern) => Regex::new(pattern)
                .map(|re| re.is_match(model_name))
                .unwrap_or(false)}
    }

    /// Get the pattern as a string
    pub fn as_str(&self) -> &str {
        match self {
            ModelPattern::Exact(s) => s,
            ModelPattern::Pattern(s) => s,
            ModelPattern::Regex(s) => s}
    }
}

impl fmt::Display for ModelPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelPattern::Exact(s) => write!(f, "{}", s),
            ModelPattern::Pattern(s) => write!(f, "pattern:{}", s),
            ModelPattern::Regex(s) => write!(f, "regex:{}", s)}
    }
}

/// A rule for model resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResolutionRule {
    /// The pattern to match against model names
    pub pattern: ModelPattern,

    /// The target provider to use
    pub provider: String,

    /// The target model name to use
    pub target: String,

    /// The priority of this rule (higher = more specific)
    pub priority: i32,

    /// Optional condition for when this rule applies
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<RuleCondition>}

/// A condition for when a rule should apply
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RuleCondition {
    /// The rule only applies if the model has the specified capability
    HasCapability { capability: String },

    /// The rule only applies if the model has the specified feature flag
    HasFeature { feature: String },

    /// The rule only applies if the specified environment variable is set
    EnvVarSet { name: String },

    /// The rule only applies if the specified feature flag is enabled
    FeatureEnabled { name: String }}

/// A model resolution result
#[derive(Debug, Clone)]
pub struct ModelResolution {
    /// The resolved provider name
    pub provider: String,

    /// The resolved model name
    pub model: String,

    /// The model info (if available)
    pub info: Option<ModelInfo>,

    /// The rule that was used for resolution (if any)
    pub rule: Option<ModelResolutionRule>,

    /// The score of the match (higher is better)
    pub score: f64}

impl ModelResolution {
    /// Create a new resolution result
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        info: Option<ModelInfo>,
        rule: Option<ModelResolutionRule>,
        score: f64,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            info,
            rule,
            score}
    }

    /// Check if the resolution is valid
    pub fn is_valid(&self) -> bool {
        self.score > 0.0 && !self.provider.is_empty() && !self.model.is_empty()
    }
}

/// A resolver for model names and providers
#[derive(Clone)]
pub struct ModelResolver {
    registry: ModelRegistry,
    rules: Vec<ModelResolutionRule>,
    aliases: HashMap<String, (String, String), RandomState>,

    // Cache for compiled regex patterns
    #[allow(clippy::type_complexity)]
    #[allow(dead_code)]
    pattern_cache: DashMap<String, (String, Regex), RandomState>}

impl Default for ModelResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelResolver {
    /// Create a new model resolver
    pub fn new() -> Self {
        Self {
            registry: ModelRegistry::new(),
            rules: Vec::new(),
            aliases: HashMap::with_hasher(RandomState::new()),
            pattern_cache: DashMap::with_hasher(RandomState::new())}
    }

    /// Add a resolution rule
    pub fn add_rule(&mut self, rule: ModelResolutionRule) {
        self.rules.push(rule);
        // Sort rules by priority (highest first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Add multiple resolution rules
    pub fn add_rules(&mut self, rules: impl IntoIterator<Item = ModelResolutionRule>) {
        self.rules.extend(rules);
        // Sort rules by priority (highest first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Add a model alias
    pub fn add_alias(
        &mut self,
        alias: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) {
        self.aliases
            .insert(alias.into(), (provider.into(), model.into()));
    }

    /// Resolve a model by name and optional provider
    pub fn resolve<M: CandleModel + 'static>(
        &self,
        model_name: &str,
        provider: Option<&str>,
    ) -> AsyncStream<ModelResolution> {
        let registry = self.registry.clone();
        let resolver = self.clone();
        let model_name = model_name.to_string();
        let provider = provider.map(|s| s.to_string());

        AsyncStream::with_channel(move |sender| {
            match resolver.resolve_with_registry::<M>(&registry, &model_name, provider.as_deref()) {
                Ok(resolution) => {
                    let _ = sender.try_send(resolution);
                }
                Err(_) => {
                    // Provide a fallback resolution
                    let fallback =
                        ModelResolution::new("fallback", model_name.clone(), None, None, 0.0);
                    let _ = sender.try_send(fallback);
                }
            }
        })
    }

    /// Resolve a model by name and optional provider using a specific registry
    pub fn resolve_with_registry<'a, M: CandleModel + 'static>(
        &'a self,
        registry: &'a ModelRegistry,
        model_name: &'a str,
        provider: Option<&'a str>,
    ) -> Result<ModelResolution> {
        // Check for exact match first (provider:model)
        if let Some(provider) = provider {
            if let Ok(Some(model)) = registry.get::<M>(provider, model_name) {
                return Ok(ModelResolution::new(
                    provider.to_string(),
                    model_name.to_string(),
                    Some(model.info().clone()),
                    None,
                    1.0,
                ));
            }
        }

        // Check for exact match in the default provider
        if let Some(default_provider) = self.get_default_provider() {
            if let Ok(Some(model)) = registry.get::<M>(default_provider, model_name) {
                return Ok(ModelResolution::new(
                    default_provider.to_string(),
                    model_name.to_string(),
                    Some(model.info().clone()),
                    None,
                    0.9,
                ));
            }
        }

        // Check for aliases
        if let Some(alias_entry) = self.aliases.get(model_name) {
            let (provider, model_name_alias) = alias_entry;
            if let Ok(Some(model)) = registry.get::<M>(provider, model_name_alias) {
                return Ok(ModelResolution::new(
                    provider.clone(),
                    model_name_alias.clone(),
                    Some(model.info().clone()),
                    None,
                    0.8,
                ));
            }
        }

        // Apply resolution rules
        for rule in &self.rules {
            if rule.pattern.matches(model_name) {
                if let Some(condition) = &rule.condition {
                    if !self.check_condition(condition) {
                        continue;
                    }
                }

                if let Ok(Some(model)) = registry.get::<M>(&rule.provider, &rule.target) {
                    return Ok(ModelResolution::new(
                        rule.provider.clone(),
                        rule.target.clone(),
                        Some(model.info().clone()),
                        Some(rule.clone()),
                        0.7,
                    ));
                }
            }
        }

        // Try fuzzy matching
        self.fuzzy_match::<M>(registry, model_name, provider)
    }

    /// Get a model by name and optional provider
    pub fn get_model<M: CandleModel + 'static>(
        &self,
        model_name: &str,
        provider: Option<&str>,
    ) -> AsyncStream<Option<RegisteredModel<M>>> {
        let resolver = self.clone();
        let model_name = model_name.to_string();
        let provider = provider.map(|s| s.to_string());

        AsyncStream::with_channel(move |sender| {
            let resolution_stream = resolver.resolve::<M>(&model_name, provider.as_deref());

            // Use proper streams-only pattern with collect() for blocking collection
            let resolutions = resolution_stream.collect();
            if let Some(resolution) = resolutions.into_iter().next() {
                if resolution.is_valid() {
                    match resolver
                        .registry
                        .get::<M>(&resolution.provider, &resolution.model)
                    {
                        Ok(Some(model)) => {
                            let _ = sender.send(Some(model));
                        }
                        _ => {
                            let _ = sender.send(None);
                        }
                    }
                } else {
                    let _ = sender.send(None);
                }
            } else {
                let _ = sender.send(None);
            }
        })
    }

    /// Get the default provider (if any)
    pub fn get_default_provider(&self) -> Option<&'static str> {
        // In a real implementation, this would check configuration
        // For now, we'll just return the first provider we find
        None
    }

    /// Check if a condition is met
    fn check_condition(&self, condition: &RuleCondition) -> bool {
        match condition {
            RuleCondition::HasCapability { capability: _ } => {
                // In a real implementation, check if the model has the capability
                false
            }
            RuleCondition::HasFeature { feature: _ } => {
                // In a real implementation, check if the feature is enabled
                false
            }
            RuleCondition::EnvVarSet { name } => std::env::var(name).is_ok(),
            RuleCondition::FeatureEnabled { name: _ } => {
                // In a real implementation, check if the feature is enabled
                false
            }
        }
    }

    /// Find the best matching model using fuzzy matching
    fn fuzzy_match<M: CandleModel + 'static>(
        &self,
        registry: &ModelRegistry,
        model_name: &str,
        provider: Option<&str>,
    ) -> Result<ModelResolution> {
        let all_models = registry.find_all::<M>();
        // Find the best match using Jaro-Winkler similarity
        let best_match = all_models
            .iter()
            .filter(|m| provider.map_or(true, |p| m.info().provider() == p))
            .map(|m| {
                let info = m.info();
                (info, strsim::jaro_winkler(info.name(), model_name))
            })
            .filter(|(_, score)| *score > 0.7)
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((info, score)) = best_match {
            Ok(ModelResolution::new(
                info.provider().to_string(),
                info.name().to_string(),
                Some(info.clone()),
                None,
                score,
            ))
        } else {
            Err(ModelError::ModelNotFound {
                provider: provider.unwrap_or("any").to_string().into(),
                name: model_name.to_string().into()}
            .into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::candle_model::info::ModelInfoBuilder;

    struct TestModel {
        info: &'static ModelInfo}

    impl crate::types::Model for TestModel {
        fn info(&self) -> &'static ModelInfo {
            self.info
        }
    }

    #[test]
    fn test_pattern_matching() {
        // Test exact matching
        let exact = ModelPattern::Exact("gpt-4".to_string());
        assert!(exact.matches("gpt-4"));
        assert!(!exact.matches("gpt-3"));

        // Test glob pattern matching
        let glob = ModelPattern::Pattern("gpt-*".to_string());
        assert!(glob.matches("gpt-4"));
        assert!(glob.matches("gpt-3.5"));
        assert!(!glob.matches("claude-2"));

        // Test regex pattern matching
        let regex = ModelPattern::Regex(r"^gpt-\d+$".to_string());
        assert!(regex.matches("gpt-4"));
        assert!(regex.matches("gpt-35"));
        assert!(!regex.matches("gpt-3.5"));
        assert!(!regex.matches("claude-2"));
    }

    #[test]
    fn test_resolution_rules() {
        let mut resolver = ModelResolver::new();

        // Add a resolution rule
        resolver.add_rule(ModelResolutionRule {
            pattern: ModelPattern::Pattern("gpt-*".to_string()),
            provider: "openai".to_string(),
            target: "gpt-3.5-turbo".to_string(),
            priority: 10,
            condition: None});

        // Add an alias
        resolver.add_alias("chat", "openai", "gpt-3.5-turbo");

        // Test resolution with a rule
        let resolution_stream = resolver.resolve::<TestModel>("gpt-4", None);
        let resolution = resolution_stream.collect().into_iter().next().unwrap();
        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");

        // Test resolution with an alias
        let resolution_stream = resolver.resolve::<TestModel>("chat", None);
        let resolution = resolution_stream.collect().into_iter().next().unwrap();
        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");
    }

    #[test]
    fn test_fuzzy_matching() {
        let mut registry = ModelRegistry::new();

        // Register some test models
        static INFO1: Lazy<ModelInfo> = Lazy::new(|| {
            ModelInfo::builder()
                .provider_name("openai")
                .name("gpt-3.5-turbo")
                .build()
                .unwrap()
        });

        static INFO2: Lazy<ModelInfo> = Lazy::new(|| {
            ModelInfo::builder()
                .provider_name("anthropic")
                .name("claude-2")
                .build()
                .unwrap()
        });

        let model1 = TestModel { info: &*INFO1 };
        let model2 = TestModel { info: &*INFO2 };

        registry.register("openai", model1).unwrap();
        registry.register("anthropic", model2).unwrap();

        let resolver = ModelResolver::new();

        // Test fuzzy matching with a typo
        let resolution = resolver
            .resolve_with_registry::<TestModel>(&registry, "gpt-3.5-turbo-16k", None)
            .unwrap();

        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");
        assert!(resolution.score > 0.7);

        // Test fuzzy matching with a different casing
        let resolution = resolver
            .resolve_with_registry::<TestModel>(&registry, "GPT-3.5-TURBO", None)
            .unwrap();

        assert_eq!(resolution.provider, "openai");
        assert_eq!(resolution.model, "gpt-3.5-turbo");
        assert!(resolution.score > 0.7);
    }
}
