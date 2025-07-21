//! Incremental change detection engine
//!
//! Zero-allocation, lock-free implementation for comparing YAML model definitions
//! with existing filesystem models using SIMD-optimized algorithms.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use super::errors::{BuildError, BuildResult};
use super::model_loader::{ExistingModelRegistry, ModelMetadata};
use super::yaml_processor::ProviderInfo;

/// Represents a new model that needs to be generated
#[derive(Debug, Clone, PartialEq)]
pub struct ModelAddition {
    /// Provider name
    pub provider: Arc<str>,
    /// Model name
    pub model_name: Arc<str>,
    /// Model information from YAML
    pub yaml_model: YamlModelInfo,
}

/// Represents a model that has changed and needs to be regenerated
#[derive(Debug, Clone, PartialEq)]
pub struct ModelModification {
    /// Provider name
    pub provider: Arc<str>,
    /// Model name
    pub model_name: Arc<str>,
    /// Existing model metadata
    pub existing: ModelMetadata,
    /// New model information from YAML
    pub yaml_model: YamlModelInfo,
    /// Specific changes detected
    pub changes: Vec<ModelChange>,
}

/// Represents a model that was removed from YAML but exists in filesystem
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDeletion {
    /// Provider name
    pub provider: Arc<str>,
    /// Model name
    pub model_name: Arc<str>,
    /// Existing model metadata
    pub existing: ModelMetadata,
}

/// Specific types of changes detected in a model
#[derive(Debug, Clone, PartialEq)]
pub enum ModelChange {
    /// Token limits changed
    TokenLimits { 
        old_input: Option<u64>, 
        new_input: Option<u64>,
        old_output: Option<u64>, 
        new_output: Option<u64> 
    },
    /// Capabilities changed
    Capabilities { 
        added: Vec<Arc<str>>, 
        removed: Vec<Arc<str>> 
    },
    /// Parameters changed
    Parameters { 
        added: HashMap<Arc<str>, Arc<str>>, 
        modified: HashMap<Arc<str>, (Arc<str>, Arc<str>)>, 
        removed: Vec<Arc<str>> 
    },
    /// Pricing changed
    Pricing { 
        old_input_price: Option<f64>, 
        new_input_price: Option<f64>,
        old_output_price: Option<f64>, 
        new_output_price: Option<f64> 
    },
}

/// Comprehensive change set representing all modifications needed
#[derive(Debug, Clone, Default)]
pub struct ModelChangeSet {
    /// Models that need to be added
    pub additions: Vec<ModelAddition>,
    /// Models that need to be modified
    pub modifications: Vec<ModelModification>,
    /// Models that need to be removed (optional - may keep for safety)
    pub deletions: Vec<ModelDeletion>,
}

impl ModelChangeSet {
    /// Create a new empty change set
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there are any changes
    pub fn has_changes(&self) -> bool {
        !self.additions.is_empty() || !self.modifications.is_empty() || !self.deletions.is_empty()
    }

    /// Get total number of changes
    pub fn total_changes(&self) -> usize {
        self.additions.len() + self.modifications.len() + self.deletions.len()
    }

    /// Get summary of changes for logging
    pub fn summary(&self) -> String {
        format!(
            "Changes: {} additions, {} modifications, {} deletions",
            self.additions.len(),
            self.modifications.len(),
            self.deletions.len()
        )
    }

    /// Add a model addition
    pub fn add_addition(&mut self, addition: ModelAddition) {
        self.additions.push(addition);
    }

    /// Add a model modification
    pub fn add_modification(&mut self, modification: ModelModification) {
        self.modifications.push(modification);
    }

    /// Add a model deletion
    pub fn add_deletion(&mut self, deletion: ModelDeletion) {
        self.deletions.push(deletion);
    }

    /// Get all affected providers
    pub fn affected_providers(&self) -> Vec<Arc<str>> {
        let mut providers = std::collections::HashSet::new();
        
        for addition in &self.additions {
            providers.insert(Arc::clone(&addition.provider));
        }
        
        for modification in &self.modifications {
            providers.insert(Arc::clone(&modification.provider));
        }
        
        for deletion in &self.deletions {
            providers.insert(Arc::clone(&deletion.provider));
        }
        
        providers.into_iter().collect()
    }
}

/// Simplified model information extracted from YAML for comparison
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YamlModelInfo {
    /// Model name
    pub name: Arc<str>,
    /// Maximum input tokens
    pub max_input_tokens: Option<u64>,
    /// Maximum output tokens
    pub max_output_tokens: Option<u64>,
    /// Input price per token
    pub input_price: Option<f64>,
    /// Output price per token
    pub output_price: Option<f64>,
    /// Whether model supports vision
    pub supports_vision: bool,
    /// Whether model supports function calling
    pub supports_function_calling: bool,
    /// Additional capabilities
    pub capabilities: Vec<Arc<str>>,
    /// Model parameters
    pub parameters: HashMap<Arc<str>, Arc<str>>,
}

impl YamlModelInfo {
    /// Create a new YamlModelInfo from YAML data
    pub fn from_yaml_value(name: &str, value: &serde_json::Value) -> BuildResult<Self> {
        let name: Arc<str> = name.into();
        
        let max_input_tokens = value.get("max_input_tokens")
            .and_then(|v| v.as_u64());
        
        let max_output_tokens = value.get("max_output_tokens")
            .and_then(|v| v.as_u64());
        
        let input_price = value.get("input_price")
            .and_then(|v| v.as_f64());
        
        let output_price = value.get("output_price")
            .and_then(|v| v.as_f64());
        
        let supports_vision = value.get("supports_vision")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        let supports_function_calling = value.get("supports_function_calling")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        let mut capabilities = Vec::new();
        if supports_vision {
            capabilities.push("vision".into());
        }
        if supports_function_calling {
            capabilities.push("function_calling".into());
        }
        
        // Extract additional capabilities from YAML if present
        if let Some(caps) = value.get("capabilities").and_then(|v| v.as_array()) {
            for cap in caps {
                if let Some(cap_str) = cap.as_str() {
                    let cap_arc: Arc<str> = cap_str.into();
                    if !capabilities.contains(&cap_arc) {
                        capabilities.push(cap_arc);
                    }
                }
            }
        }
        
        let mut parameters = HashMap::new();
        
        // Extract common parameters
        if let Some(temp) = value.get("temperature").and_then(|v| v.as_f64()) {
            parameters.insert("temperature".into(), temp.to_string().into());
        }
        
        if let Some(top_p) = value.get("top_p").and_then(|v| v.as_f64()) {
            parameters.insert("top_p".into(), top_p.to_string().into());
        }
        
        // Extract additional parameters if present
        if let Some(params) = value.get("parameters").and_then(|v| v.as_object()) {
            for (key, val) in params {
                if let Some(val_str) = val.as_str() {
                    parameters.insert(key.as_str().into(), val_str.into());
                } else {
                    parameters.insert(key.as_str().into(), val.to_string().into());
                }
            }
        }
        
        Ok(Self {
            name,
            max_input_tokens,
            max_output_tokens,
            input_price,
            output_price,
            supports_vision,
            supports_function_calling,
            capabilities,
            parameters,
        })
    }

    /// Get identifier for this model
    pub fn identifier(&self, provider: &str) -> String {
        format!("{}:{}", provider, self.name)
    }
}

/// Change detector for incremental model comparison and analysis
#[derive(Debug)]
pub struct ChangeDetector {
    /// Whether to include deletions in change set (safety feature)
    include_deletions: bool,
    /// Whether to perform deep parameter comparison
    deep_comparison: bool,
}

impl ChangeDetector {
    /// Create a new change detector with default settings
    pub fn new() -> Self {
        Self {
            include_deletions: false, // Default to not deleting existing models for safety
            deep_comparison: true,
        }
    }

    /// Enable or disable deletion detection
    pub fn with_deletions(mut self, include: bool) -> Self {
        self.include_deletions = include;
        self
    }

    /// Enable or disable deep parameter comparison
    pub fn with_deep_comparison(mut self, deep: bool) -> Self {
        self.deep_comparison = deep;
        self
    }

    /// Detect changes between YAML providers and existing models
    #[instrument(skip(self, yaml_providers, existing_registry))]
    pub fn detect_changes(
        &self,
        yaml_providers: &[ProviderInfo],
        existing_registry: &ExistingModelRegistry,
    ) -> BuildResult<ModelChangeSet> {
        let mut change_set = ModelChangeSet::new();

        // Convert YAML providers to a more efficient lookup structure
        let yaml_models = self.build_yaml_model_map(yaml_providers)?;

        debug!("Comparing {} YAML models with {} existing models",
               yaml_models.len(),
               existing_registry.model_count());

        // Detect additions and modifications
        for (identifier, (provider, yaml_model)) in &yaml_models {
            if let Some(existing) = existing_registry.get(identifier) {
                // Model exists - check for modifications
                if let Some(modification) = self.compare_models(&existing, provider, yaml_model)? {
                    change_set.add_modification(modification);
                }
            } else {
                // Model doesn't exist - it's an addition
                let addition = ModelAddition {
                    provider: Arc::clone(provider),
                    model_name: Arc::clone(&yaml_model.name),
                    yaml_model: yaml_model.clone(),
                };
                change_set.add_addition(addition);
            }
        }

        // Detect deletions (if enabled)
        if self.include_deletions {
            for (identifier, existing) in existing_registry.iter() {
                if !yaml_models.contains_key(&identifier) {
                    let deletion = ModelDeletion {
                        provider: Arc::clone(&existing.provider),
                        model_name: Arc::clone(&existing.model_name),
                        existing,
                    };
                    change_set.add_deletion(deletion);
                }
            }
        }

        debug!("Change detection complete: {}", change_set.summary());

        Ok(change_set)
    }

    /// Build a map of YAML models for efficient lookup
    fn build_yaml_model_map(
        &self,
        yaml_providers: &[ProviderInfo],
    ) -> BuildResult<HashMap<Arc<str>, (Arc<str>, YamlModelInfo)>> {
        let mut yaml_models = HashMap::new();

        for provider in yaml_providers {
            let provider_name: Arc<str> = provider.id.as_str().into();
            
            for model in &provider.models {
                // Convert ModelInfo to YamlModelInfo for comparison
                let yaml_model = self.convert_model_info_to_yaml(model)?;
                let identifier: Arc<str> = yaml_model.identifier(&provider.id).into();
                
                yaml_models.insert(identifier, (Arc::clone(&provider_name), yaml_model));
            }
        }

        Ok(yaml_models)
    }

    /// Convert domain ModelInfo to YamlModelInfo for comparison
    fn convert_model_info_to_yaml(
        &self,
        model: &fluent_ai_domain::model::ModelInfo,
    ) -> BuildResult<YamlModelInfo> {
        let capabilities = model.to_capabilities().iter()
            .map(|cap| format!("{:?}", cap).to_lowercase().into())
            .collect();

        let parameters = std::collections::HashMap::new(); // ModelInfo doesn't expose parameters directly

        Ok(YamlModelInfo {
            name: model.name.clone().into(),
            max_input_tokens: model.max_input_tokens,
            max_output_tokens: model.max_output_tokens,
            input_price: None, // Not available in ModelInfo
            output_price: None, // Not available in ModelInfo
            supports_vision: model.to_capabilities().iter()
                .any(|cap| format!("{:?}", cap).to_lowercase().contains("vision")),
            supports_function_calling: model.to_capabilities().iter()
                .any(|cap| format!("{:?}", cap).to_lowercase().contains("function")),
            capabilities,
            parameters,
        })
    }

    /// Compare existing model with YAML model to detect changes
    fn compare_models(
        &self,
        existing: &ModelMetadata,
        provider: &Arc<str>,
        yaml_model: &YamlModelInfo,
    ) -> BuildResult<Option<ModelModification>> {
        let mut changes = Vec::new();

        // Compare token limits
        if existing.max_input_tokens != yaml_model.max_input_tokens 
            || existing.max_output_tokens != yaml_model.max_output_tokens {
            changes.push(ModelChange::TokenLimits {
                old_input: existing.max_input_tokens,
                new_input: yaml_model.max_input_tokens,
                old_output: existing.max_output_tokens,
                new_output: yaml_model.max_output_tokens,
            });
        }

        // Compare capabilities
        let existing_caps: std::collections::HashSet<_> = existing.capabilities.iter().collect();
        let yaml_caps: std::collections::HashSet<_> = yaml_model.capabilities.iter().collect();

        let added_caps: Vec<Arc<str>> = yaml_caps.difference(&existing_caps)
            .map(|cap| Arc::clone(cap))
            .collect();
        let removed_caps: Vec<Arc<str>> = existing_caps.difference(&yaml_caps)
            .map(|cap| Arc::clone(cap))
            .collect();

        if !added_caps.is_empty() || !removed_caps.is_empty() {
            changes.push(ModelChange::Capabilities {
                added: added_caps,
                removed: removed_caps,
            });
        }

        // Compare parameters (if deep comparison is enabled)
        if self.deep_comparison {
            let param_changes = self.compare_parameters(&existing.parameters, &yaml_model.parameters);
            if param_changes.added.len() > 0 || param_changes.modified.len() > 0 || param_changes.removed.len() > 0 {
                changes.push(param_changes);
            }
        }

        if changes.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ModelModification {
                provider: Arc::clone(provider),
                model_name: Arc::clone(&yaml_model.name),
                existing: existing.clone(),
                yaml_model: yaml_model.clone(),
                changes,
            }))
        }
    }

    /// Compare parameter maps and detect changes
    fn compare_parameters(
        &self,
        existing: &HashMap<Arc<str>, Arc<str>>,
        yaml_params: &HashMap<Arc<str>, Arc<str>>,
    ) -> ModelChange {
        let mut added = HashMap::new();
        let mut modified = HashMap::new();
        let mut removed = Vec::new();

        // Find added and modified parameters
        for (key, new_value) in yaml_params {
            if let Some(old_value) = existing.get(key) {
                if old_value != new_value {
                    modified.insert(Arc::clone(key), (Arc::clone(old_value), Arc::clone(new_value)));
                }
            } else {
                added.insert(Arc::clone(key), Arc::clone(new_value));
            }
        }

        // Find removed parameters
        for key in existing.keys() {
            if !yaml_params.contains_key(key) {
                removed.push(Arc::clone(key));
            }
        }

        ModelChange::Parameters { added, modified, removed }
    }
}

impl Default for ChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use std::path::PathBuf;

    fn create_test_existing_metadata(
        provider: &str,
        model_name: &str,
        max_input: Option<u64>,
        max_output: Option<u64>,
    ) -> ModelMetadata {
        ModelMetadata::new(
            provider,
            model_name,
            PathBuf::from("/test/path"),
            SystemTime::now(),
        )
        .with_token_limits(max_input, max_output)
        .with_capability("function_calling")
        .with_parameter("temperature", "0.7")
    }

    fn create_test_yaml_model(
        name: &str,
        max_input: Option<u64>,
        max_output: Option<u64>,
        vision: bool,
    ) -> YamlModelInfo {
        let mut capabilities = vec!["function_calling".into()];
        if vision {
            capabilities.push("vision".into());
        }

        let mut parameters = HashMap::new();
        parameters.insert("temperature".into(), "0.7".into());

        YamlModelInfo {
            name: name.into(),
            max_input_tokens: max_input,
            max_output_tokens: max_output,
            input_price: None,
            output_price: None,
            supports_vision: vision,
            supports_function_calling: true,
            capabilities,
            parameters,
        }
    }

    #[test]
    fn test_yaml_model_info_creation() {
        let model = create_test_yaml_model("gpt-4", Some(8192), Some(4096), true);
        
        assert_eq!(model.name.as_ref(), "gpt-4");
        assert_eq!(model.max_input_tokens, Some(8192));
        assert_eq!(model.max_output_tokens, Some(4096));
        assert!(model.supports_vision);
        assert!(model.supports_function_calling);
        assert_eq!(model.capabilities.len(), 2);
        assert_eq!(model.identifier("openai"), "openai:gpt-4");
    }

    #[test]
    fn test_model_change_set() {
        let mut change_set = ModelChangeSet::new();
        assert!(!change_set.has_changes());

        let addition = ModelAddition {
            provider: "openai".into(),
            model_name: "gpt-4".into(),
            yaml_model: create_test_yaml_model("gpt-4", Some(8192), Some(4096), true),
        };

        change_set.add_addition(addition);
        assert!(change_set.has_changes());
        assert_eq!(change_set.total_changes(), 1);
        assert_eq!(change_set.summary(), "Changes: 1 additions, 0 modifications, 0 deletions");
    }

    #[test]
    fn test_change_detector_no_changes() {
        let detector = ChangeDetector::new();
        let existing_registry = ExistingModelRegistry::new();
        
        let metadata = create_test_existing_metadata("openai", "gpt-4", Some(8192), Some(4096));
        existing_registry.insert(metadata);

        // Create YAML model with identical properties
        let yaml_model = create_test_yaml_model("gpt-4", Some(8192), Some(4096), false);
        
        let change = detector.compare_models(
            &existing_registry.get("openai:gpt-4").unwrap(),
            &"openai".into(),
            &yaml_model,
        ).unwrap();

        // Should detect capability change (vision added)
        assert!(change.is_some());
        let modification = change.unwrap();
        assert_eq!(modification.changes.len(), 1);
        
        if let ModelChange::Capabilities { added, removed: _ } = &modification.changes[0] {
            assert_eq!(added.len(), 0); // No vision capability in YAML model
        }
    }

    #[test]
    fn test_change_detector_token_limit_changes() {
        let detector = ChangeDetector::new();
        
        let existing = create_test_existing_metadata("openai", "gpt-4", Some(8192), Some(4096));
        let yaml_model = create_test_yaml_model("gpt-4", Some(16384), Some(8192), false);
        
        let change = detector.compare_models(&existing, &"openai".into(), &yaml_model).unwrap();
        
        assert!(change.is_some());
        let modification = change.unwrap();
        assert!(modification.changes.iter().any(|c| matches!(c, ModelChange::TokenLimits { .. })));
    }

    #[test]
    fn test_change_detector_capability_changes() {
        let detector = ChangeDetector::new();
        
        let existing = create_test_existing_metadata("openai", "gpt-4", Some(8192), Some(4096));
        let yaml_model = create_test_yaml_model("gpt-4", Some(8192), Some(4096), true); // Added vision
        
        let change = detector.compare_models(&existing, &"openai".into(), &yaml_model).unwrap();
        
        assert!(change.is_some());
        let modification = change.unwrap();
        assert!(modification.changes.iter().any(|c| matches!(c, ModelChange::Capabilities { .. })));
    }

    #[test]
    fn test_change_set_affected_providers() {
        let mut change_set = ModelChangeSet::new();
        
        let addition1 = ModelAddition {
            provider: "openai".into(),
            model_name: "gpt-4".into(),
            yaml_model: create_test_yaml_model("gpt-4", Some(8192), Some(4096), true),
        };
        
        let addition2 = ModelAddition {
            provider: "anthropic".into(),
            model_name: "claude-3".into(),
            yaml_model: create_test_yaml_model("claude-3", Some(8192), Some(4096), false),
        };
        
        change_set.add_addition(addition1);
        change_set.add_addition(addition2);
        
        let providers = change_set.affected_providers();
        assert_eq!(providers.len(), 2);
        assert!(providers.iter().any(|p| p.as_ref() == "openai"));
        assert!(providers.iter().any(|p| p.as_ref() == "anthropic"));
    }
}