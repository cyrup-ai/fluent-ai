//! Plugin management and loading
//!
//! Provides comprehensive plugin management capabilities with zero-allocation
//! patterns and ergonomic plugin loading and execution.

use std::sync::Arc;

use super::types::{
    Plugin, PluginConfig, IntegrationResult, IntegrationError};

/// Plugin manager for handling plugin-based integrations
#[derive(Debug)]
pub struct PluginManager {
    /// Loaded plugins
    plugins: std::collections::HashMap<Arc<str>, Arc<dyn Plugin>>,
    /// Plugin configurations
    configs: std::collections::HashMap<Arc<str>, PluginConfig>}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: std::collections::HashMap::new(),
            configs: std::collections::HashMap::new()}
    }

    /// Load a plugin
    pub fn load_plugin(
        &mut self,
        plugin: Arc<dyn Plugin>,
        config: PluginConfig,
    ) -> IntegrationResult<()> {
        let plugin_id = config.plugin_id.clone();
        self.plugins.insert(plugin_id.clone(), plugin);
        self.configs.insert(plugin_id, config);
        Ok(())
    }

    /// Get a plugin by ID
    pub fn get_plugin(&self, plugin_id: &str) -> Option<&Arc<dyn Plugin>> {
        self.plugins.get(plugin_id)
    }

    /// Unload a plugin
    pub fn unload_plugin(&mut self, plugin_id: &str) -> IntegrationResult<()> {
        self.plugins.remove(plugin_id);
        self.configs.remove(plugin_id);
        Ok(())
    }

    /// List all loaded plugins
    pub fn list_plugins(&self) -> Vec<&PluginConfig> {
        self.configs.values().collect()
    }

    /// Get plugin configuration
    pub fn get_plugin_config(&self, plugin_id: &str) -> Option<&PluginConfig> {
        self.configs.get(plugin_id)
    }

    /// Update plugin configuration
    pub fn update_plugin_config(
        &mut self,
        plugin_id: &str,
        config: PluginConfig,
    ) -> IntegrationResult<()> {
        if !self.plugins.contains_key(plugin_id) {
            return Err(IntegrationError::PluginError {
                detail: Arc::from("Plugin not found")});
        }

        self.configs.insert(Arc::from(plugin_id), config);
        Ok(())
    }

    /// Check if plugin is loaded
    pub fn is_plugin_loaded(&self, plugin_id: &str) -> bool {
        self.plugins.contains_key(plugin_id)
    }

    /// Get number of loaded plugins
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Clear all plugins
    pub fn clear_all_plugins(&mut self) {
        self.plugins.clear();
        self.configs.clear();
    }

    /// Initialize all plugins
    pub fn initialize_all_plugins(&mut self) -> IntegrationResult<()> {
        for (plugin_id, config) in &self.configs {
            if let Some(plugin) = self.plugins.get_mut(plugin_id) {
                if let Some(plugin_mut) = Arc::get_mut(plugin) {
                    plugin_mut.initialize(config).map_err(|e| {
                        IntegrationError::PluginError {
                            detail: Arc::from(format!("Failed to initialize plugin {}: {}", plugin_id, e))}
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Cleanup all plugins
    pub fn cleanup_all_plugins(&mut self) -> IntegrationResult<()> {
        for (plugin_id, plugin) in &mut self.plugins {
            if let Some(plugin_mut) = Arc::get_mut(plugin) {
                plugin_mut.cleanup().map_err(|e| {
                    IntegrationError::PluginError {
                        detail: Arc::from(format!("Failed to cleanup plugin {}: {}", plugin_id, e))}
                })?;
            }
        }
        Ok(())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}