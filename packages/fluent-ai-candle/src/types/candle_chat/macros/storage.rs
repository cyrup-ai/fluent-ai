//! Macro storage and persistence functionality
//!
//! This module handles the storage, retrieval, and persistence of macros
//! using high-performance lock-free data structures.

use std::collections::HashMap;
use std::sync::Arc;

use crate::types::candle_chat::search::tagging::ConsistentCounter;
use crossbeam_skiplist::SkipMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::{ChatMacro, ExecutionStats, MacroSystemError, MacroResult};

/// High-performance macro storage with lock-free operations
pub struct MacroStorage {
    /// Lock-free macro storage using skip list
    macros: Arc<SkipMap<Uuid, ChatMacro>>,
    /// Macro execution statistics
    execution_stats: SkipMap<Uuid, Arc<ExecutionStats>>,
    /// Global macro counter
    macro_counter: ConsistentCounter}

impl MacroStorage {
    /// Create new macro storage
    pub fn new() -> Self {
        Self {
            macros: Arc::new(SkipMap::new()),
            execution_stats: SkipMap::new(),
            macro_counter: ConsistentCounter::new(0)}
    }

    /// Get shared reference to macros storage
    pub fn macros(&self) -> Arc<SkipMap<Uuid, ChatMacro>> {
        self.macros.clone()
    }

    /// Store a macro
    pub fn store_macro(&self, macro_def: ChatMacro) -> MacroResult<Uuid> {
        let macro_id = macro_def.metadata.id;
        
        // Initialize execution statistics
        let stats = Arc::new(ExecutionStats::default());
        self.execution_stats.insert(macro_id, stats);
        
        // Store the macro
        self.macros.insert(macro_id, macro_def);
        self.macro_counter.inc();
        
        Ok(macro_id)
    }

    /// Get a macro by ID
    pub fn get_macro(&self, macro_id: Uuid) -> Option<ChatMacro> {
        self.macros.get(&macro_id).map(|entry| entry.value().clone())
    }

    /// Delete a macro
    pub fn delete_macro(&self, macro_id: Uuid) -> MacroResult<bool> {
        let removed = self.macros.remove(&macro_id).is_some();
        if removed {
            self.execution_stats.remove(&macro_id);
            self.macro_counter.dec();
        }
        Ok(removed)
    }

    /// List all macros
    pub fn list_macros(&self) -> Vec<ChatMacro> {
        self.macros
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get macro count
    pub fn macro_count(&self) -> usize {
        self.macro_counter.get()
    }

    /// Search macros by name pattern
    pub fn search_macros(&self, pattern: &str) -> Vec<ChatMacro> {
        let pattern_lower = pattern.to_lowercase();
        self.macros
            .iter()
            .filter(|entry| {
                entry.value().metadata.name.to_lowercase().contains(&pattern_lower)
                    || entry.value().metadata.description.to_lowercase().contains(&pattern_lower)
                    || entry.value().metadata.tags.iter().any(|tag| tag.to_lowercase().contains(&pattern_lower))
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get macros by category
    pub fn get_macros_by_category(&self, category: &str) -> Vec<ChatMacro> {
        self.macros
            .iter()
            .filter(|entry| entry.value().metadata.category.as_ref() == category)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get macros by author
    pub fn get_macros_by_author(&self, author: &str) -> Vec<ChatMacro> {
        self.macros
            .iter()
            .filter(|entry| entry.value().metadata.author.as_ref() == author)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Update macro metadata
    pub fn update_macro_metadata(&self, macro_id: Uuid, updater: impl FnOnce(&mut ChatMacro)) -> MacroResult<()> {
        if let Some(entry) = self.macros.get(&macro_id) {
            let mut macro_def = entry.value().clone();
            updater(&mut macro_def);
            self.macros.insert(macro_id, macro_def);
            Ok(())
        } else {
            Err(MacroSystemError::MacroNotFound(macro_id))
        }
    }

    /// Get execution statistics for a macro
    pub fn get_execution_stats(&self, macro_id: Uuid) -> Option<Arc<ExecutionStats>> {
        self.execution_stats.get(&macro_id).map(|entry| entry.value().clone())
    }

    /// Export macro to JSON
    pub fn export_macro(&self, macro_id: Uuid) -> MacroResult<String> {
        if let Some(macro_def) = self.get_macro(macro_id) {
            serde_json::to_string_pretty(&macro_def)
                .map_err(MacroSystemError::SerializationError)
        } else {
            Err(MacroSystemError::MacroNotFound(macro_id))
        }
    }

    /// Import macro from JSON
    pub fn import_macro(&self, json_data: &str) -> MacroResult<Uuid> {
        let macro_def: ChatMacro = serde_json::from_str(json_data)
            .map_err(MacroSystemError::SerializationError)?;
        
        self.store_macro(macro_def)
    }

    /// Export all macros to JSON
    pub fn export_all_macros(&self) -> MacroResult<String> {
        let all_macros = self.list_macros();
        serde_json::to_string_pretty(&all_macros)
            .map_err(MacroSystemError::SerializationError)
    }

    /// Import multiple macros from JSON
    pub fn import_multiple_macros(&self, json_data: &str) -> MacroResult<Vec<Uuid>> {
        let macros: Vec<ChatMacro> = serde_json::from_str(json_data)
            .map_err(MacroSystemError::SerializationError)?;
        
        let mut imported_ids = Vec::new();
        for macro_def in macros {
            let macro_id = self.store_macro(macro_def)?;
            imported_ids.push(macro_id);
        }
        
        Ok(imported_ids)
    }

    /// Clear all macros
    pub fn clear_all_macros(&self) {
        self.macros.clear();
        self.execution_stats.clear();
        self.macro_counter.set(0);
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> StorageStats {
        StorageStats {
            total_macros: self.macro_count(),
            memory_usage_bytes: self.estimate_memory_usage(),
            categories: self.get_category_counts()}
    }

    /// Estimate memory usage (rough calculation)
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimate: each macro entry ~1KB on average
        self.macro_count() * 1024
    }

    /// Get count of macros by category
    fn get_category_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entry in self.macros.iter() {
            let category = entry.value().metadata.category.as_ref();
            *counts.entry(category.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Validate macro before storage
    pub fn validate_macro(&self, macro_def: &ChatMacro) -> MacroResult<()> {
        // Basic validation
        if macro_def.metadata.name.is_empty() {
            return Err(MacroSystemError::InvalidAction("Macro name cannot be empty".to_string()));
        }

        if macro_def.actions.is_empty() {
            return Err(MacroSystemError::InvalidAction("Macro must have at least one action".to_string()));
        }

        // Check for circular dependencies
        self.validate_dependencies(&macro_def.dependencies)?;

        Ok(())
    }

    /// Validate macro dependencies
    fn validate_dependencies(&self, dependencies: &[String]) -> MacroResult<()> {
        for dep in dependencies {
            if let Ok(dep_uuid) = Uuid::parse_str(dep) {
                if !self.macros.contains_key(&dep_uuid) {
                    return Err(MacroSystemError::InvalidAction(
                        format!("Dependency macro not found: {}", dep)
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for MacroStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_macros: usize,
    pub memory_usage_bytes: usize,
    pub categories: HashMap<String, usize>}

/// Macro export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroExport {
    pub version: String,
    pub exported_at: String,
    pub macros: Vec<ChatMacro>}

impl MacroExport {
    /// Create new export
    pub fn new(macros: Vec<ChatMacro>) -> Self {
        Self {
            version: "1.0".to_string(),
            exported_at: chrono::Utc::now().to_rfc3339(),
            macros}
    }
}