use std::collections::HashMap;
use std::sync::Arc;

// Removed unused import: std::time::SystemTime
use crate::memory::primitives::{
    BaseMemory, MemoryContent, MemoryNode as NewMemoryNode, MemoryTypeEnum,
};
use crate::memory::types_legacy;

/// Bridge pattern for backward compatibility during memory type migration
///
/// Zero-allocation conversion functions with inline optimization
/// Provides seamless migration path from legacy types to new domain types

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompatibilityMode {
    /// Strict mode: Only allow exact matches
    Strict,
    /// Flexible mode: Allow best-effort conversions
    Flexible,
    /// Hybrid mode: Support both legacy and new types simultaneously
    Hybrid,
}

/// Convert legacy MemoryType to new MemoryTypeEnum with zero allocation
#[inline(always)]
pub const fn convert_memory_type(legacy_type: &types_legacy::MemoryType) -> MemoryTypeEnum {
    match legacy_type {
        types_legacy::MemoryType::ShortTerm => MemoryTypeEnum::Working,
        types_legacy::MemoryType::LongTerm => MemoryTypeEnum::LongTerm,
        types_legacy::MemoryType::Semantic => MemoryTypeEnum::Semantic,
        types_legacy::MemoryType::Episodic => MemoryTypeEnum::Episodic,
        types_legacy::MemoryType::Conversation => MemoryTypeEnum::Working,
        types_legacy::MemoryType::Context => MemoryTypeEnum::Semantic,
        types_legacy::MemoryType::Document => MemoryTypeEnum::LongTerm,
    }
}

/// Convert new MemoryTypeEnum back to legacy MemoryType with zero allocation
#[inline(always)]
pub const fn convert_memory_type_back(
    new_type: &MemoryTypeEnum,
) -> Option<types_legacy::MemoryType> {
    match new_type {
        MemoryTypeEnum::Working => Some(types_legacy::MemoryType::ShortTerm),
        MemoryTypeEnum::LongTerm => Some(types_legacy::MemoryType::LongTerm),
        MemoryTypeEnum::Semantic => Some(types_legacy::MemoryType::Semantic),
        MemoryTypeEnum::Episodic => Some(types_legacy::MemoryType::Episodic),
        // New types that don't have legacy equivalents
        MemoryTypeEnum::Fact
        | MemoryTypeEnum::Episode
        | MemoryTypeEnum::Procedural
        | MemoryTypeEnum::Declarative
        | MemoryTypeEnum::Implicit
        | MemoryTypeEnum::Explicit
        | MemoryTypeEnum::Contextual
        | MemoryTypeEnum::Temporal
        | MemoryTypeEnum::Spatial
        | MemoryTypeEnum::Associative
        | MemoryTypeEnum::Emotional => None,
    }
}

/// Convert legacy MemoryNode to new MemoryNode with zero-copy content sharing
pub fn convert_memory_node(
    legacy_node: &types_legacy::MemoryNode,
    _mode: CompatibilityMode,
) -> Result<NewMemoryNode, CompatibilityError> {
    let memory_type = convert_memory_type(&legacy_node.memory_type);

    // Convert content to new MemoryContent format
    let content = if legacy_node.content.is_empty() {
        MemoryContent::Empty
    } else {
        MemoryContent::Text(Arc::from(legacy_node.content.as_str()))
    };

    // Create base memory with metadata preservation
    let base_memory = BaseMemory::new(
        uuid::Uuid::new_v4(), // Generate new UUID for compatibility
        memory_type,
        content,
    );

    // Build new memory node with preserved metadata
    let mut builder = NewMemoryNode::builder()
        .with_base_memory(base_memory)
        .with_importance(legacy_node.metadata.importance)
        .with_creation_time(legacy_node.metadata.creation_time)
        .with_last_accessed(legacy_node.metadata.last_accessed);

    // Add embedding if present
    if let Some(embedding) = &legacy_node.embedding {
        builder = builder.with_embedding(embedding.clone());
    }

    builder
        .build()
        .map_err(|e| CompatibilityError::ConversionFailed(e.to_string()))
}

/// Convert new MemoryNode back to legacy format for backward compatibility
pub fn convert_memory_node_back(
    new_node: &NewMemoryNode,
    mode: CompatibilityMode,
) -> Result<types_legacy::MemoryNode, CompatibilityError> {
    // Try to convert memory type back to legacy format
    let legacy_type =
        convert_memory_type_back(&new_node.base_memory().memory_type).ok_or_else(|| {
            CompatibilityError::TypeNotSupported(format!(
                "MemoryType {:?} has no legacy equivalent",
                new_node.base_memory().memory_type
            ))
        })?;

    // Convert content back to string
    let content = match new_node.base_memory().content {
        MemoryContent::Text(ref text) => text.to_string(),
        MemoryContent::Empty => String::new(),
        _ => {
            if matches!(mode, CompatibilityMode::Strict) {
                return Err(CompatibilityError::ContentNotSupported(
                    "Non-text content not supported in legacy format".to_string(),
                ));
            }
            // In flexible mode, use debug representation
            format!("{:?}", new_node.base_memory().content)
        }
    };

    // Build legacy metadata
    let metadata = types_legacy::MemoryMetadata {
        importance: new_node.importance(),
        last_accessed: new_node.last_accessed(),
        creation_time: new_node.creation_time(),
    };

    Ok(types_legacy::MemoryNode {
        id: new_node.id().as_u128() as u64, // Convert UUID to u64 (may lose precision)
        content,
        memory_type: legacy_type,
        metadata,
        embedding: new_node.embedding().map(|e| e.to_vec()),
    })
}

/// Error types for compatibility conversion operations
#[derive(Debug, Clone)]
pub enum CompatibilityError {
    ConversionFailed(String),
    TypeNotSupported(String),
    ContentNotSupported(String),
    ValidationFailed(String),
}

impl std::fmt::Display for CompatibilityError {
    #[cold]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompatibilityError::ConversionFailed(msg) => write!(f, "Conversion failed: {}", msg),
            CompatibilityError::TypeNotSupported(msg) => write!(f, "Type not supported: {}", msg),
            CompatibilityError::ContentNotSupported(msg) => {
                write!(f, "Content not supported: {}", msg)
            }
            CompatibilityError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
        }
    }
}

impl std::error::Error for CompatibilityError {}

/// Compatibility layer for managing migration between legacy and new memory systems
pub struct CompatibilityLayer {
    mode: CompatibilityMode,
    conversion_stats: HashMap<String, u64>,
}

impl CompatibilityLayer {
    /// Create new compatibility layer with specified mode
    #[inline]
    pub fn new(mode: CompatibilityMode) -> Self {
        Self {
            mode,
            conversion_stats: HashMap::new(),
        }
    }

    /// Convert batch of legacy nodes to new format with statistics tracking
    pub fn convert_batch_legacy_to_new(
        &mut self,
        legacy_nodes: Vec<types_legacy::MemoryNode>,
    ) -> Result<Vec<NewMemoryNode>, CompatibilityError> {
        let mut converted = Vec::with_capacity(legacy_nodes.len());
        let mut successful_conversions = 0u64;

        for legacy_node in legacy_nodes {
            match convert_memory_node(&legacy_node, self.mode) {
                Ok(new_node) => {
                    converted.push(new_node);
                    successful_conversions += 1;
                }
                Err(e) => {
                    if matches!(self.mode, CompatibilityMode::Strict) {
                        return Err(e);
                    }
                    // In flexible mode, skip failed conversions and continue
                    continue;
                }
            }
        }

        self.conversion_stats
            .insert("legacy_to_new".to_string(), successful_conversions);
        Ok(converted)
    }

    /// Convert batch of new nodes to legacy format with statistics tracking
    pub fn convert_batch_new_to_legacy(
        &mut self,
        new_nodes: Vec<NewMemoryNode>,
    ) -> Result<Vec<types_legacy::MemoryNode>, CompatibilityError> {
        let mut converted = Vec::with_capacity(new_nodes.len());
        let mut successful_conversions = 0u64;

        for new_node in new_nodes {
            match convert_memory_node_back(&new_node, self.mode) {
                Ok(legacy_node) => {
                    converted.push(legacy_node);
                    successful_conversions += 1;
                }
                Err(e) => {
                    if matches!(self.mode, CompatibilityMode::Strict) {
                        return Err(e);
                    }
                    // In flexible mode, skip failed conversions and continue
                    continue;
                }
            }
        }

        self.conversion_stats
            .insert("new_to_legacy".to_string(), successful_conversions);
        Ok(converted)
    }

    /// Get conversion statistics for monitoring migration progress
    #[inline]
    pub fn stats(&self) -> &HashMap<String, u64> {
        &self.conversion_stats
    }

    /// Reset conversion statistics
    #[inline]
    pub fn reset_stats(&mut self) {
        self.conversion_stats.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_conversion() {
        let legacy_short_term = types_legacy::MemoryType::ShortTerm;
        let converted = convert_memory_type(&legacy_short_term);
        assert_eq!(converted, MemoryTypeEnum::Working);

        let back_converted = convert_memory_type_back(&converted);
        assert_eq!(back_converted, Some(types_legacy::MemoryType::ShortTerm));
    }

    #[test]
    fn test_compatibility_layer() {
        let mut layer = CompatibilityLayer::new(CompatibilityMode::Flexible);
        assert_eq!(layer.stats().len(), 0);

        layer.reset_stats();
        assert_eq!(layer.stats().len(), 0);
    }
}
