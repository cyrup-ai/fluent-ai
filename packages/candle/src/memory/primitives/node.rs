use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::SystemTime;

use crossbeam_skiplist::SkipMap;
use crossbeam_utils::CachePadded;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{MapAccess, Visitor},
    ser::SerializeStruct};
use uuid::Uuid;

use super::types::{
    BaseMemory, MemoryContent, MemoryError, MemoryResult, MemoryTypeEnum, RelationshipType};

/// High-performance memory node with concurrent design
///
/// Features:
/// - UUID-based node identification with inline generation
/// - SIMD-aligned embedding vectors for AVX2/NEON optimization
/// - CachePadded metadata structure to prevent false sharing
/// - AtomicU64 for concurrent access statistics and version tracking
/// - Lock-free relationship tracking with crossbeam-skiplist
#[derive(Debug, Clone)]
pub struct MemoryNode {
    /// Base memory with core data
    pub base_memory: BaseMemory,

    /// SIMD-aligned embedding vector for AVX2/NEON optimization
    pub embedding: Option<AlignedEmbedding>,

    /// Cache-padded metadata to prevent false sharing
    pub metadata: Arc<CachePadded<MemoryNodeMetadata>>,

    /// Lock-free relationship tracking with skip-list
    pub relationships: Arc<SkipMap<Uuid, MemoryRelationshipEntry>>,

    /// Atomic access statistics for concurrent monitoring
    pub stats: Arc<CachePadded<MemoryNodeStats>>}

/// SIMD-aligned embedding vector for optimal performance
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(align(32))] // Align to 32 bytes for AVX2 SIMD operations
pub struct AlignedEmbedding {
    /// Embedding vector data aligned for SIMD operations
    pub data: Vec<f32>,
    /// Vector dimension for validation
    pub dimension: usize}

impl AlignedEmbedding {
    /// Create new aligned embedding with SIMD optimization
    #[inline]
    pub fn new(data: Vec<f32>) -> Self {
        let dimension = data.len();
        Self { data, dimension }
    }

    /// Get embedding data as slice for SIMD operations
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Convert to Vec for compatibility
    #[inline]
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Calculate dot product with SIMD optimization hint
    #[inline]
    pub fn dot_product(&self, other: &Self) -> Option<f32> {
        if self.dimension != other.dimension {
            return None;
        }

        // Hint to compiler for SIMD optimization
        Some(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .sum(),
        )
    }

    /// Calculate cosine similarity with SIMD optimization
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> Option<f32> {
        if self.dimension != other.dimension {
            return None;
        }

        let dot = self.dot_product(other)?;
        let norm_self = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_other = other.data.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_self == 0.0 || norm_other == 0.0 {
            Some(0.0)
        } else {
            Some(dot / (norm_self * norm_other))
        }
    }
}

/// Cache-padded metadata to prevent false sharing between CPU cores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeMetadata {
    /// Importance score (0.0 to 1.0)
    pub importance: f32,
    /// Keywords for search optimization
    pub keywords: Vec<Arc<str>>,
    /// Classification tags
    pub tags: Vec<Arc<str>>,
    /// Custom metadata with zero-copy keys
    pub custom: HashMap<Arc<str>, Arc<serde_json::Value>>,
    /// Version for optimistic concurrency control
    pub version: u64}

impl MemoryNodeMetadata {
    /// Create new metadata with default values
    #[inline]
    pub fn new() -> Self {
        Self {
            importance: 0.5,
            keywords: Vec::new(),
            tags: Vec::new(),
            custom: HashMap::new(),
            version: 1}
    }

    /// Add keyword with zero-copy sharing
    #[inline]
    pub fn add_keyword(&mut self, keyword: impl Into<Arc<str>>) {
        self.keywords.push(keyword.into());
        self.version += 1;
    }

    /// Add tag with zero-copy sharing
    #[inline]
    pub fn add_tag(&mut self, tag: impl Into<Arc<str>>) {
        self.tags.push(tag.into());
        self.version += 1;
    }

    /// Set custom metadata with zero-copy key
    #[inline]
    pub fn set_custom(&mut self, key: impl Into<Arc<str>>, value: serde_json::Value) {
        self.custom.insert(key.into(), Arc::new(value));
        self.version += 1;
    }
}

impl Default for MemoryNodeMetadata {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Atomic access statistics for concurrent monitoring
#[derive(Debug)]
pub struct MemoryNodeStats {
    /// Total access count
    pub access_count: AtomicU64,
    /// Read operation count
    pub read_count: AtomicU64,
    /// Write operation count
    pub write_count: AtomicU64,
    /// Relationship access count
    pub relationship_count: AtomicUsize,
    /// Last access timestamp (as nanos since UNIX_EPOCH)
    pub last_access_nanos: AtomicU64}

impl MemoryNodeStats {
    /// Create new stats with zero counters
    #[inline]
    pub fn new() -> Self {
        Self {
            access_count: AtomicU64::new(0),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            relationship_count: AtomicUsize::new(0),
            last_access_nanos: AtomicU64::new(0)}
    }

    /// Record read access atomically
    #[inline]
    pub fn record_read(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.read_count.fetch_add(1, Ordering::Relaxed);
        self.update_last_access();
    }

    /// Record write access atomically
    #[inline]
    pub fn record_write(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.write_count.fetch_add(1, Ordering::Relaxed);
        self.update_last_access();
    }

    /// Update last access timestamp atomically
    #[inline]
    fn update_last_access(&self) {
        let now_nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.last_access_nanos.store(now_nanos, Ordering::Relaxed);
    }

    /// Get access count
    #[inline]
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Get read count
    #[inline]
    pub fn read_count(&self) -> u64 {
        self.read_count.load(Ordering::Relaxed)
    }

    /// Get write count
    #[inline]
    pub fn write_count(&self) -> u64 {
        self.write_count.load(Ordering::Relaxed)
    }

    /// Get relationship count
    #[inline]
    pub fn relationship_count(&self) -> usize {
        self.relationship_count.load(Ordering::Relaxed)
    }

    /// Get last access time
    #[inline]
    pub fn last_access_time(&self) -> Option<SystemTime> {
        let nanos = self.last_access_nanos.load(Ordering::Relaxed);
        if nanos == 0 {
            None
        } else {
            SystemTime::UNIX_EPOCH.checked_add(std::time::Duration::from_nanos(nanos))
        }
    }
}

/// Lock-free relationship entry for concurrent access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRelationshipEntry {
    /// Target node ID
    pub target_id: Uuid,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength (0.0 to 1.0)
    pub strength: f32,
    /// Creation timestamp
    pub created_at: SystemTime}

impl MemoryRelationshipEntry {
    /// Create new relationship entry
    #[inline]
    pub fn new(target_id: Uuid, relationship_type: RelationshipType, strength: f32) -> Self {
        Self {
            target_id,
            relationship_type,
            strength: strength.clamp(0.0, 1.0),
            created_at: SystemTime::now()}
    }
}

impl MemoryNode {
    /// Create new memory node with generated UUID
    #[inline]
    pub fn new(memory_type: MemoryTypeEnum, content: MemoryContent) -> Self {
        let id = Uuid::new_v4();
        let base_memory = BaseMemory::new(id, memory_type, content);

        Self {
            base_memory,
            embedding: None,
            metadata: Arc::new(CachePadded::new(MemoryNodeMetadata::new())),
            relationships: Arc::new(SkipMap::new()),
            stats: Arc::new(CachePadded::new(MemoryNodeStats::new()))}
    }

    /// Create memory node with specific UUID
    #[inline]
    pub fn with_id(id: Uuid, memory_type: MemoryTypeEnum, content: MemoryContent) -> Self {
        let base_memory = BaseMemory::new(id, memory_type, content);

        Self {
            base_memory,
            embedding: None,
            metadata: Arc::new(CachePadded::new(MemoryNodeMetadata::new())),
            relationships: Arc::new(SkipMap::new()),
            stats: Arc::new(CachePadded::new(MemoryNodeStats::new()))}
    }

    /// Get node ID
    #[inline]
    pub fn id(&self) -> Uuid {
        self.stats.record_read();
        self.base_memory.id
    }

    /// Get base memory reference
    #[inline]
    pub fn base_memory(&self) -> &BaseMemory {
        self.stats.record_read();
        &self.base_memory
    }

    /// Get memory type
    #[inline]
    pub fn memory_type(&self) -> MemoryTypeEnum {
        self.stats.record_read();
        self.base_memory.memory_type
    }

    /// Get content reference
    #[inline]
    pub fn content(&self) -> &MemoryContent {
        self.stats.record_read();
        &self.base_memory.content
    }

    /// Get embedding reference
    #[inline]
    pub fn embedding(&self) -> Option<&AlignedEmbedding> {
        self.stats.record_read();
        self.embedding.as_ref()
    }

    /// Get creation time
    #[inline]
    pub fn creation_time(&self) -> SystemTime {
        self.stats.record_read();
        self.base_memory.created_at
    }

    /// Get last accessed time from stats
    #[inline]
    pub fn last_accessed(&self) -> SystemTime {
        self.stats
            .last_access_time()
            .unwrap_or_else(|| self.base_memory.created_at)
    }

    /// Get importance from metadata
    #[inline]
    pub fn importance(&self) -> f32 {
        self.stats.record_read();
        self.metadata.importance
    }

    /// Set embedding with SIMD alignment
    pub fn set_embedding(&mut self, embedding: Vec<f32>) -> MemoryResult<()> {
        if embedding.is_empty() {
            return Err(MemoryError::invalid_content("Embedding cannot be empty"));
        }

        self.stats.record_write();
        self.embedding = Some(AlignedEmbedding::new(embedding));
        Ok(())
    }

    /// Set importance with validation
    pub fn set_importance(&mut self, importance: f32) -> MemoryResult<()> {
        if !(0.0..=1.0).contains(&importance) {
            return Err(MemoryError::invalid_content(
                "Importance must be between 0.0 and 1.0",
            ));
        }

        self.stats.record_write();

        // Update metadata atomically by cloning and replacing
        let mut new_metadata = (**self.metadata).clone();
        new_metadata.importance = importance;
        new_metadata.version += 1;

        self.metadata = Arc::new(CachePadded::new(new_metadata));
        Ok(())
    }

    /// Add keyword to metadata
    pub fn add_keyword(&mut self, keyword: impl Into<Arc<str>>) {
        self.stats.record_write();

        let mut new_metadata = (**self.metadata).clone();
        new_metadata.add_keyword(keyword);

        self.metadata = Arc::new(CachePadded::new(new_metadata));
    }

    /// Add tag to metadata
    pub fn add_tag(&mut self, tag: impl Into<Arc<str>>) {
        self.stats.record_write();

        let mut new_metadata = (**self.metadata).clone();
        new_metadata.add_tag(tag);

        self.metadata = Arc::new(CachePadded::new(new_metadata));
    }

    /// Set custom metadata value
    pub fn set_custom_metadata(&mut self, key: impl Into<Arc<str>>, value: serde_json::Value) {
        self.stats.record_write();

        let mut new_metadata = (**self.metadata).clone();
        new_metadata.set_custom(key, value);

        self.metadata = Arc::new(CachePadded::new(new_metadata));
    }

    /// Add relationship with lock-free skip-list
    pub fn add_relationship(
        &self,
        target_id: Uuid,
        relationship_type: RelationshipType,
        strength: f32,
    ) -> MemoryResult<()> {
        if target_id == self.base_memory.id {
            return Err(MemoryError::invalid_content(
                "Cannot create self-relationship",
            ));
        }

        let entry = MemoryRelationshipEntry::new(target_id, relationship_type, strength);
        self.relationships.insert(target_id, entry);
        self.stats
            .relationship_count
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Remove relationship by target ID
    pub fn remove_relationship(&self, target_id: Uuid) -> bool {
        let removed = self.relationships.remove(&target_id).is_some();
        if removed {
            self.stats
                .relationship_count
                .fetch_sub(1, Ordering::Relaxed);
        }
        removed
    }

    /// Get relationship to specific target
    pub fn get_relationship(&self, target_id: Uuid) -> Option<MemoryRelationshipEntry> {
        self.stats.record_read();
        self.relationships
            .get(&target_id)
            .map(|entry| entry.value().clone())
    }

    /// List all relationships
    pub fn list_relationships(&self) -> Vec<(Uuid, MemoryRelationshipEntry)> {
        self.stats.record_read();
        self.relationships
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    /// Get access statistics
    #[inline]
    pub fn stats(&self) -> &MemoryNodeStats {
        &self.stats
    }

    /// Calculate similarity with another node using embeddings
    pub fn calculate_similarity(&self, other: &Self) -> Option<f32> {
        self.stats.record_read();
        other.stats.record_read();

        match (&self.embedding, &other.embedding) {
            (Some(embedding1), Some(embedding2)) => embedding1.cosine_similarity(embedding2),
            _ => None}
    }

}

impl Serialize for MemoryNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("MemoryNode", 2)?;
        state.serialize_field("base_memory", &self.base_memory)?;
        state.serialize_field("embedding", &self.embedding)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for MemoryNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            BaseMemory,
            Embedding}

        struct MemoryNodeVisitor;

        impl<'de> Visitor<'de> for MemoryNodeVisitor {
            type Value = MemoryNode;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct MemoryNode")
            }

            fn visit_map<V>(self, mut map: V) -> Result<MemoryNode, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut base_memory = None;
                let mut embedding = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::BaseMemory => {
                            if base_memory.is_some() {
                                return Err(serde::de::Error::duplicate_field("base_memory"));
                            }
                            base_memory = Some(map.next_value()?);
                        }
                        Field::Embedding => {
                            if embedding.is_some() {
                                return Err(serde::de::Error::duplicate_field("embedding"));
                            }
                            embedding = Some(map.next_value()?);
                        }
                    }
                }

                let base_memory =
                    base_memory.ok_or_else(|| serde::de::Error::missing_field("base_memory"))?;
                let embedding = embedding.unwrap_or(None);

                Ok(MemoryNode {
                    base_memory,
                    embedding,
                    metadata: Arc::new(CachePadded::new(MemoryNodeMetadata::new())),
                    relationships: Arc::new(SkipMap::new()),
                    stats: Arc::new(CachePadded::new(MemoryNodeStats::new()))})
            }
        }

        const FIELDS: &[&str] = &["base_memory", "embedding"];
        deserializer.deserialize_struct("MemoryNode", FIELDS, MemoryNodeVisitor)
    }
}

impl PartialEq for MemoryNode {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.base_memory.id == other.base_memory.id
    }
}

impl Eq for MemoryNode {}

impl std::hash::Hash for MemoryNode {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.base_memory.id.hash(state);
    }
}

