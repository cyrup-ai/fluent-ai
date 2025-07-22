use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crossbeam_utils::CachePadded;
use serde::{Deserialize, Serialize};

use super::super::primitives::types::{MemoryError, MemoryResult};

/// Vector store configuration with SIMD optimization settings
///
/// Features:
/// - Dimension validation with compile-time checks where possible
/// - Distance metric selection with SIMD-optimized implementations
/// - Index parameters with memory usage estimation
/// - Similarity threshold with floating-point precision handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Vector store type
    pub store_type: VectorStoreType,
    /// Embedding model configuration
    pub embedding_config: EmbeddingConfig,
    /// Vector dimension (validated at runtime)
    pub dimension: usize,
    /// Distance metric for similarity calculations
    pub distance_metric: DistanceMetric,
    /// Index configuration for performance optimization
    pub index_config: IndexConfig,
    /// SIMD optimization settings
    pub simd_config: SimdConfig,
    /// Connection configuration for external stores
    pub connection_config: Option<VectorConnectionConfig>,
    /// Performance tuning parameters
    pub performance_config: PerformanceConfig,
    /// Memory usage configuration
    pub memory_config: MemoryConfig,
}

/// Vector store types with performance characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum VectorStoreType {
    /// SurrealDB vector store - Multi-model database with vector support
    SurrealDB = 0,
    /// In-memory vector store - Fastest for small datasets
    Memory = 1,
    /// FAISS vector store - Facebook AI Similarity Search
    FAISS = 2,
    /// HNSW vector store - Hierarchical Navigable Small World
    HNSW = 3,
    /// Annoy vector store - Approximate Nearest Neighbors Oh Yeah
    Annoy = 4,
    /// Milvus vector store - Purpose-built vector database
    Milvus = 5,
    /// Pinecone vector store - Managed vector database
    Pinecone = 6,
}

impl VectorStoreType {
    /// Get recommended index type for vector store
    #[inline]
    pub const fn recommended_index_type(&self) -> IndexType {
        match self {
            Self::SurrealDB | Self::Memory => IndexType::FlatIP,
            Self::FAISS => IndexType::IVFPQ,
            Self::HNSW => IndexType::HNSW,
            Self::Annoy => IndexType::Annoy,
            Self::Milvus => IndexType::IVFPQ,
            Self::Pinecone => IndexType::FlatIP,
        }
    }

    /// Check if store supports batch operations
    #[inline]
    pub const fn supports_batch_operations(&self) -> bool {
        match self {
            Self::SurrealDB | Self::FAISS | Self::HNSW | Self::Milvus | Self::Pinecone => true,
            Self::Memory | Self::Annoy => false,
        }
    }

    /// Get optimal batch size for store type
    #[inline]
    pub const fn optimal_batch_size(&self) -> usize {
        match self {
            Self::SurrealDB => 1000,
            Self::Memory => 100,
            Self::FAISS => 10000,
            Self::HNSW => 1000,
            Self::Annoy => 1,
            Self::Milvus => 2000,
            Self::Pinecone => 100,
        }
    }

    /// Check if store supports real-time updates
    #[inline]
    pub const fn supports_realtime_updates(&self) -> bool {
        match self {
            Self::SurrealDB | Self::Memory | Self::Milvus | Self::Pinecone => true,
            Self::FAISS | Self::HNSW | Self::Annoy => false,
        }
    }
}

impl std::fmt::Display for VectorStoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SurrealDB => write!(f, "surrealdb"),
            Self::Memory => write!(f, "memory"),
            Self::FAISS => write!(f, "faiss"),
            Self::HNSW => write!(f, "hnsw"),
            Self::Annoy => write!(f, "annoy"),
            Self::Milvus => write!(f, "milvus"),
            Self::Pinecone => write!(f, "pinecone"),
        }
    }
}

/// Distance metrics with SIMD optimization support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DistanceMetric {
    /// Cosine similarity - Most common for semantic search
    Cosine = 0,
    /// Euclidean distance - L2 norm
    Euclidean = 1,
    /// Manhattan distance - L1 norm
    Manhattan = 2,
    /// Dot product - Inner product similarity
    DotProduct = 3,
    /// Hamming distance - For binary vectors
    Hamming = 4,
    /// Jaccard similarity - For sparse vectors
    Jaccard = 5,
}

impl DistanceMetric {
    /// Check if metric supports SIMD optimization
    #[inline]
    pub const fn supports_simd(&self) -> bool {
        match self {
            Self::Cosine | Self::Euclidean | Self::DotProduct => true,
            Self::Manhattan | Self::Hamming | Self::Jaccard => false,
        }
    }

    /// Get recommended SIMD instruction set for metric
    #[inline]
    pub const fn recommended_simd_set(&self) -> SimdInstructionSet {
        match self {
            Self::Cosine | Self::DotProduct => SimdInstructionSet::AVX2,
            Self::Euclidean => SimdInstructionSet::AVX512,
            _ => SimdInstructionSet::None,
        }
    }

    /// Check if metric is symmetric
    #[inline]
    pub const fn is_symmetric(&self) -> bool {
        match self {
            Self::Cosine | Self::Euclidean | Self::Manhattan | Self::Hamming | Self::Jaccard => {
                true
            }
            Self::DotProduct => false,
        }
    }

    /// Get value range for metric
    #[inline]
    pub const fn value_range(&self) -> (f32, f32) {
        match self {
            Self::Cosine => (-1.0, 1.0),
            Self::Euclidean | Self::Manhattan => (0.0, f32::INFINITY),
            Self::DotProduct => (f32::NEG_INFINITY, f32::INFINITY),
            Self::Hamming => (0.0, f32::INFINITY),
            Self::Jaccard => (0.0, 1.0),
        }
    }
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cosine => write!(f, "cosine"),
            Self::Euclidean => write!(f, "euclidean"),
            Self::Manhattan => write!(f, "manhattan"),
            Self::DotProduct => write!(f, "dot_product"),
            Self::Hamming => write!(f, "hamming"),
            Self::Jaccard => write!(f, "jaccard"),
        }
    }
}

/// SIMD instruction sets for vector operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SimdInstructionSet {
    /// No SIMD optimization
    None = 0,
    /// SSE (128-bit)
    SSE = 1,
    /// AVX (256-bit)
    AVX = 2,
    /// AVX2 (256-bit with integer support)
    AVX2 = 3,
    /// AVX-512 (512-bit)
    AVX512 = 4,
    /// ARM NEON (128-bit)
    NEON = 5,
}

impl SimdInstructionSet {
    /// Get vector width in bytes
    #[inline]
    pub const fn vector_width_bytes(&self) -> usize {
        match self {
            Self::None => 4,              // Single f32
            Self::SSE | Self::NEON => 16, // 128-bit
            Self::AVX | Self::AVX2 => 32, // 256-bit
            Self::AVX512 => 64,           // 512-bit
        }
    }

    /// Get number of f32 elements per vector
    #[inline]
    pub const fn f32_elements_per_vector(&self) -> usize {
        self.vector_width_bytes() / 4
    }

    /// Check if instruction set is available on current CPU
    pub fn is_available(&self) -> bool {
        match self {
            Self::None => true,
            #[cfg(target_arch = "x86_64")]
            Self::SSE => is_x86_feature_detected!("sse"),
            #[cfg(target_arch = "x86_64")]
            Self::AVX => is_x86_feature_detected!("avx"),
            #[cfg(target_arch = "x86_64")]
            Self::AVX2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Self::AVX512 => is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "aarch64")]
            Self::NEON => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(target_arch = "x86_64"))]
            Self::SSE | Self::AVX | Self::AVX2 | Self::AVX512 => false,
            #[cfg(not(target_arch = "aarch64"))]
            Self::NEON => false,
        }
    }

    /// Detect best available SIMD instruction set
    pub fn detect_best_available() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::AVX512.is_available() {
                Self::AVX512
            } else if Self::AVX2.is_available() {
                Self::AVX2
            } else if Self::AVX.is_available() {
                Self::AVX
            } else if Self::SSE.is_available() {
                Self::SSE
            } else {
                Self::None
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if Self::NEON.is_available() {
                Self::NEON
            } else {
                Self::None
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        Self::None
    }
}

/// Index types for vector stores
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum IndexType {
    /// Flat index with inner product
    FlatIP = 0,
    /// Flat index with L2 distance
    FlatL2 = 1,
    /// Inverted file with product quantization
    IVFPQ = 2,
    /// Hierarchical Navigable Small World
    HNSW = 3,
    /// Approximate nearest neighbors optimized for memory
    Annoy = 4,
    /// Locality sensitive hashing
    LSH = 5,
    /// Scalar quantization
    SQ = 6,
}

impl IndexType {
    /// Check if index type supports exact search
    #[inline]
    pub const fn supports_exact_search(&self) -> bool {
        match self {
            Self::FlatIP | Self::FlatL2 => true,
            Self::IVFPQ | Self::HNSW | Self::Annoy | Self::LSH | Self::SQ => false,
        }
    }

    /// Get memory usage multiplier compared to raw vectors
    #[inline]
    pub const fn memory_multiplier(&self) -> f32 {
        match self {
            Self::FlatIP | Self::FlatL2 => 1.0,
            Self::IVFPQ => 0.1,
            Self::HNSW => 1.5,
            Self::Annoy => 2.0,
            Self::LSH => 1.2,
            Self::SQ => 0.25,
        }
    }

    /// Get build time complexity relative to vector count
    #[inline]
    pub const fn build_complexity(&self) -> &'static str {
        match self {
            Self::FlatIP | Self::FlatL2 => "O(1)",
            Self::IVFPQ => "O(n)",
            Self::HNSW => "O(n log n)",
            Self::Annoy => "O(n log n)",
            Self::LSH => "O(n)",
            Self::SQ => "O(n)",
        }
    }
}

/// SIMD configuration for vector operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdConfig {
    /// Target SIMD instruction set
    pub instruction_set: SimdInstructionSet,
    /// Enable auto-detection of best SIMD support
    pub auto_detect: bool,
    /// Force specific alignment for vectors
    pub force_alignment: Option<usize>,
    /// Enable SIMD for distance calculations
    pub enable_distance_simd: bool,
    /// Enable SIMD for vector normalization
    pub enable_normalization_simd: bool,
    /// Minimum vector dimension to use SIMD
    pub simd_threshold: usize,
}

impl SimdConfig {
    /// Create optimized SIMD configuration
    #[inline]
    pub fn optimized() -> Self {
        let instruction_set = {
            // Removed unexpected cfg condition "simd-auto-detect" - feature does not exist
            SimdInstructionSet::None
        };

        Self {
            instruction_set,
            auto_detect: true,
            force_alignment: Some(32), // 256-bit alignment for AVX2
            enable_distance_simd: true,
            enable_normalization_simd: true,
            simd_threshold: 16, // Use SIMD for vectors >= 16 dimensions
        }
    }

    /// Create disabled SIMD configuration
    #[inline]
    pub fn disabled() -> Self {
        Self {
            instruction_set: SimdInstructionSet::None,
            auto_detect: false,
            force_alignment: None,
            enable_distance_simd: false,
            enable_normalization_simd: false,
            simd_threshold: usize::MAX,
        }
    }

    /// Check if SIMD should be used for given dimension
    #[inline]
    pub fn should_use_simd(&self, dimension: usize) -> bool {
        self.instruction_set != SimdInstructionSet::None
            && dimension >= self.simd_threshold
            && (self.enable_distance_simd || self.enable_normalization_simd)
    }

    /// Get optimal alignment for vectors
    #[inline]
    pub fn optimal_alignment(&self) -> usize {
        self.force_alignment
            .unwrap_or_else(|| self.instruction_set.vector_width_bytes())
            .max(4) // Minimum f32 alignment
    }
}

impl Default for SimdConfig {
    #[inline]
    fn default() -> Self {
        Self::optimized()
    }
}

/// Index configuration for performance optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Index type
    pub index_type: IndexType,
    /// Number of clusters for IVF-based indices
    pub num_clusters: Option<usize>,
    /// Number of bits for PQ compression
    pub pq_bits: Option<u8>,
    /// Number of subspaces for PQ
    pub pq_subspaces: Option<usize>,
    /// HNSW maximum connections per node
    pub hnsw_max_connections: Option<usize>,
    /// HNSW construction parameter
    pub hnsw_ef_construction: Option<usize>,
    /// Search parameter for approximate indices
    pub search_ef: Option<usize>,
    /// Number of trees for Annoy
    pub annoy_trees: Option<usize>,
}

impl IndexConfig {
    /// Create optimized configuration for index type
    pub fn optimized(index_type: IndexType, dimension: usize, expected_vectors: usize) -> Self {
        match index_type {
            IndexType::FlatIP | IndexType::FlatL2 => Self {
                index_type,
                num_clusters: None,
                pq_bits: None,
                pq_subspaces: None,
                hnsw_max_connections: None,
                hnsw_ef_construction: None,
                search_ef: None,
                annoy_trees: None,
            },
            IndexType::IVFPQ => {
                let num_clusters = (expected_vectors as f64).sqrt() as usize;
                let pq_subspaces = (dimension / 4).max(1).min(64);
                Self {
                    index_type,
                    num_clusters: Some(num_clusters),
                    pq_bits: Some(8),
                    pq_subspaces: Some(pq_subspaces),
                    hnsw_max_connections: None,
                    hnsw_ef_construction: None,
                    search_ef: Some(num_clusters / 4),
                    annoy_trees: None,
                }
            }
            IndexType::HNSW => Self {
                index_type,
                num_clusters: None,
                pq_bits: None,
                pq_subspaces: None,
                hnsw_max_connections: Some(16),
                hnsw_ef_construction: Some(200),
                search_ef: Some(50),
                annoy_trees: None,
            },
            IndexType::Annoy => Self {
                index_type,
                num_clusters: None,
                pq_bits: None,
                pq_subspaces: None,
                hnsw_max_connections: None,
                hnsw_ef_construction: None,
                search_ef: None,
                annoy_trees: Some((expected_vectors as f64).log2() as usize * 2),
            },
            IndexType::LSH | IndexType::SQ => Self {
                index_type,
                num_clusters: None,
                pq_bits: Some(8),
                pq_subspaces: None,
                hnsw_max_connections: None,
                hnsw_ef_construction: None,
                search_ef: None,
                annoy_trees: None,
            },
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(&self, dimension: usize, num_vectors: usize) -> usize {
        let base_size = num_vectors * dimension * 4; // f32 vectors
        let multiplier = self.index_type.memory_multiplier();
        (base_size as f32 * multiplier) as usize
    }
}

impl Default for IndexConfig {
    #[inline]
    fn default() -> Self {
        Self::optimized(IndexType::FlatIP, 768, 10000)
    }
}

/// Connection configuration for external vector stores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConnectionConfig {
    /// Connection URL
    pub url: Arc<str>,
    /// API key for authentication
    pub api_key: Option<Arc<str>>,
    /// Connection timeout
    pub timeout: Duration,
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    /// Connection idle timeout
    pub idle_timeout: Duration,
    /// Enable TLS/SSL
    pub enable_tls: bool,
    /// Custom headers for requests
    pub headers: Option<Arc<serde_json::Value>>,
}

impl VectorConnectionConfig {
    /// Create new connection configuration
    #[inline]
    pub fn new(url: impl Into<Arc<str>>) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            timeout: Duration::from_secs(30),
            max_connections: 10,
            idle_timeout: Duration::from_secs(300),
            enable_tls: true,
            headers: None,
        }
    }

    /// Set API key
    #[inline]
    pub fn with_api_key(mut self, api_key: impl Into<Arc<str>>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set timeout
    #[inline]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Performance configuration for vector operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads for parallel operations
    pub num_threads: usize,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Enable prefetching for sequential access
    pub enable_prefetch: bool,
    /// Cache size for frequently accessed vectors
    pub cache_size: usize,
    /// Enable compression for storage
    pub enable_compression: bool,
    /// Quantization precision for reduced memory usage
    pub quantization_bits: Option<u8>,
}

impl PerformanceConfig {
    /// Create optimized performance configuration
    #[inline]
    pub fn optimized(store_type: VectorStoreType) -> Self {
        Self {
            num_threads: num_cpus::get(),
            batch_size: store_type.optimal_batch_size(),
            enable_prefetch: true,
            cache_size: 10000,
            enable_compression: matches!(
                store_type,
                VectorStoreType::FAISS | VectorStoreType::Milvus
            ),
            quantization_bits: None,
        }
    }

    /// Create minimal performance configuration for testing
    #[inline]
    pub fn minimal() -> Self {
        Self {
            num_threads: 1,
            batch_size: 10,
            enable_prefetch: false,
            cache_size: 100,
            enable_compression: false,
            quantization_bits: None,
        }
    }
}

impl Default for PerformanceConfig {
    #[inline]
    fn default() -> Self {
        Self::optimized(VectorStoreType::Memory)
    }
}

/// Memory configuration for vector operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Memory allocation strategy
    pub allocation_strategy: AllocationStrategy,
    /// Enable memory mapping for large indices
    pub enable_mmap: bool,
    /// Memory usage tracking
    pub track_usage: bool,
    /// Atomic memory usage counter
    #[serde(skip)]
    pub current_usage: Arc<CachePadded<AtomicUsize>>,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AllocationStrategy {
    /// Standard system allocator
    System = 0,
    /// Memory pool allocator for frequent allocations
    Pool = 1,
    /// Arena allocator for batch operations
    Arena = 2,
    /// NUMA-aware allocator
    NUMA = 3,
}

impl MemoryConfig {
    /// Create new memory configuration
    #[inline]
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            max_memory_bytes,
            allocation_strategy: AllocationStrategy::System,
            enable_mmap: max_memory_bytes > 1024 * 1024 * 1024, // Enable for >1GB
            track_usage: true,
            current_usage: Arc::new(CachePadded::new(AtomicUsize::new(0))),
        }
    }

    /// Record memory allocation
    #[inline]
    pub fn allocate(&self, bytes: usize) -> bool {
        if self.track_usage {
            let current = self.current_usage.fetch_add(bytes, Ordering::Relaxed);
            current + bytes <= self.max_memory_bytes
        } else {
            true
        }
    }

    /// Record memory deallocation
    #[inline]
    pub fn deallocate(&self, bytes: usize) {
        if self.track_usage {
            self.current_usage.fetch_sub(bytes, Ordering::Relaxed);
        }
    }

    /// Get current memory usage
    #[inline]
    pub fn current_usage(&self) -> usize {
        if self.track_usage {
            self.current_usage.load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// Get available memory
    #[inline]
    pub fn available_memory(&self) -> usize {
        self.max_memory_bytes.saturating_sub(self.current_usage())
    }

    /// Check if allocation would exceed limit
    #[inline]
    pub fn would_exceed_limit(&self, bytes: usize) -> bool {
        if self.track_usage {
            self.current_usage() + bytes > self.max_memory_bytes
        } else {
            false
        }
    }
}

impl Default for MemoryConfig {
    #[inline]
    fn default() -> Self {
        // Default to 1GB memory limit
        Self::new(1024 * 1024 * 1024)
    }
}

// Use shared EmbeddingConfig from super::shared
use super::shared::EmbeddingConfig;

impl VectorStoreConfig {
    /// Create new vector store configuration with validation
    pub fn new(
        store_type: VectorStoreType,
        embedding_config: EmbeddingConfig,
        dimension: usize,
    ) -> MemoryResult<Self> {
        // Validate dimension
        if dimension == 0 {
            return Err(MemoryError::invalid_content(
                "Vector dimension must be greater than 0",
            ));
        }

        if dimension > 65536 {
            return Err(MemoryError::invalid_content(
                "Vector dimension must be <= 65536",
            ));
        }

        // Validate dimension matches embedding config
        if dimension != embedding_config.dimension {
            return Err(MemoryError::invalid_content(
                "Vector dimension must match embedding configuration dimension",
            ));
        }

        Ok(Self {
            store_type,
            embedding_config,
            dimension,
            distance_metric: DistanceMetric::Cosine,
            index_config: IndexConfig::optimized(
                store_type.recommended_index_type(),
                dimension,
                10000,
            ),
            simd_config: SimdConfig::optimized(),
            connection_config: None,
            performance_config: PerformanceConfig::optimized(store_type),
            memory_config: MemoryConfig::default(),
        })
    }

    /// Set distance metric
    #[inline]
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set index configuration
    #[inline]
    pub fn with_index_config(mut self, config: IndexConfig) -> Self {
        self.index_config = config;
        self
    }

    /// Set SIMD configuration
    #[inline]
    pub fn with_simd_config(mut self, config: SimdConfig) -> Self {
        self.simd_config = config;
        self
    }

    /// Set connection configuration
    #[inline]
    pub fn with_connection_config(mut self, config: VectorConnectionConfig) -> Self {
        self.connection_config = Some(config);
        self
    }

    /// Set performance configuration
    #[inline]
    pub fn with_performance_config(mut self, config: PerformanceConfig) -> Self {
        self.performance_config = config;
        self
    }

    /// Estimate memory usage for given number of vectors
    pub fn estimate_memory_usage(&self, num_vectors: usize) -> usize {
        self.index_config
            .estimate_memory_usage(self.dimension, num_vectors)
    }

    /// Check if configuration supports SIMD operations
    #[inline]
    pub fn supports_simd(&self) -> bool {
        self.distance_metric.supports_simd() && self.simd_config.should_use_simd(self.dimension)
    }

    /// Get optimal batch size for operations
    #[inline]
    pub fn optimal_batch_size(&self) -> usize {
        self.performance_config.batch_size
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> MemoryResult<()> {
        // Check SIMD compatibility
        if self.simd_config.enable_distance_simd && !self.distance_metric.supports_simd() {
            return Err(MemoryError::validation(
                "Distance metric does not support SIMD optimization",
            ));
        }

        // Check index type compatibility with store type
        match (self.store_type, self.index_config.index_type) {
            (VectorStoreType::Memory, IndexType::IVFPQ) => {
                return Err(MemoryError::validation(
                    "Memory store does not support IVFPQ index",
                ));
            }
            (VectorStoreType::HNSW, index_type) if index_type != IndexType::HNSW => {
                return Err(MemoryError::validation(
                    "HNSW store only supports HNSW index type",
                ));
            }
            _ => {}
        }

        Ok(())
    }
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self::new(VectorStoreType::Memory, EmbeddingConfig::default(), 1536)
            .expect("Default vector store configuration should be valid")
    }
}
