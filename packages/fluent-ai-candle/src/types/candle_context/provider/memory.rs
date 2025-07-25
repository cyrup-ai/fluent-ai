//! Memory Integration Layer with Atomic Operations
//!
//! Zero-allocation memory management with atomic performance tracking and owned string storage.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

/// Memory node representation with owned strings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub embedding: Option<Vec<f32>>,
    pub timestamp: SystemTime,
}

impl MemoryNode {
    /// Create new memory node with owned strings
    pub fn new(id: String, content: String) -> Self {
        Self {
            id,
            content,
            metadata: HashMap::new(),
            embedding: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Add metadata entry
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Get content length
    pub fn content_length(&self) -> usize {
        self.content.len()
    }

    /// Check if has embedding
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }
}

/// Memory integration layer with atomic operations
#[derive(Debug)]
pub struct MemoryIntegration {
    /// Memory manager identifier
    pub manager_id: String,
    /// Embedding model identifier
    pub embedding_model: String,
    /// Vector dimension
    pub vector_dimension: usize,
    /// Performance tracking
    pub memory_requests: AtomicU64,
    pub successful_operations: AtomicU64,
    pub failed_operations: AtomicU64,
    pub total_processing_time_nanos: AtomicU64,
}

impl Clone for MemoryIntegration {
    fn clone(&self) -> Self {
        Self {
            manager_id: self.manager_id.clone(),
            embedding_model: self.embedding_model.clone(),
            vector_dimension: self.vector_dimension,
            memory_requests: AtomicU64::new(
                self.memory_requests
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            successful_operations: AtomicU64::new(
                self.successful_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            failed_operations: AtomicU64::new(
                self.failed_operations
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_processing_time_nanos: AtomicU64::new(
                self.total_processing_time_nanos
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

impl MemoryIntegration {
    /// Create new memory integration with owned strings
    #[inline]
    pub fn new(manager_id: String, embedding_model: String, vector_dimension: usize) -> Self {
        Self {
            manager_id,
            embedding_model,
            vector_dimension,
            memory_requests: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            failed_operations: AtomicU64::new(0),
            total_processing_time_nanos: AtomicU64::new(0),
        }
    }

    /// Record successful operation
    #[inline]
    pub fn record_success(&self, duration_nanos: u64) {
        self.successful_operations.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_nanos
            .fetch_add(duration_nanos, Ordering::Relaxed);
    }

    /// Record failed operation
    #[inline]
    pub fn record_failure(&self) {
        self.failed_operations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get success rate (0.0 to 1.0)
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let successful = self.successful_operations.load(Ordering::Relaxed);
        let failed = self.failed_operations.load(Ordering::Relaxed);
        let total = successful + failed;
        if total == 0 {
            1.0
        } else {
            successful as f64 / total as f64
        }
    }

    /// Get average processing time in nanoseconds
    #[inline]
    pub fn average_processing_time_nanos(&self) -> u64 {
        let total_time = self.total_processing_time_nanos.load(Ordering::Relaxed);
        let successful = self.successful_operations.load(Ordering::Relaxed);
        if successful == 0 {
            0
        } else {
            total_time / successful
        }
    }

    /// Record memory request
    #[inline]
    pub fn record_request(&self) {
        self.memory_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total memory requests
    #[inline]
    pub fn total_requests(&self) -> u64 {
        self.memory_requests.load(Ordering::Relaxed)
    }
}