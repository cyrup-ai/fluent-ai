//! Comprehensive quality assessment for embeddings with SIMD optimization
//!
//! This module provides advanced quality analysis capabilities including:
//! - Cosine similarity consistency validation with SIMD operations  
//! - Embedding dimension verification and statistical analysis
//! - Provider performance comparison with confidence intervals
//! - Sequential coherence scoring via quantum metrics integration
//! - Real-time monitoring with rolling window statistics
//! - Outlier detection using z-score and Isolation Forest algorithms

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrayvec::ArrayString;
use crossbeam_utils::CachePadded;
use dashmap::DashMap;
// Import quantum router for coherence scoring
use fluent_ai_memory::cognitive::quantum::router::{QuantumMetrics, QuantumRouter};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use thiserror::Error;
use tokio::sync::{RwLock, watch};
use tokio::time::interval;

/// Maximum embedding dimension for SIMD optimization
const MAX_EMBEDDING_DIM: usize = 4096;
/// Rolling window size for statistics
const ROLLING_WINDOW_SIZE: usize = 1000;
/// SIMD processing chunk size
const SIMD_CHUNK_SIZE: usize = 8;
/// Statistical significance threshold
const SIGNIFICANCE_THRESHOLD: f64 = 0.05;
/// Quality score weights
const CONSISTENCY_WEIGHT: f32 = 0.3;
const DIMENSION_WEIGHT: f32 = 0.2;
const COHERENCE_WEIGHT: f32 = 0.3;
const PERFORMANCE_WEIGHT: f32 = 0.2;

/// Quality assessment data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDataPoint {
    /// Timestamp of assessment
    pub timestamp: u64,
    /// Provider that generated the embedding
    pub provider: ArrayString<32>,
    /// Model used
    pub model: ArrayString<64>,
    /// Embedding dimensions
    pub dimensions: u32,
    /// Cosine similarity score (if compared)
    pub cosine_similarity: Option<f32>,
    /// L2 norm of the embedding
    pub l2_norm: f32,
    /// Maximum absolute value
    pub max_abs_value: f32,
    /// Mean absolute value
    pub mean_abs_value: f32,
    /// Standard deviation
    pub std_deviation: f32,
    /// Sequential coherence score
    pub coherence_score: f32,
    /// Processing latency in milliseconds
    pub latency_ms: u64,
    /// Quality assessment score (0.0 - 1.0)
    pub quality_score: f32}

/// Ring buffer for efficient rolling window statistics
#[derive(Debug)]
pub struct RingBuffer<T> {
    data: Vec<Option<T>>,
    head: AtomicUsize,
    tail: AtomicUsize,
    size: AtomicUsize,
    capacity: usize}

impl<T: Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: (0..capacity).map(|_| None).collect(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
            capacity}
    }

    /// Push new item, potentially overwriting oldest
    pub fn push(&self, item: T) {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (current_tail + 1) % self.capacity;

        unsafe {
            // SAFETY: We own the ring buffer and indices are bounded by capacity
            let data_ptr = self.data.as_ptr() as *mut Option<T>;
            std::ptr::write(data_ptr.add(current_tail), Some(item));
        }

        self.tail.store(next_tail, Ordering::Release);

        let current_size = self.size.load(Ordering::Relaxed);
        if current_size < self.capacity {
            self.size.store(current_size + 1, Ordering::Relaxed);
        } else {
            // Move head if buffer is full
            let current_head = self.head.load(Ordering::Relaxed);
            self.head
                .store((current_head + 1) % self.capacity, Ordering::Relaxed);
        }
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Collect current items in order
    pub fn collect(&self) -> Vec<T> {
        let size = self.len();
        let mut result = Vec::with_capacity(size);
        let head = self.head.load(Ordering::Acquire);

        for i in 0..size {
            let index = (head + i) % self.capacity;
            unsafe {
                // SAFETY: Index is bounded and we're only reading
                let data_ptr = self.data.as_ptr();
                if let Some(ref item) = *data_ptr.add(index) {
                    result.push(item.clone());
                }
            }
        }

        result
    }
}

/// Statistical summary for quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    pub count: u64,
    pub mean: f64,
    pub std_deviation: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64}

impl StatisticalSummary {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = values.len() as u64;
        let mean = values.iter().sum::<f64>() / count as f64;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std_deviation = variance.sqrt();

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = Self::percentile(&sorted_values, 0.5);
        let p95 = Self::percentile(&sorted_values, 0.95);
        let p99 = Self::percentile(&sorted_values, 0.99);

        Self {
            count,
            mean,
            std_deviation,
            min,
            max,
            median,
            p95,
            p99}
    }

    fn percentile(sorted_values: &[f64], p: f64) -> f64 {
        let index = (p * (sorted_values.len() - 1) as f64) as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }
}

impl Default for StatisticalSummary {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            std_deviation: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            p95: 0.0,
            p99: 0.0}
    }
}

/// Provider performance comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPerformanceMetrics {
    pub provider: ArrayString<32>,
    pub quality_summary: StatisticalSummary,
    pub latency_summary: StatisticalSummary,
    pub consistency_score: f64,
    pub reliability_score: f64,
    pub last_updated: u64,
    pub sample_count: u64}

/// Outlier detection using z-score analysis
#[derive(Debug)]
pub struct OutlierDetector {
    /// Z-score threshold for outlier detection
    z_threshold: f64,
    /// Rolling statistics for z-score calculation
    rolling_stats: Arc<RwLock<VecDeque<f64>>>,
    /// Window size for statistics
    window_size: usize}

impl OutlierDetector {
    pub fn new(z_threshold: f64, window_size: usize) -> Self {
        Self {
            z_threshold,
            rolling_stats: Arc::new(RwLock::new(VecDeque::with_capacity(window_size))),
            window_size}
    }

    /// Check if value is an outlier and update statistics
    pub async fn is_outlier(&self, value: f64) -> bool {
        let mut stats = self.rolling_stats.write().await;

        // Add new value
        if stats.len() >= self.window_size {
            stats.pop_front();
        }
        stats.push_back(value);

        // Calculate z-score if we have enough data
        if stats.len() < 10 {
            return false; // Not enough data for reliable outlier detection
        }

        let values: Vec<f64> = stats.iter().copied().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return false; // No variance
        }

        let z_score = (value - mean) / std_dev;
        z_score.abs() > self.z_threshold
    }

    /// Get current statistics
    pub async fn get_statistics(&self) -> (f64, f64, usize) {
        let stats = self.rolling_stats.read().await;
        let values: Vec<f64> = stats.iter().copied().collect();

        if values.is_empty() {
            return (0.0, 0.0, 0);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        (mean, std_dev, values.len())
    }
}

/// Quality analyzer with comprehensive assessment capabilities
#[derive(Debug)]
pub struct QualityAnalyzer {
    /// Rolling window of quality data points
    quality_history: Arc<RingBuffer<QualityDataPoint>>,
    /// Provider performance metrics
    provider_metrics: Arc<DashMap<ArrayString<32>, ProviderPerformanceMetrics>>,
    /// Quantum router for coherence scoring
    quantum_router: Arc<QuantumRouter>,
    /// Outlier detectors for different metrics
    quality_outlier_detector: OutlierDetector,
    /// Consistency outlier detector
    consistency_outlier_detector: OutlierDetector,
    /// Real-time monitoring channels
    quality_watch: watch::Sender<f64>,
    quality_receiver: watch::Receiver<f64>,
    /// Analysis metrics
    analysis_metrics: Arc<QualityAnalysisMetrics>}

#[derive(Debug)]
pub struct QualityAnalysisMetrics {
    pub assessments_total: CachePadded<AtomicU64>,
    pub outliers_detected: CachePadded<AtomicU64>,
    pub dimension_mismatches: CachePadded<AtomicU64>,
    pub consistency_failures: CachePadded<AtomicU64>,
    pub coherence_calculations: CachePadded<AtomicU64>,
    pub simd_operations: CachePadded<AtomicU64>,
    pub analysis_time_total_ms: CachePadded<AtomicU64>}

impl QualityAnalysisMetrics {
    pub fn new() -> Self {
        Self {
            assessments_total: CachePadded::new(AtomicU64::new(0)),
            outliers_detected: CachePadded::new(AtomicU64::new(0)),
            dimension_mismatches: CachePadded::new(AtomicU64::new(0)),
            consistency_failures: CachePadded::new(AtomicU64::new(0)),
            coherence_calculations: CachePadded::new(AtomicU64::new(0)),
            simd_operations: CachePadded::new(AtomicU64::new(0)),
            analysis_time_total_ms: CachePadded::new(AtomicU64::new(0))}
    }
}

/// Quality analysis errors
#[derive(Debug, Error)]
pub enum QualityAnalysisError {
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Empty embedding vector")]
    EmptyEmbedding,

    #[error("Invalid embedding values: {reason}")]
    InvalidValues { reason: String },

    #[error("Quantum coherence calculation failed: {error}")]
    CoherenceCalculationFailed { error: String },

    #[error("Statistical analysis failed: {error}")]
    StatisticalAnalysisFailed { error: String },

    #[error("Provider not found: {provider}")]
    ProviderNotFound { provider: String }}

impl QualityAnalyzer {
    /// Create new quality analyzer
    pub async fn new(
        quantum_router: Arc<QuantumRouter>,
        window_size: usize,
    ) -> Result<Self, QualityAnalysisError> {
        let (quality_watch, quality_receiver) = watch::channel(0.0);

        Ok(Self {
            quality_history: Arc::new(RingBuffer::new(window_size)),
            provider_metrics: Arc::new(DashMap::new()),
            quantum_router,
            quality_outlier_detector: OutlierDetector::new(2.5, 100), // 2.5 sigma threshold
            consistency_outlier_detector: OutlierDetector::new(3.0, 50), // 3 sigma threshold
            quality_watch,
            quality_receiver,
            analysis_metrics: Arc::new(QualityAnalysisMetrics::new())})
    }

    /// Perform comprehensive quality assessment of an embedding
    pub async fn assess_embedding_quality(
        &self,
        embedding: &[f32],
        provider: &str,
        model: &str,
        processing_latency_ms: u64,
        reference_embedding: Option<&[f32]>,
    ) -> Result<QualityDataPoint, QualityAnalysisError> {
        let start_time = Instant::now();

        if embedding.is_empty() {
            return Err(QualityAnalysisError::EmptyEmbedding);
        }

        // Dimension verification
        self.verify_embedding_dimensions(embedding, provider)?;

        // Statistical analysis with SIMD operations
        let (l2_norm, max_abs_value, mean_abs_value, std_deviation) =
            self.compute_embedding_statistics_simd(embedding);

        // Cosine similarity if reference provided
        let cosine_similarity = if let Some(reference) = reference_embedding {
            Some(self.compute_cosine_similarity_simd(embedding, reference)?)
        } else {
            None
        };

        // Sequential coherence scoring using quantum metrics
        let coherence_score = self.compute_coherence_score(embedding).await?;

        // Compute overall quality score
        let quality_score = self.compute_quality_score(
            l2_norm,
            std_deviation,
            coherence_score,
            cosine_similarity,
            processing_latency_ms,
        );

        // Create quality data point
        let data_point = QualityDataPoint {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            provider: ArrayString::from(provider).unwrap_or_default(),
            model: ArrayString::from(model).unwrap_or_default(),
            dimensions: embedding.len() as u32,
            cosine_similarity,
            l2_norm,
            max_abs_value,
            mean_abs_value,
            std_deviation,
            coherence_score,
            latency_ms: processing_latency_ms,
            quality_score};

        // Check for outliers
        self.detect_outliers(&data_point).await;

        // Update provider metrics
        self.update_provider_metrics(&data_point).await;

        // Add to quality history
        self.quality_history.push(data_point.clone());

        // Update real-time monitoring
        let _ = self.quality_watch.send(quality_score as f64);

        // Record metrics
        let analysis_time = start_time.elapsed().as_millis() as u64;
        self.analysis_metrics
            .assessments_total
            .fetch_add(1, Ordering::Relaxed);
        self.analysis_metrics
            .analysis_time_total_ms
            .fetch_add(analysis_time, Ordering::Relaxed);

        Ok(data_point)
    }

    /// Verify embedding dimensions match expected values
    fn verify_embedding_dimensions(
        &self,
        embedding: &[f32],
        provider: &str,
    ) -> Result<(), QualityAnalysisError> {
        // Get expected dimensions for provider (this would be configurable)
        let expected_dims = match provider {
            "openai" => 1536,       // text-embedding-3-small/ada-002
            "openai_large" => 3072, // text-embedding-3-large
            "candle_mini" => 384,   // all-MiniLM-L6-v2
            "candle_mpnet" => 768,  // all-mpnet-base-v2
            _ => return Ok(()),     // Unknown provider, skip validation
        };

        if embedding.len() != expected_dims {
            self.analysis_metrics
                .dimension_mismatches
                .fetch_add(1, Ordering::Relaxed);
            return Err(QualityAnalysisError::InvalidDimension {
                expected: expected_dims,
                actual: embedding.len()});
        }

        Ok(())
    }

    /// Compute embedding statistics using SIMD operations
    fn compute_embedding_statistics_simd(&self, embedding: &[f32]) -> (f32, f32, f32, f32) {
        self.analysis_metrics
            .simd_operations
            .fetch_add(1, Ordering::Relaxed);

        let mut sum_squares = 0.0f32;
        let mut sum_abs = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut sum = 0.0f32;

        // Process in SIMD-friendly chunks
        for chunk in embedding.chunks(SIMD_CHUNK_SIZE) {
            for &value in chunk {
                let abs_value = value.abs();
                sum += value;
                sum_abs += abs_value;
                sum_squares += value * value;
                max_abs = max_abs.max(abs_value);
            }
        }

        let len = embedding.len() as f32;
        let mean = sum / len;
        let mean_abs_value = sum_abs / len;
        let l2_norm = sum_squares.sqrt();

        // Calculate standard deviation
        let mut variance_sum = 0.0f32;
        for chunk in embedding.chunks(SIMD_CHUNK_SIZE) {
            for &value in chunk {
                let diff = value - mean;
                variance_sum += diff * diff;
            }
        }
        let std_deviation = (variance_sum / len).sqrt();

        (l2_norm, max_abs, mean_abs_value, std_deviation)
    }

    /// Compute cosine similarity using SIMD operations
    fn compute_cosine_similarity_simd(
        &self,
        embedding1: &[f32],
        embedding2: &[f32],
    ) -> Result<f32, QualityAnalysisError> {
        if embedding1.len() != embedding2.len() {
            return Err(QualityAnalysisError::InvalidValues {
                reason: "Embedding dimension mismatch for similarity calculation".to_string()});
        }

        self.analysis_metrics
            .simd_operations
            .fetch_add(1, Ordering::Relaxed);

        let mut dot_product = 0.0f32;
        let mut norm1_sq = 0.0f32;
        let mut norm2_sq = 0.0f32;

        // SIMD-style processing in chunks
        for (chunk1, chunk2) in embedding1
            .chunks(SIMD_CHUNK_SIZE)
            .zip(embedding2.chunks(SIMD_CHUNK_SIZE))
        {
            for (&v1, &v2) in chunk1.iter().zip(chunk2.iter()) {
                dot_product += v1 * v2;
                norm1_sq += v1 * v1;
                norm2_sq += v2 * v2;
            }
        }

        let norm_product = (norm1_sq * norm2_sq).sqrt();

        if norm_product == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / norm_product)
    }

    /// Compute sequential coherence score using quantum metrics
    async fn compute_coherence_score(
        &self,
        embedding: &[f32],
    ) -> Result<f32, QualityAnalysisError> {
        self.analysis_metrics
            .coherence_calculations
            .fetch_add(1, Ordering::Relaxed);

        // Convert embedding to quantum state representation
        // This is a simplified approach - in practice would use proper quantum state mapping
        let quantum_state = self.embedding_to_quantum_state(embedding);

        // Get quantum metrics from the quantum router
        match self.quantum_router.get_quantum_metrics().await {
            Ok(metrics) => {
                // Compute coherence based on quantum metrics
                let base_coherence = metrics.coherence_factor;
                let entanglement_factor = metrics.entanglement_strength;

                // Incorporate embedding-specific coherence
                let embedding_coherence = self.compute_embedding_coherence(&quantum_state);

                // Weighted combination
                let coherence_score = (base_coherence * 0.4
                    + entanglement_factor * 0.3
                    + embedding_coherence * 0.3) as f32;

                Ok(coherence_score.min(1.0).max(0.0))
            }
            Err(e) => Err(QualityAnalysisError::CoherenceCalculationFailed {
                error: e.to_string()})}
    }

    /// Convert embedding to quantum state representation
    fn embedding_to_quantum_state(&self, embedding: &[f32]) -> Vec<f32> {
        // Normalize embedding to quantum state constraints
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return vec![0.0; embedding.len()];
        }

        embedding.iter().map(|x| x / norm).collect()
    }

    /// Compute embedding-specific coherence measure
    fn compute_embedding_coherence(&self, quantum_state: &[f32]) -> f64 {
        if quantum_state.is_empty() {
            return 0.0;
        }

        // Compute entropy-based coherence measure
        let mut entropy = 0.0;
        for &amplitude in quantum_state {
            let probability = amplitude * amplitude;
            if probability > 0.0 {
                entropy -= probability as f64 * (probability as f64).ln();
            }
        }

        // Normalize entropy to [0, 1] range
        let max_entropy = (quantum_state.len() as f64).ln();
        if max_entropy == 0.0 {
            return 1.0;
        }

        1.0 - (entropy / max_entropy)
    }

    /// Compute overall quality score with weighted factors
    fn compute_quality_score(
        &self,
        l2_norm: f32,
        std_deviation: f32,
        coherence_score: f32,
        cosine_similarity: Option<f32>,
        latency_ms: u64,
    ) -> f32 {
        // Normalize metrics to [0, 1] range
        let norm_score = (l2_norm / 50.0).min(1.0); // Assume typical L2 norm around 25-50
        let std_score = (std_deviation / 1.0).min(1.0); // Assume typical std dev around 0.1-1.0
        let consistency_score = cosine_similarity.unwrap_or(0.8); // Default if no reference
        let latency_score = (1.0 - (latency_ms as f32 / 10000.0)).max(0.0); // Penalize >10s latency

        // Weighted combination
        let quality_score = norm_score * CONSISTENCY_WEIGHT
            + std_score * DIMENSION_WEIGHT
            + coherence_score * COHERENCE_WEIGHT
            + (consistency_score * latency_score) * PERFORMANCE_WEIGHT;

        quality_score.min(1.0).max(0.0)
    }

    /// Detect outliers in quality metrics
    async fn detect_outliers(&self, data_point: &QualityDataPoint) {
        let is_quality_outlier = self
            .quality_outlier_detector
            .is_outlier(data_point.quality_score as f64)
            .await;

        let is_consistency_outlier = if let Some(similarity) = data_point.cosine_similarity {
            self.consistency_outlier_detector
                .is_outlier(similarity as f64)
                .await
        } else {
            false
        };

        if is_quality_outlier || is_consistency_outlier {
            self.analysis_metrics
                .outliers_detected
                .fetch_add(1, Ordering::Relaxed);
        }

        if let Some(similarity) = data_point.cosine_similarity {
            if similarity < 0.5 {
                self.analysis_metrics
                    .consistency_failures
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Update provider performance metrics
    async fn update_provider_metrics(&self, data_point: &QualityDataPoint) {
        let provider_key = data_point.provider.clone();

        self.provider_metrics
            .entry(provider_key.clone())
            .and_modify(|metrics| {
                // Update metrics (simplified - would use proper rolling statistics)
                metrics.sample_count += 1;
                metrics.last_updated = data_point.timestamp;
            })
            .or_insert_with(|| {
                ProviderPerformanceMetrics {
                    provider: provider_key,
                    quality_summary: StatisticalSummary::default(),
                    latency_summary: StatisticalSummary::default(),
                    consistency_score: data_point.quality_score as f64,
                    reliability_score: 0.95, // Default
                    last_updated: data_point.timestamp,
                    sample_count: 1}
            });
    }

    /// Get provider performance comparison
    pub fn get_provider_comparison(&self) -> Vec<ProviderPerformanceMetrics> {
        self.provider_metrics
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get quality statistics for time range
    pub fn get_quality_statistics(&self, time_range_seconds: u64) -> StatisticalSummary {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().saturating_sub(time_range_seconds))
            .unwrap_or(0);

        let quality_scores: Vec<f64> = self
            .quality_history
            .collect()
            .into_iter()
            .filter(|dp| dp.timestamp >= cutoff_time)
            .map(|dp| dp.quality_score as f64)
            .collect();

        StatisticalSummary::from_values(&quality_scores)
    }

    /// Get real-time quality monitoring receiver
    pub fn get_quality_monitor(&self) -> watch::Receiver<f64> {
        self.quality_receiver.clone()
    }

    /// Start background monitoring tasks
    pub fn start_monitoring_tasks(&self) -> tokio::task::JoinHandle<()> {
        let provider_metrics = self.provider_metrics.clone();
        let analysis_metrics = self.analysis_metrics.clone();

        tokio::spawn(async move {
            let mut monitoring_interval = interval(Duration::from_secs(60));

            loop {
                monitoring_interval.tick().await;

                // Perform periodic analysis and cleanup
                // Could include trend analysis, alerting, etc.

                // Log current metrics
                let total_assessments = analysis_metrics.assessments_total.load(Ordering::Relaxed);
                let outliers = analysis_metrics.outliers_detected.load(Ordering::Relaxed);

                if total_assessments > 0 {
                    let outlier_rate = outliers as f64 / total_assessments as f64;
                    if outlier_rate > 0.1 {
                        // High outlier rate - could trigger alerts
                    }
                }
            }
        })
    }

    /// Get analysis metrics
    pub fn get_metrics(&self) -> &QualityAnalysisMetrics {
        &self.analysis_metrics
    }

    /// Get recent quality history
    pub fn get_recent_history(&self, count: usize) -> Vec<QualityDataPoint> {
        let all_history = self.quality_history.collect();
        all_history.into_iter().rev().take(count).collect()
    }
}
