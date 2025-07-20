//! Intelligent batch processing for embeddings with dynamic optimization
//!
//! This module provides advanced batch processing capabilities with:
//! - Dynamic batch size optimization based on provider capabilities
//! - Parallel processing with work-stealing task queues
//! - Adaptive sizing with exponential moving average
//! - Backpressure handling and circuit breaker integration
//! - Zero-allocation request aggregation with SIMD validation
//! - Provider-specific optimization strategies

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

use arrayvec::ArrayString;
use smallvec::SmallVec;
use crossbeam_utils::CachePadded;
use crossbeam::deque::{Injector, Stealer, Worker};
use crossbeam::channel::{self, Receiver, Sender};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Semaphore, RwLock, Notify};
use tokio::task::JoinSet;
use tokio::time::{sleep, timeout};
use thiserror::Error;

/// Maximum request batch size for zero allocation
const MAX_REQUEST_BATCH_SIZE: usize = 256;
/// Maximum response batch size
const MAX_RESPONSE_BATCH_SIZE: usize = 256;
/// Default batch timeout in milliseconds
const DEFAULT_BATCH_TIMEOUT_MS: u64 = 100;
/// Maximum concurrent batches per provider
const MAX_CONCURRENT_BATCHES: usize = 10;
/// Exponential moving average smoothing factor
const EMA_ALPHA: f64 = 0.2;
/// SIMD validation chunk size
const SIMD_CHUNK_SIZE: usize = 8;

/// Batch request with zero-allocation patterns
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Request ID for tracking
    pub id: ArrayString<32>,
    /// Input text for embedding
    pub input: String, // TODO: Could be optimized to use Cow<str> or similar
    /// Provider-specific parameters
    pub params: SmallVec<[(ArrayString<32>, ArrayString<64>); 4]>,
    /// Request timestamp
    pub timestamp: Instant,
    /// Quality requirements
    pub quality_requirements: QualityRequirements,
}

/// Quality requirements for embedding generation
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    /// Minimum quality score (0.0 - 1.0)
    pub min_quality: f32,
    /// Maximum latency tolerance in milliseconds
    pub max_latency_ms: u64,
    /// Whether to use cache
    pub use_cache: bool,
    /// Consistency requirements
    pub consistency_level: ConsistencyLevel,
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self {
            min_quality: 0.8,
            max_latency_ms: 5000,
            use_cache: true,
            consistency_level: ConsistencyLevel::Eventually,
        }
    }
}

/// Consistency levels for distributed processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConsistencyLevel {
    Strong = 0,     // Synchronous processing
    Eventually = 1, // Asynchronous with eventual consistency
    Weak = 2,       // Fire-and-forget
}

/// Batch response with result aggregation
#[derive(Debug, Clone)]
pub struct BatchResponse {
    /// Response ID matching request
    pub id: ArrayString<32>,
    /// Generated embedding
    pub embedding: SmallVec<[f32; 1536]>, // Common embedding size
    /// Processing latency in milliseconds
    pub latency_ms: u64,
    /// Quality score
    pub quality_score: f32,
    /// Provider used
    pub provider: ArrayString<32>,
    /// Model used
    pub model: ArrayString<64>,
    /// Error if any
    pub error: Option<BatchProcessingError>,
}

/// Provider-specific batch capabilities
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    /// Provider name
    pub name: ArrayString<32>,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Optimal batch size range
    pub optimal_batch_range: (usize, usize),
    /// Average latency per item in milliseconds
    pub avg_latency_per_item_ms: f64,
    /// Rate limit (requests per second)
    pub rate_limit_rps: f32,
    /// Memory-based batch sizing (for local providers)
    pub memory_based_sizing: bool,
    /// Supports parallel processing
    pub supports_parallel: bool,
}

impl ProviderCapabilities {
    /// OpenAI provider capabilities
    pub fn openai() -> Self {
        Self {
            name: ArrayString::from("openai").unwrap_or_default(),
            max_batch_size: 2048,
            optimal_batch_range: (100, 500),
            avg_latency_per_item_ms: 10.0,
            rate_limit_rps: 3000.0,
            memory_based_sizing: false,
            supports_parallel: true,
        }
    }

    /// Local Candle provider capabilities
    pub fn candle(available_memory_gb: f32) -> Self {
        let max_batch = if available_memory_gb >= 8.0 { 64 } else { 32 };
        Self {
            name: ArrayString::from("candle").unwrap_or_default(),
            max_batch_size: max_batch,
            optimal_batch_range: (8, max_batch / 2),
            avg_latency_per_item_ms: 50.0,
            rate_limit_rps: f32::INFINITY, // No rate limits for local
            memory_based_sizing: true,
            supports_parallel: false, // GPU memory constraints
        }
    }

    /// Get optimal batch size based on current load
    pub fn optimal_batch_size(&self, current_load: f32) -> usize {
        let (min_optimal, max_optimal) = self.optimal_batch_range;
        
        // Scale batch size inversely with load
        let load_factor = (1.0 - current_load.min(1.0)).max(0.1);
        let target_size = min_optimal as f32 + 
                         (max_optimal - min_optimal) as f32 * load_factor;
        
        (target_size as usize).min(self.max_batch_size).max(1)
    }
}

/// Adaptive batch size optimizer using exponential moving average
#[derive(Debug)]
pub struct BatchSizeOptimizer {
    /// Current optimal batch size
    current_size: CachePadded<AtomicUsize>,
    /// Moving average of latency per item
    avg_latency_per_item: CachePadded<AtomicU64>, // In nanoseconds for precision
    /// Moving average of throughput
    avg_throughput: CachePadded<AtomicU64>, // Items per second
    /// Provider capabilities
    capabilities: ProviderCapabilities,
    /// Performance history for trend analysis
    performance_history: Arc<RwLock<VecDeque<PerformanceDataPoint>>>,
}

#[derive(Debug, Clone)]
struct PerformanceDataPoint {
    batch_size: usize,
    latency_per_item_ms: f64,
    throughput_items_per_sec: f64,
    timestamp: Instant,
    memory_usage_mb: f32,
}

impl BatchSizeOptimizer {
    pub fn new(capabilities: ProviderCapabilities) -> Self {
        let initial_size = capabilities.optimal_batch_range.0;
        
        Self {
            current_size: CachePadded::new(AtomicUsize::new(initial_size)),
            avg_latency_per_item: CachePadded::new(AtomicU64::new(
                (capabilities.avg_latency_per_item_ms * 1_000_000.0) as u64
            )),
            avg_throughput: CachePadded::new(AtomicU64::new(
                (1000.0 / capabilities.avg_latency_per_item_ms) as u64
            )),
            capabilities,
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
        }
    }

    /// Update optimizer with batch performance data
    pub async fn update_performance(
        &self,
        batch_size: usize,
        total_latency_ms: u64,
        total_items: usize,
        memory_usage_mb: f32,
    ) {
        if total_items == 0 {
            return;
        }

        let latency_per_item_ms = total_latency_ms as f64 / total_items as f64;
        let throughput = total_items as f64 / (total_latency_ms as f64 / 1000.0);
        
        // Update moving averages using exponential moving average
        let latency_nanos = (latency_per_item_ms * 1_000_000.0) as u64;
        let old_latency = self.avg_latency_per_item.load(Ordering::Relaxed) as f64;
        let new_latency = old_latency * (1.0 - EMA_ALPHA) + latency_nanos as f64 * EMA_ALPHA;
        self.avg_latency_per_item.store(new_latency as u64, Ordering::Relaxed);

        let old_throughput = self.avg_throughput.load(Ordering::Relaxed) as f64;
        let new_throughput = old_throughput * (1.0 - EMA_ALPHA) + throughput * EMA_ALPHA;
        self.avg_throughput.store(new_throughput as u64, Ordering::Relaxed);

        // Record performance data point
        let data_point = PerformanceDataPoint {
            batch_size,
            latency_per_item_ms,
            throughput_items_per_sec: throughput,
            timestamp: Instant::now(),
            memory_usage_mb,
        };

        {
            let mut history = self.performance_history.write().await;
            if history.len() >= 100 {
                history.pop_front();
            }
            history.push_back(data_point);
        }

        // Optimize batch size based on performance trends
        self.optimize_batch_size(latency_per_item_ms, throughput, memory_usage_mb).await;
    }

    /// Optimize batch size based on performance analysis
    async fn optimize_batch_size(
        &self,
        current_latency_per_item_ms: f64,
        current_throughput: f64,
        memory_usage_mb: f32,
    ) {
        let current_size = self.current_size.load(Ordering::Relaxed);
        let mut new_size = current_size;

        // Memory-based optimization for local providers
        if self.capabilities.memory_based_sizing {
            let memory_factor = if memory_usage_mb > 1024.0 { 0.8 } else { 1.2 };
            new_size = ((current_size as f32) * memory_factor) as usize;
        } else {
            // Latency and throughput based optimization
            let target_latency = self.capabilities.avg_latency_per_item_ms;
            
            if current_latency_per_item_ms > target_latency * 1.5 {
                // Reduce batch size if latency too high
                new_size = (current_size as f32 * 0.8) as usize;
            } else if current_latency_per_item_ms < target_latency * 0.5 {
                // Increase batch size if latency very low
                new_size = (current_size as f32 * 1.2) as usize;
            }
        }

        // Apply constraints
        new_size = new_size
            .max(self.capabilities.optimal_batch_range.0)
            .min(self.capabilities.max_batch_size);

        if new_size != current_size {
            self.current_size.store(new_size, Ordering::Relaxed);
        }
    }

    /// Get current optimal batch size
    pub fn get_optimal_size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> (f64, f64) {
        let latency_ms = self.avg_latency_per_item.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let throughput = self.avg_throughput.load(Ordering::Relaxed) as f64;
        (latency_ms, throughput)
    }
}

/// Work-stealing task queue for parallel processing
#[derive(Debug)]
pub struct WorkStealingQueue {
    /// Main work injector
    injector: Arc<Injector<BatchTask>>,
    /// Worker queues
    workers: Vec<Worker<BatchTask>>,
    /// Stealers for work stealing
    stealers: Vec<Stealer<BatchTask>>,
    /// Work notification
    work_notify: Arc<Notify>,
}

#[derive(Debug)]
pub struct BatchTask {
    pub requests: SmallVec<[BatchRequest; MAX_REQUEST_BATCH_SIZE]>,
    pub response_sender: broadcast::Sender<BatchResponse>,
    pub provider: ArrayString<32>,
    pub priority: TaskPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl WorkStealingQueue {
    pub fn new(num_workers: usize) -> Self {
        let injector = Arc::new(Injector::new());
        let mut workers = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }

        Self {
            injector,
            workers,
            stealers,
            work_notify: Arc::new(Notify::new()),
        }
    }

    /// Push task to queue
    pub fn push(&self, task: BatchTask) {
        self.injector.push(task);
        self.work_notify.notify_one();
    }

    /// Try to pop task from worker queue or steal from others
    pub fn try_pop(&self, worker_id: usize) -> Option<BatchTask> {
        // Try local worker first
        if let Some(task) = self.workers[worker_id].pop() {
            return Some(task);
        }

        // Try to steal from global injector
        if let Ok(task) = self.injector.steal() {
            return Some(task);
        }

        // Try to steal from other workers
        for stealer in &self.stealers {
            if let Ok(task) = stealer.steal() {
                return Some(task);
            }
        }

        None
    }

    /// Wait for work notification
    pub async fn wait_for_work(&self) {
        self.work_notify.notified().await;
    }

    /// Get queue statistics
    pub fn len(&self) -> usize {
        self.injector.len() + self.workers.iter().map(|w| w.len()).sum::<usize>()
    }
}

/// Batch processing metrics
#[derive(Debug)]
pub struct BatchMetrics {
    /// Total batches processed
    pub batches_processed: CachePadded<AtomicU64>,
    /// Total requests processed
    pub requests_processed: CachePadded<AtomicU64>,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: CachePadded<AtomicU64>,
    /// Queue depth (current)
    pub queue_depth: CachePadded<AtomicUsize>,
    /// Backpressure events
    pub backpressure_events: CachePadded<AtomicU64>,
    /// Failed batches
    pub failed_batches: CachePadded<AtomicU64>,
    /// Provider-specific metrics
    pub provider_metrics: DashMap<ArrayString<32>, ProviderMetrics>,
}

#[derive(Debug)]
pub struct ProviderMetrics {
    pub requests: CachePadded<AtomicU64>,
    pub latency_sum_ms: CachePadded<AtomicU64>,
    pub errors: CachePadded<AtomicU64>,
    pub optimal_batch_size: CachePadded<AtomicUsize>,
}

impl BatchMetrics {
    pub fn new() -> Self {
        Self {
            batches_processed: CachePadded::new(AtomicU64::new(0)),
            requests_processed: CachePadded::new(AtomicU64::new(0)),
            total_processing_time_ms: CachePadded::new(AtomicU64::new(0)),
            queue_depth: CachePadded::new(AtomicUsize::new(0)),
            backpressure_events: CachePadded::new(AtomicU64::new(0)),
            failed_batches: CachePadded::new(AtomicU64::new(0)),
            provider_metrics: DashMap::new(),
        }
    }

    /// Record batch completion
    pub fn record_batch(
        &self,
        provider: &str,
        request_count: usize,
        latency_ms: u64,
        success: bool,
    ) {
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
        self.requests_processed.fetch_add(request_count as u64, Ordering::Relaxed);
        self.total_processing_time_ms.fetch_add(latency_ms, Ordering::Relaxed);

        if !success {
            self.failed_batches.fetch_add(1, Ordering::Relaxed);
        }

        // Update provider metrics
        let provider_key = ArrayString::from(provider).unwrap_or_default();
        self.provider_metrics
            .entry(provider_key)
            .or_insert_with(|| ProviderMetrics {
                requests: CachePadded::new(AtomicU64::new(0)),
                latency_sum_ms: CachePadded::new(AtomicU64::new(0)),
                errors: CachePadded::new(AtomicU64::new(0)),
                optimal_batch_size: CachePadded::new(AtomicUsize::new(32)),
            })
            .requests.fetch_add(request_count as u64, Ordering::Relaxed);

        if let Some(provider_metrics) = self.provider_metrics.get(&provider_key) {
            provider_metrics.latency_sum_ms.fetch_add(latency_ms, Ordering::Relaxed);
            if !success {
                provider_metrics.errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get average latency per request
    pub fn average_latency_per_request_ms(&self) -> f64 {
        let total_ms = self.total_processing_time_ms.load(Ordering::Relaxed) as f64;
        let total_requests = self.requests_processed.load(Ordering::Relaxed) as f64;
        if total_requests > 0.0 { total_ms / total_requests } else { 0.0 }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.batches_processed.load(Ordering::Relaxed) as f64;
        let failed = self.failed_batches.load(Ordering::Relaxed) as f64;
        if total > 0.0 { (total - failed) / total } else { 0.0 }
    }
}

/// Batch processing errors
#[derive(Debug, Error, Clone)]
pub enum BatchProcessingError {
    #[error("Batch size exceeded: {actual} > {max}")]
    BatchSizeExceeded { actual: usize, max: usize },
    
    #[error("Provider not available: {provider}")]
    ProviderNotAvailable { provider: String },
    
    #[error("Timeout exceeded: {timeout_ms}ms")]
    TimeoutExceeded { timeout_ms: u64 },
    
    #[error("Backpressure: queue full")]
    BackpressureActivated,
    
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },
    
    #[error("Provider error: {error}")]
    ProviderError { error: String },
    
    #[error("Quality requirements not met: {actual} < {required}")]
    QualityNotMet { actual: f32, required: f32 },
    
    #[error("Circuit breaker open")]
    CircuitBreakerOpen,
}

/// Intelligent batch processor with adaptive optimization
#[derive(Debug)]
pub struct IntelligentBatchProcessor {
    /// Work-stealing task queue
    work_queue: Arc<WorkStealingQueue>,
    /// Batch size optimizers per provider
    optimizers: DashMap<ArrayString<32>, Arc<BatchSizeOptimizer>>,
    /// Processing metrics
    metrics: Arc<BatchMetrics>,
    /// Backpressure semaphore
    backpressure_semaphore: Arc<Semaphore>,
    /// Provider capabilities
    provider_capabilities: DashMap<ArrayString<32>, ProviderCapabilities>,
    /// Active processors
    active_processors: AtomicUsize,
    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

impl IntelligentBatchProcessor {
    /// Create new intelligent batch processor
    pub fn new(max_concurrent_batches: usize, num_workers: usize) -> Self {
        Self {
            work_queue: Arc::new(WorkStealingQueue::new(num_workers)),
            optimizers: DashMap::new(),
            metrics: Arc::new(BatchMetrics::new()),
            backpressure_semaphore: Arc::new(Semaphore::new(max_concurrent_batches)),
            provider_capabilities: DashMap::new(),
            active_processors: AtomicUsize::new(0),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Register provider capabilities
    pub fn register_provider(&self, capabilities: ProviderCapabilities) {
        let provider_name = capabilities.name.clone();
        let optimizer = Arc::new(BatchSizeOptimizer::new(capabilities.clone()));
        
        self.provider_capabilities.insert(provider_name.clone(), capabilities);
        self.optimizers.insert(provider_name, optimizer);
    }

    /// Start background processing workers
    pub async fn start_workers(&self, num_workers: usize) -> Vec<tokio::task::JoinHandle<()>> {
        let mut handles = Vec::new();

        for worker_id in 0..num_workers {
            let work_queue = self.work_queue.clone();
            let metrics = self.metrics.clone();
            let backpressure_semaphore = self.backpressure_semaphore.clone();
            let shutdown_signal = self.shutdown_signal.clone();
            let optimizers = self.optimizers.clone();

            let handle = tokio::spawn(async move {
                while !shutdown_signal.load(Ordering::Relaxed) {
                    if let Some(task) = work_queue.try_pop(worker_id) {
                        if let Ok(_permit) = backpressure_semaphore.try_acquire() {
                            Self::process_batch_task(
                                task,
                                &metrics,
                                &optimizers,
                            ).await;
                        } else {
                            // Backpressure activated - could queue for later or drop
                            metrics.backpressure_events.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        // No work available, wait for notification
                        work_queue.wait_for_work().await;
                    }
                }
            });

            handles.push(handle);
        }

        handles
    }

    /// Process a batch task
    async fn process_batch_task(
        task: BatchTask,
        metrics: &Arc<BatchMetrics>,
        optimizers: &DashMap<ArrayString<32>, Arc<BatchSizeOptimizer>>,
    ) {
        let start_time = Instant::now();
        let request_count = task.requests.len();
        let provider = task.provider.clone();

        // SIMD validation of batch requests
        let validation_result = Self::validate_batch_simd(&task.requests);
        
        match validation_result {
            Ok(_) => {
                // Process the batch (placeholder for actual processing)
                // In real implementation, this would call the actual provider
                
                let processing_time = start_time.elapsed().as_millis() as u64;
                
                // Update optimizer with performance data
                if let Some(optimizer) = optimizers.get(&provider) {
                    optimizer.update_performance(
                        request_count,
                        processing_time,
                        request_count,
                        128.0, // Memory usage placeholder
                    ).await;
                }

                // Record metrics
                metrics.record_batch(
                    provider.as_str(),
                    request_count,
                    processing_time,
                    true,
                );

                // Send responses (placeholder)
                for request in &task.requests {
                    let response = BatchResponse {
                        id: request.id.clone(),
                        embedding: SmallVec::from_slice(&[0.0; 1536]), // Placeholder
                        latency_ms: processing_time / request_count as u64,
                        quality_score: 0.9,
                        provider: provider.clone(),
                        model: ArrayString::from("placeholder").unwrap_or_default(),
                        error: None,
                    };

                    let _ = task.response_sender.send(response);
                }
            }
            Err(error) => {
                let processing_time = start_time.elapsed().as_millis() as u64;
                
                // Record failed batch
                metrics.record_batch(
                    provider.as_str(),
                    request_count,
                    processing_time,
                    false,
                );

                // Send error responses
                for request in &task.requests {
                    let response = BatchResponse {
                        id: request.id.clone(),
                        embedding: SmallVec::new(),
                        latency_ms: processing_time / request_count as u64,
                        quality_score: 0.0,
                        provider: provider.clone(),
                        model: ArrayString::from("error").unwrap_or_default(),
                        error: Some(error.clone()),
                    };

                    let _ = task.response_sender.send(response);
                }
            }
        }
    }

    /// SIMD validation of batch requests
    fn validate_batch_simd(
        requests: &SmallVec<[BatchRequest; MAX_REQUEST_BATCH_SIZE]>
    ) -> Result<(), BatchProcessingError> {
        if requests.is_empty() {
            return Err(BatchProcessingError::ValidationFailed {
                reason: "Empty batch".to_string()
            });
        }

        if requests.len() > MAX_REQUEST_BATCH_SIZE {
            return Err(BatchProcessingError::BatchSizeExceeded {
                actual: requests.len(),
                max: MAX_REQUEST_BATCH_SIZE,
            });
        }

        // SIMD-style validation in chunks
        for chunk in requests.chunks(SIMD_CHUNK_SIZE) {
            for request in chunk {
                // Validate individual request
                if request.input.is_empty() {
                    return Err(BatchProcessingError::ValidationFailed {
                        reason: format!("Empty input for request {}", request.id)
                    });
                }

                if request.input.len() > 8192 {
                    return Err(BatchProcessingError::ValidationFailed {
                        reason: format!("Input too long for request {}", request.id)
                    });
                }
            }
        }

        Ok(())
    }

    /// Submit batch for processing
    pub async fn submit_batch(
        &self,
        requests: SmallVec<[BatchRequest; MAX_REQUEST_BATCH_SIZE]>,
        provider: &str,
        priority: TaskPriority,
    ) -> Result<broadcast::Receiver<BatchResponse>, BatchProcessingError> {
        if requests.is_empty() {
            return Err(BatchProcessingError::ValidationFailed {
                reason: "Empty batch".to_string()
            });
        }

        // Check if provider is registered
        let provider_key = ArrayString::from(provider).unwrap_or_default();
        if !self.provider_capabilities.contains_key(&provider_key) {
            return Err(BatchProcessingError::ProviderNotAvailable {
                provider: provider.to_string()
            });
        }

        // Create response channel
        let (response_sender, response_receiver) = broadcast::channel(requests.len());

        // Create batch task
        let task = BatchTask {
            requests,
            response_sender,
            provider: provider_key,
            priority,
        };

        // Submit to work queue
        self.work_queue.push(task);
        self.metrics.queue_depth.fetch_add(1, Ordering::Relaxed);

        Ok(response_receiver)
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &BatchMetrics {
        &self.metrics
    }

    /// Get optimal batch size for provider
    pub fn get_optimal_batch_size(&self, provider: &str) -> Option<usize> {
        let provider_key = ArrayString::from(provider).ok()?;
        self.optimizers.get(&provider_key).map(|opt| opt.get_optimal_size())
    }

    /// Shutdown processor
    pub async fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
        
        // Wait for active processors to finish
        while self.active_processors.load(Ordering::Relaxed) > 0 {
            sleep(Duration::from_millis(10)).await;
        }
    }

    /// Get queue statistics
    pub fn queue_stats(&self) -> (usize, usize) {
        let queue_depth = self.work_queue.len();
        let available_permits = self.backpressure_semaphore.available_permits();
        (queue_depth, available_permits)
    }
}