//! Work-stealing thread pool - NO Future usage!
//!
//! High-performance, crossbeam-based thread pool with work-stealing algorithm
//! Eliminates thread creation overhead while maintaining pure closure execution

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use crossbeam_channel::{Receiver, bounded};
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_utils::Backoff;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Work-stealing statistics for monitoring thread pool performance
#[derive(Debug, Clone)]
pub struct WorkStealingStats {
    /// Total number of jobs in the global injector queue
    pub total_pending: usize,
    /// Number of jobs in each worker's local queue
    pub worker_queue_lengths: Vec<usize>,
    /// Total number of worker threads
    pub worker_count: usize,
}

/// Work-stealing thread pool with fixed number of worker threads
struct WorkerPool {
    injector: Arc<Injector<Job>>,
    stealers: Vec<Stealer<Job>>,
    shutdown: Arc<AtomicBool>,
    workers_count: usize,
}

impl WorkerPool {
    /// Create new work-stealing thread pool
    fn new() -> Self {
        let worker_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let injector = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut stealers = Vec::with_capacity(worker_count);

        // Create worker threads
        for worker_id in 0..worker_count {
            let worker = Worker::new_fifo();
            let stealer = worker.stealer();
            stealers.push(stealer.clone());

            let injector_clone = Arc::clone(&injector);
            let shutdown_clone = Arc::clone(&shutdown);
            let stealers_clone = stealers.clone();

            thread::Builder::new()
                .name(format!("fluent-ai-worker-{}", worker_id))
                .spawn(move || {
                    run_worker(worker, injector_clone, stealers_clone, shutdown_clone);
                })
                .expect("Failed to spawn worker thread");
        }

        Self {
            injector,
            stealers,
            shutdown,
            workers_count: worker_count,
        }
    }

    /// Submit a job to the thread pool
    fn submit<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let boxed_job: Job = Box::new(job);
        self.injector.push(boxed_job);
    }

    /// Get number of worker threads
    fn worker_count(&self) -> usize {
        self.workers_count
    }

    /// Get work-stealing statistics for monitoring and debugging
    fn get_statistics(&self) -> WorkStealingStats {
        let total_pending = self.injector.len();
        let worker_queue_lengths: Vec<usize> =
            self.stealers.iter().map(|stealer| stealer.len()).collect();

        WorkStealingStats {
            total_pending,
            worker_queue_lengths,
            worker_count: self.workers_count,
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Signal shutdown to all workers
        self.shutdown.store(true, Ordering::Release);

        // Submit poison pills to wake up sleeping workers
        for _ in 0..self.workers_count {
            self.injector.push(Box::new(|| {}));
        }
    }
}

/// Worker thread main loop with work-stealing algorithm
fn run_worker(
    worker: Worker<Job>,
    injector: Arc<Injector<Job>>,
    stealers: Vec<Stealer<Job>>,
    shutdown: Arc<AtomicBool>,
) {
    let backoff = Backoff::new();

    loop {
        // Check for shutdown signal
        if shutdown.load(Ordering::Acquire) {
            break;
        }

        // Try to find work in this order:
        // 1. Local worker queue
        // 2. Global injector queue
        // 3. Steal from other workers
        let job = worker
            .pop()
            .or_else(|| injector.steal().success())
            .or_else(|| {
                // Try to steal from other workers
                stealers
                    .iter()
                    .map(|s| s.steal())
                    .find(|s| s.is_success())
                    .and_then(|s| s.success())
            });

        if let Some(job) = job {
            // Reset backoff when we find work
            backoff.reset();
            // Execute the job
            job();
        } else {
            // No work found, use backoff strategy
            backoff.snooze();
        }
    }
}

/// Global executor that maintains the same API but uses work-stealing pool
pub struct GlobalExecutor {
    pool: WorkerPool,
}

impl GlobalExecutor {
    pub fn new() -> Self {
        Self {
            pool: WorkerPool::new(),
        }
    }

    /// Execute closure on thread pool - NO Future usage!
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.pool.submit(f);
    }

    /// Execute closure and send result to channel
    pub fn execute_with_result<F, T>(&self, f: F) -> Receiver<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = bounded(1);
        self.pool.submit(move || {
            let result = f();
            let _ = tx.send(result);
        });
        rx
    }

    /// Get number of worker threads in the pool
    pub fn worker_count(&self) -> usize {
        self.pool.worker_count()
    }

    /// Get work-stealing statistics for monitoring and debugging
    pub fn get_statistics(&self) -> WorkStealingStats {
        self.pool.get_statistics()
    }
}

impl Default for GlobalExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// Static instance for global access - SAFE, no mutable static
static GLOBAL_EXECUTOR_INSTANCE: std::sync::OnceLock<GlobalExecutor> = std::sync::OnceLock::new();

pub fn global_executor() -> &'static GlobalExecutor {
    GLOBAL_EXECUTOR_INSTANCE.get_or_init(GlobalExecutor::new)
}
