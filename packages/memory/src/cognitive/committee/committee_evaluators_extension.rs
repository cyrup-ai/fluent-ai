//! Additional types for zero-allocation committee evaluators

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::committee_types::{CommitteeEvaluation, CommitteeResult, ModelType};
use super::relaxed_counter::RelaxedCounter;
use crate::cognitive::types::OptimizationSpec;

/// Lock-free evaluation task for task queue distribution
#[derive(Debug)]
pub struct EvaluationTask {
    /// Model type for this evaluation
    pub model_type: ModelType,
    /// Evaluation request payload
    pub request: EvaluationRequest,
    /// Result sender for async completion
    pub result_sender: tokio::sync::oneshot::Sender<CommitteeResult<CommitteeEvaluation>>,
}

/// Evaluation request with zero-allocation data
#[derive(Debug, Clone)]
pub struct EvaluationRequest {
    /// Optimization specification
    pub optimization_spec: OptimizationSpec,
    /// Current code reference (no allocation)
    pub current_code: Arc<str>,
    /// Proposed code reference (no allocation)
    pub proposed_code: Arc<str>,
    /// Evaluation timeout
    pub timeout: Duration,
}

/// Lock-free evaluator pool metrics
#[derive(Debug)]
pub struct EvaluatorPoolMetrics {
    /// Total evaluators added to pool
    pub evaluators_added: RelaxedCounter,
    /// Total evaluator access operations
    pub evaluators_accessed: RelaxedCounter,
    /// Task queue operations
    pub tasks_queued: RelaxedCounter,
    /// Task completion count
    pub tasks_completed: RelaxedCounter,
    /// Pool creation timestamp
    pub created_at: Instant,
}

impl EvaluatorPoolMetrics {
    /// Create new pool metrics
    #[inline]
    pub fn new() -> Self {
        Self {
            evaluators_added: RelaxedCounter::new(0),
            evaluators_accessed: RelaxedCounter::new(0),
            tasks_queued: RelaxedCounter::new(0),
            tasks_completed: RelaxedCounter::new(0),
            created_at: Instant::now(),
        }
    }

    /// Get metrics snapshot
    #[inline]
    pub fn snapshot(&self) -> EvaluatorPoolSnapshot {
        EvaluatorPoolSnapshot {
            evaluators_added: self.evaluators_added.get() as usize,
            evaluators_accessed: self.evaluators_accessed.get() as usize,
            tasks_queued: self.tasks_queued.get() as usize,
            tasks_completed: self.tasks_completed.get() as usize,
            uptime_seconds: self.created_at.elapsed().as_secs(),
        }
    }
}

/// Immutable evaluator pool metrics snapshot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvaluatorPoolSnapshot {
    pub evaluators_added: usize,
    pub evaluators_accessed: usize,
    pub tasks_queued: usize,
    pub tasks_completed: usize,
    pub uptime_seconds: u64,
}

/// Lock-free evaluation session metrics
#[derive(Debug)]
pub struct EvaluationSessionMetrics {
    /// Total evaluations completed in this session
    pub evaluations_completed: RelaxedCounter,
    /// Total evaluation failures
    pub evaluation_failures: RelaxedCounter,
    /// Session creation timestamp
    pub created_at: Instant,
}

impl EvaluationSessionMetrics {
    /// Create new session metrics
    #[inline]
    pub fn new() -> Self {
        Self {
            evaluations_completed: RelaxedCounter::new(0),
            evaluation_failures: RelaxedCounter::new(0),
            created_at: Instant::now(),
        }
    }

    /// Record evaluation failure
    #[inline]
    pub fn record_failure(&self) {
        self.evaluation_failures.inc();
    }

    /// Get success rate
    #[inline]
    pub fn success_rate(&self) -> f64 {
        let completed = self.evaluations_completed.get() as f64;
        let failed = self.evaluation_failures.get() as f64;
        let total = completed + failed;
        if total > 0.0 { completed / total } else { 1.0 }
    }

    /// Get metrics snapshot
    #[inline]
    pub fn snapshot(&self) -> EvaluationSessionSnapshot {
        EvaluationSessionSnapshot {
            evaluations_completed: self.evaluations_completed.get() as usize,
            evaluation_failures: self.evaluation_failures.get() as usize,
            success_rate: self.success_rate(),
            session_duration_seconds: self.created_at.elapsed().as_secs(),
        }
    }
}

/// Immutable evaluation session metrics snapshot
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvaluationSessionSnapshot {
    pub evaluations_completed: usize,
    pub evaluation_failures: usize,
    pub success_rate: f64,
    pub session_duration_seconds: u64,
}
