//! Progress tracking for model loading operations
//!
//! Provides progress tracking types specifically for model loading with detailed stage reporting.

use std::sync::Arc;

use crate::error::CandleResult;

/// Stages of model loading process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadingStage {
    /// Initializing model loading infrastructure
    Initializing,
    /// Loading model weights from storage
    LoadingWeights,
    /// Processing and optimizing loaded weights
    Processing,
    /// Validating loaded model structure
    Validating,
    /// Model loading completed successfully
    Completed}

impl LoadingStage {
    /// Get human-readable description of the loading stage
    pub fn description(&self) -> &'static str {
        match self {
            Self::Initializing => "Initializing model loading",
            Self::LoadingWeights => "Loading model weights",
            Self::Processing => "Processing weights",
            Self::Validating => "Validating model",
            Self::Completed => "Loading completed"}
    }

    /// Get estimated progress percentage for this stage
    pub fn progress_estimate(&self) -> f32 {
        match self {
            Self::Initializing => 0.05,
            Self::LoadingWeights => 0.70,
            Self::Processing => 0.20,
            Self::Validating => 0.04,
            Self::Completed => 0.01}
    }
}

/// Callback function type for progress updates
pub type ProgressCallback = Arc<dyn Fn(LoadingStage, f32) + Send + Sync>;

/// Progress tracker for model loading operations
#[derive(Clone)]
pub struct ProgressTracker {
    /// Current loading stage
    current_stage: LoadingStage,
    /// Progress callback function
    callback: Option<ProgressCallback>}

/// Custom Debug implementation with zero-allocation formatting
impl std::fmt::Debug for ProgressTracker {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgressTracker")
            .field("current_stage", &self.current_stage)
            .field("callback", &self.callback.as_ref().map(|_| "<closure>"))
            .finish()
    }
}

impl ProgressTracker {
    /// Create a new progress tracker without callback
    pub fn new() -> Self {
        Self {
            current_stage: LoadingStage::Initializing,
            callback: None}
    }

    /// Create a new progress tracker with callback
    pub fn with_callback(callback: ProgressCallback) -> Self {
        Self {
            current_stage: LoadingStage::Initializing,
            callback: Some(callback)}
    }

    /// Set the current loading stage
    pub fn set_stage(&mut self, stage: LoadingStage) -> CandleResult<()> {
        self.current_stage = stage;
        if let Some(ref callback) = self.callback {
            callback(stage, stage.progress_estimate());
        }
        Ok(())
    }

    /// Get the current loading stage
    pub fn current_stage(&self) -> LoadingStage {
        self.current_stage
    }

    /// Create a sub-progress tracker for a specific stage
    pub fn subtracker(&self, stage: LoadingStage) -> SubProgressTracker {
        SubProgressTracker::new(stage, self.callback.clone())
    }

    /// Report progress for the current stage
    pub fn report_progress(&self, progress: f32) -> CandleResult<()> {
        if let Some(ref callback) = self.callback {
            callback(self.current_stage, progress);
        }
        Ok(())
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Sub-progress tracker for tracking progress within a specific loading stage
#[derive(Clone)]
pub struct SubProgressTracker {
    /// The loading stage this sub-tracker is for
    stage: LoadingStage,
    /// Progress callback function
    callback: Option<ProgressCallback>}

/// Custom Debug implementation with zero-allocation formatting
impl std::fmt::Debug for SubProgressTracker {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubProgressTracker")
            .field("stage", &self.stage)
            .field("callback", &self.callback.as_ref().map(|_| "<closure>"))
            .finish()
    }
}

impl SubProgressTracker {
    /// Create a new sub-progress tracker
    pub fn new(stage: LoadingStage, callback: Option<ProgressCallback>) -> Self {
        Self { stage, callback }
    }

    /// Report sub-progress (0.0 to 1.0) within this stage
    pub fn report_progress(&self, progress: f32) -> CandleResult<()> {
        if let Some(ref callback) = self.callback {
            // Scale progress within the stage's overall contribution
            let scaled_progress = progress * self.stage.progress_estimate();
            callback(self.stage, scaled_progress);
        }
        Ok(())
    }

    /// Get the stage this sub-tracker is for
    pub fn stage(&self) -> LoadingStage {
        self.stage
    }
}
