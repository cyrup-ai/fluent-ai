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
    Completed,
}

impl LoadingStage {
    /// Get human-readable description of the loading stage
    pub fn description(&self) -> &'static str {
        match self {
            Self::Initializing => "Initializing model loading",
            Self::LoadingWeights => "Loading model weights",
            Self::Processing => "Processing weights",
            Self::Validating => "Validating model",
            Self::Completed => "Loading completed",
        }
    }

    /// Get estimated progress percentage for this stage
    pub fn progress_estimate(&self) -> f32 {
        match self {
            Self::Initializing => 0.05,
            Self::LoadingWeights => 0.70,
            Self::Processing => 0.20,
            Self::Validating => 0.04,
            Self::Completed => 0.01,
        }
    }
}

/// Callback function type for progress updates
pub type ProgressCallback = Arc<dyn Fn(LoadingStage, f32) + Send + Sync>;

/// Progress tracker for model loading operations
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    /// Current loading stage
    current_stage: LoadingStage,
    /// Progress callback function
    callback: Option<ProgressCallback>,
}

impl ProgressTracker {
    /// Create a new progress tracker without callback
    pub fn new() -> Self {
        Self {
            current_stage: LoadingStage::Initializing,
            callback: None,
        }
    }

    /// Create a new progress tracker with callback
    pub fn with_callback(callback: ProgressCallback) -> Self {
        Self {
            current_stage: LoadingStage::Initializing,
            callback: Some(callback),
        }
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
#[derive(Debug, Clone)]
pub struct SubProgressTracker {
    /// The loading stage this sub-tracker is for
    stage: LoadingStage,
    /// Progress callback function
    callback: Option<ProgressCallback>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_loading_stage_descriptions() {
        assert_eq!(LoadingStage::Initializing.description(), "Initializing model loading");
        assert_eq!(LoadingStage::LoadingWeights.description(), "Loading model weights");
        assert_eq!(LoadingStage::Processing.description(), "Processing weights");
        assert_eq!(LoadingStage::Validating.description(), "Validating model");
        assert_eq!(LoadingStage::Completed.description(), "Loading completed");
    }

    #[test]
    fn test_progress_estimates_sum_to_one() {
        let total = LoadingStage::Initializing.progress_estimate()
            + LoadingStage::LoadingWeights.progress_estimate()
            + LoadingStage::Processing.progress_estimate()
            + LoadingStage::Validating.progress_estimate()
            + LoadingStage::Completed.progress_estimate();
        
        // Allow for small floating point errors
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_progress_tracker_without_callback() {
        let mut tracker = ProgressTracker::new();
        assert_eq!(tracker.current_stage(), LoadingStage::Initializing);
        
        tracker.set_stage(LoadingStage::LoadingWeights).unwrap();
        assert_eq!(tracker.current_stage(), LoadingStage::LoadingWeights);
    }

    #[test]
    fn test_progress_tracker_with_callback() {
        let received_stages: Arc<Mutex<Vec<LoadingStage>>> = Arc::new(Mutex::new(Vec::new()));
        let received_stages_clone = received_stages.clone();
        
        let callback: ProgressCallback = Arc::new(move |stage, _progress| {
            received_stages_clone.lock().unwrap().push(stage);
        });
        
        let mut tracker = ProgressTracker::with_callback(callback);
        tracker.set_stage(LoadingStage::LoadingWeights).unwrap();
        tracker.set_stage(LoadingStage::Processing).unwrap();
        
        let stages = received_stages.lock().unwrap();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0], LoadingStage::LoadingWeights);
        assert_eq!(stages[1], LoadingStage::Processing);
    }

    #[test]
    fn test_sub_progress_tracker() {
        let received_progress: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let received_progress_clone = received_progress.clone();
        
        let callback: ProgressCallback = Arc::new(move |_stage, progress| {
            received_progress_clone.lock().unwrap().push(progress);
        });
        
        let sub_tracker = SubProgressTracker::new(LoadingStage::LoadingWeights, Some(callback));
        assert_eq!(sub_tracker.stage(), LoadingStage::LoadingWeights);
        
        sub_tracker.report_progress(0.5).unwrap();
        
        let progress_values = received_progress.lock().unwrap();
        assert_eq!(progress_values.len(), 1);
        // Should be 0.5 * 0.70 (LoadingWeights stage estimate)
        assert!((progress_values[0] - 0.35).abs() < 0.001);
    }
}