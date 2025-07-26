use fluent_ai_candle::model::loading::progress::*;
use fluent_ai_candle::model::loading::*;

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
