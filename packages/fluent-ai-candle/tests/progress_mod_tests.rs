use fluent_ai_candle::progress::mod::*;
use fluent_ai_candle::progress::*;

use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_progress_reporting() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test basic progress reporting
        reporter.report_progress("Test progress", 0.5).expect("Failed to report progress");
        reporter.report_stage_completion("Test stage").expect("Failed to report completion");
        
        assert!(reporter.is_active());
        assert_eq!(reporter.update_count(), 2);
    }

    #[test]
    fn test_session_management() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test session management
        reporter.start_session("test_session", "Test Operation").expect("Failed to start session");
        assert_eq!(reporter.current_session_id(), Some("test_session".to_string()));
        
        reporter.end_session().expect("Failed to end session");
        assert_eq!(reporter.current_session_id(), None);
    }

    #[test]
    fn test_concurrent_reporting() {
        let reporter = Arc::new(create_reporter().expect("Failed to create reporter"));
        let mut handles = vec![];

        // Test concurrent access
        for i in 0..10 {
            let reporter_clone = Arc::clone(&reporter);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let progress = (j as f64) / 100.0;
                    let message = format!("Worker {} progress", i);
                    reporter_clone.report_progress(&message, progress).unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify metrics were updated
        let stats = reporter.metrics().get_stats();
        assert!(stats.total_operations > 0);
    }

    #[test]
    fn test_metrics_aggregation() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test generation metrics
        reporter.report_generation_metrics(42.5, 0.85, 125_000).expect("Failed to report metrics");
        
        let metrics = reporter.inference_metrics();
        assert!(metrics.total_operations > 0);
        assert!(metrics.avg_tokens_per_sec > 0.0);
    }

    #[test]
    fn test_configuration_variants() {
        // Test different configuration presets
        let low_latency = create_low_latency_reporter().expect("Failed to create low latency reporter");
        assert!(low_latency.config().is_performance_optimized());

        let high_throughput = create_high_throughput_reporter().expect("Failed to create high throughput reporter");
        assert_eq!(high_throughput.config().max_concurrent_sessions, 1000);

        let minimal = create_minimal_reporter().expect("Failed to create minimal reporter");
        assert!(minimal.config().is_resource_efficient());
    }

    #[test]
    fn test_error_handling() {
        let reporter = create_reporter().expect("Failed to create reporter");
        
        // Test error reporting
        reporter.report_error("Test error", "test_context").expect("Failed to report error");
        
        let stats = reporter.metrics().get_stats();
        assert!(stats.failed_operations > 0);
    }
