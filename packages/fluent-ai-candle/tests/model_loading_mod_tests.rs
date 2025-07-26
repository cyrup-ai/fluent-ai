use fluent_ai_candle::model::loading::mod::*;
use fluent_ai_candle::model::loading::*;

use tempfile::tempdir;

    

    #[test]
    fn test_model_loader_default() {
        let loader = ModelLoader::default();
        assert_eq!(loader.config.dtype, DType::F32);
        assert!(loader.config.use_mmap);
        assert!(!loader.config.keep_original);
        assert!(loader.config.validate);
    }

    #[test]
    fn test_model_loader_with_callback() {
        let callback_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_called_clone = callback_called.clone();

        let callback = move |_progress: f64, _message: &str| {
            callback_called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        };

        let loader = ModelLoader::default().with_progress_callback(callback);

        // Trigger a progress update
        loader.progress_tracker().update_progress(0.5).unwrap();

        assert!(callback_called.load(std::sync::atomic::Ordering::SeqCst));
    }
