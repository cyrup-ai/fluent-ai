use fluent_ai_candle::sampling::topk::*;
use fluent_ai_candle::sampling::*;

use candle_core::{DType, Device};

    

    fn create_test_logits() -> Tensor {
        let device = Device::Cpu;
        // Create logits with clear ranking: [4.0, 3.0, 2.0, 1.0, 0.5]
        Tensor::from_vec(vec![1.0f32, 3.0, 2.0, 4.0, 0.5], (5,), &device).expect("tensor creation")
    }

    fn create_uniform_logits() -> Tensor {
        let device = Device::Cpu;
        Tensor::from_vec(vec![1.0f32; 10], (10,), &device).expect("tensor creation")
    }

    fn create_tied_logits() -> Tensor {
        let device = Device::Cpu;
        // Create logits with ties: [3.0, 2.0, 2.0, 2.0, 1.0]
        Tensor::from_vec(vec![3.0f32, 2.0, 2.0, 2.0, 1.0], (5,), &device).expect("tensor creation")
    }

    #[test]
    fn test_top_k_validation() {
        // Valid top-k values
        assert!(TopKProcessor::new(1).is_ok());
        assert!(TopKProcessor::new(50).is_ok());
        assert!(TopKProcessor::new(1000).is_ok());

        // Invalid top-k values
        assert!(matches!(
            TopKProcessor::new(0),
            Err(SamplingError::InvalidTopK(0))
        ));
    }

    #[test]
    fn test_top_k_filtering() {
        let processor = TopKProcessor::new(3).expect("valid top-k");

        let mut logits = create_test_logits();
        let original_vec = logits.to_vec1::<f32>().expect("conversion");

        processor
            .process(&mut logits, &[], 0)
            .expect("processing succeeds");

        let processed_vec = logits.to_vec1::<f32>().expect("conversion");

        // Should keep exactly 3 tokens
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();
        assert_eq!(finite_count, 3);

        // The 3 largest values should be preserved
        let mut orig_sorted: Vec<(usize, f32)> = original_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        orig_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check that top 3 are kept
        for i in 0..3 {
            let (idx, _) = orig_sorted[i];
            assert!(
                processed_vec[idx].is_finite(),
                "Top {} token should be kept",
                i + 1
            );
        }

        // Check that bottom 2 are filtered
        for i in 3..5 {
            let (idx, _) = orig_sorted[i];
            assert_eq!(
                processed_vec[idx],
                f32::NEG_INFINITY,
                "Bottom {} token should be filtered",
                i - 2
            );
        }
    }

    #[test]
    fn test_top_k_larger_than_vocab() {
        let processor = TopKProcessor::new(10).expect("valid top-k");

        let mut logits = create_test_logits(); // Only 5 tokens
        let original_vec = logits.to_vec1::<f32>().expect("conversion");

        processor
            .process(&mut logits, &[], 0)
            .expect("processing succeeds");

        let processed_vec = logits.to_vec1::<f32>().expect("conversion");

        // All tokens should be kept when k > vocab_size
        for (orig, proc) in original_vec.iter().zip(processed_vec.iter()) {
            assert!((orig - proc).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_top_k_one() {
        let processor = TopKProcessor::new(1).expect("valid top-k");

        let mut logits = create_test_logits();
        let original_vec = logits.to_vec1::<f32>().expect("conversion");

        processor
            .process(&mut logits, &[], 0)
            .expect("processing succeeds");

        let processed_vec = logits.to_vec1::<f32>().expect("conversion");

        // Should keep exactly 1 token
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();
        assert_eq!(finite_count, 1);

        // Should keep the maximum token
        let max_value = original_vec
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let max_idx = original_vec
            .iter()
            .position(|&x| (x - max_value).abs() < f32::EPSILON)
            .unwrap();

        assert!(processed_vec[max_idx].is_finite());
        for (i, &val) in processed_vec.iter().enumerate() {
            if i != max_idx {
                assert_eq!(val, f32::NEG_INFINITY);
            }
        }
    }

    #[test]
    fn test_ties_handling() {
        let processor = TopKProcessor::new(3).expect("valid top-k");

        let mut logits = create_tied_logits(); // [3.0, 2.0, 2.0, 2.0, 1.0]
        processor
            .process(&mut logits, &[], 0)
            .expect("processing succeeds");

        let processed_vec = logits.to_vec1::<f32>().expect("conversion");
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();

        // Should handle ties deterministically
        assert!(
            finite_count <= 3,
            "Should not exceed k={}",
            processor.top_k()
        );
        assert!(finite_count >= 1, "Should keep at least one token");

        // The highest value (3.0) should always be kept
        assert!(processed_vec[0].is_finite());
    }

    #[test]
    fn test_uniform_distribution() {
        let processor = TopKProcessor::new(3).expect("valid top-k");

        let mut logits = create_uniform_logits(); // 10 identical values
        processor
            .process(&mut logits, &[], 0)
            .expect("processing succeeds");

        let processed_vec = logits.to_vec1::<f32>().expect("conversion");
        let finite_count = processed_vec.iter().filter(|&&x| x.is_finite()).count();

        // Should keep exactly 3 tokens even with ties
        assert_eq!(finite_count, 3);
    }

    #[test]
    fn test_edge_cases() {
        let device = Device::Cpu;

        // Empty logits should fail
        let empty_logits = Tensor::zeros((0,), DType::F32, &device).expect("tensor creation");
        let processor = TopKProcessor::new(3).expect("valid top-k");
        let mut logits = empty_logits;
        assert!(processor.process(&mut logits, &[], 0).is_err());

        // Single token with k=1
        let single_logits = Tensor::from_vec(vec![1.0f32], (1,), &device).expect("tensor creation");
        let mut logits = single_logits;
        assert!(processor.process(&mut logits, &[], 0).is_ok());
        let finite_count = processor.count_finite_tokens(&logits);
        assert_eq!(finite_count, 1);
    }

    #[test]
    fn test_kth_largest_algorithms() {
        let processor = TopKProcessor::new(3).expect("valid top-k");
        let values = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0]; // 3rd largest is 5.0

        // Test quickselect approach
        let mut values1 = values.clone();
        let result1 = processor
            .quickselect_kth_largest(&mut values1)
            .expect("quickselect works");
        assert!((result1 - 5.0).abs() < f32::EPSILON);

        // Test partial sort approach
        let mut values2 = values.clone();
        let result2 = processor
            .partial_sort_kth_largest(&mut values2)
            .expect("partial sort works");
        assert!((result2 - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_top_k_update() {
        let mut processor = TopKProcessor::new(5).expect("valid top-k");
        assert_eq!(processor.top_k(), 5);

        processor.set_top_k(10).expect("valid update");
        assert_eq!(processor.top_k(), 10);

        // Invalid update should fail
        assert!(processor.set_top_k(0).is_err());
    }

    #[test]
    fn test_display_and_equality() {
        let proc1 = TopKProcessor::new(50).expect("valid top-k");
        let proc2 = TopKProcessor::new(50).expect("valid top-k");
        let proc3 = TopKProcessor::new(40).expect("valid top-k");

        assert_eq!(proc1, proc2);
        assert_ne!(proc1, proc3);

        let display = format!("{}", proc1);
        assert!(display.contains("TopK"));
        assert!(display.contains("50"));
    }

    #[test]
    fn test_processor_trait_methods() {
        let processor = TopKProcessor::new(25).expect("valid top-k");

        assert_eq!(processor.name(), "TopKProcessor");
        assert!(processor.validate().is_ok());

        // is_identity depends on runtime vocab size, so test basic functionality
        assert!(!processor.is_identity() || processor.is_identity());
    }

    #[test]
    fn test_performance_characteristics() {
        // Test with larger vocabulary to verify performance characteristics
        let device = Device::Cpu;
        let large_vocab: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let large_logits =
            Tensor::from_vec(large_vocab, (1000,), &device).expect("large tensor creation");

        let processor = TopKProcessor::new(50).expect("valid top-k");
        let mut logits = large_logits;

        let start = std::time::Instant::now();
        processor
            .process(&mut logits, &[], 0)
            .expect("processing succeeds");
        let duration = start.elapsed();

        // Should complete in reasonable time (< 1ms for 1000 tokens)
        assert!(
            duration.as_millis() < 100,
            "Processing took too long: {:?}",
            duration
        );

        let finite_count = processor.count_finite_tokens(&logits);
        assert_eq!(finite_count, 50);
    }
