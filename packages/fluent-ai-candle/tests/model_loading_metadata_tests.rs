use fluent_ai_candle::model::loading::metadata::*;
use fluent_ai_candle::model::loading::*;

#[test]
    fn test_detect_architecture() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            TensorInfo {
                dtype: DType::F32,
                shape: vec![4096, 4096],
                quantized: None},
        );

        let arch = detect_architecture_from_tensors(&tensors);
        assert_eq!(arch, "llama");
    }
