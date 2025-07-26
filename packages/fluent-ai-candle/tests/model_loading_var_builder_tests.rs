use fluent_ai_candle::model::loading::var_builder::*;
use fluent_ai_candle::model::loading::*;

use tempfile::tempdir;
    
    #[test]
    fn test_var_builder_factory() -> CandleResult<()> {
        // Create a test safetensors file
        let dir = tempdir()?;
        let path = dir.path().join("test.safetensors");
        
        // Create a simple tensor
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let tensors = vec![("weight".to_string(), tensor)];
        
        // Save to safetensors
        candle_core::safetensors::save_safetensors(&tensors, &path)?;
        
        // Load back using VarBuilderFactory
        let config = VarBuilderConfig {
            device: Device::Cpu,
            dtype: DType::F32,
            use_mmap: false,
            keep_original: false,
            progress: None,
            transforms: HashMap::new()};
        
        let factory = VarBuilderFactory::from_safetensors(&path, config)?;
        let var_builder = factory.into_var_builder();
        
        // Verify the tensor was loaded correctly
        let loaded_tensor = var_builder.get("weight").unwrap();
        assert_eq!(loaded_tensor.dims(), &[4]);
        
        Ok(())
    }
    
    #[test]
    fn test_hot_swappable_var_builder() {
        // Create a simple var builder
        let tensors = HashMap::new();
        let device = Device::Cpu;
        let var_builder = VarBuilder::from_tensors(tensors, DType::F32, &device);
        
        // Create a hot-swappable wrapper
        let swappable = HotSwappableVarBuilder::new(var_builder);
        
        // Create a new var builder
        let new_tensors = HashMap::new();
        let new_var_builder = VarBuilder::from_tensors(new_tensors, DType::F32, &device);
        
        // Swap the var builder
        swappable.swap(new_var_builder);
        
        // Verify we can still get a reference
        let _builder = swappable.get();
    }
