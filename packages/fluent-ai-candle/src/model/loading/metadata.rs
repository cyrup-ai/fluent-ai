//! Model metadata extraction and validation
//!
//! This module handles extraction and validation of model metadata from SafeTensors.

use std::collections::HashMap;

use candle_core::{DType, Device, Result as CandleResult};
use candle_core::safetensors::MmapedSafetensors;

use crate::error::CandleError;

/// Information about a tensor in the model
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Data type of the tensor
    pub dtype: DType,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Optional quantization information
    pub quantized: Option<String>,
}

/// Model metadata extracted from SafeTensors
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model architecture type (e.g., "llama", "mistral")
    pub architecture: String,
    /// Model version
    pub version: String,
    /// Framework used to train the model
    pub framework: String,
    /// Number of parameters in the model
    pub num_parameters: u64,
    /// Size in bytes of the model
    pub size_bytes: u64,
    /// Whether the model is quantized
    pub is_quantized: bool,
    /// Quantization type if applicable
    pub quantization_type: Option<String>,
    /// Tensor information map
    pub tensors: HashMap<String, TensorInfo>,
    /// Model configuration parameters
    pub config: HashMap<String, String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            architecture: String::from("unknown"),
            version: String::from("1.0"),
            framework: String::from("safetensors"),
            num_parameters: 0,
            size_bytes: 0,
            is_quantized: false,
            quantization_type: None,
            tensors: HashMap::new(),
            config: HashMap::new(),
        }
    }
}

impl ModelMetadata {
    /// Extract model metadata from SafeTensors
    pub fn from_safetensors(
        safetensors: &MmapedSafetensors,
    ) -> CandleResult<Self> {
        let mut metadata = ModelMetadata::default();
        
        // Extract tensor information
        for (name, tensor) in safetensors.tensors() {
            let info = TensorInfo {
                dtype: tensor.dtype().into(),
                shape: tensor.shape().to_vec(),
                quantized: None, // Will be detected during loading
            };
            metadata.tensors.insert(name.to_string(), info);
            
            // Update total size
            metadata.size_bytes += tensor.size_bytes() as u64;
        }
        
        // Try to detect architecture from tensor names
        metadata.architecture = detect_architecture_from_tensors(&metadata.tensors);
        
        // Extract any metadata from the header
        if let Some(config) = safetensors.header().get("config") {
            if let Some(config_map) = config.as_object() {
                for (k, v) in config_map {
                    if let Some(v_str) = v.as_str() {
                        metadata.config.insert(k.clone(), v_str.to_string());
                    }
                }
            }
        }
        
        // Update parameter count based on architecture
        metadata.num_parameters = match metadata.architecture.as_str() {
            "llama" | "llama2" | "llama3" => {
                // Calculate parameters for LLaMA architecture
                let hidden_size = metadata.config.get("hidden_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let num_hidden_layers = metadata.config.get("num_hidden_layers")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                
                if hidden_size > 0 && num_hidden_layers > 0 {
                    // Approximate parameter count for LLaMA
                    let hidden_size = hidden_size as u64;
                    let num_hidden_layers = num_hidden_layers as u64;
                    
                    // Self-attention parameters
                    let attn_params = 4 * hidden_size * hidden_size * num_hidden_layers;
                    
                    // MLP parameters (assuming 4x hidden size for intermediate)
                    let mlp_params = 8 * hidden_size * hidden_size * num_hidden_layers;
                    
                    // Embedding and output layer
                    let emb_params = 2 * hidden_size * hidden_size;
                    
                    attn_params + mlp_params + emb_params
                } else {
                    0
                }
            }
            _ => 0, // Unknown architecture
        };
        
        Ok(metadata)
    }
}

/// Detect model architecture from tensor naming patterns
fn detect_architecture_from_tensors(tensors: &HashMap<String, TensorInfo>) -> String {
    // Check for LLaMA patterns
    if tensors.keys().any(|k| k.contains("model.layers.0.self_attn")) {
        return "llama".to_string();
    }
    
    // Check for Mistral patterns
    if tensors.keys().any(|k| k.contains("model.layers.0.self_attn.q_proj")) {
        return "mistral".to_string();
    }
    
    // Check for GPT patterns
    if tensors.keys().any(|k| k.contains("h.0.attn.c_attn")) {
        return "gpt2".to_string();
    }
    
    // Default to unknown
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype;
    use std::collections::HashMap;
    
    #[test]
    fn test_detect_architecture() {
        let mut tensors = HashMap::new();
        tensors.insert("model.layers.0.self_attn.q_proj.weight".to_string(), 
            TensorInfo {
                dtype: DType::F32,
                shape: vec![4096, 4096],
                quantized: None,
            });
            
        let arch = detect_architecture_from_tensors(&tensors);
        assert_eq!(arch, "llama");
    }
}