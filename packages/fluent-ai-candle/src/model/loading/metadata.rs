//! Model metadata extraction and validation
//!
//! This module handles extraction and validation of model metadata from SafeTensors.

use std::collections::HashMap;

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Result as CandleResult};

// Removed unused import: CandleError

/// Zero-allocation conversion function for DType conversion
/// Provides const-time conversion with compile-time optimization
#[inline(always)]
pub fn convert_safetensors_dtype(dtype: safetensors::tensor::Dtype) -> DType {
    match dtype {
        safetensors::tensor::Dtype::BOOL => DType::U8, // Bool maps to U8 in candle
        safetensors::tensor::Dtype::U8 => DType::U8,
        safetensors::tensor::Dtype::I8 => DType::U8, // I8 maps to U8 in candle
        safetensors::tensor::Dtype::U16 => DType::U32, // U16 maps to U32 (candle limitation)
        safetensors::tensor::Dtype::I16 => DType::U32, // I16 maps to U32 (candle limitation)
        safetensors::tensor::Dtype::U32 => DType::U32,
        safetensors::tensor::Dtype::I32 => DType::U32, // I32 maps to U32 in candle
        safetensors::tensor::Dtype::U64 => DType::U32, // U64 maps to U32 (candle limitation)
        safetensors::tensor::Dtype::I64 => DType::U32, // I64 maps to U32 (candle limitation)
        safetensors::tensor::Dtype::F16 => DType::F16,
        safetensors::tensor::Dtype::BF16 => DType::BF16,
        safetensors::tensor::Dtype::F32 => DType::F32,
        safetensors::tensor::Dtype::F64 => DType::F64,
        _ => DType::F32, // Default fallback for any new or unsupported types
    }
}

/// Information about a tensor in the model
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Data type of the tensor
    pub dtype: DType,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Optional quantization information
    pub quantized: Option<String>}

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
    pub config: HashMap<String, String>}

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
            config: HashMap::new()}
    }
}

impl ModelMetadata {
    /// Extract model metadata from SafeTensors
    pub fn from_safetensors(safetensors: &MmapedSafetensors) -> CandleResult<Self> {
        let mut metadata = ModelMetadata::default();

        // Extract tensor information
        for (name, tensor) in safetensors.tensors() {
            let info = TensorInfo {
                dtype: convert_safetensors_dtype(tensor.dtype()),
                shape: tensor.shape().to_vec(),
                quantized: None, // Will be detected during loading
            };
            metadata.tensors.insert(name.to_string(), info);

            // Update total size with zero-allocation calculation
            let tensor_size = tensor.data().len() as u64;
            metadata.size_bytes += tensor_size;
        }

        // Try to detect architecture from tensor names
        metadata.architecture = detect_architecture_from_tensors(&metadata.tensors);

        // TODO: Extract metadata - safetensors 0.4.5 API differs from 0.6.0
        // Metadata extraction temporarily disabled due to API compatibility
        // Will be re-enabled once proper safetensors 0.4.5 metadata access is implemented

        // Update parameter count based on architecture
        metadata.num_parameters = match metadata.architecture.as_str() {
            "llama" | "llama2" | "llama3" => {
                // Calculate parameters for LLaMA architecture
                let hidden_size = metadata
                    .config
                    .get("hidden_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let num_hidden_layers = metadata
                    .config
                    .get("num_hidden_layers")
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
    if tensors
        .keys()
        .any(|k| k.contains("model.layers.0.self_attn"))
    {
        return "llama".to_string();
    }

    // Check for Mistral patterns
    if tensors
        .keys()
        .any(|k| k.contains("model.layers.0.self_attn.q_proj"))
    {
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
}
