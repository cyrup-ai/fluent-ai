//! Performance-oriented constants for zero-allocation operations

/// Default KV cache size in bytes (512 MB)
pub const DEFAULT_KV_CACHE_SIZE: usize = 512 * 1024 * 1024;

/// Default token buffer size for streaming operations
pub const DEFAULT_TOKEN_BUFFER_SIZE: usize = 8192;

/// Maximum model file size for memory mapping (16 GB)
pub const MAX_MODEL_FILE_SIZE: usize = 16 * 1024 * 1024 * 1024;

/// Maximum logits dimensions for SIMD optimization
pub const MAX_LOGITS_DIM: usize = 512 * 1024;

/// Default temperature for sampling
pub const DEFAULT_TEMPERATURE: f32 = 1.0;

/// Default top-p value for nucleus sampling
pub const DEFAULT_TOP_P: f32 = 0.9;

/// Default top-k value for top-k sampling
pub const DEFAULT_TOP_K: u32 = 50;

/// Default repetition penalty
pub const DEFAULT_REPETITION_PENALTY: f32 = 1.1;

/// Maximum sequence length for context
pub const MAX_SEQUENCE_LENGTH: usize = 32768;

/// Probability buffer size for sampling operations
pub const PROBABILITY_BUFFER_SIZE: usize = 65536;

/// Maximum number of stop sequences
pub const MAX_STOP_SEQUENCES: usize = 16;

/// Maximum length of a stop sequence
pub const MAX_STOP_SEQUENCE_LENGTH: usize = 64;

/// Streaming chunk size for optimal performance
pub const STREAMING_CHUNK_SIZE: usize = 1024;