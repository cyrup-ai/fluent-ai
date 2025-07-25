//! Ultra-High-Performance VarBuilder Pattern for Efficient Weight Loading

pub mod builder;
pub mod config;
pub mod metadata;
pub mod types;

// Re-export public types and functions
pub use builder::CandleVarBuilder;
pub use config::{VarBuilderConfig, VarBuilderConfigBuilder};
pub use metadata::{ModelMetadata, TensorEntry};
pub use types::{
    convert_dtype, DeviceHint, LoadingStats, TensorLoadStrategy, TensorMetadata,
    TensorName, ConfigKey, ConfigValue,
    MAX_TENSOR_NAME_LEN, MAX_TENSORS, MAX_CONFIG_ENTRIES, MAX_FILE_PATHS, CACHE_LINE_SIZE,
    ERR_TENSOR_NOT_FOUND, ERR_INVALID_SHAPE, ERR_DEVICE_MISMATCH, ERR_MODEL_LOADING, ERR_METADATA_PARSING};