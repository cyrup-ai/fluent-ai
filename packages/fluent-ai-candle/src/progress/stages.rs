//! Progress stage definitions for different ML operations

/// Stages of model download and loading process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DownloadStage {
    /// Initializing download connection
    Initializing,
    /// Actively downloading model data
    Downloading,
    /// Verifying downloaded data integrity
    Verifying,
    /// Extracting compressed model files
    Extracting,
    /// Loading model into memory
    Loading,
    /// Download and loading completed
    Completed,
}

impl DownloadStage {
    /// Get human-readable description of the stage
    pub fn description(&self) -> &'static str {
        match self {
            Self::Initializing => "Initializing download connection",
            Self::Downloading => "Downloading model data",
            Self::Verifying => "Verifying data integrity",
            Self::Extracting => "Extracting model files",
            Self::Loading => "Loading model into memory",
            Self::Completed => "Download completed",
        }
    }

    /// Get typical progress range for this stage (start, end)
    pub fn progress_range(&self) -> (f64, f64) {
        match self {
            Self::Initializing => (0.0, 0.05),
            Self::Downloading => (0.05, 0.80),
            Self::Verifying => (0.80, 0.85),
            Self::Extracting => (0.85, 0.95),
            Self::Loading => (0.95, 0.99),
            Self::Completed => (1.0, 1.0),
        }
    }

    /// Get estimated duration percentage for this stage
    pub fn duration_weight(&self) -> f64 {
        match self {
            Self::Initializing => 0.02,
            Self::Downloading => 0.70,
            Self::Verifying => 0.08,
            Self::Extracting => 0.15,
            Self::Loading => 0.04,
            Self::Completed => 0.01,
        }
    }
}

/// Stages of weight loading process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightLoadingStage {
    /// Preparing weight loading infrastructure
    Preparing,
    /// Loading individual layers sequentially
    LoadingLayers,
    /// Optimizing loaded weights for inference
    Optimizing,
    /// Weight loading completed
    Completed,
}

impl WeightLoadingStage {
    /// Get human-readable description of the stage
    pub fn description(&self) -> &'static str {
        match self {
            Self::Preparing => "Preparing weight loading infrastructure",
            Self::LoadingLayers => "Loading model layers",
            Self::Optimizing => "Optimizing weights for inference",
            Self::Completed => "Weight loading completed",
        }
    }

    /// Get typical progress range for this stage (start, end)
    pub fn progress_range(&self) -> (f64, f64) {
        match self {
            Self::Preparing => (0.0, 0.10),
            Self::LoadingLayers => (0.10, 0.85),
            Self::Optimizing => (0.85, 0.98),
            Self::Completed => (1.0, 1.0),
        }
    }

    /// Get estimated duration percentage for this stage
    pub fn duration_weight(&self) -> f64 {
        match self {
            Self::Preparing => 0.05,
            Self::LoadingLayers => 0.80,
            Self::Optimizing => 0.14,
            Self::Completed => 0.01,
        }
    }
}

/// Stages of model quantization process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationStage {
    /// Analyzing model for quantization opportunities
    Analyzing,
    /// Actively quantizing model weights
    Quantizing,
    /// Validating quantized model accuracy
    Validating,
    /// Quantization process completed
    Completed,
}

impl QuantizationStage {
    /// Get human-readable description of the stage
    pub fn description(&self) -> &'static str {
        match self {
            Self::Analyzing => "Analyzing model for quantization",
            Self::Quantizing => "Quantizing model weights",
            Self::Validating => "Validating quantized model",
            Self::Completed => "Quantization completed",
        }
    }

    /// Get typical progress range for this stage (start, end)
    pub fn progress_range(&self) -> (f64, f64) {
        match self {
            Self::Analyzing => (0.0, 0.15),
            Self::Quantizing => (0.15, 0.85),
            Self::Validating => (0.85, 0.98),
            Self::Completed => (1.0, 1.0),
        }
    }

    /// Get estimated duration percentage for this stage
    pub fn duration_weight(&self) -> f64 {
        match self {
            Self::Analyzing => 0.10,
            Self::Quantizing => 0.75,
            Self::Validating => 0.14,
            Self::Completed => 0.01,
        }
    }

    /// Get memory usage multiplier for this stage
    pub fn memory_multiplier(&self) -> f64 {
        match self {
            Self::Analyzing => 1.0,
            Self::Quantizing => 1.5, // Temporary extra memory during quantization
            Self::Validating => 1.2,
            Self::Completed => 0.7, // Quantized model uses less memory
        }
    }
}

/// Inference stages for token generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceStage {
    /// Preparing input tokens
    InputPreparation,
    /// Forward pass through model
    ForwardPass,
    /// Applying sampling strategies
    Sampling,
    /// Updating KV cache
    CacheUpdate,
    /// Post-processing output
    PostProcessing,
    /// Inference step completed
    Completed,
}

impl InferenceStage {
    /// Get human-readable description of the stage
    pub fn description(&self) -> &'static str {
        match self {
            Self::InputPreparation => "Preparing input tokens",
            Self::ForwardPass => "Forward pass through model",
            Self::Sampling => "Applying sampling strategies",
            Self::CacheUpdate => "Updating KV cache",
            Self::PostProcessing => "Post-processing output",
            Self::Completed => "Inference completed",
        }
    }

    /// Get typical duration percentage for this stage
    pub fn duration_weight(&self) -> f64 {
        match self {
            Self::InputPreparation => 0.05,
            Self::ForwardPass => 0.70,
            Self::Sampling => 0.15,
            Self::CacheUpdate => 0.08,
            Self::PostProcessing => 0.01,
            Self::Completed => 0.01,
        }
    }

    /// Check if this stage is compute-intensive
    pub fn is_compute_intensive(&self) -> bool {
        matches!(self, Self::ForwardPass | Self::Sampling)
    }

    /// Check if this stage involves memory operations
    pub fn involves_memory_ops(&self) -> bool {
        matches!(self, Self::CacheUpdate | Self::InputPreparation)
    }
}