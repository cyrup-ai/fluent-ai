//! Model-related builders extracted from fluent_ai_domain

pub mod model_builder;
pub mod model_info_builder;

// Re-export for convenience
pub use model_builder::CandleModelBuilder;
pub use model_info_builder::CandleModelInfoBuilder;