//! Fluent AI Provider Library
//!
//! This crate provides provider and model traits and definitions for AI services.
//! The enum variants are auto-generated from the AiChat models.yaml file.

// Module declarations
pub mod model;
pub mod model_info;
pub mod models;
pub mod provider;
pub mod providers;

// Re-export all types for convenience
pub use cyrup_sugars::ZeroOneOrMany;
pub use model::Model;
pub use model_info::ModelInfoData;
pub use models::Models;
pub use provider::Provider;
pub use providers::Providers;
