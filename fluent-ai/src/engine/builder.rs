//! Typesafe typestate immutable builder for engine configuration
//!
//! This module provides a compile-time safe builder pattern for configuring engines.
//! The builder uses phantom types to enforce configuration order and completeness at compile time.

use super::{register_engine, set_default_engine, Engine};
use std::error::Error as StdError;
use std::marker::PhantomData;
use std::sync::Arc;

/// Marker types for typestate pattern
pub mod states {
    /// Indicates the engine type has not been specified
    pub struct NoEngine;

    /// Indicates an engine has been configured
    pub struct EngineConfigured;

    /// Indicates the engine name has not been specified
    pub struct NoName;

    /// Indicates the engine name has been configured
    pub struct NameConfigured;

    /// Indicates whether the engine should be set as default
    pub struct DefaultNotSet;

    /// Indicates the engine is configured to be the default
    pub struct DefaultSet;
}

/// Typesafe immutable engine builder using typestate pattern
///
/// The builder enforces compile-time safety by using phantom types to track
/// which configuration options have been set. This prevents runtime errors
/// from incomplete or invalid configurations.
pub struct EngineBuilder<E, N, D> {
    engine: Option<Arc<dyn Engine>>,
    name: Option<String>,
    set_as_default: bool,
    _phantom: PhantomData<(E, N, D)>,
}

impl EngineBuilder<states::NoEngine, states::NoName, states::DefaultNotSet> {
    /// Create a new engine builder
    ///
    /// # Examples
    /// ```rust,ignore
    /// let builder = EngineBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            engine: None,
            name: None,
            set_as_default: false,
            _phantom: PhantomData,
        }
    }
}

impl<N, D> EngineBuilder<states::NoEngine, N, D> {
    /// Set the engine instance
    ///
    /// This transitions the builder to the `EngineConfigured` state,
    /// enabling further configuration options.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let builder = EngineBuilder::new()
    ///     .engine(Arc::new(MyEngine::new()));
    /// ```
    pub fn engine(self, engine: Arc<dyn Engine>) -> EngineBuilder<states::EngineConfigured, N, D> {
        EngineBuilder {
            engine: Some(engine),
            name: self.name,
            set_as_default: self.set_as_default,
            _phantom: PhantomData,
        }
    }
}

impl<E, D> EngineBuilder<E, states::NoName, D> {
    /// Set the engine name for registry
    ///
    /// This transitions the builder to the `NameConfigured` state,
    /// which is required for building the final configuration.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let builder = EngineBuilder::new()
    ///     .engine(Arc::new(MyEngine::new()))
    ///     .name("my_engine");
    /// ```
    pub fn name<S: Into<String>>(self, name: S) -> EngineBuilder<E, states::NameConfigured, D> {
        EngineBuilder {
            engine: self.engine,
            name: Some(name.into()),
            set_as_default: self.set_as_default,
            _phantom: PhantomData,
        }
    }
}

impl<E, N> EngineBuilder<E, N, states::DefaultNotSet> {
    /// Configure this engine to be set as the default
    ///
    /// This transitions the builder to the `DefaultSet` state.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let builder = EngineBuilder::new()
    ///     .engine(Arc::new(MyEngine::new()))
    ///     .name("my_engine")
    ///     .as_default();
    /// ```
    pub fn as_default(self) -> EngineBuilder<E, N, states::DefaultSet> {
        EngineBuilder {
            engine: self.engine,
            name: self.name,
            set_as_default: true,
            _phantom: PhantomData,
        }
    }
}

/// Result of building an engine configuration
pub struct EngineConfig {
    pub engine: Arc<dyn Engine>,
    pub name: String,
    pub is_default: bool,
}

impl EngineBuilder<states::EngineConfigured, states::NameConfigured, states::DefaultNotSet> {
    /// Build the engine configuration (without setting as default)
    ///
    /// This method is only available when both engine and name are configured.
    /// The typestate pattern ensures compile-time safety.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let config = EngineBuilder::new()
    ///     .engine(Arc::new(MyEngine::new()))
    ///     .name("my_engine")
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<EngineConfig, Box<dyn StdError + Send + Sync>> {
        let engine = self
            .engine
            .expect("Engine should be set in EngineConfigured state");
        let name = self
            .name
            .expect("Name should be set in NameConfigured state");

        Ok(EngineConfig {
            engine,
            name,
            is_default: self.set_as_default,
        })
    }

    /// Build and register the engine in the global registry
    ///
    /// This convenience method builds the configuration and immediately
    /// registers it in the global engine registry.
    ///
    /// # Examples
    /// ```rust,ignore
    /// EngineBuilder::new()
    ///     .engine(Arc::new(MyEngine::new()))
    ///     .name("my_engine")
    ///     .build_and_register()?;
    /// ```
    pub fn build_and_register(self) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let config = self.build()?;
        register_engine(&config.name, config.engine)?;
        Ok(())
    }
}

impl EngineBuilder<states::EngineConfigured, states::NameConfigured, states::DefaultSet> {
    /// Build the engine configuration (with default setting)
    ///
    /// This method is available when the engine is fully configured
    /// and set to be the default.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let config = EngineBuilder::new()
    ///     .engine(Arc::new(MyEngine::new()))
    ///     .name("my_engine")
    ///     .as_default()
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<EngineConfig, Box<dyn StdError + Send + Sync>> {
        let engine = self
            .engine
            .expect("Engine should be set in EngineConfigured state");
        let name = self
            .name
            .expect("Name should be set in NameConfigured state");

        Ok(EngineConfig {
            engine,
            name,
            is_default: self.set_as_default,
        })
    }

    /// Build and register the engine, setting it as default
    ///
    /// This convenience method builds the configuration, registers it,
    /// and sets it as the default engine.
    ///
    /// # Examples
    /// ```rust,ignore
    /// EngineBuilder::new()
    ///     .engine(Arc::new(MyEngine::new()))
    ///     .name("my_engine")
    ///     .as_default()
    ///     .build_and_register()?;
    /// ```
    pub fn build_and_register(self) -> Result<(), Box<dyn StdError + Send + Sync>> {
        let config = self.build()?;
        register_engine(&config.name, config.engine.clone())?;

        if config.is_default {
            set_default_engine(config.engine)?;
        }

        Ok(())
    }
}

/// Convenience function to start building an engine configuration
///
/// # Examples
/// ```rust,ignore
/// let config = engine_builder()
///     .engine(Arc::new(MyEngine::new()))
///     .name("my_engine")
///     .build()?;
/// ```
pub fn engine_builder() -> EngineBuilder<states::NoEngine, states::NoName, states::DefaultNotSet> {
    EngineBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::super::NoOpEngine;
    use super::*;

    #[tokio::test]
    async fn test_builder_basic_flow() {
        let engine = Arc::new(NoOpEngine);

        let config = engine_builder()
            .engine(engine.clone())
            .name("test_engine")
            .build()
            .unwrap();

        assert_eq!(config.name, "test_engine");
        assert!(!config.is_default);

        // Test that the engine works by calling a method
        let tools = config.engine.available_tools().await;
        assert!(tools.is_ok());
    }

    #[tokio::test]
    async fn test_builder_with_default() {
        let engine = Arc::new(NoOpEngine);

        let config = engine_builder()
            .engine(engine.clone())
            .name("test_engine")
            .as_default()
            .build()
            .unwrap();

        assert_eq!(config.name, "test_engine");
        assert!(config.is_default);

        // Test that the engine works by calling a method
        let tools = config.engine.available_tools().await;
        assert!(tools.is_ok());
    }

    #[tokio::test]
    async fn test_builder_register() {
        let engine = Arc::new(NoOpEngine);

        // Test basic registration
        engine_builder()
            .engine(engine.clone())
            .name("test_register_engine")
            .build_and_register()
            .unwrap();

        // Verify it was registered
        let retrieved = super::super::get_engine("test_register_engine").unwrap();
        let tools = retrieved.available_tools().await;
        assert!(tools.is_ok());
    }

    #[tokio::test]
    async fn test_builder_register_as_default() {
        let engine = Arc::new(NoOpEngine);

        // Test registration as default
        engine_builder()
            .engine(engine.clone())
            .name("test_default_engine")
            .as_default()
            .build_and_register()
            .unwrap();

        // Verify it was set as default
        let default = super::super::get_default_engine().unwrap();
        let tools = default.available_tools().await;
        assert!(tools.is_ok());
    }
}
