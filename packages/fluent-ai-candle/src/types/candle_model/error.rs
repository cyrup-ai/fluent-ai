//! Error types for the model system

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

/// Error type for model operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// Model not found in registry
    ModelNotFound {
        provider: Cow<'static, str>,
        name: Cow<'static, str>},

    /// Provider not found in registry
    ProviderNotFound(Cow<'static, str>),

    /// Model already exists in registry
    ModelAlreadyExists {
        provider: Cow<'static, str>,
        name: Cow<'static, str>},

    /// Invalid model configuration
    InvalidConfiguration(Cow<'static, str>),

    /// Operation not supported by model
    OperationNotSupported(Cow<'static, str>),

    /// Invalid input data
    InvalidInput(Cow<'static, str>),

    /// Internal error (should be used sparingly)
    Internal(Cow<'static, str>)}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound { provider, name } => {
                write!(f, "Model not found: {}:{}", provider, name)
            }
            Self::ProviderNotFound(provider) => write!(f, "Provider not found: {}", provider),
            Self::ModelAlreadyExists { provider, name } => {
                write!(f, "Model already registered: {}:{}", provider, name)
            }
            Self::InvalidConfiguration(msg) => write!(f, "Invalid model configuration: {}", msg),
            Self::OperationNotSupported(msg) => {
                write!(f, "Operation not supported by model: {}", msg)
            }
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg)}
    }
}

impl Error for ModelError {}

/// Result type for model operations
pub type Result<T> = std::result::Result<T, ModelError>;

/// Extension trait for converting Option to ModelError
pub trait OptionExt<T> {
    /// Convert an Option to a Result with a ModelError::ModelNotFound
    fn or_model_not_found<P, N>(self, provider: P, name: N) -> Result<T>
    where
        P: Into<Cow<'static, str>>,
        N: Into<Cow<'static, str>>;
}

impl<T> OptionExt<T> for Option<T> {
    fn or_model_not_found<P, N>(self, provider: P, name: N) -> Result<T>
    where
        P: Into<Cow<'static, str>>,
        N: Into<Cow<'static, str>>,
    {
        self.ok_or_else(|| ModelError::ModelNotFound {
            provider: provider.into(),
            name: name.into()})
    }
}

/// Extension trait for converting Result to ModelError
pub trait ResultExt<T, E> {
    /// Map an error to a ModelError::InvalidConfiguration
    fn invalid_config<M>(self, msg: M) -> Result<T>
    where
        M: Into<Cow<'static, str>>;

    /// Map an error to a ModelError::OperationNotSupported
    fn not_supported<M>(self, msg: M) -> Result<T>
    where
        M: Into<Cow<'static, str>>;
}

impl<T, E: Error> ResultExt<T, E> for std::result::Result<T, E> {
    fn invalid_config<M>(self, msg: M) -> Result<T>
    where
        M: Into<Cow<'static, str>>,
    {
        self.map_err(|_| ModelError::InvalidConfiguration(msg.into()))
    }

    fn not_supported<M>(self, msg: M) -> Result<T>
    where
        M: Into<Cow<'static, str>>,
    {
        self.map_err(|_| ModelError::OperationNotSupported(msg.into()))
    }
}

/// Helper for creating error messages with static strings
#[macro_export]
macro_rules! model_err {
    (not_found: $provider:expr, $name:expr) => {
        $crate::model::error::ModelError::ModelNotFound {
            provider: $provider.into(),
            name: $name.into()}
    };
    (provider_not_found: $provider:expr) => {
        $crate::model::error::ModelError::ProviderNotFound($provider.into())
    };
    (already_exists: $provider:expr, $name:expr) => {
        $crate::model::error::ModelError::ModelAlreadyExists {
            provider: $provider.into(),
            name: $name.into()}
    };
    (invalid_config: $msg:expr) => {
        $crate::model::error::ModelError::InvalidConfiguration($msg.into())
    };
    (not_supported: $msg:expr) => {
        $crate::model::error::ModelError::OperationNotSupported($msg.into())
    };
    (invalid_input: $msg:expr) => {
        $crate::model::error::ModelError::InvalidInput($msg.into())
    };
    (internal: $msg:expr) => {
        $crate::model::error::ModelError::Internal($msg.into())
    };
}

/// Helper for creating error results
#[macro_export]
macro_rules! bail_model_err {
    ($($tokens:tt)*) => {
        return Err(model_err!($($tokens)*))
    };
}
