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
        name: Cow<'static, str>,
    },

    /// Provider not found in registry
    ProviderNotFound(Cow<'static, str>),

    /// Model already exists in registry
    ModelAlreadyExists {
        provider: Cow<'static, str>,
        name: Cow<'static, str>,
    },

    /// Invalid model configuration
    InvalidConfiguration(Cow<'static, str>),

    /// Operation not supported by model
    OperationNotSupported(Cow<'static, str>),

    /// Invalid input data
    InvalidInput(Cow<'static, str>),

    /// Internal error (should be used sparingly)
    Internal(Cow<'static, str>),
}

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
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
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
            name: name.into(),
        })
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
            name: $name.into(),
        }
    };
    (provider_not_found: $provider:expr) => {
        $crate::model::error::ModelError::ProviderNotFound($provider.into())
    };
    (already_exists: $provider:expr, $name:expr) => {
        $crate::model::error::ModelError::ModelAlreadyExists {
            provider: $provider.into(),
            name: $name.into(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_error_display() {
        assert_eq!(
            ModelError::ModelNotFound {
                provider: "test",
                name: "test"
            }
            .to_string(),
            "Model not found: test:test"
        );
        assert_eq!(
            ModelError::ProviderNotFound("test").to_string(),
            "Provider not found: test"
        );
        assert_eq!(
            ModelError::ModelAlreadyExists {
                provider: "test",
                name: "test"
            }
            .to_string(),
            "Model already registered: test:test"
        );
        assert_eq!(
            ModelError::InvalidConfiguration("test").to_string(),
            "Invalid model configuration: test"
        );
        assert_eq!(
            ModelError::OperationNotSupported("test").to_string(),
            "Operation not supported by model: test"
        );
        assert_eq!(
            ModelError::InvalidInput("test").to_string(),
            "Invalid input: test"
        );
        assert_eq!(
            ModelError::Internal("test").to_string(),
            "Internal error: test"
        );
    }

    #[test]
    fn test_option_ext() {
        let some: Option<u32> = Some(42);
        assert_eq!(some.or_model_not_found("test", "test").unwrap(), 42);

        let none: Option<u32> = None;
        assert!(matches!(
            none.or_model_not_found("test", "test"),
            Err(ModelError::ModelNotFound {
                provider: "test",
                name: "test"
            })
        ));
    }

    #[test]
    fn test_result_ext() {
        let ok: std::result::Result<u32, &str> = Ok(42);
        assert_eq!(ok.invalid_config("test").unwrap(), 42);
        assert_eq!(ok.not_supported("test").unwrap(), 42);

        let err: std::result::Result<u32, &str> = Err("error");
        assert!(matches!(
            err.invalid_config("test"),
            Err(ModelError::InvalidConfiguration("test"))
        ));
        assert!(matches!(
            err.not_supported("test"),
            Err(ModelError::OperationNotSupported("test"))
        ));
    }

    #[test]
    fn test_model_err_macro() {
        assert!(matches!(
            model_err!(not_found: "test", "test"),
            ModelError::ModelNotFound {
                provider: "test",
                name: "test"
            }
        ));
        assert!(matches!(
            model_err!(provider_not_found: "test"),
            ModelError::ProviderNotFound("test")
        ));
        assert!(matches!(
            model_err!(already_exists: "test", "test"),
            ModelError::ModelAlreadyExists {
                provider: "test",
                name: "test"
            }
        ));
        assert!(matches!(
            model_err!(invalid_config: "test"),
            ModelError::InvalidConfiguration("test")
        ));
        assert!(matches!(
            model_err!(not_supported: "test"),
            ModelError::OperationNotSupported("test")
        ));
        assert!(matches!(
            model_err!(invalid_input: "test"),
            ModelError::InvalidInput("test")
        ));
        assert!(matches!(
            model_err!(internal: "test"),
            ModelError::Internal("test")
        ));
    }
}
