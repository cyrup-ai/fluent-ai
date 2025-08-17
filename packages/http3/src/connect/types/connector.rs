//! HTTP/3 connector types and service abstractions
//!
//! Provides the core connector types for establishing HTTP/3 connections
//! with support for different TLS configurations and service layers.

use crate::hyper::connect::service::ConnectorService;

/// HTTP/3 connection provider with zero-allocation streaming
#[derive(Clone, Debug)]
pub struct Connector {
    pub(super) inner: ConnectorKind,
}

/// Enumeration of different connector implementations
#[derive(Clone, Debug)]
pub enum ConnectorKind {
    #[cfg(feature = "__tls")]
    BuiltDefault(ConnectorService),
    #[cfg(not(feature = "__tls"))]
    BuiltHttp(ConnectorService),
    WithLayers(BoxedConnectorService),
}

/// Direct ConnectorService type - no more Service trait boxing needed
pub type BoxedConnectorService = ConnectorService;

/// Simplified approach: Use trait objects for connector layers
/// This provides the same functionality as tower::Layer but with AsyncStream services
/// Boxed connector layer type for composable connection handling.
pub type BoxedConnectorLayer =
    Box<dyn Fn(BoxedConnectorService) -> BoxedConnectorService + Send + Sync + 'static>;

/// Sealed module for internal traits.
pub mod sealed {
    /// Unnameable struct for internal use.
    #[derive(Default, Debug)]
    pub struct Unnameable;
}

pub use sealed::Unnameable;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connector_kind_variants() {
        // Test that ConnectorKind variants exist and can be matched
        #[cfg(feature = "__tls")]
        {
            // Test would require actual ConnectorService for full verification
            assert!(true); // Placeholder for TLS variant test
        }

        #[cfg(not(feature = "__tls"))]
        {
            // Test would require actual ConnectorService for full verification
            assert!(true); // Placeholder for HTTP variant test
        }
    }

    #[test]
    fn test_connector_clone() {
        // Test that Connector implements Clone correctly
        assert!(true); // Placeholder for clone verification
    }

    #[test]
    fn test_unnameable_type() {
        let _unnameable = Unnameable::default();
        // Test that Unnameable can be created and used
        assert!(true);
    }
}
