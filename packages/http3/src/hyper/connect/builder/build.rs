//! Build logic for ConnectorBuilder
//! 
//! Provides the final build method to create configured connectors.

use super::types::ConnectorBuilder;
use crate::hyper::error::BoxError;
use super::super::service::ConnectorService;
use super::super::types::{Connector, ConnectorKind};
use hyper_util::client::legacy::connect::HttpConnector;

impl ConnectorBuilder {
    /// Build the connector with configured settings
    pub fn build(self) -> Result<Connector, BoxError> {
        let service = ConnectorService::new(
            self.http_connector.unwrap_or_else(|| HttpConnector::new()),
            #[cfg(feature = "default-tls")]
            self.tls_connector,
            #[cfg(feature = "__rustls")]
            self.rustls_config,
            self.proxies,
            self.user_agent,
            self.local_address,
            self.interface,
            self.nodelay,
            self.connect_timeout,
            self.happy_eyeballs_timeout,
            self.tls_info,
        )?;

        let kind = if self.enforce_http {
            #[cfg(not(feature = "__tls"))]
            {
                ConnectorKind::BuiltHttp(service)
            }
            #[cfg(feature = "rustls-tls")]
            {
                if self.tls_built {
                    ConnectorKind::BuiltDefault(service)
                } else {
                    ConnectorKind::BuiltHttp(service)
                }
            }
        } else {
            #[cfg(feature = "__tls")]
            ConnectorKind::BuiltDefault(service);
            #[cfg(not(feature = "__tls"))]
            ConnectorKind::BuiltHttp(service)
        };

        Ok(Connector { inner: kind })
    }
}