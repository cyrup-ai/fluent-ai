//! Identity implementation for client authentication
//!
//! Production-quality client identity management with PKCS#8/PKCS#12 support.

use super::types::{Identity, ClientCert};

impl Identity {
    /// Parses a DER-formatted PKCS #12 archive, using the specified password to decrypt the key.
    ///
    /// The archive should contain a leaf certificate and its private key, as well any intermediate
    /// certificates that allow clients to build a chain to a trusted root.
    /// The chain certificates should be in order from the leaf certificate towards the root.
    ///
    /// # Optional
    ///
    /// This requires the `native-tls` Cargo feature enabled.
    #[cfg(feature = "native-tls")]
    pub fn from_pkcs12_der(der: &[u8], password: &str) -> crate::Result<Identity> {
        let native = native_tls_crate::Identity::from_pkcs12(der, password).map_err(crate::error::builder)?;
        let inner = ClientCert::Pkcs12(native.clone());
        Ok(Identity { native, inner })
    }

    /// Parses a chain of PEM encoded certificates, with the leaf certificate first.
    /// `key` is a PEM encoded PKCS #8 formatted private key.
    ///
    /// The certificate chain should contain any intermediate certs that should be sent to
    /// clients to allow them to build a chain to a trusted root.
    ///
    /// # Optional
    ///
    /// This requires the `rustls-tls(-...)` Cargo feature enabled.
    #[cfg(feature = "__rustls")]
    pub fn from_pem(buf: &[u8]) -> crate::Result<Identity> {
        use std::io::Cursor;

        let mut cursor = Cursor::new(buf);
        let mut reader = std::io::BufReader::new(&mut cursor);

        let certs = rustls::pki_types::CertificateDer::pem_reader_iter(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| crate::HttpError::Tls { message: format!("invalid certificate: {}", e) })?;

        if certs.is_empty() {
            return Err(crate::HttpError::Tls { message: "no certificates found".to_string() });
        }

        cursor.set_position(0);
        let mut reader = std::io::BufReader::new(&mut cursor);

        let key = rustls::pki_types::PrivateKeyDer::from_pem_reader(&mut reader)
            .map_err(|e| crate::HttpError::Tls { message: format!("invalid private key: {}", e) })?;

        let inner = ClientCert::Pem {
            key,
            certs: certs.into_iter().map(|c| c.into_owned()).collect(),
        };

        Ok(Identity {
            #[cfg(feature = "default-tls")]
            native: {
                // For default-tls, we need to convert to a format native-tls can understand
                // This is a simplified approach - in practice, you might need more sophisticated conversion
                let pkcs8_der = match &inner {
                    ClientCert::Pem { key, certs } => {
                        // This is a placeholder - actual implementation would need proper PKCS#12 generation
                        return Err(crate::HttpError::Tls { message: "PEM identity not supported with default-tls backend".to_string() });
                    }
                    _ => unreachable!(),
                };
                pkcs8_der
            },
            inner,
        })
    }

    /// Parses a PKCS #8 private key and certificate chain, with the leaf certificate first.
    ///
    /// # Optional
    ///
    /// This requires the `native-tls` Cargo feature enabled.
    #[cfg(feature = "native-tls")]
    pub fn from_pkcs8_pem(buf: &[u8]) -> crate::Result<Identity> {
        let native = native_tls_crate::Identity::from_pkcs8(buf).map_err(crate::error::builder)?;
        let inner = ClientCert::Pkcs8(native.clone());
        Ok(Identity { native, inner })
    }

    #[cfg(feature = "default-tls")]
    pub(crate) fn add_to_native_tls(
        self,
        tls: &mut native_tls_crate::TlsConnectorBuilder,
    ) -> crate::Result<()> {
        tls.identity(self.native);
        Ok(())
    }

    #[cfg(feature = "__rustls")]
    pub(crate) fn add_to_rustls(
        self,
        config_builder: rustls::ConfigBuilder<
            rustls::ClientConfig,
            // Not sure here
            rustls::client::WantsClientCert,
        >,
    ) -> crate::Result<rustls::ClientConfig> {
        match self.inner {
            ClientCert::Pem { key, certs } => config_builder
                .with_client_auth_cert(certs, key)
                .map_err(|e| crate::HttpError::Tls { message: format!("TLS client cert error: {}", e) }),
            #[cfg(feature = "native-tls")]
            ClientCert::Pkcs12(..) | ClientCert::Pkcs8(..) => {
                Err(crate::error::builder("incompatible TLS identity type"))
            }
        }
    }
}