//! Certificate implementation and management
//!
//! Production-quality certificate handling with DER/PEM support and backend integration.

use std::io::{BufRead, Cursor};

use super::types::{Certificate, Cert};

impl Certificate {
    /// Create a `Certificate` from a binary DER encoded certificate
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::fs::File;
    /// # use std::io::Read;
    /// # fn cert() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut buf = Vec::new();
    /// File::open("my_cert.der")?
    ///     .read_to_end(&mut buf)?;
    /// let cert = crate::hyper::Certificate::from_der(&buf)?;
    /// # drop(cert);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_der(der: &[u8]) -> crate::Result<Certificate> {
        Ok(Certificate {
            #[cfg(feature = "default-tls")]
            native: native_tls_crate::Certificate::from_der(der).map_err(crate::error::builder)?,
            #[cfg(feature = "__rustls")]
            original: Cert::Der(der.to_owned()),
        })
    }

    /// Create a `Certificate` from a PEM encoded certificate
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::fs::File;
    /// # use std::io::Read;
    /// # fn cert() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut buf = Vec::new();
    /// File::open("my_cert.pem")?
    ///     .read_to_end(&mut buf)?;
    /// let cert = crate::hyper::Certificate::from_pem(&buf)?;
    /// # drop(cert);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_pem(pem: &[u8]) -> crate::Result<Certificate> {
        Ok(Certificate {
            #[cfg(feature = "default-tls")]
            native: native_tls_crate::Certificate::from_pem(pem).map_err(crate::error::builder)?,
            #[cfg(feature = "__rustls")]
            original: Cert::Pem(pem.to_owned()),
        })
    }

    /// Create a collection of `Certificate`s from a PEM encoded certificate bundle.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::fs::File;
    /// # use std::io::Read;
    /// # fn cert() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut buf = Vec::new();
    /// File::open("my_cert_bundle.pem")?
    ///     .read_to_end(&mut buf)?;
    /// let certs = crate::hyper::Certificate::from_pem_bundle(&buf)?;
    /// # drop(certs);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_pem_bundle(pem_bundle: &[u8]) -> crate::Result<Vec<Certificate>> {
        let mut reader = Cursor::new(pem_bundle);
        let certs = Self::read_pem_certs(&mut reader)?;

        certs
            .into_iter()
            .map(|c| Certificate::from_der(&c))
            .collect()
    }

    #[cfg(feature = "default-tls")]
    pub(crate) fn add_to_native_tls(self, tls: &mut native_tls_crate::TlsConnectorBuilder) {
        tls.add_root_certificate(self.native);
    }

    #[cfg(feature = "__rustls")]
    pub(crate) fn add_to_rustls(
        self,
        root_cert_store: &mut rustls::RootCertStore,
    ) -> crate::Result<()> {
        use std::io::Cursor;

        match self.original {
            Cert::Der(buf) => root_cert_store
                .add(buf.into())
                .map_err(crate::error::builder)?,
            Cert::Pem(buf) => {
                let mut reader = Cursor::new(buf);
                let certs = Self::read_pem_certs(&mut reader)?;
                for c in certs {
                    root_cert_store
                        .add(c.into())
                        .map_err(crate::error::builder)?;
                }
            }
        }
        Ok(())
    }
}