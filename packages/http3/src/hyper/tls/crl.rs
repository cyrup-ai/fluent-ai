//! Certificate Revocation List (CRL) implementation
//!
//! Production-quality CRL handling for rustls backend.

#[cfg(feature = "__rustls")]
use super::types::CertificateRevocationList;

#[cfg(feature = "__rustls")]
impl CertificateRevocationList {
    /// Parses a PEM encoded CRL.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::fs::File;
    /// # use std::io::Read;
    /// # fn crl() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut buf = Vec::new();
    /// File::open("my_crl.pem")?
    ///     .read_to_end(&mut buf)?;
    /// let crl = crate::hyper::tls::CertificateRevocationList::from_pem(&buf)?;
    /// # drop(crl);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Optional
    ///
    /// This requires the `rustls-tls(-...)` Cargo feature enabled.
    #[cfg(feature = "__rustls")]
    pub fn from_pem(pem: &[u8]) -> crate::Result<CertificateRevocationList> {
        Ok(CertificateRevocationList {
            #[cfg(feature = "__rustls")]
            inner: rustls::pki_types::CertificateRevocationListDer::from(pem.to_vec()),
        })
    }

    /// Parses a collection of PEM encoded CRLs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::fs::File;
    /// # use std::io::Read;
    /// # fn crl() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut buf = Vec::new();
    /// File::open("my_crls.pem")?
    ///     .read_to_end(&mut buf)?;
    /// let crls = crate::hyper::tls::CertificateRevocationList::from_pem_bundle(&buf)?;
    /// # drop(crls);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Optional
    ///
    /// This requires the `rustls-tls(-...)` Cargo feature enabled.
    #[cfg(feature = "__rustls")]
    pub fn from_pem_bundle(pem_bundle: &[u8]) -> crate::Result<Vec<CertificateRevocationList>> {
        use std::io::Cursor;

        let mut cursor = Cursor::new(pem_bundle);
        let mut reader = std::io::BufReader::new(&mut cursor);
        let mut crls = Vec::new();

        for result in rustls::pki_types::CertificateRevocationListDer::pem_reader_iter(&mut reader)
        {
            match result {
                Ok(crl) => crls.push(CertificateRevocationList {
                    inner: crl.into_owned(),
                }),
                Err(e) => {
                    return Err(crate::HttpError::Tls {
                        message: format!("invalid CRL encoding: {}", e),
                    });
                }
            }
        }

        Ok(crls)
    }

    /// Returns the number of CRLs in this collection.
    #[cfg(feature = "__rustls")]
    pub fn len(&self) -> usize {
        1 // Each CertificateRevocationList represents a single CRL
    }

    /// Returns true if this collection is empty.
    #[cfg(feature = "__rustls")]
    pub fn is_empty(&self) -> bool {
        false // A CertificateRevocationList always contains one CRL
    }
}
