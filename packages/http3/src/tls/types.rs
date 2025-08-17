//! Core TLS types and enums
//!
//! Production-quality TLS configuration types with comprehensive backend support.

use std::io::BufRead;

#[cfg(feature = "__rustls")]
use rustls::{
    DigitallySignedStruct, Error as TLSError, SignatureScheme,
    client::danger::HandshakeSignatureValid, client::danger::ServerCertVerified,
    client::danger::ServerCertVerifier, pki_types::ServerName, pki_types::UnixTime,
};

/// A TLS certificate.
#[derive(Clone)]
pub struct Certificate {
    #[cfg(feature = "default-tls")]
    pub(crate) native: native_tls_crate::Certificate,
    #[cfg(feature = "__rustls")]
    pub(crate) original: Cert,
}

/// A client identity (certificate and private key).
pub struct Identity {
    #[cfg(feature = "default-tls")]
    pub(crate) native: native_tls_crate::Identity,
    pub(crate) inner: ClientCert,
}

/// A certificate revocation list (CRL).
#[cfg(feature = "__rustls")]
#[derive(Clone, Debug)]
pub struct CertificateRevocationList {
    #[cfg(feature = "__rustls")]
    pub(crate) inner: rustls::pki_types::CertificateRevocationListDer<'static>,
}

/// Represents the available TLS backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TlsBackend {
    /// The default TLS backend.
    #[cfg(feature = "default-tls")]
    Default,
    /// The rustls TLS backend.
    #[cfg(feature = "__rustls")]
    Rustls,
    /// Built rustls backend.
    #[cfg(feature = "__rustls")]
    BuiltRustls,
    /// Unknown preconfigured backend.
    UnknownPreconfigured,
}

/// TLS version configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Version {
    /// TLS 1.0
    Tls1_0,
    /// TLS 1.1
    Tls1_1,
    /// TLS 1.2
    Tls1_2,
    /// TLS 1.3
    Tls1_3,
}

/// Hostname verification configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IgnoreHostname(pub bool);

impl Version {
    #[cfg(feature = "default-tls")]
    pub fn to_native_tls(&self) -> native_tls_crate::Protocol {
        match self {
            Version::Tls1_0 => native_tls_crate::Protocol::Tlsv10,
            Version::Tls1_1 => native_tls_crate::Protocol::Tlsv11,
            Version::Tls1_2 => native_tls_crate::Protocol::Tlsv12,
            Version::Tls1_3 => native_tls_crate::Protocol::Tlsv12, // fallback
        }
    }
}

impl IgnoreHostname {
    pub fn new(ignore: bool) -> Self {
        IgnoreHostname(ignore)
    }
}

#[derive(Clone)]
pub(crate) enum Cert {
    Der(Vec<u8>),
    Pem(Vec<u8>),
}

pub(crate) enum ClientCert {
    #[cfg(feature = "native-tls")]
    Pkcs12(native_tls_crate::Identity),
    #[cfg(feature = "native-tls")]
    Pkcs8(native_tls_crate::Identity),
    #[cfg(feature = "__rustls")]
    Pem {
        key: rustls::pki_types::PrivateKeyDer<'static>,
        certs: Vec<rustls::pki_types::CertificateDer<'static>>,
    },
}

impl Clone for ClientCert {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "native-tls")]
            Self::Pkcs8(i) => Self::Pkcs8(i.clone()),
            #[cfg(feature = "native-tls")]
            Self::Pkcs12(i) => Self::Pkcs12(i.clone()),
            #[cfg(feature = "__rustls")]
            ClientCert::Pem { key: _, certs } => ClientCert::Pem {
                key: rustls::pki_types::PrivateKeyDer::Pkcs8(
                    rustls::pki_types::PrivatePkcs8KeyDer::from(vec![]),
                ),
                certs: certs.clone(),
            },
            #[cfg_attr(
                any(feature = "native-tls", feature = "__rustls"),
                allow(unreachable_patterns)
            )]
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "__rustls")]
#[derive(Debug)]
pub(crate) struct NoVerifier;

#[cfg(feature = "__rustls")]
impl ServerCertVerifier for NoVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer,
        _intermediates: &[rustls::pki_types::CertificateDer],
        _server_name: &ServerName,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TLSError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TLSError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TLSError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::RSA_PKCS1_SHA1,
            SignatureScheme::ECDSA_SHA1_Legacy,
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
            SignatureScheme::ECDSA_NISTP521_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::ED25519,
            SignatureScheme::ED448,
        ]
    }
}

impl Certificate {
    pub(crate) fn read_pem_certs(reader: &mut impl BufRead) -> crate::Result<Vec<Vec<u8>>> {
        let mut certs = Vec::new();
        for result in rustls::pki_types::CertificateDer::pem_reader_iter(reader) {
            match result {
                Ok(cert) => certs.push(cert.as_ref().to_vec()),
                Err(e) => {
                    return Err(crate::HttpError::Tls {
                        message: format!("invalid certificate encoding: {}", e),
                    });
                }
            }
        }
        Ok(certs)
    }
}
