//! Certificate Authority domain object and builders

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::tls::errors::TlsError;

/// Certificate Authority domain object with serialization support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateAuthority {
    pub name: String,
    pub certificate_pem: String,
    pub private_key_pem: String,
    pub metadata: CaMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaMetadata {
    pub subject: String,
    pub issuer: String,
    pub serial_number: String,
    pub valid_from: SystemTime,
    pub valid_until: SystemTime,
    pub key_algorithm: String,
    pub key_size: Option<u32>,
    pub created_at: SystemTime,
    pub source: CaSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CaSource {
    Filesystem { path: PathBuf },
    Keychain,
    Remote { url: String },
    Generated,
}

impl CertificateAuthority {
    /// Check if the certificate authority is currently valid
    pub fn is_valid(&self) -> bool {
        let now = SystemTime::now();
        now >= self.metadata.valid_from && now <= self.metadata.valid_until
    }

    /// Get duration until expiry
    pub fn expires_in(&self) -> Result<Duration, TlsError> {
        let now = SystemTime::now();
        self.metadata.valid_until.duration_since(now).map_err(|_| {
            TlsError::CertificateExpired("Certificate authority has expired".to_string())
        })
    }

    /// Check if this CA can sign certificates for the given domain
    pub fn can_sign_for_domain(&self, _domain: &str) -> bool {
        // TODO: Implement domain validation logic
        self.is_valid()
    }
}

/// Builder for certificate authority operations
#[derive(Debug, Clone)]
pub struct AuthorityBuilder {
    name: String,
}

impl AuthorityBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    /// Work with filesystem-based certificate authority
    pub fn path<P: AsRef<Path>>(self, path: P) -> AuthorityFilesystemBuilder {
        AuthorityFilesystemBuilder {
            name: self.name,
            path: path.as_ref().to_path_buf(),
            common_name: None,
            valid_for_years: 10,
            key_size: 2048,
        }
    }

    /// Work with keychain-based certificate authority (macOS/Windows)
    pub fn keychain(self) -> AuthorityKeychainBuilder {
        AuthorityKeychainBuilder { name: self.name }
    }

    /// Work with remote certificate authority
    pub fn url(self, url: &str) -> AuthorityRemoteBuilder {
        AuthorityRemoteBuilder {
            name: self.name,
            url: url.to_string(),
            timeout: Duration::from_secs(30),
        }
    }
}

/// Builder for filesystem certificate authority operations
#[derive(Debug, Clone)]
pub struct AuthorityFilesystemBuilder {
    name: String,
    path: PathBuf,
    common_name: Option<String>,
    valid_for_years: u32,
    key_size: u32,
}

impl AuthorityFilesystemBuilder {
    /// Set common name for certificate authority creation
    pub fn common_name(self, cn: &str) -> Self {
        Self {
            common_name: Some(cn.to_string()),
            ..self
        }
    }

    /// Set validity period in years for certificate authority creation
    pub fn valid_for_years(self, years: u32) -> Self {
        Self {
            valid_for_years: years,
            ..self
        }
    }

    /// Set key size for certificate authority creation
    pub fn key_size(self, bits: u32) -> Self {
        Self {
            key_size: bits,
            ..self
        }
    }

    /// Create a new certificate authority
    pub async fn create(self) -> super::responses::CertificateAuthorityResponse {
        use std::time::SystemTime;

        use rcgen::{CertificateParams, DistinguishedName, DnType, KeyPair};

        // Create directory if it doesn't exist
        if let Err(e) = tokio::fs::create_dir_all(&self.path).await {
            return super::responses::CertificateAuthorityResponse {
                success: false,
                authority: None,
                operation: super::responses::CaOperation::CreateFailed,
                issues: vec![format!("Failed to create directory: {}", e)],
                files_created: vec![],
            };
        }

        // Generate CA certificate
        let mut params = CertificateParams::new(vec![]);
        params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);

        let mut distinguished_name = DistinguishedName::new();
        let common_name = self.common_name.unwrap_or_else(|| self.name.clone());
        distinguished_name.push(DnType::CommonName, &common_name);
        params.distinguished_name = distinguished_name;

        // Set validity period
        let now = SystemTime::now();
        params.not_before = now.into();
        params.not_after = (now
            + std::time::Duration::from_secs(365 * 24 * 3600 * self.valid_for_years as u64))
        .into();

        // Generate key pair
        let key_pair = KeyPair::generate(&rcgen::PKCS_RSA_SHA256)
            .map_err(|e| format!("Failed to generate key pair: {}", e));

        let key_pair = match key_pair {
            Ok(kp) => kp,
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::CreateFailed,
                    issues: vec![e],
                    files_created: vec![],
                };
            }
        };

        params.key_pair = Some(key_pair);

        let cert = match rcgen::Certificate::from_params(params) {
            Ok(c) => c,
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::CreateFailed,
                    issues: vec![format!("Failed to generate certificate: {}", e)],
                    files_created: vec![],
                };
            }
        };

        let cert_pem = cert
            .serialize_pem()
            .map_err(|e| format!("Failed to serialize certificate: {}", e));
        let key_pem = cert.serialize_private_key_pem();

        let (cert_pem, key_pem) = match (cert_pem, key_pem) {
            (Ok(c), k) => (c, k),
            (Err(e), _) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::CreateFailed,
                    issues: vec![e],
                    files_created: vec![],
                };
            }
        };

        // Save files
        let cert_path = self.path.join("ca.crt");
        let key_path = self.path.join("ca.key");
        let mut files_created = vec![];

        if let Err(e) = tokio::fs::write(&cert_path, &cert_pem).await {
            return super::responses::CertificateAuthorityResponse {
                success: false,
                authority: None,
                operation: super::responses::CaOperation::CreateFailed,
                issues: vec![format!("Failed to write certificate: {}", e)],
                files_created,
            };
        }
        files_created.push(cert_path);

        if let Err(e) = tokio::fs::write(&key_path, &key_pem).await {
            return super::responses::CertificateAuthorityResponse {
                success: false,
                authority: None,
                operation: super::responses::CaOperation::CreateFailed,
                issues: vec![format!("Failed to write private key: {}", e)],
                files_created,
            };
        }
        files_created.push(key_path);

        // Create authority object
        let authority = CertificateAuthority {
            name: self.name.clone(),
            certificate_pem: cert_pem,
            private_key_pem: key_pem,
            metadata: CaMetadata {
                subject: common_name.clone(),
                issuer: common_name,
                serial_number: "1".to_string(), // CA serial number
                valid_from: now,
                valid_until: now
                    + std::time::Duration::from_secs(365 * 24 * 3600 * self.valid_for_years as u64),
                key_algorithm: "RSA".to_string(),
                key_size: Some(self.key_size),
                created_at: now,
                source: CaSource::Generated,
            },
        };

        super::responses::CertificateAuthorityResponse {
            success: true,
            authority: Some(authority),
            operation: super::responses::CaOperation::Created,
            issues: vec![],
            files_created,
        }
    }

    /// Load existing certificate authority from filesystem
    pub async fn load(self) -> super::responses::CertificateAuthorityResponse {
        use std::time::SystemTime;

        use crate::tls::certificate::parse_certificate_from_pem;

        let cert_path = self.path.join("ca.crt");
        let key_path = self.path.join("ca.key");

        // Check if both files exist
        if !cert_path.exists() || !key_path.exists() {
            return super::responses::CertificateAuthorityResponse {
                success: false,
                authority: None,
                operation: super::responses::CaOperation::LoadFailed,
                issues: vec![format!("CA files not found at {:?}", self.path)],
                files_created: vec![],
            };
        }

        // Read certificate and key files
        let cert_pem = match tokio::fs::read_to_string(&cert_path).await {
            Ok(content) => content,
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec![format!("Failed to read certificate: {}", e)],
                    files_created: vec![],
                };
            }
        };

        let key_pem = match tokio::fs::read_to_string(&key_path).await {
            Ok(content) => content,
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec![format!("Failed to read private key: {}", e)],
                    files_created: vec![],
                };
            }
        };

        // Parse certificate to extract metadata
        let parsed_cert = match parse_certificate_from_pem(&cert_pem) {
            Ok(cert) => cert,
            Err(e) => {
                return super::responses::CertificateAuthorityResponse {
                    success: false,
                    authority: None,
                    operation: super::responses::CaOperation::LoadFailed,
                    issues: vec![format!("Failed to parse certificate: {}", e)],
                    files_created: vec![],
                };
            }
        };

        let authority = CertificateAuthority {
            name: self.name.clone(),
            certificate_pem: cert_pem,
            private_key_pem: key_pem,
            metadata: CaMetadata {
                subject: parsed_cert.subject.clone(),
                issuer: parsed_cert.issuer.clone(),
                serial_number: parsed_cert.serial_number.clone(),
                valid_from: parsed_cert.not_before,
                valid_until: parsed_cert.not_after,
                key_algorithm: "RSA".to_string(), // TODO: Extract from parsed cert
                key_size: None,                   // TODO: Extract from parsed cert
                created_at: SystemTime::now(),
                source: CaSource::Filesystem {
                    path: self.path.clone(),
                },
            },
        };

        super::responses::CertificateAuthorityResponse {
            success: true,
            authority: Some(authority),
            operation: super::responses::CaOperation::Loaded,
            issues: vec![],
            files_created: vec![],
        }
    }
}

/// Builder for keychain certificate authority operations
#[derive(Debug, Clone)]
pub struct AuthorityKeychainBuilder {
    name: String,
}

impl AuthorityKeychainBuilder {
    /// Load certificate authority from system keychain
    pub async fn load(self) -> super::responses::CertificateAuthorityResponse {
        // TODO: Implement keychain CA loading
        super::responses::CertificateAuthorityResponse {
            success: false,
            authority: None,
            operation: super::responses::CaOperation::LoadFailed,
            issues: vec!["Keychain loading not yet implemented".to_string()],
            files_created: vec![],
        }
    }
}

/// Builder for remote certificate authority operations
#[derive(Debug, Clone)]
pub struct AuthorityRemoteBuilder {
    name: String,
    url: String,
    timeout: Duration,
}

impl AuthorityRemoteBuilder {
    /// Set timeout for remote operations
    pub fn with_timeout(self, timeout: Duration) -> Self {
        Self { timeout, ..self }
    }

    /// Load certificate authority from remote URL
    pub async fn load(self) -> super::responses::CertificateAuthorityResponse {
        // TODO: Implement remote CA loading
        super::responses::CertificateAuthorityResponse {
            success: false,
            authority: None,
            operation: super::responses::CaOperation::LoadFailed,
            issues: vec!["Remote loading not yet implemented".to_string()],
            files_created: vec![],
        }
    }
}
