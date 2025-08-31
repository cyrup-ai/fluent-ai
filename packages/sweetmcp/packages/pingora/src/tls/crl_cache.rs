//! CRL cache implementation and validation logic

use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use base64::engine::Engine;
use reqwest::Client;
use tokio::time::timeout;
use x509_parser::prelude::*;

use super::errors::TlsError;
use super::types::{CrlCacheEntry, ParsedCertificate};

pub struct CrlCache {
    cache: Arc<RwLock<std::collections::HashMap<String, CrlCacheEntry>>>,
    http_client: Client,
}

impl CrlCache {
    pub fn new() -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30)) // CRL files can be large
            .user_agent("SweetMCP/1.0 CRL Client")
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            cache: Arc::new(RwLock::new(std::collections::HashMap::with_capacity(64))),
            http_client,
        }
    }

    /// Check if certificate serial number is revoked using CRL
    pub async fn check_certificate_revocation(
        &self,
        cert: &ParsedCertificate,
    ) -> Result<bool, TlsError> {
        if cert.crl_urls.is_empty() {
            tracing::warn!("No CRL URLs found in certificate, skipping CRL validation");
            return Ok(false); // Not revoked (no CRL available)
        }

        // Try each CRL URL until one succeeds
        for crl_url in &cert.crl_urls {
            match self.check_against_crl(cert, crl_url).await {
                Ok(is_revoked) => {
                    if is_revoked {
                        tracing::warn!(
                            "Certificate serial {:?} found in CRL from {}",
                            hex::encode(&cert.serial_number),
                            crl_url
                        );
                        return Ok(true);
                    }
                    tracing::info!(
                        "Certificate serial {:?} not found in CRL from {}",
                        hex::encode(&cert.serial_number),
                        crl_url
                    );
                }
                Err(e) => {
                    tracing::warn!("CRL validation failed for URL {}: {}", crl_url, e);
                    continue;
                }
            }
        }

        // If all CRLs were checked and certificate not found in any, it's not revoked
        Ok(false)
    }

    async fn check_against_crl(
        &self,
        cert: &ParsedCertificate,
        crl_url: &str,
    ) -> Result<bool, TlsError> {
        let cache_key = crl_url.to_string();

        // Check cache first
        if let Some(cached_crl) = self.get_cached_crl(&cache_key) {
            if !Self::is_crl_cache_expired(&cached_crl) {
                tracing::debug!("CRL cache hit for URL: {}", crl_url);
                return Ok(cached_crl.revoked_serials.contains(&cert.serial_number));
            }
        }

        // Download and parse CRL
        let crl_entry = self.download_and_parse_crl(crl_url).await?;

        // Cache the CRL
        self.cache_crl(cache_key, crl_entry.clone());

        // Check if certificate is revoked
        Ok(crl_entry.revoked_serials.contains(&cert.serial_number))
    }

    #[inline]
    fn get_cached_crl(&self, cache_key: &str) -> Option<CrlCacheEntry> {
        match self.cache.read() {
            Ok(cache) => cache.get(cache_key).cloned(),
            Err(poisoned) => {
                tracing::warn!("CRL cache read lock poisoned, recovering");
                poisoned.into_inner().get(cache_key).cloned()
            }
        }
    }

    fn is_crl_cache_expired(entry: &CrlCacheEntry) -> bool {
        let now = SystemTime::now();

        // Check if we have next_update time and it's passed
        if let Some(next_update) = entry.next_update {
            return now > next_update;
        }

        // Default cache expiry: 24 hours (CRLs are typically updated daily)
        let cache_duration = Duration::from_secs(24 * 3600);
        now.duration_since(entry.cached_at)
            .unwrap_or(Duration::ZERO)
            > cache_duration
    }

    #[inline]
    fn cache_crl(&self, cache_key: String, entry: CrlCacheEntry) {
        match self.cache.write() {
            Ok(mut cache) => {
                cache.insert(cache_key, entry);
            }
            Err(poisoned) => {
                tracing::warn!("CRL cache write lock poisoned, recovering");
                poisoned.into_inner().insert(cache_key, entry);
            }
        }
    }

    async fn download_and_parse_crl(&self, crl_url: &str) -> Result<CrlCacheEntry, TlsError> {
        // Download CRL with timeout
        let response = timeout(
            Duration::from_secs(30),
            self.http_client.get(crl_url).send(),
        )
        .await
        .map_err(|_| TlsError::NetworkError("CRL download timeout".to_string()))?
        .map_err(|e| TlsError::NetworkError(format!("CRL HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(TlsError::CrlValidation(format!(
                "CRL download failed with status: {}",
                response.status()
            )));
        }

        let crl_bytes = response
            .bytes()
            .await
            .map_err(|e| TlsError::NetworkError(format!("Failed to read CRL data: {}", e)))?;

        // Parse CRL
        self.parse_crl_data(&crl_bytes)
    }

    fn parse_crl_data(&self, crl_bytes: &[u8]) -> Result<CrlCacheEntry, TlsError> {
        // Parse PEM if it starts with "-----BEGIN"
        let der_bytes = if crl_bytes.starts_with(b"-----BEGIN") {
            let crl_pem = std::str::from_utf8(crl_bytes)
                .map_err(|_| TlsError::CrlValidation("Invalid UTF-8 in PEM CRL".to_string()))?;

            // Extract DER from PEM
            let mut der_data = Vec::new();
            let mut in_crl = false;
            for line in crl_pem.lines() {
                if line.contains("-----BEGIN") && line.contains("CRL") {
                    in_crl = true;
                    continue;
                }
                if line.contains("-----END") && line.contains("CRL") {
                    break;
                }
                if in_crl {
                    if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(line) {
                        der_data.extend(decoded);
                    }
                }
            }

            if der_data.is_empty() {
                return Err(TlsError::CrlValidation(
                    "No CRL data found in PEM".to_string(),
                ));
            }

            der_data
        } else {
            // Assume DER format
            crl_bytes.to_vec()
        };

        // Parse X.509 CRL using x509-parser
        let (_, crl) = parse_x509_crl(&der_bytes)
            .map_err(|e| TlsError::CrlValidation(format!("CRL parsing failed: {}", e)))?;

        // Extract revoked certificate serial numbers
        let mut revoked_serials = HashSet::new();
        for revoked_cert in crl.iter_revoked_certificates() {
            revoked_serials.insert(revoked_cert.user_certificate.to_bytes_be());
        }

        // Extract next update time
        let next_update = crl.next_update().map(|time| {
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(time.timestamp() as u64)
        });

        tracing::info!(
            "Parsed CRL with {} revoked certificates, next update: {:?}",
            revoked_serials.len(),
            next_update
        );

        Ok(CrlCacheEntry {
            revoked_serials,
            cached_at: SystemTime::now(),
            next_update,
        })
    }

    /// Cleanup expired CRL cache entries
    pub fn cleanup_cache(&self) {
        let mut cache = match self.cache.write() {
            Ok(cache) => cache,
            Err(poisoned) => {
                tracing::warn!("CRL cache write lock poisoned during cleanup, recovering");
                poisoned.into_inner()
            }
        };

        cache.retain(|_url, entry| !Self::is_crl_cache_expired(entry));

        tracing::debug!(
            "CRL cache cleanup completed, {} CRLs remaining",
            cache.len()
        );
    }
}
