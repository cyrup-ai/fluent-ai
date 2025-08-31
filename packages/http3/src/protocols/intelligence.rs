//! Domain protocol intelligence cache
//!
//! Per-domain protocol capability tracking with lock-free atomic operations.
//! Maintains hot cache of domains and their supported protocols ensuring we never
//! try the incorrect protocol twice for a domain.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::protocols::core::HttpVersion;

/// Domain protocol intelligence cache with atomic operations for lock-free access
///
/// Tracks protocol capabilities, success/failure rates, and last attempt timestamps
/// for each domain to enable intelligent protocol selection and prevent retrying
/// failed protocols.
#[derive(Debug)]
pub struct ProtocolIntelligence {
    /// Domain capability cache with atomic access
    domains: Arc<RwLock<HashMap<String, Arc<DomainCapabilities>>>>,
    /// Global statistics for cache performance
    stats: ProtocolIntelligenceStats,
    /// Cache configuration
    config: IntelligenceConfig,
}

/// Per-domain protocol capabilities with atomic tracking
#[derive(Debug)]
pub struct DomainCapabilities {
    /// Domain name
    pub domain: String,
    /// HTTP/3 support tracking
    pub h3_support: AtomicProtocolSupport,
    /// HTTP/2 support tracking
    pub h2_support: AtomicProtocolSupport,
    /// HTTP/1.1 support tracking (always assumed available)
    pub h1_support: AtomicProtocolSupport,
    /// Last successful protocol used
    pub last_successful_protocol: Arc<RwLock<Option<HttpVersion>>>,
    /// Domain discovery timestamp
    pub discovered_at: SystemTime,
    /// Last update timestamp
    pub last_updated: Arc<RwLock<SystemTime>>,
}

/// Atomic protocol support tracking
#[derive(Debug)]
pub struct AtomicProtocolSupport {
    /// Whether protocol is supported (None=unknown, Some(true)=supported, Some(false)=not supported)
    pub is_supported: AtomicBool,
    /// Whether support status is known
    pub is_known: AtomicBool,
    /// Success count
    pub success_count: AtomicUsize,
    /// Failure count
    pub failure_count: AtomicUsize,
    /// Last attempt timestamp (nanoseconds since UNIX_EPOCH)
    pub last_attempt: AtomicU64,
    /// Last success timestamp (nanoseconds since UNIX_EPOCH)
    pub last_success: AtomicU64,
}

/// Global protocol intelligence statistics
#[derive(Debug)]
pub struct ProtocolIntelligenceStats {
    /// Total domains tracked
    pub domains_tracked: AtomicUsize,
    /// Cache hits
    pub cache_hits: AtomicUsize,
    /// Cache misses
    pub cache_misses: AtomicUsize,
    /// Protocol discoveries
    pub protocol_discoveries: AtomicUsize,
    /// Failed protocol attempts prevented
    pub failed_attempts_prevented: AtomicUsize,
}

/// Configuration for protocol intelligence cache
#[derive(Debug, Clone)]
pub struct IntelligenceConfig {
    /// Maximum domains to track in cache
    pub max_domains: usize,
    /// Time before retrying a failed protocol
    pub retry_after_failure: Duration,
    /// Time before considering cached data stale
    pub cache_expiry: Duration,
    /// Minimum attempts before marking protocol as unsupported
    pub min_attempts_for_failure: usize,
}

impl Default for IntelligenceConfig {
    fn default() -> Self {
        Self {
            max_domains: 10000,
            retry_after_failure: Duration::from_secs(300), // 5 minutes
            cache_expiry: Duration::from_secs(3600),       // 1 hour
            min_attempts_for_failure: 3,
        }
    }
}

impl AtomicProtocolSupport {
    /// Create new atomic protocol support tracker
    pub fn new() -> Self {
        Self {
            is_supported: AtomicBool::new(false),
            is_known: AtomicBool::new(false),
            success_count: AtomicUsize::new(0),
            failure_count: AtomicUsize::new(0),
            last_attempt: AtomicU64::new(0),
            last_success: AtomicU64::new(0),
        }
    }

    /// Mark protocol as supported
    pub fn mark_supported(&self) {
        self.is_supported.store(true, Ordering::Relaxed);
        self.is_known.store(true, Ordering::Relaxed);
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.last_success.store(current_timestamp_nanos(), Ordering::Relaxed);
        self.last_attempt.store(current_timestamp_nanos(), Ordering::Relaxed);
    }

    /// Mark protocol as not supported
    pub fn mark_not_supported(&self) {
        self.is_supported.store(false, Ordering::Relaxed);
        self.is_known.store(true, Ordering::Relaxed);
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        self.last_attempt.store(current_timestamp_nanos(), Ordering::Relaxed);
    }

    /// Check if protocol is known to be supported
    pub fn is_supported(&self) -> Option<bool> {
        if self.is_known.load(Ordering::Relaxed) {
            Some(self.is_supported.load(Ordering::Relaxed))
        } else {
            None
        }
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let successes = self.success_count.load(Ordering::Relaxed);
        let failures = self.failure_count.load(Ordering::Relaxed);
        let total = successes + failures;

        if total == 0 {
            0.0
        } else {
            successes as f64 / total as f64
        }
    }

    /// Check if enough time has passed since last failure to retry
    pub fn can_retry_after_failure(&self, retry_duration: Duration) -> bool {
        let last_attempt = self.last_attempt.load(Ordering::Relaxed);
        let now = current_timestamp_nanos();
        let elapsed_nanos = now.saturating_sub(last_attempt);
        let elapsed_duration = Duration::from_nanos(elapsed_nanos);

        elapsed_duration >= retry_duration
    }
}

impl DomainCapabilities {
    /// Create new domain capabilities tracker
    pub fn new(domain: String) -> Self {
        Self {
            domain,
            h3_support: AtomicProtocolSupport::new(),
            h2_support: AtomicProtocolSupport::new(),
            h1_support: AtomicProtocolSupport::new(),
            last_successful_protocol: Arc::new(RwLock::new(None)),
            discovered_at: SystemTime::now(),
            last_updated: Arc::new(RwLock::new(SystemTime::now())),
        }
    }

    /// Get protocol support for specific version
    pub fn get_protocol_support(&self, version: HttpVersion) -> &AtomicProtocolSupport {
        match version {
            HttpVersion::Http3 => &self.h3_support,
            HttpVersion::Http2 => &self.h2_support,
        }
    }

    /// Track successful protocol usage
    pub fn track_success(&self, protocol: HttpVersion) {
        self.get_protocol_support(protocol).mark_supported();
        
        // Update last successful protocol
        if let Ok(mut last_successful) = self.last_successful_protocol.write() {
            *last_successful = Some(protocol);
        }
        
        // Update last updated timestamp
        if let Ok(mut last_updated) = self.last_updated.write() {
            *last_updated = SystemTime::now();
        }
    }

    /// Track failed protocol attempt
    pub fn track_failure(&self, protocol: HttpVersion) {
        self.get_protocol_support(protocol).mark_not_supported();
        
        // Update last updated timestamp
        if let Ok(mut last_updated) = self.last_updated.write() {
            *last_updated = SystemTime::now();
        }
    }

    /// Get preferred protocol order based on historical success
    pub fn get_preferred_protocols(&self) -> Vec<HttpVersion> {
        let mut protocols = vec![
            (HttpVersion::Http3, self.h3_support.success_rate()),
            (HttpVersion::Http2, self.h2_support.success_rate()),
        ];

        // Sort by success rate (descending)
        protocols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        protocols.into_iter().map(|(version, _)| version).collect()
    }

    /// Check if protocol should be retried based on failure history
    pub fn should_retry_protocol(&self, protocol: HttpVersion, config: &IntelligenceConfig) -> bool {
        let support = self.get_protocol_support(protocol);
        
        // If protocol is known to be supported, always retry
        if let Some(true) = support.is_supported() {
            return true;
        }

        // If protocol is known to be unsupported, check retry conditions
        if let Some(false) = support.is_supported() {
            let failure_count = support.failure_count.load(Ordering::Relaxed);
            
            // If we haven't reached minimum attempts threshold, allow retry
            if failure_count < config.min_attempts_for_failure {
                return true;
            }
            
            // Check if enough time has passed since last failure
            return support.can_retry_after_failure(config.retry_after_failure);
        }

        // Protocol support is unknown, allow attempt
        true
    }
}

impl ProtocolIntelligence {
    /// Create new protocol intelligence cache
    pub fn new() -> Self {
        Self::with_config(IntelligenceConfig::default())
    }

    /// Create protocol intelligence cache with custom configuration
    pub fn with_config(config: IntelligenceConfig) -> Self {
        Self {
            domains: Arc::new(RwLock::new(HashMap::new())),
            stats: ProtocolIntelligenceStats {
                domains_tracked: AtomicUsize::new(0),
                cache_hits: AtomicUsize::new(0),
                cache_misses: AtomicUsize::new(0),
                protocol_discoveries: AtomicUsize::new(0),
                failed_attempts_prevented: AtomicUsize::new(0),
            },
            config,
        }
    }

    /// Track successful protocol usage for domain
    pub fn track_success(&self, domain: &str, protocol: HttpVersion) {
        let capabilities = self.get_or_create_domain_capabilities(domain);
        capabilities.track_success(protocol);
        self.stats.protocol_discoveries.fetch_add(1, Ordering::Relaxed);
    }

    /// Track failed protocol attempt for domain
    pub fn track_failure(&self, domain: &str, protocol: HttpVersion) {
        let capabilities = self.get_or_create_domain_capabilities(domain);
        capabilities.track_failure(protocol);
    }

    /// Get preferred protocol for domain based on historical data
    pub fn get_preferred_protocol(&self, domain: &str) -> HttpVersion {
        if let Some(capabilities) = self.get_domain_capabilities(domain) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            let preferred_protocols = capabilities.get_preferred_protocols();
            
            // Return first protocol that should be retried
            for protocol in preferred_protocols {
                if capabilities.should_retry_protocol(protocol, &self.config) {
                    return protocol;
                }
            }
        } else {
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Default fallback: HTTP/3 -> HTTP/2 -> HTTP/1.1
        HttpVersion::Http3
    }

    /// Check if protocol should be retried for domain
    pub fn should_retry_protocol(&self, domain: &str, protocol: HttpVersion) -> bool {
        if let Some(capabilities) = self.get_domain_capabilities(domain) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            let should_retry = capabilities.should_retry_protocol(protocol, &self.config);
            
            if !should_retry {
                self.stats.failed_attempts_prevented.fetch_add(1, Ordering::Relaxed);
            }
            
            should_retry
        } else {
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            true // Allow attempt for unknown domains
        }
    }

    /// Get domain capabilities (internal)
    fn get_domain_capabilities(&self, domain: &str) -> Option<Arc<DomainCapabilities>> {
        self.domains.read().ok()?.get(domain).cloned()
    }

    /// Get or create domain capabilities (internal)
    fn get_or_create_domain_capabilities(&self, domain: &str) -> Arc<DomainCapabilities> {
        // Fast path: try to get existing capabilities
        if let Some(capabilities) = self.get_domain_capabilities(domain) {
            return capabilities;
        }

        // Slow path: create new capabilities
        let mut domains = self.domains.write().unwrap();
        
        // Double-check after acquiring write lock
        if let Some(capabilities) = domains.get(domain) {
            return capabilities.clone();
        }

        // Create new capabilities
        let capabilities = Arc::new(DomainCapabilities::new(domain.to_string()));
        domains.insert(domain.to_string(), capabilities.clone());
        self.stats.domains_tracked.fetch_add(1, Ordering::Relaxed);

        // Enforce cache size limit
        if domains.len() > self.config.max_domains {
            self.evict_oldest_domains(&mut domains);
        }

        capabilities
    }

    /// Evict oldest domains to maintain cache size limit
    fn evict_oldest_domains(&self, domains: &mut HashMap<String, Arc<DomainCapabilities>>) {
        // Remove 10% of cache when limit is exceeded
        let target_size = (self.config.max_domains as f64 * 0.9) as usize;
        let to_remove = domains.len().saturating_sub(target_size);

        if to_remove == 0 {
            return;
        }

        // Collect domains with their last updated times
        let mut domain_times: Vec<(String, SystemTime)> = domains
            .iter()
            .filter_map(|(domain, capabilities)| {
                capabilities.last_updated.read().ok()
                    .map(|last_updated| (domain.clone(), *last_updated))
            })
            .collect();

        // Sort by last updated time (oldest first)
        domain_times.sort_by_key(|(_, time)| *time);

        // Remove oldest domains
        for (domain, _) in domain_times.into_iter().take(to_remove) {
            domains.remove(&domain);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> ProtocolIntelligenceStats {
        ProtocolIntelligenceStats {
            domains_tracked: AtomicUsize::new(self.stats.domains_tracked.load(Ordering::Relaxed)),
            cache_hits: AtomicUsize::new(self.stats.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicUsize::new(self.stats.cache_misses.load(Ordering::Relaxed)),
            protocol_discoveries: AtomicUsize::new(self.stats.protocol_discoveries.load(Ordering::Relaxed)),
            failed_attempts_prevented: AtomicUsize::new(self.stats.failed_attempts_prevented.load(Ordering::Relaxed)),
        }
    }

    /// Clear cache (for testing or maintenance)
    pub fn clear(&self) {
        if let Ok(mut domains) = self.domains.write() {
            domains.clear();
            self.stats.domains_tracked.store(0, Ordering::Relaxed);
        }
    }
}

/// Get current timestamp in nanoseconds since UNIX_EPOCH
fn current_timestamp_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

impl Default for ProtocolIntelligence {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ProtocolIntelligenceStats {
    fn clone(&self) -> Self {
        Self {
            domains_tracked: AtomicUsize::new(self.domains_tracked.load(Ordering::Relaxed)),
            cache_hits: AtomicUsize::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicUsize::new(self.cache_misses.load(Ordering::Relaxed)),
            protocol_discoveries: AtomicUsize::new(self.protocol_discoveries.load(Ordering::Relaxed)),
            failed_attempts_prevented: AtomicUsize::new(self.failed_attempts_prevented.load(Ordering::Relaxed)),
        }
    }
}