//! DNS resolver with hostname overrides
//!
//! Zero-allocation DNS resolver with hostname overrides for testing
//! and custom routing scenarios.

use fluent_ai_async::{AsyncStream, emit, spawn_task};
use std::sync::Arc;
use std::net::SocketAddr;

use super::types::{Resolve, DnsResult, HyperName};

/// Zero-allocation DNS resolver with hostname overrides for testing and custom routing.
pub(crate) struct DnsResolverWithOverridesImpl {
    pub dns_resolver: Arc<dyn Resolve>,
    pub overrides: Arc<std::collections::HashMap<String, arrayvec::ArrayVec<SocketAddr, 8>>>,
}

impl Resolve for DnsResolverWithOverridesImpl {
    fn resolve(&self, name: HyperName) -> AsyncStream<DnsResult> {
        let hostname = name.as_str().to_string();
        let overrides = self.overrides.clone();
        let dns_resolver = self.dns_resolver.clone();
        
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                // Check for override first
                if let Some(addrs) = overrides.get(&hostname) {
                    emit!(sender, DnsResult { addrs: addrs.clone() });
                    return;
                }
                
                // Fall back to actual DNS resolution
                let resolver_stream = dns_resolver.resolve(name);
                match resolver_stream.try_next() {
                    Some(result) => {
                        if result.is_error() {
                            emit!(sender, DnsResult::bad_chunk("DNS resolution failed".to_string()));
                        } else {
                            emit!(sender, result);
                        }
                    },
                    None => {
                        emit!(sender, DnsResult::bad_chunk("DNS resolver stream ended without producing addresses".to_string()));
                    }
                }
            });
        })
    }
}