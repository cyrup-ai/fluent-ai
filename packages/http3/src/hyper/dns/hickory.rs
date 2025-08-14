//! DNS resolution via the [hickory-resolver](https://github.com/hickory-dns/hickory-dns) crate

use hickory_resolver::{
    config::LookupIpStrategy, lookup_ip::LookupIpIntoIter, ResolveError, AsyncResolver,
};
use once_cell::sync::OnceCell;

use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;

use super::{Addrs, Name, Resolve};
use crate::hyper::error::BoxError;
use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};

/// Wrapper around an `AsyncResolver`, which implements the `Resolve` trait.
#[derive(Debug, Default, Clone)]
pub(crate) struct HickoryDnsResolver {
    /// Since we use AsyncResolver directly without tokio runtime,
    /// we delay the actual construction of the resolver.
    state: Arc<OnceCell<AsyncResolver>>,
}

struct SocketAddrs {
    iter: LookupIpIntoIter,
}

#[derive(Debug)]
struct HickoryDnsSystemConfError(ResolveError);

impl Resolve for HickoryDnsResolver {
    fn resolve(&self, name: Name) -> AsyncStream<Result<Addrs, BoxError>> {
        let resolver = self.clone();
        let hostname = name.as_str().to_string();
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Addrs, hickory_resolver::error::ResolveError> {
                // Initialize resolver if needed
                let resolver_instance = resolver.state.get_or_try_init(new_resolver)?;
                
                // Use synchronous polling pattern instead of tokio runtime
                // Create a minimal future executor using std::task components
                use std::task::{Context, Poll, Waker};
                use std::future::Future;
                use std::pin::Pin;
                
                // Create a dummy waker for polling
                let raw_waker = std::task::RawWaker::new(
                    std::ptr::null(),
                    &std::task::RawWakerVTable::new(
                        |_| std::task::RawWaker::new(std::ptr::null(), &std::task::RawWakerVTable::new(|_| panic!(), |_| panic!(), |_| panic!(), |_| panic!())),
                        |_| {},
                        |_| {},
                        |_| {},
                    )
                );
                let waker = unsafe { Waker::from_raw(raw_waker) };
                let mut context = Context::from_waker(&waker);
                
                // Convert async lookup to synchronous polling
                let mut future = Box::pin(resolver_instance.lookup_ip(hostname.as_str()));
                
                // Poll the future until completion with sleep intervals
                loop {
                    match future.as_mut().poll(&mut context) {
                        Poll::Ready(result) => {
                            let lookup = result?;
                            let addrs: Addrs = Box::new(SocketAddrs {
                                iter: lookup.into_iter(),
                            });
                            return Ok(addrs);
                        }
                        Poll::Pending => {
                            // Sleep and try again - this is a blocking operation
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                    }
                }
            });
            
            match task.collect() {
                Ok(addrs) => emit!(sender, Ok(addrs)),
                Err(e) => handle_error!(e, "hickory DNS lookup"),
            }
        })
    }
}

impl Iterator for SocketAddrs {
    type Item = SocketAddr;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|ip_addr| SocketAddr::new(ip_addr, 0))
    }
}

/// Create a new resolver with the default configuration,
/// which reads from `/etc/resolve.conf`. The options are
/// overridden to look up for both IPv4 and IPv6 addresses
/// to work with "happy eyeballs" algorithm.
fn new_resolver() -> Result<AsyncResolver, HickoryDnsSystemConfError> {
    let mut builder = AsyncResolver::builder().map_err(HickoryDnsSystemConfError)?;
    builder.options_mut().ip_strategy = LookupIpStrategy::Ipv4AndIpv6;
    Ok(builder.build())
}

impl fmt::Display for HickoryDnsSystemConfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("error reading DNS system conf for hickory-dns")
    }
}

impl std::error::Error for HickoryDnsSystemConfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
