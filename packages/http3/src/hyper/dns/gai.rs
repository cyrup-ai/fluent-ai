use crate::hyper::dns::resolve::{Resolve, Name, DnsResult};
use fluent_ai_async::{AsyncStream, emit, spawn_task};

#[derive(Debug)]
pub struct GaiResolver {
    // Pure implementation without hyper_util dependency
}

impl GaiResolver {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for GaiResolver {
    fn default() -> Self {
        GaiResolver::new()
    }
}

impl Resolve for GaiResolver {
    fn resolve(&self, name: Name) -> AsyncStream<DnsResult> {
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                // Use synchronous DNS resolution via ToSocketAddrs
                let host_str = format!("{}", name);
                match std::net::ToSocketAddrs::to_socket_addrs(&host_str.as_str()) {
                    Ok(addrs_iter) => {
                        let addrs: arrayvec::ArrayVec<std::net::SocketAddr, 8> = 
                            addrs_iter.take(8).collect();
                        emit!(sender, DnsResult { addrs });
                    },
                    Err(_e) => {
                        emit!(sender, DnsResult::new()); // Empty result for error case
                    }
                }
            });
        })
    }
}
