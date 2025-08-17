use std::thread;
use fluent_ai_async::{AsyncStream, emit};

use super::{Addrs, Name, Resolve};
use crate::error::HttpError;

struct GaiAddrs {
    addrs: std::vec::IntoIter<std::net::SocketAddr>,
}

impl Iterator for GaiAddrs {
    type Item = std::net::SocketAddr;

    fn next(&mut self) -> Option<Self::Item> {
        self.addrs.next()
    }
}

#[derive(Debug)]
pub struct GaiResolver {
    // Pure implementation without external dependencies
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
    fn resolve(&self, name: Name) -> AsyncStream<Result<Addrs, HttpError>, 1024> {
        let hostname = name.as_str().to_string();
        
        AsyncStream::with_channel(move |sender| {
            thread::spawn(move || {
                // Use synchronous DNS resolution via ToSocketAddrs
                let host_port = format!("{}:0", hostname);
                match std::net::ToSocketAddrs::to_socket_addrs(&host_port) {
                    Ok(addrs_iter) => {
                        let socket_addrs: Vec<std::net::SocketAddr> = addrs_iter.collect();
                        if socket_addrs.is_empty() {
                            emit!(sender, Err(HttpError::DnsError(format!("No addresses found for {}", hostname))));
                        } else {
                            let addrs: Addrs = Box::new(GaiAddrs {
                                addrs: socket_addrs.into_iter(),
                            });
                            emit!(sender, Ok(addrs));
                        }
                    }
                    Err(e) => {
                        emit!(sender, Err(HttpError::DnsError(format!("GAI resolution failed for {}: {}", hostname, e))));
                    }
                }
            });
        })
    }
}
