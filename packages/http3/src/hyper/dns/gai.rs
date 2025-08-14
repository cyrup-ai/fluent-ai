use crate::hyper::dns::resolve::{Addrs, Resolve, Name};
use crate::hyper::error::BoxError;
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
    fn resolve(&self, name: Name) -> AsyncStream<Result<Addrs, BoxError>> {
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Result<Addrs, BoxError> {
                // Use synchronous DNS resolution via ToSocketAddrs
                let host_str = format!("{}", name);
                std::net::ToSocketAddrs::to_socket_addrs(&host_str.as_str())
                    .map_err(|err| -> BoxError { Box::new(err) })
                    .map(|addrs_iter| addrs_iter.collect::<Vec<_>>().into_iter())
            });
            
            match task.collect() {
                Ok(addrs) => emit!(sender, Ok(addrs)),
                Err(e) => emit!(sender, Err(e)),
            }
        })
    }
}
