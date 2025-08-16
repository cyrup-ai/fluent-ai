use std::net::SocketAddr;

use fluent_ai_async::prelude::MessageChunk;
use fluent_ai_async::{AsyncStream, emit, spawn_task};

// Simplified Name type for H3 DNS resolution - removing hyper_util dependency
#[derive(Debug, Clone)]
pub struct Name(String);

impl Name {
    pub fn from_str(name: &str) -> Result<Self, &'static str> {
        if name.is_empty() {
            Err("Name cannot be empty")
        } else {
            Ok(Name(name.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// Wrapper type for socket address lists to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct SocketAddrListWrapper(pub Vec<SocketAddr>);

impl MessageChunk for SocketAddrListWrapper {
    fn bad_chunk(error: String) -> Self {
        // Return empty list for error case
        Self(Vec::new())
    }

    fn is_error(&self) -> bool {
        self.0.is_empty()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("DNS resolution failed")
        } else {
            None
        }
    }
}

// Trait for DNS resolution using pure AsyncStream architecture for HTTP/3 client.
// Completely eliminates Future/Service abstractions for streams-first design.
pub trait Resolve {
    /// Resolve a name to socket addresses using pure AsyncStream
    /// Returns unwrapped stream - no Result wrapping per async-stream architecture
    fn resolve(&mut self, name: Name) -> AsyncStream<SocketAddrListWrapper>;
}

/// Standard DNS resolver implementation for H3 client
#[derive(Clone)]
pub struct StandardResolver;

impl StandardResolver {
    pub fn new() -> Self {
        StandardResolver
    }
}

impl Resolve for StandardResolver {
    fn resolve(&mut self, name: Name) -> AsyncStream<SocketAddrListWrapper> {
        let hostname = name.as_str().to_string();

        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || -> Vec<SocketAddr> {
                // Use std::net::ToSocketAddrs for synchronous resolution
                let dummy_port = 80; // Port will be set by caller
                let host_with_port = format!("{}:{}", hostname, dummy_port);

                match host_with_port.parse::<SocketAddr>() {
                    Ok(addr) => vec![addr],
                    Err(_) => {
                        // Try DNS resolution
                        match std::net::ToSocketAddrs::to_socket_addrs(&host_with_port) {
                            Ok(addrs) => addrs.collect(),
                            Err(_) => {
                                Vec::new() // Return empty vec on error
                            }
                        }
                    }
                }
            });

            match task.collect() {
                Ok(addrs) => emit!(sender, SocketAddrListWrapper(addrs)),
                Err(e) => {
                    emit!(
                        sender,
                        SocketAddrListWrapper::bad_chunk(format!("DNS task error: {}", e))
                    );
                }
            }
        })
    }
}

/// Pure AsyncStream DNS resolution function - zero allocation streaming architecture
/// Eliminates all Future abstractions for streams-first pattern
pub(super) fn resolve<R>(mut resolver: R, name: Name) -> AsyncStream<SocketAddrVecWrapper>
where
    R: Resolve + Send + 'static,
{
    use fluent_ai_async::{AsyncStream, emit};

    AsyncStream::with_channel(move |sender| {
        let wrapper_stream = resolver.resolve(name);

        // Convert SocketAddrListWrapper to SocketAddrVecWrapper
        for wrapper in wrapper_stream.collect() {
            emit!(sender, SocketAddrVecWrapper(wrapper.0));
        }
    })
}

// Wrapper for Vec<SocketAddr> to implement MessageChunk for direct use
#[derive(Debug, Clone)]
pub struct SocketAddrVecWrapper(pub Vec<SocketAddr>);

impl MessageChunk for SocketAddrVecWrapper {
    fn bad_chunk(error: String) -> Self {
        Self(Vec::new())
    }

    fn is_error(&self) -> bool {
        self.0.is_empty()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("DNS resolution failed")
        } else {
            None
        }
    }
}

impl Default for SocketAddrVecWrapper {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl From<Vec<SocketAddr>> for SocketAddrVecWrapper {
    fn from(addrs: Vec<SocketAddr>) -> Self {
        Self(addrs)
    }
}
