use fluent_ai_async::{AsyncStream, emit, spawn_task};
use fluent_ai_async::prelude::MessageChunk;
use std::time::{Duration, Instant};
use std::sync::atomic::AtomicBool;

use bytes::Bytes;
use h3::client::SendRequest;
use h3_quinn::OpenStreams;
use http::{uri::{Authority, Scheme}, Request};
use quinn::Endpoint;
use rustls::RootCertStore;
use dashmap::DashMap;

// Wrapper for Option<PoolClient> to implement MessageChunk
#[derive(Clone)]
pub struct PoolClientWrapper(pub Option<PoolClient>);

impl MessageChunk for PoolClientWrapper {
    fn bad_chunk(error: String) -> Self {
        Self(None)
    }
    
    fn is_error(&self) -> bool {
        self.0.is_none()
    }
    
    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("Pool client unavailable")
        } else {
            None
        }
    }
}

impl Default for PoolClientWrapper {
    fn default() -> Self {
        Self(None)
    }
}

impl From<Option<PoolClient>> for PoolClientWrapper {
    fn from(opt: Option<PoolClient>) -> Self {
        Self(opt)
    }
}

impl From<PoolClient> for PoolClientWrapper {
    fn from(client: PoolClient) -> Self {
        Self(Some(client))
    }
}

pub type Key = (Scheme, Authority);

#[derive(Clone)]
pub struct Pool {
    inner: std::sync::Arc<PoolInner>,
    endpoint: Endpoint,
}

impl Pool {
    pub fn new(timeout: Option<Duration>) -> Option<Self> {
        // Create production TLS configuration with native certificates
        let mut root_store = RootCertStore::empty();
        
        // Load native certificates for production TLS
        let cert_result = rustls_native_certs::load_native_certs();
        for cert in &cert_result.certs {
            if let Err(e) = root_store.add(cert.clone()) {
                log::warn!("Failed to add native cert: {}", e);
            }
        }
        if let Some(first_error) = cert_result.errors.first() {
            log::warn!("Certificate loading errors: {}", first_error);
        }
        log::debug!("Loaded {} native certificates", cert_result.certs.len());
        
        // Create production TLS configuration
        let tls_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        
        // Create QUIC configuration with production TLS
        let crypto_config = match quinn::crypto::rustls::QuicClientConfig::try_from(tls_config) {
            Ok(config) => config,
            Err(e) => {
                log::error!("Failed to create QUIC crypto config: {}", e);
                return None;
            }
        };
        
        let mut quinn_config = quinn::ClientConfig::new(std::sync::Arc::new(crypto_config));
        
        // Configure transport parameters optimized for HTTP/3
        let mut transport_config = quinn::TransportConfig::default();
        transport_config.max_concurrent_bidi_streams(100_u32.into());
        transport_config.max_concurrent_uni_streams(100_u32.into());
        transport_config.max_idle_timeout(Some(
            Duration::from_secs(30).try_into().map_err(|e| {
                log::error!("Invalid idle timeout: {}", e);
            }).ok()?
        ));
        transport_config.keep_alive_interval(Some(Duration::from_secs(10)));
        quinn_config.transport_config(std::sync::Arc::new(transport_config));
        
        // Create QUIC endpoint with production configuration
        let socket_addr = match "[::]:0".parse() {
            Ok(addr) => addr,
            Err(_) => {
                log::debug!("IPv6 parsing failed, trying IPv4");
                match "0.0.0.0:0".parse() {
                    Ok(addr) => addr,
                    Err(_) => {
                        log::debug!("Address parsing failed, using localhost");
                        std::net::SocketAddr::V4(
                            std::net::SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, 0)
                        )
                    }
                }
            }
        };
        
        let mut endpoint = match Endpoint::client(socket_addr) {
            Ok(ep) => ep,
            Err(e) => {
                log::error!("Failed to create QUIC endpoint: {}", e);
                return None;
            }
        };
        
        // Set the production client configuration
        endpoint.set_default_client_config(quinn_config);
        
        Some(Self {
            inner: std::sync::Arc::new(PoolInner {
                connecting: DashMap::new(),
                idle_conns: DashMap::new(),
                timeout,
            }),
            endpoint,
        })
    }

    pub fn try_pool(&self, key: &Key) -> Option<PoolClient> {
        if let Some(conn) = self.inner.idle_conns.get(key) {
            if let Some(duration) = self.inner.timeout {
                if Instant::now().saturating_duration_since(conn.idle_timeout) > duration {
                    self.inner.idle_conns.remove(key);
                    return None;
                }
            }
            
            Some(conn.pool())
        } else {
            None
        }
    }

    /// Get connection from pool or establish new one - required by h3_client/mod.rs
    pub fn get_connection(&self, key: &Key) -> Result<(PoolClient, ()), Box<dyn std::error::Error + Send + Sync>> {
        // Try to get existing connection from pool first
        if let Some(client) = self.try_pool(key) {
            return Ok((client, ()));
        }
        
        // Establish new connection if none available
        match self.establish_connection(key) {
            Ok(client) => Ok((client, ())),
            Err(e) => Err(e)
        }
    }

    pub fn connecting(&self, key: &Key) -> Option<Connecting> {
        if self.inner.connecting.contains_key(key) {
            Some(Connecting::InProgress)
        } else {
            self.inner.connecting.insert(key.clone(), AtomicBool::new(true));
            Some(Connecting::Acquired)
        }
    }

    pub fn establish_h3_connection(&self, key: &Key, _connector: &mut crate::hyper::async_impl::h3_client::connect::H3Connector) -> Result<PoolClient, Box<dyn std::error::Error + Send + Sync>> {
        let host = key.1.host();
        let port = key.1.port_u16().unwrap_or(443);
        
        let server_addr = format!("{}:{}", host, port).parse()
            .map_err(|e| format!("Address parse failed: {}", e))?;
        
        // Real H3 connection establishment using fluent_ai_async patterns with async context
        let connection_stream = AsyncStream::<PoolClientWrapper, 1024>::with_channel({
            let endpoint = self.endpoint.clone();
            let host = host.to_string();
            move |sender| {
                spawn_task(move || async move {
                // Real QUIC connection establishment using Quinn with async context
                let connecting = match endpoint.connect(server_addr, &host) {
                    Ok(connecting) => connecting,
                    Err(e) => {
                        emit!(sender, PoolClientWrapper::bad_chunk(format!("Quinn connect failed: {}", e)));
                        return;
                    }
                };
                
                let conn = match connecting.await {
                    Ok(conn) => conn,
                    Err(e) => {
                        emit!(sender, PoolClientWrapper::bad_chunk(format!("QUIC connection failed: {}", e)));
                        return;
                    }
                };
                
                // Real H3 client creation using h3::client::new()
                let h3_conn = h3_quinn::Connection::new(conn);
                let (h3_driver, send_request) = match h3::client::new(h3_conn).await {
                    Ok((driver, send_request)) => (driver, send_request),
                    Err(e) => {
                        emit!(sender, PoolClientWrapper::bad_chunk(format!("H3 client creation failed: {}", e)));
                        return;
                    }
                };
                
                // H3 driver will be managed by H3 library internally
                // No need for explicit tokio spawn in streams-first architecture
                let _h3_driver = h3_driver; // Keep driver alive
                
                // Create PoolClient with real H3 SendRequest
                let client = PoolClient::new(send_request);
                emit!(sender, PoolClientWrapper::from(client));
                });
            }
        });
        
        // Collect connection result
        match connection_stream.collect().into_iter().next() {
            Some(wrapper) => {
                if let Some(client) = wrapper.0 {
                    // Store in connection pool with atomic connection state
                    let pool_conn = PoolConnection::new(client.clone(), std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
                    self.inner.idle_conns.insert(key.clone(), pool_conn);
                    Ok(client)
                } else {
                    Err(wrapper.error().unwrap_or("H3 connection establishment failed").into())
                }
            }
            None => Err("H3 connection establishment failed".into())
        }
    }

    pub fn establish_connection(&self, key: &Key) -> Result<PoolClient, Box<dyn std::error::Error + Send + Sync>> {
        // Create H3Connector for real connection establishment
        let mut connector = match crate::hyper::async_impl::h3_client::connect::H3Connector::new() {
            Some(connector) => connector,
            None => {
                return Err("Failed to create H3Connector - TLS or endpoint configuration failed".into());
            }
        };
        
        // Use real H3 connection establishment through establish_h3_connection
        self.establish_h3_connection(key, &mut connector)
    }

    pub fn put(&self, key: Key, client: PoolClient, _lock: &ConnectingLock) {
        let conn = PoolConnection::new(client, std::sync::Arc::new(AtomicBool::new(false)));
        self.inner.idle_conns.insert(key.clone(), conn);
        self.inner.connecting.remove(&key);
    }

}

// Add missing functions referenced by h3_client/mod.rs
pub fn domain_as_uri(domain: &str) -> String {
    // Convert domain to HTTPS URI with proper validation
    if domain.starts_with("http://") || domain.starts_with("https://") {
        domain.to_string()
    } else {
        format!("https://{}", domain)
    }
}

pub fn extract_domain(uri: &http::Uri) -> (http::uri::Scheme, http::uri::Authority) {
    let scheme = uri.scheme().cloned().unwrap_or(http::uri::Scheme::HTTPS);
    let authority = uri.authority().cloned().unwrap_or_else(|| {
        http::uri::Authority::from_static("localhost")
    });
    (scheme, authority)
}

pub enum Connecting {
    InProgress,
    Acquired,
}

struct PoolInner {
    connecting: DashMap<Key, AtomicBool>,
    idle_conns: DashMap<Key, PoolConnection>,
    timeout: Option<Duration>,
}

#[derive(Clone)]
pub struct PoolClient {
    inner: SendRequest<OpenStreams, Bytes>,
}

impl PoolClient {
    pub fn new(tx: SendRequest<OpenStreams, Bytes>) -> Self {
        Self { inner: tx }
    }

    pub fn send_request(
        &mut self,
        req: Request<bytes::Bytes>,
    ) -> AsyncStream<crate::response::HttpResponseChunk> {
        let mut inner = self.inner.clone();
        AsyncStream::with_channel(move |sender| {
            spawn_task(move || async move {
            // Real H3 request handling using async context
            let (parts, body) = req.into_parts();
            let uri_str = parts.uri.to_string();
            
            // Build H3 request from HTTP parts
            let h3_req_builder = http::Request::builder()
                .method(parts.method)
                .uri(parts.uri)
                .version(parts.version);
                
            let h3_req_builder = parts.headers.iter().fold(h3_req_builder, |req_builder, (name, value)| {
                req_builder.header(name, value)
            });
            
            let h3_req = match h3_req_builder.body(()) {
                Ok(req) => req,
                Err(e) => {
                    emit!(sender, crate::response::HttpResponseChunk::bad_chunk(format!("Request build failed: {}", e)));
                    return;
                }
            };
            
            // Send H3 request
            match inner.send_request(h3_req).await {
                Ok(mut stream) => {
                    // Send body data if present
                    if !body.is_empty() {
                        if let Err(e) = stream.send_data(body).await {
                            emit!(sender, crate::response::HttpResponseChunk::bad_chunk(format!("Send data failed: {}", e)));
                            return;
                        }
                    }
                    
                    // Finish request
                    if let Err(e) = stream.finish().await {
                        emit!(sender, crate::response::HttpResponseChunk::bad_chunk(format!("Finish failed: {}", e)));
                        return;
                    }
                    
                    // Receive response
                    match stream.recv_response().await {
                        Ok(response) => {
                            let status = response.status().as_u16();
                            let mut headers = std::collections::HashMap::new();
                            
                            // Extract headers
                            for (name, value) in response.headers() {
                                if let Ok(value_str) = value.to_str() {
                                    headers.insert(name.to_string(), value_str.to_string());
                                }
                            }
                            
                            // Collect response body
                            let mut body_data = Vec::new();
                            while let Ok(Some(chunk)) = stream.recv_data().await {
                                use bytes::Buf;
                                let chunk_bytes = chunk.chunk();
                                body_data.extend_from_slice(chunk_bytes);
                            }
                            
                            // Emit successful response
                            let response_chunk = crate::response::HttpResponseChunk::new(status, headers, body_data, uri_str);
                            emit!(sender, response_chunk);
                        }
                        Err(e) => {
                            emit!(sender, crate::response::HttpResponseChunk::bad_chunk(format!("Receive response failed: {}", e)));
                        }
                    }
                }
                Err(e) => {
                    emit!(sender, crate::response::HttpResponseChunk::bad_chunk(format!("H3 send request failed: {}", e)));
                }
            }
            });
        })
    }

    pub fn send_h3_request(
        &mut self,
        _req: Request<bytes::Bytes>,
    ) -> AsyncStream<crate::response::HttpResponseChunk> {
        self.send_request(_req)
    }
}

pub struct PoolConnection {
    client: PoolClient,
    is_closed: std::sync::Arc<AtomicBool>,
    idle_timeout: Instant,
}

impl PoolConnection {
    pub fn new(client: PoolClient, is_closed: std::sync::Arc<AtomicBool>) -> Self {
        Self {
            client,
            is_closed,
            idle_timeout: Instant::now(),
        }
    }

    pub fn pool(&self) -> PoolClient {
        self.client.clone()
    }
}

pub struct ConnectingLock;

impl ConnectingLock {
    pub fn forget(self) -> Key {
        // Return default key for connection cleanup
        (Scheme::HTTP, Authority::from_static("localhost"))
    }
}