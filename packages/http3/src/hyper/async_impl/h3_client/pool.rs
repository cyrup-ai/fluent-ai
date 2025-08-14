use std::collections::HashMap;
use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
use std::sync::mpsc::{Receiver, TryRecvError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Buf;
use bytes::Bytes;
use h3::client::SendRequest;
use h3_quinn::{Connection, OpenStreams};
use http::uri::{Authority, Scheme};
use http::{Request, Response, Uri};
use log::trace;

use super::super::body::ResponseBody;
use crate::Body;
use crate::hyper::error::{BoxError, Error, Kind};

pub(super) type Key = (Scheme, Authority);

#[derive(Clone)]
pub struct Pool {
    inner: Arc<Mutex<PoolInner>>,
}

struct ConnectingLockInner {
    key: Key,
    pool: Arc<Mutex<PoolInner>>,
}

/// A lock that ensures only one HTTP/3 connection is established per host at a
/// time. The lock is automatically released when dropped.
pub struct ConnectingLock(Option<ConnectingLockInner>);

/// A waiter that allows subscribers to receive updates when a new connection is
/// established or when the connection attempt fails. For example, when
/// connection lock is dropped due to an error.
pub struct ConnectingWaiter {
    receiver: std::sync::mpsc::Receiver<Option<PoolClient>>,
}

pub enum Connecting {
    /// A connection attempt is already in progress.
    /// You must subscribe to updates instead of initiating a new connection.
    InProgress(ConnectingWaiter),
    /// The connection lock has been acquired, allowing you to initiate a
    /// new connection.
    Acquired(ConnectingLock),
}

impl ConnectingLock {
    fn new(key: Key, pool: Arc<Mutex<PoolInner>>) -> Self {
        Self(Some(ConnectingLockInner { key, pool }))
    }

    /// Forget the lock and return corresponding Key
    fn forget(mut self) -> Key {
        // Option is guaranteed to be Some until dropped
        self.0.take().expect("ConnectingLock should not be used after drop").key
    }
}

impl Drop for ConnectingLock {
    fn drop(&mut self) {
        if let Some(ConnectingLockInner { key, pool }) = self.0.take() {
            let mut pool = pool.lock().expect("pool mutex should not be poisoned");
            pool.connecting.remove(&key);
            trace!("HTTP/3 connecting lock for {:?} is dropped", key);
        }
    }
}

impl ConnectingWaiter {
    pub fn receive(mut self) -> AsyncStream<Option<PoolClient>> {
        use fluent_ai_async::{AsyncStream, emit, handle_error, spawn_task};
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                // Use blocking receive on std::sync::mpsc instead of watch channel
                let start = std::time::Instant::now();
                let timeout = std::time::Duration::from_secs(30); // 30 second timeout
                
                loop {
                    match self.receiver.try_recv() {
                        Ok(client) => {
                            return Ok::<Option<PoolClient>, Box<dyn std::error::Error + Send + Sync>>(client);
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            // Check timeout
                            if start.elapsed() > timeout {
                                return Ok::<Option<PoolClient>, Box<dyn std::error::Error + Send + Sync>>(None);
                            }
                            // Sleep a bit and try again
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            return Ok::<Option<PoolClient>, Box<dyn std::error::Error + Send + Sync>>(None);
                        }
                    }
                }
            });
            
            match task.collect() {
                Ok(client) => emit!(sender, client),
                Err(e) => handle_error!(e, "connection waiting"),
            }
        })
    }
}

impl Pool {
    pub fn new(timeout: Option<Duration>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolInner {
                connecting: HashMap::new(),
                idle_conns: HashMap::new(),
                timeout,
            })),
        }
    }

    /// Acquire a connecting lock. This is to ensure that we have only one HTTP3
    /// connection per host.
    pub fn connecting(&self, key: &Key) -> Connecting {
        let mut inner = self.inner.lock().expect("pool mutex should not be poisoned");

        if let Some(_senders) = inner.connecting.get(key) {
            // Create a new receiver for this waiter
            let (tx, rx) = std::sync::mpsc::channel();
            // Add this sender to the list for this key
            inner.connecting.get_mut(key).unwrap().push(tx);
            
            Connecting::InProgress(ConnectingWaiter {
                receiver: rx,
            })
        } else {
            // Start new connection attempt - initialize empty sender list
            inner.connecting.insert(key.clone(), Vec::new());
            Connecting::Acquired(ConnectingLock::new(key.clone(), Arc::clone(&self.inner)))
        }
    }

    pub fn try_pool(&self, key: &Key) -> Option<PoolClient> {
        let mut inner = self.inner.lock().expect("pool mutex should not be poisoned");
        let timeout = inner.timeout;
        if let Some(conn) = inner.idle_conns.get(&key) {
            // We check first if the connection still valid
            // and if not, we remove it from the pool.
            if conn.is_invalid() {
                trace!("pooled HTTP/3 connection is invalid so removing it...");
                inner.idle_conns.remove(&key);
                return None;
            }

            if let Some(duration) = timeout {
                if Instant::now().saturating_duration_since(conn.idle_timeout) > duration {
                    trace!("pooled connection expired");
                    return None;
                }
            }
        }

        inner
            .idle_conns
            .get_mut(&key)
            .and_then(|conn| Some(conn.pool()))
    }

    pub fn new_connection(
        &mut self,
        lock: ConnectingLock,
        mut driver: h3::client::Connection<Connection, Bytes>,
        tx: SendRequest<OpenStreams, Bytes>,
    ) -> PoolClient {
        let (close_tx, close_rx) = std::sync::mpsc::channel();
        {
            use fluent_ai_async::spawn_task;
            let _task = spawn_task(move || {
                // Use safe synchronous pattern with no unsafe waker creation
                use std::task::{Context, Poll, Waker};
                use std::thread;
                use std::time::Duration;
                
                // Create safe noop waker - no unsafe operations needed
                let waker = Waker::noop();
                let mut context = Context::from_waker(&waker);
                
                // Poll in a loop with sleeps instead of async await
                loop {
                    match driver.poll_close(&mut context) {
                        Poll::Ready(e) => {
                            trace!("poll_close returned error {e:?}");
                            close_tx.send(e).ok();
                            break;
                        }
                        Poll::Pending => {
                            thread::sleep(Duration::from_millis(10));
                        }
                    }
                }
            });
        }

        let mut inner = self.inner.lock().expect("pool mutex should not be poisoned");

        // We clean up "connecting" here so we don't have to acquire the lock again.
        let key = lock.forget();
        let Some(senders) = inner.connecting.remove(&key) else {
            unreachable!("there should be one connecting lock at a time");
        };
        let client = PoolClient::new(tx);

        // Send the client to all our awaiters
        for sender in senders {
            let _ = sender.send(Some(client.clone()));
        }

        let conn = PoolConnection::new(client.clone(), close_rx);
        inner.insert(key, conn);

        client
    }
}

struct PoolInner {
    connecting: HashMap<Key, Vec<std::sync::mpsc::Sender<Option<PoolClient>>>>,
    idle_conns: HashMap<Key, PoolConnection>,
    timeout: Option<Duration>,
}

impl PoolInner {
    fn insert(&mut self, key: Key, conn: PoolConnection) {
        if self.idle_conns.contains_key(&key) {
            trace!("connection already exists for key {key:?}");
        }

        self.idle_conns.insert(key, conn);
    }
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
        req: Request<Body>,
    ) -> AsyncStream<Result<Response<ResponseBody>, BoxError>> {
        use fluent_ai_async::{AsyncStream, emit};
        
        let mut inner = self.inner.clone();
        
        AsyncStream::with_channel(move |sender| {
            use fluent_ai_async::spawn_task;
            
            let task = spawn_task(move || {
                // Use pure synchronous pattern instead of tokio runtime
                // Convert the async operations to synchronous polling patterns
                    use hyper::body::Body as _;

                    let (head, mut req_body) = req.into_parts();
                    let mut req = Request::from_parts(head, ());

                    if let Some(n) = req_body.size_hint().exact() {
                        if n > 0 {
                            req.headers_mut()
                                .insert(http::header::CONTENT_LENGTH, n.into());
                        }
                    }

                    // For now, return an error indicating this needs more sophisticated conversion
                    // The async h3 operations require significant reworking to be truly synchronous
                    // This is a complex conversion that may require changes to the h3 library usage
                    Err("HTTP/3 async operations not yet converted to synchronous patterns - requires more work".into())
            });
            
            match task.collect() {
                Ok(response) => emit!(sender, Ok(response)),
                Err(e) => emit!(sender, Err(e.into())),
            }
        })
    }
}

pub struct PoolConnection {
    // This receives errors from polling h3 driver.
    close_rx: Receiver<h3::error::ConnectionError>,
    client: PoolClient,
    idle_timeout: Instant,
}

impl PoolConnection {
    pub fn new(client: PoolClient, close_rx: Receiver<h3::error::ConnectionError>) -> Self {
        Self {
            close_rx,
            client,
            idle_timeout: Instant::now(),
        }
    }

    pub fn pool(&mut self) -> PoolClient {
        self.idle_timeout = Instant::now();
        self.client.clone()
    }

    pub fn is_invalid(&self) -> bool {
        match self.close_rx.try_recv() {
            Err(TryRecvError::Empty) => false,
            Err(TryRecvError::Disconnected) => true,
            Ok(_) => true,
        }
    }
}

struct Incoming<S, B> {
    inner: h3::client::RequestStream<S, B>,
    content_length: Option<u64>,
    send_rx: std::sync::mpsc::Receiver<Result<(), BoxError>>,
}

impl<S, B> Incoming<S, B> {
    fn new(
        stream: h3::client::RequestStream<S, B>,
        headers: &http::header::HeaderMap,
        send_rx: std::sync::mpsc::Receiver<Result<(), BoxError>>,
    ) -> Self {
        Self {
            inner: stream,
            content_length: headers
                .get(http::header::CONTENT_LENGTH)
                .and_then(|h| h.to_str().ok())
                .and_then(|v| v.parse().ok()),
            send_rx,
        }
    }
}

impl<S, B> http_body::Body for Incoming<S, B>
where
    S: h3::quic::RecvStream,
{
    type Data = Bytes;
    type Error = crate::hyper::error::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        if let Ok(Err(e)) = self.send_rx.try_recv() {
            return Poll::Ready(Some(Err(crate::error::body(e))));
        }

        // Convert async polling to synchronous pattern for pure AsyncStream architecture
        // This would need significant rework to properly integrate with h3 library
        match std::task::Poll::Pending {
            Ok(Some(mut b)) => Poll::Ready(Some(Ok(hyper::body::Frame::data(
                b.copy_to_bytes(b.remaining()),
            )))),
            Ok(None) => Poll::Ready(None),
            Err(e) => Poll::Ready(Some(Err(crate::error::body(e)))),
        }
    }

    fn size_hint(&self) -> hyper::body::SizeHint {
        if let Some(content_length) = self.content_length {
            hyper::body::SizeHint::with_exact(content_length)
        } else {
            hyper::body::SizeHint::default()
        }
    }
}

pub(crate) fn extract_domain(uri: &mut Uri) -> Result<Key, Error> {
    let uri_clone = uri.clone();
    match (uri_clone.scheme(), uri_clone.authority()) {
        (Some(scheme), Some(auth)) => Ok((scheme.clone(), auth.clone())),
        _ => Err(Error::new(Kind::Request, None::<Error>)),
    }
}

pub(crate) fn domain_as_uri((scheme, auth): Key) -> Uri {
    http::uri::Builder::new()
        .scheme(scheme)
        .authority(auth)
        .path_and_query("/")
        .build()
        .expect("domain is valid Uri")
}
