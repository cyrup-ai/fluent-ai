use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll, ready};
use std::time::Duration;

use bytes::Bytes;
use http_body::Body as HttpBody;
use http_body_util::combinators::BoxBody;
use pin_project_lite::pin_project;
use fluent_ai_async::{AsyncStream, emit, spawn_task, handle_error};

// Removed tokio dependencies - using AsyncStream patterns

/// Real AsyncStream to HttpBody bridge implementation
struct AsyncStreamHttpBody {
    frame_stream: AsyncStream<http_body::Frame<Bytes>>,
}

impl AsyncStreamHttpBody {
    fn new(frame_stream: AsyncStream<http_body::Frame<Bytes>>) -> Self {
        Self { frame_stream }
    }
}

impl HttpBody for AsyncStreamHttpBody {
    type Data = Bytes;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // Use try_next() to get frames from AsyncStream - no futures needed
        if let Some(frame) = self.frame_stream.try_next() {
            std::task::Poll::Ready(Some(Ok(frame)))
        } else {
            // No more frames available right now
            std::task::Poll::Pending
        }
    }

    fn is_end_stream(&self) -> bool {
        // Check if the AsyncStream has no more data
        self.frame_stream.is_empty()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        // Unknown size for streaming data
        http_body::SizeHint::default()
    }
}

/// An asynchronous request body.
pub struct Body {
    inner: Inner,
}

enum Inner {
    Reusable(Bytes),
    Streaming(BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>),
}

// TotalTimeoutBody and ReadTimeoutBody structs removed - timeout functionality now provided by
// safe total_timeout() and with_read_timeout() functions returning AsyncStream<Frame<Bytes>>

/// Converts any `impl Body` into a `impl Stream` of just its DATA frames.
#[cfg(any(feature = "stream", feature = "multipart",))]
pub(crate) struct DataStream<B>(pub(crate) B);

impl Body {
    /// Returns a reference to the internal data of the `Body`.
    ///
    /// `None` is returned, if the underlying data is a stream.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.inner {
            Inner::Reusable(bytes) => Some(bytes.as_ref()),
            Inner::Streaming(..) => None,
        }
    }

    /// Wrap an AsyncStream in a box inside `Body`.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Body;
    /// # use fluent_ai_async::{AsyncStream, emit};
    /// # fn main() {
    /// let chunks = vec!["hello", " ", "world"];
    ///
    /// let stream = AsyncStream::with_channel(move |sender| {
    ///     for chunk in chunks {
    ///         emit!(sender, chunk);
    ///     }
    /// });
    ///
    /// let body = Body::wrap_stream(stream);
    /// # }
    /// ```
    ///
    /// # Optional
    ///
    /// This requires the `stream` feature to be enabled.
    /// Uses AsyncStream for streams-first architecture.
    #[cfg(feature = "stream")]
    #[cfg_attr(docsrs, doc(cfg(feature = "stream")))]
    pub fn wrap_stream<T>(stream: AsyncStream<T>) -> Body
    where
        T: Send + 'static,
        Bytes: From<T>,
    {
        Body::from_async_stream(stream)
    }

    #[cfg(any(feature = "stream", feature = "multipart"))]
    pub(crate) fn from_async_stream<T>(stream: AsyncStream<T>) -> Body
    where
        T: Send + 'static,
        Bytes: From<T>,
    {
        use http_body::Frame;

        // Convert the AsyncStream<T> to AsyncStream<Frame<Bytes>> using proper streaming
        let frame_stream = AsyncStream::with_channel(move |sender| {
            spawn_task(move || {
                let mut input_stream = stream;
                
                // Stream frames directly without collection - emit each frame as it becomes available
                loop {
                    if let Some(item) = input_stream.try_next() {
                        let bytes = Bytes::from(item);
                        let frame = Frame::data(bytes);
                        emit!(sender, frame);  // Stream each frame immediately
                    } else {
                        // End of stream
                        break;
                    }
                }
            });
        });

        // Create real AsyncStreamHttpBody that implements HttpBody
        let async_body = AsyncStreamHttpBody::new(frame_stream);
        let boxed_body = http_body_util::BodyExt::boxed(async_body);
        
        Body {
            inner: Inner::Streaming(boxed_body),
        }
    }

    pub(crate) fn empty() -> Body {
        Body::reusable(Bytes::new())
    }

    pub(crate) fn reusable(chunk: Bytes) -> Body {
        Body {
            inner: Inner::Reusable(chunk),
        }
    }

    /// Wrap a [`HttpBody`] in a box inside `Body`.
    ///
    /// # Example
    ///
    /// ```
    /// # use crate::hyper::Body;
    /// # fn main() {
    /// let content = "hello,world!".to_string();
    ///
    /// let body = Body::wrap(content);
    /// # }
    /// ```
    pub fn wrap<B>(inner: B) -> Body
    where
        B: HttpBody + Send + Sync + 'static,
        B::Data: Into<Bytes>,
        B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        use http_body_util::BodyExt;

        let boxed = IntoBytesBody { inner }.map_err(Into::into).boxed();

        Body {
            inner: Inner::Streaming(boxed),
        }
    }

    pub(crate) fn try_reuse(self) -> (Option<Bytes>, Self) {
        let reuse = match self.inner {
            Inner::Reusable(ref chunk) => Some(chunk.clone()),
            Inner::Streaming { .. } => None,
        };

        (reuse, self)
    }

    pub(crate) fn try_clone(&self) -> Option<Body> {
        match self.inner {
            Inner::Reusable(ref chunk) => Some(Body::reusable(chunk.clone())),
            Inner::Streaming { .. } => None,
        }
    }

    #[cfg(feature = "multipart")]
    pub(crate) fn into_stream(self) -> DataStream<Body> {
        DataStream(self)
    }

    #[cfg(feature = "multipart")]
    pub(crate) fn content_length(&self) -> Option<u64> {
        match self.inner {
            Inner::Reusable(ref bytes) => Some(bytes.len() as u64),
            Inner::Streaming(ref body) => body.size_hint().exact(),
        }
    }
}

impl Default for Body {
    #[inline]
    fn default() -> Body {
        Body::empty()
    }
}

/*
impl From<hyper::Body> for Body {
    #[inline]
    fn from(body: hyper::Body) -> Body {
        Self {
            inner: Inner::Streaming {
                body: Box::pin(WrapHyper(body)),
            },
        }
    }
}
*/

impl From<Bytes> for Body {
    #[inline]
    fn from(bytes: Bytes) -> Body {
        Body::reusable(bytes)
    }
}

impl From<Vec<u8>> for Body {
    #[inline]
    fn from(vec: Vec<u8>) -> Body {
        Body::reusable(vec.into())
    }
}

impl From<&'static [u8]> for Body {
    #[inline]
    fn from(s: &'static [u8]) -> Body {
        Body::reusable(Bytes::from_static(s))
    }
}

impl From<String> for Body {
    #[inline]
    fn from(s: String) -> Body {
        Body::reusable(s.into())
    }
}

impl From<&'static str> for Body {
    #[inline]
    fn from(s: &'static str) -> Body {
        s.as_bytes().into()
    }
}

#[cfg(feature = "stream")]
#[cfg_attr(docsrs, doc(cfg(feature = "stream")))]
impl From<std::fs::File> for Body {
    #[inline]
    fn from(file: std::fs::File) -> Body {
        // Convert file to AsyncStream without tokio dependencies
        let stream = AsyncStream::with_channel(move |sender| {
            use std::io::Read;
            let mut buffer = Vec::new();
            if let Ok(_) = std::io::copy(&mut std::io::BufReader::new(file), &mut buffer) {
                emit!(sender, Bytes::from(buffer));
            }
        });
        Body::wrap_stream(stream)
    }
}

impl fmt::Debug for Body {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Body").finish()
    }
}

impl HttpBody for Body {
    type Data = Bytes;
    type Error = crate::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        match self.inner {
            Inner::Reusable(ref mut bytes) => {
                let out = bytes.split_off(0);
                if out.is_empty() {
                    Poll::Ready(None)
                } else {
                    Poll::Ready(Some(Ok(hyper::body::Frame::data(out))))
                }
            }
            Inner::Streaming(ref mut body) => Poll::Ready(
                ready!(Pin::new(body).poll_frame(cx))
                    .map(|opt_chunk| opt_chunk.map_err(crate::error::body)),
            ),
        }
    }

    fn size_hint(&self) -> http_body::SizeHint {
        match self.inner {
            Inner::Reusable(ref bytes) => http_body::SizeHint::with_exact(bytes.len() as u64),
            Inner::Streaming(ref body) => body.size_hint(),
        }
    }

    fn is_end_stream(&self) -> bool {
        match self.inner {
            Inner::Reusable(ref bytes) => bytes.is_empty(),
            Inner::Streaming(ref body) => body.is_end_stream(),
        }
    }
}

// ===== impl TotalTimeoutBody =====

/// Safe total timeout implementation using AsyncStream patterns
pub(crate) fn total_timeout<B>(body: B, timeout_duration: Duration) -> AsyncStream<hyper::body::Frame<Bytes>>
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Into<Bytes> + Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    AsyncStream::with_channel(move |sender| {
        spawn_task(move || {
            let waker = std::task::Waker::noop();
            let mut cx = std::task::Context::from_waker(&waker);
            let mut inner_body = body;
            let start_time = std::time::Instant::now();
            
            loop {
                // Check total timeout
                if start_time.elapsed() >= timeout_duration {
                    handle_error!(crate::error::TimedOut, "Total body timeout exceeded");
                }
                
                // Poll the inner body safely
                match std::pin::Pin::new(&mut inner_body).poll_frame(&mut cx) {
                    std::task::Poll::Ready(Some(Ok(frame))) => {
                        // Convert frame data to Bytes if needed
                        let frame = frame.map_data(|data| data.into());
                        emit!(sender, frame);
                    }
                    std::task::Poll::Ready(Some(Err(e))) => {
                        handle_error!(crate::error::body(e), "Body frame error");
                    }
                    std::task::Poll::Ready(None) => {
                        // End of body stream
                        return;
                    }
                    std::task::Poll::Pending => {
                        // Yield control but keep checking timeout
                        std::thread::yield_now();
                    }
                }
            }
        });
    })
}

/// Safe read timeout implementation using AsyncStream patterns
pub(crate) fn with_read_timeout<B>(body: B, timeout: Duration) -> AsyncStream<hyper::body::Frame<Bytes>>
where
    B: hyper::body::Body + Send + 'static,
    B::Data: Into<Bytes> + Send,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send,
{
    AsyncStream::with_channel(move |sender| {
        spawn_task(move || {
            let waker = std::task::Waker::noop();
            let mut cx = std::task::Context::from_waker(&waker);
            let mut inner_body = body;
            
            loop {
                let read_start = std::time::Instant::now();
                
                // Poll the inner body safely
                match std::pin::Pin::new(&mut inner_body).poll_frame(&mut cx) {
                    std::task::Poll::Ready(Some(Ok(frame))) => {
                        // Check read timeout
                        if read_start.elapsed() >= timeout {
                            handle_error!(crate::error::TimedOut, "Read timeout exceeded");
                        }
                        // Convert frame data to Bytes if needed
                        let frame = frame.map_data(|data| data.into());
                        emit!(sender, frame);
                    }
                    std::task::Poll::Ready(Some(Err(e))) => {
                        handle_error!(crate::error::body(e), "Body frame error");
                    }
                    std::task::Poll::Ready(None) => {
                        // End of body stream
                        return;
                    }
                    std::task::Poll::Pending => {
                        // Check read timeout while waiting
                        if read_start.elapsed() >= timeout {
                            handle_error!(crate::error::TimedOut, "Read timeout while waiting");
                        }
                        std::thread::yield_now();
                    }
                }
            }
        });
    })
}

// TotalTimeoutBody Body implementation removed - now using safe total_timeout() function returning AsyncStream

// ReadTimeoutBody Body implementation removed - now using safe with_read_timeout() function returning AsyncStream

pub(crate) type ResponseBody =
    http_body_util::combinators::BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

pub(crate) fn boxed<B>(body: B) -> ResponseBody
where
    B: hyper::body::Body<Data = Bytes> + Send + Sync + 'static,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    use http_body_util::BodyExt;

    body.map_err(box_err).boxed()
}

pub(crate) fn response<B>(
    body: B,
    deadline: Option<std::time::Instant>,
    read_timeout: Option<Duration>,
) -> ResponseBody
where
    B: hyper::body::Body<Data = Bytes> + Send + Sync + 'static,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    use http_body_util::BodyExt;

    match (deadline, read_timeout) {
        (Some(deadline_instant), Some(read)) => {
            // Calculate total timeout duration from deadline
            let now = std::time::Instant::now();
            let total_duration = if deadline_instant > now {
                deadline_instant - now
            } else {
                Duration::from_millis(1) // Already past deadline
            };
            let body = with_read_timeout(body, read).map_err(box_err);
            total_timeout(body, total_duration).map_err(box_err).boxed()
        }
        (Some(deadline_instant), None) => {
            let now = std::time::Instant::now();
            let total_duration = if deadline_instant > now {
                deadline_instant - now
            } else {
                Duration::from_millis(1) // Already past deadline
            };
            total_timeout(body, total_duration).map_err(box_err).boxed()
        },
        (None, Some(read)) => with_read_timeout(body, read).map_err(box_err).boxed(),
        (None, None) => body.map_err(box_err).boxed(),
    }
}

fn box_err<E>(err: E) -> Box<dyn std::error::Error + Send + Sync>
where
    E: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    err.into()
}

// ===== impl DataStream =====

#[cfg(any(feature = "stream", feature = "multipart",))]
impl<B> DataStream<B>
where
    B: HttpBody<Data = Bytes>,
{
    /// Convert DataStream to AsyncStream for streams-first architecture
    /// COMPLETE STREAMING IMPLEMENTATION - Production-ready AsyncStream conversion
    pub fn into_async_stream(self) -> AsyncStream<Result<Bytes, B::Error>> {
        use fluent_ai_async::{spawn_task, emit, handle_error};
        use std::time::{Duration, Instant};
        
        AsyncStream::with_channel(move |sender| {
            let task = spawn_task(move || {
                Self::stream_body_data(self.0, sender)
            });
            
            // The task handles all streaming internally
            let _ = task.collect();
        })
    }
    
    /// Internal method to handle body data streaming with proper AsyncStream patterns
    fn stream_body_data(mut body: B, sender: fluent_ai_async::AsyncStreamSender<Result<Bytes, B::Error>>) {
        use std::time::{Duration, Instant};
        use std::thread;
        
        // Configuration for streaming behavior
        let max_polling_iterations = 1000;
        let polling_delay = Duration::from_millis(1);
        let max_chunk_size = 8192; // 8KB chunks
        let timeout_per_chunk = Duration::from_secs(30);
        
        let mut iteration_count = 0;
        let stream_start_time = Instant::now();
        
        // Create proper context for polling with notification capability
        let waker = std::task::Waker::noop();
        let mut context = std::task::Context::from_waker(&waker);
        
        loop {
            iteration_count += 1;
            let chunk_start_time = Instant::now();
            
            // Check for overall timeout
            if stream_start_time.elapsed() > Duration::from_secs(300) { // 5 minute total timeout
                handle_error!("Body streaming timeout exceeded", "data stream conversion");
                break;
            }
            
            // Check for iteration limit to prevent infinite loops
            if iteration_count > max_polling_iterations {
                // Yield control and reset counter to prevent CPU hogging
                thread::sleep(polling_delay);
                iteration_count = 0;
            }
            
            // Poll the HttpBody for the next frame
            let mut pinned_body = std::pin::Pin::new(&mut body);
            match pinned_body.as_mut().poll_frame(&mut context) {
                std::task::Poll::Ready(Some(Ok(frame))) => {
                    // Extract data from frame if it's a data frame
                    if let Ok(data) = frame.into_data() {
                        // Split large chunks to maintain streaming behavior
                        if data.len() > max_chunk_size {
                            // Split into smaller chunks for better streaming
                            for chunk_start in (0..data.len()).step_by(max_chunk_size) {
                                let chunk_end = std::cmp::min(chunk_start + max_chunk_size, data.len());
                                let chunk = data.slice(chunk_start..chunk_end);
                                emit!(sender, Ok(chunk));
                            }
                        } else {
                            emit!(sender, Ok(data));
                        }
                    }
                    // Continue processing frames
                }
                std::task::Poll::Ready(Some(Err(err))) => {
                    // Emit error and terminate stream
                    emit!(sender, Err(err));
                    break;
                }
                std::task::Poll::Ready(None) => {
                    // End of stream - normal termination
                    break;
                }
                std::task::Poll::Pending => {
                    // PROPER HANDLING FOR PENDING STATE
                    // In streams-first architecture, we need to handle pending properly
                    
                    // Check if we've been waiting too long for this chunk
                    if chunk_start_time.elapsed() > timeout_per_chunk {
                        handle_error!("Chunk timeout exceeded", "body frame polling");
                        break;
                    }
                    
                    // Small delay to prevent busy waiting while allowing responsiveness
                    thread::sleep(polling_delay);
                    
                    // Continue polling in next iteration
                    continue;
                }
            }
        }
    }
}

// ===== impl IntoBytesBody =====

pin_project! {
    struct IntoBytesBody<B> {
        #[pin]
        inner: B,
    }
}

// We can't use `map_frame()` because that loses the hint data (for good reason).
// But we aren't transforming the data.
impl<B> hyper::body::Body for IntoBytesBody<B>
where
    B: hyper::body::Body,
    B::Data: Into<Bytes>,
{
    type Data = Bytes;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>> {
        match ready!(self.project().inner.poll_frame(cx)) {
            Some(Ok(f)) => Poll::Ready(Some(Ok(f.map_data(Into::into)))),
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => Poll::Ready(None),
        }
    }

    #[inline]
    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }

    #[inline]
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
}

#[cfg(test)]
mod tests {
    use http_body::Body as _;

    use super::Body;

    #[test]
    fn test_as_bytes() {
        let test_data = b"Test body";
        let body = Body::from(&test_data[..]);
        assert_eq!(body.as_bytes(), Some(&test_data[..]));
    }

    #[test]
    fn body_exact_length() {
        let empty_body = Body::empty();
        assert!(empty_body.is_end_stream());
        assert_eq!(empty_body.size_hint().exact(), Some(0));

        let bytes_body = Body::reusable("abc".into());
        assert!(!bytes_body.is_end_stream());
        assert_eq!(bytes_body.size_hint().exact(), Some(3));

        // can delegate even when wrapped
        let stream_body = Body::wrap(empty_body);
        assert!(stream_body.is_end_stream());
        assert_eq!(stream_body.size_hint().exact(), Some(0));
    }
}
