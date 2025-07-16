// ============================================================================
// Zero-alloc JSONL → AsyncStream adapter (Better-RIG style)
// ============================================================================

#![allow(clippy::type_complexity)]

use core::{
    pin::Pin,
    task::{Context, Poll},
};
use futures::{Stream, StreamExt};
use serde::de::DeserializeOwned;
use std::{marker::PhantomData, str};

use crate::{
    runtime::{self as rt, AsyncStream},
    streaming::AsyncStreamDyn, // only for the blanket impls
};

use super::line::LineDecoder;

// ----- Errors ---------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum JSONLDecoderError {
    #[error("stream error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),
}

// ----- Public façade --------------------------------------------------------

/// Capacity of the internal ring.  Tune at build-time if benchmarks dictate.
pub const CAP: usize = 256;

/// Turn **any** byte-stream of JSON-Lines into an `AsyncStream` of parsed items.
///
/// ```rust
/// let byte_stream = reqwest::get(url).await?.bytes_stream();
/// let lines: AsyncStream<Result<MyRow, _>, 256> = decode_jsonl(byte_stream);
/// ```
#[inline(always)]
pub fn decode_jsonl<T, S>(stream: S) -> AsyncStream<Result<T, JSONLDecoderError>, CAP>
where
    T: DeserializeOwned + Send + 'static,
    S: Stream<Item = Result<Vec<u8>, std::io::Error>> + Send + Unpin + 'static,
{
    // bounded ring – one allocation, ever
    let (tx, rx) = crate::runtime::channel::<Result<T, JSONLDecoderError>, CAP>();

    // spawn the parser task (single allocation for the boxed future)
    rt::spawn_async(async move {
        let mut decoder = LineDecoder::new();
        let mut buf = String::new(); // reused for every successful parse

        futures::pin_mut!(stream);
        while let Some(chunk_res) = stream.next().await {
            match chunk_res {
                Ok(chunk) => {
                    for line in decoder.decode(&chunk) {
                        if line.trim().is_empty() {
                            continue;
                        }
                        // zero-alloc convert: reuse the String buffer
                        buf.clear();
                        buf.push_str(&line);
                        let parsed =
                            serde_json::from_str::<T>(&buf).map_err(JSONLDecoderError::from);
                        // best-effort back-pressure
                        let _ = tx.try_send(parsed);
                    }
                }
                Err(e) => {
                    let _ = tx.try_send(Err(e.into()));
                }
            }
        }

        // flush tail
        for line in decoder.flush() {
            if line.trim().is_empty() {
                continue;
            }
            buf.clear();
            buf.push_str(&line);
            let parsed = serde_json::from_str::<T>(&buf).map_err(JSONLDecoderError::from);
            let _ = tx.try_send(parsed);
        }
    });

    AsyncStream::new(rx)
}
