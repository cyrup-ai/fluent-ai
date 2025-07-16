// ============================================================================
// File: src/decoders/sse.rs       (fully replaces your previous file)
// ----------------------------------------------------------------------------
// Zero-alloc Server-Sent-Event decoder wired to Better-RIG primitives.
// ============================================================================

#![allow(clippy::type_complexity)]

use super::line::{find_double_newline_index, LineDecoder};
use crate::runtime::{self as rt, AsyncStream};
use crossbeam_channel::bounded;
use futures::{Stream, StreamExt};
use std::fmt::Debug;
use thiserror::Error;

/* ───────────────────────────── errors ─────────────────────────────────── */

#[derive(Debug, Error)]
pub enum SSEDecoderError {
    #[error("Failed to parse SSE: {0}")]
    Parse(String),
    #[error("UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

/* ───────────────────────────── event value ────────────────────────────── */

#[derive(Debug, Clone)]
pub struct ServerSentEvent {
    pub event: Option<String>,
    pub data: String,
    pub raw: Vec<String>,
}

/* ───────────────────────────── low-level state machine ────────────────── */

#[derive(Default)]
struct InnerSSEDecoder {
    data: Vec<String>,
    event: Option<String>,
    raw: Vec<String>,
}

impl InnerSSEDecoder {
    fn feed_line(&mut self, line: &str) -> Option<ServerSentEvent> {
        // cut trailing CR (spec quirk)
        let mut ln = line;
        if ln.ends_with('\r') {
            ln = &ln[..ln.len() - 1];
        }

        // blank line --> event boundary
        if ln.is_empty() {
            if self.event.is_none() && self.data.is_empty() {
                return None;
            }
            let ev = ServerSentEvent {
                event: self.event.take(),
                data: self.data.join("\n"),
                raw: std::mem::take(&mut self.raw),
            };
            self.data.clear();
            return Some(ev);
        }

        self.raw.push(ln.to_owned());

        // ignore comments
        if ln.starts_with(':') {
            return None;
        }

        // field:value split (max 1 ':')
        let mut split = ln.splitn(2, ':');
        let field = split.next().unwrap_or_default();
        let mut val = split.next().unwrap_or_default();
        if let Some(stripped) = val.strip_prefix(' ') {
            val = stripped;
        }

        match field {
            "event" => self.event = Some(val.to_owned()),
            "data" => self.data.push(val.to_owned()),
            _ => {}
        }
        None
    }

    fn flush_final(mut self) -> Option<ServerSentEvent> {
        if self.event.is_some() || !self.data.is_empty() {
            Some(ServerSentEvent {
                event: self.event.take(),
                data: self.data.join("\n"),
                raw: std::mem::take(&mut self.raw),
            })
        } else {
            None
        }
    }
}

/* ───────────────────────────── public stream API ─────────────────────── */

const RING_CAP: usize = 256;
type SSEStream = AsyncStream<Result<ServerSentEvent, SSEDecoderError>, RING_CAP>;

/// Convert **any** byte-stream (`Vec<u8>` chunks) into an `AsyncStream` of
/// decoded `ServerSentEvent`s – zero-alloc after spawn.
pub fn decode_sse_stream<B>(stream: B) -> SSEStream
where
    B: Stream<Item = Result<Vec<u8>, std::io::Error>> + Send + 'static,
{
    let (tx, rx) = bounded::<Result<ServerSentEvent, SSEDecoderError>>(RING_CAP);

    rt::spawn_async(async move {
        let mut sse = InnerSSEDecoder::default();
        let mut lines = LineDecoder::new();
        let mut buf = Vec::<u8>::new();
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(e.into()));
                    continue;
                }
            };
            buf.extend_from_slice(&chunk);

            // split buffer on **double** newline boundaries
            while let Some((front, rest)) = split_sse_chunk(&buf) {
                buf = rest; // remaining bytes (move)
                for line in lines.decode(&front) {
                    if let Some(ev) = sse.feed_line(&line) {
                        let _ = tx.send(Ok(ev));
                    }
                }
            }
        }

        // drain tail
        for line in lines.flush() {
            if let Some(ev) = sse.feed_line(&line) {
                let _ = tx.send(Ok(ev));
            }
        }
        if let Some(final_ev) = sse.flush_final() {
            let _ = tx.send(Ok(final_ev));
        }
    });

    AsyncStream::new(rx)
}

/// Convenience wrapper for `reqwest` responses.
pub fn from_response(resp: reqwest::Response) -> SSEStream {
    let byte_stream = resp
        .bytes_stream()
        .map(|r| r.map(|b| b.to_vec()).map_err(std::io::Error::other));
    decode_sse_stream(byte_stream)
}

/* ───────────────────────────── helpers ───────────────────────────────── */

/// Try to cut `buffer` at the first **double** newline sequence.
/// Returns `(chunk_before_delimiter, remaining_tail)`.
fn split_sse_chunk(buf: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let idx = find_double_newline_index(buf);
    if idx <= 0 {
        return None;
    }
    let i = idx as usize;
    Some((buf[..i].to_vec(), buf[i..].to_vec()))
}
