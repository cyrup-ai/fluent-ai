// ============================================================================
// File: src/streaming.rs
// ----------------------------------------------------------------------------
// Uniform streaming abstraction for completion-style providers.
// Public surface = AsyncTask / AsyncStream only.
// ============================================================================

#![allow(clippy::type_complexity)]

use core::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::{Stream, StreamExt};
use serde_json::Value;

use crate::{
    OneOrMany,
    agent::Agent,
    completion::{
        CompletionError, CompletionModel, CompletionRequestBuilder, CompletionResponse, Message,
        message::{AssistantContent, ToolCall, ToolFunction},
    },
    runtime::async_stream::AsyncStream,
    runtime::{AsyncTask, spawn_async},
};

// ============================================================================
// 0.  Low-level ring stream (imported from runtime)
// ============================================================================

/// Capacity for every provider-emitted stream.
const STREAM_CAP: usize = 256;

/// Helper: convert any dynamic `Stream` into our bounded ring.
fn to_ring_stream<S, T>(mut dyn_stream: S) -> AsyncStream<T, STREAM_CAP>
where
    S: Stream + Send + Unpin + 'static,
    S::Item: Send + 'static,
    T: From<S::Item> + Send + 'static,
{
    AsyncStream::from_dyn(dyn_stream)
}

// ============================================================================
// 1.  Provider-agnostic streaming payload
// ============================================================================

#[derive(Clone, Debug)]
pub enum RawStreamingChoice<R: Clone> {
    Message(String),
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    FinalResponse(R),
}

pub type StreamingResult<R> =
    AsyncStream<Result<RawStreamingChoice<R>, CompletionError>, STREAM_CAP>;

// ============================================================================
// 2.  High-level aggregated response
// ============================================================================

pub struct StreamingCompletionResponse<R: Clone + Unpin> {
    inner: StreamingResult<R>,
    text_buf: String,
    tool_buf: Vec<ToolCall>,
    pub choice: OneOrMany<AssistantContent>,
    pub response: Option<R>,
}

impl<R: Clone + Unpin> StreamingCompletionResponse<R> {
    #[inline]
    pub fn new(inner: StreamingResult<R>) -> Self {
        Self {
            inner,
            text_buf: String::new(),
            tool_buf: Vec::new(),
            choice: OneOrMany::one(AssistantContent::text("")),
            response: None,
        }
    }
}

impl<R: Clone + Unpin> From<StreamingCompletionResponse<R>> for CompletionResponse<Option<R>> {
    fn from(v: StreamingCompletionResponse<R>) -> Self {
        CompletionResponse {
            choice: v.choice,
            raw_response: v.response,
        }
    }
}

impl<R: Clone + Unpin> Stream for StreamingCompletionResponse<R> {
    type Item = Result<AssistantContent, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let me = self.get_mut();

        match Pin::new(&mut me.inner).poll_next(cx) {
            Poll::Pending | Poll::Ready(None) => {
                // When the inner stream ends, flush buffered text / tool calls.
                if matches!(Pin::new(&mut me.inner).poll_next(cx), Poll::Ready(None)) {
                    let mut aggregated: Vec<AssistantContent> = me
                        .tool_buf
                        .iter()
                        .cloned()
                        .map(AssistantContent::ToolCall)
                        .collect();

                    if !me.text_buf.trim().is_empty() {
                        aggregated.insert(0, AssistantContent::Text(me.text_buf.clone().into()));
                    }
                    me.choice = match OneOrMany::many(aggregated) {
                        Ok(messages) => messages,
                        Err(_) => OneOrMany::One(AssistantContent::Text("".into())),
                    };
                }
                Poll::Pending
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(Some(Ok(chunk))) => match chunk {
                RawStreamingChoice::Message(t) => {
                    me.text_buf.push_str(&t);
                    Poll::Ready(Some(Ok(AssistantContent::text(t))))
                }
                RawStreamingChoice::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    let call = ToolCall {
                        id: id.clone(),
                        function: ToolFunction { name, arguments },
                    };
                    me.tool_buf.push(call.clone());
                    Poll::Ready(Some(Ok(AssistantContent::ToolCall(call))))
                }
                RawStreamingChoice::FinalResponse(r) => {
                    me.response = Some(r);
                    Poll::Pending
                }
            },
        }
    }
}

// dyn-erased helper used by CompletionModelHandle
pub(crate) struct StreamingResultDyn<R: Clone + Unpin> {
    inner: StreamingResult<R>,
}

impl<R: Clone + Unpin> From<StreamingResult<R>> for StreamingResultDyn<R> {
    #[inline(always)]
    fn from(inner: StreamingResult<R>) -> Self {
        Self { inner }
    }
}

impl<R: Clone + Unpin> Stream for StreamingResultDyn<R> {
    type Item = Result<RawStreamingChoice<()>, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let me = self.get_mut();
        match Pin::new(&mut me.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Ok(RawStreamingChoice::FinalResponse(_)))) => {
                Poll::Ready(Some(Ok(RawStreamingChoice::FinalResponse(()))))
            }
            Poll::Ready(Some(Ok(RawStreamingChoice::Message(m)))) => {
                Poll::Ready(Some(Ok(RawStreamingChoice::Message(m))))
            }
            Poll::Ready(Some(Ok(RawStreamingChoice::ToolCall {
                id,
                name,
                arguments,
            }))) => Poll::Ready(Some(Ok(RawStreamingChoice::ToolCall {
                id,
                name,
                arguments,
            }))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
        }
    }
}

// ============================================================================
// 3.  Public high-level traits
// ============================================================================

pub trait StreamingPrompt<R: Clone + Unpin>: Send + Sync {
    fn stream_prompt(
        &self,
        prompt: impl Into<Message> + Send,
    ) -> AsyncTask<Result<StreamingCompletionResponse<R>, CompletionError>>;
}

pub trait StreamingChat<R: Clone + Unpin>: Send + Sync {
    fn stream_chat(
        &self,
        prompt: impl Into<Message> + Send,
        history: Vec<Message>,
    ) -> AsyncTask<Result<StreamingCompletionResponse<R>, CompletionError>>;
}

pub trait StreamingCompletion<M: CompletionModel> {
    fn stream_completion(
        &self,
        prompt: impl Into<Message> + Send,
        history: Vec<Message>,
    ) -> AsyncTask<Result<CompletionRequestBuilder<M>, CompletionError>>;
}

// ============================================================================
// 4.  Utility: print stream to stdout & invoke tools live
// ============================================================================

/// Stream completion to stdout with tool execution - returns AsyncTask
pub fn stream_to_stdout<M: CompletionModel>(
    agent: &Agent<M>,
    stream: StreamingCompletionResponse<M::StreamingResponse>,
) -> AsyncTask<Result<(), std::io::Error>> {
    spawn_async(async move {
        use std::io::{self, Write};

        use termcolor::{ColoredMessage, colored_print, colored_println, info};

        let mut stream = stream;
        colored_print!(primary: "Response: ");
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(AssistantContent::Text(t)) => {
                    colored_print!(text_primary: "{}", t.text);
                    io::stdout().flush()?;
                }
                Ok(AssistantContent::ToolCall(tc)) => {
                    // Note: This would need the agent tools to be async-compatible
                    ColoredMessage::new()
                        .newline()
                        .info("Tool call: ")
                        .accent(&tc.function.name)
                        .text_secondary(" with args: ")
                        .text_muted(&tc.function.arguments.to_string())
                        .println()
                        .ok();
                }
                Err(e) => {
                    colored_println!(error: "Error: {e}");
                    break;
                }
            }
        }
        colored_println!();
        Ok(())
    })
}
