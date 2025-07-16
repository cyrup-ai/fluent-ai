// ============================================================================
// File: src/providers/anthropic/streaming.rs      (FULL REWRITE)
// ----------------------------------------------------------------------------
// Anthropic streaming (“messages” endpoint with `"stream":true`) wired to the
// Better-RIG runtime: public API stays sync, returns AsyncTask + opaque stream.
// ============================================================================

#![allow(clippy::type_complexity)]

use async_stream::stream as async_stream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;

use super::{
    client::Client,
    completion::{CompletionModel, Content, Message, ToolChoice, ToolDefinition, Usage},
    decoders::sse::from_response as sse_from_response,
};
use crate::{
    completion::{CompletionError, CompletionRequest},
    json_util::merge_inplace,
    rt::{self, AsyncTask},
    streaming::{RawStreamingChoice, StreamingCompletionResponse as RigStreaming, StreamingResult},
};

/* ─────────────────────────── Anthropic SSE payloads ───────────────────── */

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamingEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: Content,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: PartialUsage,
    },
    MessageStop,
    Ping,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct MessageStart {
    pub id: String,
    pub role: String,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct PartialUsage {
    pub output_tokens: usize,
    #[serde(default)]
    pub input_tokens: Option<usize>,
}

/* ─────────────────────────── internal state helpers ───────────────────── */

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    input_json: String,
}

#[derive(Clone)]
pub struct StreamingCompletionResponse {
    pub usage: PartialUsage,
}

/* ───────────────────────── CompletionModel::stream ────────────────────── */

impl CompletionModel {
    /// Public sync façade: returns **one** AsyncTask that resolves to a
    /// `StreamingCompletionResponse`, hiding *all* async machinery.
    #[inline]
    pub fn stream(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<RigStreaming<StreamingCompletionResponse>, CompletionError>> {
        rt::spawn_async(self.clone().drive_stream(req))
    }

    /* ---------------- internal async driver (NOT public) ---------------- */

    async fn drive_stream(
        self,
        completion_request: CompletionRequest,
    ) -> Result<RigStreaming<StreamingCompletionResponse>, CompletionError> {
        /* -------- build request JSON (identical to non-streaming path) --- */
        let max_tokens = completion_request
            .max_tokens
            .or(self.default_max_tokens)
            .ok_or_else(|| {
                CompletionError::RequestError("`max_tokens` must be set for Anthropic".into())
            })?;

        // merge docs + history
        let mut hist = Vec::<Message>::new();
        if let Some(docs) = completion_request.normalized_documents() {
            hist.push(docs);
        }
        hist.extend(
            completion_request
                .chat_history
                .into_iter()
                .map(Message::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        );

        let mut payload = json!({
            "model": self.model,
            "messages": hist,
            "max_tokens": max_tokens,
            "system": completion_request.preamble.unwrap_or_default(),
            "stream": true,
        });

        if let Some(t) = completion_request.temperature {
            merge_inplace(&mut payload, json!({ "temperature": t }));
        }
        if !completion_request.tools.is_empty() {
            merge_inplace(
                &mut payload,
                json!({
                    "tools": completion_request.tools.into_iter().map(|t| ToolDefinition {
                        name:         t.name,
                        description:  Some(t.description),
                        input_schema: t.parameters,
                    }).collect::<Vec<_>>(),
                    "tool_choice": ToolChoice::Auto,
                }),
            );
        }
        if let Some(extra) = completion_request.additional_params {
            merge_inplace(&mut payload, extra);
        }

        /* -------- fire HTTP request ------------------------------------- */
        let resp = self
            .client
            .post("/v1/messages")
            .json(&payload)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(CompletionError::ProviderError(resp.text().await?));
        }

        /* -------- convert SSE → StreamingResult ------------------------- */
        let sse_stream = sse_from_response(resp);

        let result_stream: StreamingResult<StreamingCompletionResponse> = Box::pin(async_stream! {
            let mut current_tool: Option<ToolCallState> = None;
            let mut in_tokens = 0usize;

            futures::pin_mut!(sse_stream);
            while let Some(evt) = sse_stream.next().await {
                let sse = match evt {
                    Ok(s) => s,
                    Err(e) => { yield Err(CompletionError::ResponseError(e.to_string())); break; }
                };

                // empty keep-alive lines are ignored
                if sse.data.trim().is_empty() { continue; }

                let parsed: StreamingEvent =
                    match serde_json::from_str(&sse.data) {
                        Ok(v) => v,
                        Err(e) => { yield Err(CompletionError::ResponseError(
                                        format!("Bad JSON: {e} / {}", sse.data))); continue; }
                    };

                /* token bookkeeping & final-usage emission */
                if let StreamingEvent::MessageStart { message } = &parsed {
                    in_tokens = message.usage.input_tokens as usize;
                }
                if let StreamingEvent::MessageDelta { usage, delta } = &parsed {
                    if delta.stop_reason.is_some() {
                        yield Ok(RawStreamingChoice::FinalResponse(
                            StreamingCompletionResponse {
                                usage: PartialUsage {
                                    input_tokens: Some(in_tokens),
                                    output_tokens: usage.output_tokens,
                                },
                            }));
                    }
                }

                /* regular delta handling */
                if let Some(out) = handle_event(&parsed, &mut current_tool) {
                    yield out;
                }
            }
        });

        Ok(RigStreaming::stream(result_stream))
    }
}

/* ─────────────────────────── event handler (unchanged) ────────────────── */

fn handle_event(
    evt: &StreamingEvent,
    state: &mut Option<ToolCallState>,
) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
    match evt {
        StreamingEvent::ContentBlockDelta { delta, .. } => match delta {
            ContentDelta::TextDelta { text } => {
                if state.is_none() {
                    Some(Ok(RawStreamingChoice::Message(text.clone())))
                } else {
                    None
                }
            }
            ContentDelta::InputJsonDelta { partial_json } => {
                if let Some(ref mut tc) = state {
                    tc.input_json.push_str(partial_json);
                }
                None
            }
        },

        StreamingEvent::ContentBlockStart { content_block, .. } => match content_block {
            Content::ToolUse { id, name, .. } => {
                *state = Some(ToolCallState {
                    id: id.clone(),
                    name: name.clone(),
                    input_json: String::new(),
                });
                None
            }
            _ => None,
        },

        StreamingEvent::ContentBlockStop { .. } => state.take().map(|tc| {
            let json_str = if tc.input_json.is_empty() {
                "{}"
            } else {
                &tc.input_json
            };
            match serde_json::from_str(json_str) {
                Ok(args) => Ok(RawStreamingChoice::ToolCall {
                    name: tc.name,
                    id: tc.id,
                    arguments: args,
                }),
                Err(e) => Err(CompletionError::from(e)),
            }
        }),

        _ => None,
    }
}
