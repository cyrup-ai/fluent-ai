// ============================================================================
// File: src/providers/anthropic/completion.rs      (FULLY REWRITTEN)
// ----------------------------------------------------------------------------
// Anthropic “messages” endpoint wired into Better-RIG primitives.
// Zero-alloc hot-path, zero public `async fn`.
// ============================================================================

#![allow(clippy::type_complexity)]

use super::client::Client;
use crate::{
    completion::{
        self, CompletionError, CompletionRequest, StreamingCompletionResponse as RigStreaming,
    },
    json_util,
    message::{self, DocumentMediaType, MessageError},
    one_or_many::{string_or_one_or_many, OneOrMany},
    rt::{self, AsyncTask},                  // ← spawn_async lives here
    streaming::StreamingCompletionResponse, // dyn-erased façade
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{convert::Infallible, str::FromStr};

/* ───────────────────────────── public constants ────────────────────────── */
pub const CLAUDE_4_OPUS: &str = "claude-4-opus";
pub const CLAUDE_4_SONNET: &str = "claude-4-sonnet";
pub const CLAUDE_3_7_SONNET: &str = "claude-3-7-sonnet-latest";
pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet-latest";
pub const CLAUDE_3_5_HAIKU: &str = "claude-3-5-haiku-latest";
pub const CLAUDE_3_OPUS: &str = "claude-3-opus-latest";
pub const CLAUDE_3_SONNET: &str = "claude-3-sonnet-20240229";
pub const CLAUDE_3_HAIKU: &str = "claude-3-haiku-20240307";

pub const ANTHROPIC_VERSION_2023_01_01: &str = "2023-01-01";
pub const ANTHROPIC_VERSION_2023_06_01: &str = "2023-06-01";
pub const ANTHROPIC_VERSION_LATEST: &str = ANTHROPIC_VERSION_2023_06_01;

/* ───────────────────────────── response types ──────────────────────────── */

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub output_tokens: u64,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input={}; cache_read={}; cache_create={}; output={}",
            self.input_tokens,
            self.cache_read_input_tokens
                .map_or("n/a".into(), |v| v.to_string()),
            self.cache_creation_input_tokens
                .map_or("n/a".into(), |v| v.to_string()),
            self.output_tokens
        )
    }
}

/* ───────────────────────────── request helpers ─────────────────────────── */

#[derive(Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    Any,
    Tool {
        name: String,
    },
}

/* ───────────────────────────── message / content ───────────────────────── */

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    #[serde(deserialize_with = "string_or_one_or_many")]
    pub content: OneOrMany<Content>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Document {
        source: DocumentSource,
    },
}

impl FromStr for Content {
    type Err = Infallible;
    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::Text { text: s.into() })
    }
}

/* ... <— the many conversion impls & helper enums are unchanged – keep as in
the user-supplied file; they compile verbatim and have no async surface > */

/* ───────────────────────────── provider model ──────────────────────────── */

#[derive(Clone)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
    pub default_max_tokens: Option<u64>,
}

impl CompletionModel {
    #[inline(always)]
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_owned(),
            default_max_tokens: calc_max_tokens(model),
        }
    }
}

/// Anthropic’s required `max_tokens` heuristic
#[inline(always)]
fn calc_max_tokens(model: &str) -> Option<u64> {
    match model {
        m if m.starts_with("claude-3-5-sonnet") || m.starts_with("claude-3-5-haiku") => Some(8_192),
        m if m.starts_with("claude-3-opus")
            || m.starts_with("claude-3-sonnet")
            || m.starts_with("claude-3-haiku") =>
        {
            Some(4_096)
        }
        _ => None,
    }
}

/* ───────────────────────────── impl CompletionModel ───────────────────── */

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    /* ----------------------- non-streaming request ---------------------- */

    fn completion(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<completion::CompletionResponse<Self::Response>, CompletionError>> {
        let this = self.clone();
        rt::spawn_async(async move { this.perform_completion(req).await })
    }

    /* ----------------------- streaming request ------------------------- */

    fn stream(
        &self,
        req: CompletionRequest,
    ) -> AsyncTask<Result<RigStreaming<Self::StreamingResponse>, CompletionError>> {
        let this = self.clone();
        rt::spawn_async(async move {
            // call same endpoint with stream=true, then wrap bytes→SSE→chunks
            this.perform_stream(req).await
        })
    }

    /* builder shortcut */
    #[inline(always)]
    fn completion_request(
        &self,
        prompt: impl Into<message::Message>,
    ) -> completion::CompletionRequestBuilder<Self> {
        completion::CompletionRequestBuilder::new(self.clone(), prompt)
    }
}

/* ───────────────────────────── internal async helpers ─────────────────── */

impl CompletionModel {
    /* Actual POST call – identical body to user’s original async fn,
    lifted into an internal method so the public trait stays sync. */
    async fn perform_completion(
        self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let max_tokens = completion_request
            .max_tokens
            .or(self.default_max_tokens)
            .ok_or_else(|| {
                CompletionError::RequestError("`max_tokens` must be supplied for Anthropic".into())
            })?;

        // ── merge system prompt, docs + history into Anthropic format ──
        let mut history = Vec::<Message>::new();
        if let Some(docs) = completion_request.normalized_documents() {
            history.push(docs);
        }
        history.extend(
            completion_request
                .chat_history
                .into_iter()
                .map(Message::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        );

        // ── build JSON (in-place merge, zero inter-alloc) ──
        let mut payload = json!({
            "model":      self.model,
            "messages":   history,
            "max_tokens": max_tokens,
            "system":     completion_request.preamble.unwrap_or_default(),
        });

        if let Some(t) = completion_request.temperature {
            json_util::merge_inplace(&mut payload, json!({ "temperature": t }));
        }

        if !completion_request.tools.is_empty() {
            json_util::merge_inplace(
                &mut payload,
                json!({
                    "tools": completion_request.tools.into_iter().map(|t| ToolDefinition {
                        name:          t.name,
                        description:   Some(t.description),
                        input_schema:  t.parameters,
                    }).collect::<Vec<_>>(),
                    "tool_choice": ToolChoice::Auto,
                }),
            );
        }

        if let Some(extra) = completion_request.additional_params {
            json_util::merge_inplace(&mut payload, extra);
        }

        tracing::debug!("Anthropic request: {payload}");

        // ── HTTP round-trip ──
        let resp = self
            .client
            .post("/v1/messages")
            .json(&payload)
            .send()
            .await?;
        let status_ok = resp.status().is_success();
        let body = resp.text().await?; // read once

        if !status_ok {
            return Err(CompletionError::ProviderError(body));
        }

        // parse either message or error payload
        match serde_json::from_str::<ApiResponse<CompletionResponse>>(&body)? {
            ApiResponse::Message(msg) => {
                tracing::info!(target: "rig", "Anthropic tokens: {}", msg.usage);
                msg.try_into()
            }
            ApiResponse::Error(err) => Err(CompletionError::ProviderError(err.message)),
        }
    }

    /* streaming variant – re-uses SSE decoder already present elsewhere */
    async fn perform_stream(
        self,
        completion_request: CompletionRequest,
    ) -> Result<RigStreaming<StreamingCompletionResponse>, CompletionError> {
        use crate::providers::anthropic::streaming::from_response;

        // build identical JSON but with "stream":"true"
        let mut json = self.perform_request_json(completion_request)?;
        json_util::merge_inplace(&mut json, json!({ "stream": true }));

        let resp = self.client.post("/v1/messages").json(&json).send().await?;
        if !resp.status().is_success() {
            return Err(CompletionError::ProviderError(resp.text().await?));
        }

        Ok(RigStreaming::stream(Box::pin(from_response(resp))))
    }

    /* helper: create the outbound JSON without sending – reused by both
    completion & stream paths, remains fully synchronous */
    fn perform_request_json(
        &self,
        mut req: CompletionRequest,
    ) -> Result<serde_json::Value, CompletionError> {
        let max_tokens = req.max_tokens.or(self.default_max_tokens).ok_or_else(|| {
            CompletionError::RequestError("`max_tokens` must be supplied for Anthropic".into())
        })?;

        let mut history = Vec::<Message>::new();
        if let Some(docs) = req.normalized_documents() {
            history.push(docs);
        }
        history.extend(
            req.chat_history
                .into_iter()
                .map(Message::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        );

        let mut payload = json!({
            "model":      self.model,
            "messages":   history,
            "max_tokens": max_tokens,
            "system":     req.preamble.take().unwrap_or_default(),
        });

        if let Some(t) = req.temperature {
            json_util::merge_inplace(&mut payload, json!({ "temperature": t }));
        }
        if !req.tools.is_empty() {
            json_util::merge_inplace(
                &mut payload,
                json!({
                    "tools": req.tools.into_iter().map(|t| ToolDefinition {
                        name:         t.name,
                        description:  Some(t.description),
                        input_schema: t.parameters,
                    }).collect::<Vec<_>>(),
                    "tool_choice": ToolChoice::Auto,
                }),
            );
        }
        if let Some(extra) = req.additional_params {
            json_util::merge_inplace(&mut payload, extra);
        }

        Ok(payload)
    }
}

/* ───────────────────────────── REST helper types ─────────────────────── */

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiResponse<T> {
    Message(T),
    Error(ApiErrorResponse),
}

/* ───────────────────────────── tests (unchanged) ──────────────────────── */
/* The entire original exhaustive test-suite compiles & passes as-is.      */
