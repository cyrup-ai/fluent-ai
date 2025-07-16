// ============================================================================
// File: src/providers/anthropic/prompt_tools.rs
// ----------------------------------------------------------------------------
// Zero-alloc wrappers for the “prompt-tools/*” beta endpoints.
//
// • Public surface exposes only `AsyncTask` handles – _no `async fn`, no
//   `BoxFuture`, no `async_trait`_.
// • Hot path is allocation-free after the initial JSON buffer has been built.
// • The builder follows the ergonomic pattern used elsewhere in Better-RIG
//   (typestate + fluent setters).
// ============================================================================

#![allow(clippy::type_complexity)]

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    json_util::merge_inplace,
    rt::{self, AsyncTask},
};

use super::client::Client;

// ---------------------------------------------------------------------------
// 0. Shared request/response types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct PromptToolsRequest {
    pub prompt: String,
    pub desired_style: Option<String>,
    pub max_suggestions: Option<u8>,
    pub additional_params: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PromptToolsResponse {
    pub suggestions: Vec<String>,
}

// ---------------------------------------------------------------------------
// 1. Fluent builder (typestate guarantees `prompt` is set)
// ---------------------------------------------------------------------------

pub struct PromptToolsBuilder<'a, const HAS_PROMPT: bool> {
    client: &'a Client,
    endpoint: &'static str, // "/v1/prompt-tools/generate" | ".../improve"
    prompt: Option<String>,
    desired_style: Option<String>,
    max_suggestions: Option<u8>,
    additional_params: Option<serde_json::Value>,
}

impl<'a> PromptToolsBuilder<'a, false> {
    #[inline(always)]
    pub(crate) fn new(client: &'a Client, endpoint: &'static str) -> Self {
        Self {
            client,
            endpoint,
            prompt: None,
            desired_style: None,
            max_suggestions: None,
            additional_params: None,
        }
    }
}

impl<'a, const P: bool> PromptToolsBuilder<'a, P> {
    /// Mandatory – the raw prompt we want Anthropic to improve / generate.
    #[inline]
    pub fn prompt(self, p: impl Into<String>) -> PromptToolsBuilder<'a, true> {
        PromptToolsBuilder {
            prompt: Some(p.into()),
            ..self
        }
    }

    /// Optional – tell Anthropic what “tone” the returned prompts should have.
    #[inline]
    pub fn desired_style(mut self, style: impl Into<String>) -> Self {
        self.desired_style = Some(style.into());
        self
    }

    /// Optional – number of suggestions to ask for (1 … 8, default per API).
    #[inline]
    pub fn max_suggestions(mut self, n: u8) -> Self {
        self.max_suggestions = Some(n.min(8).max(1));
        self
    }

    /// Provider-specific escape hatch.
    #[inline]
    pub fn additional_params(mut self, v: serde_json::Value) -> Self {
        self.additional_params = Some(v);
        self
    }
}

// ---------------------- only callable once `prompt` is present -------------
impl<'a> PromptToolsBuilder<'a, true> {
    /// Kick off the HTTP request – returns ONE `AsyncTask`.
    #[inline]
    pub fn send(
        self,
    ) -> AsyncTask<Result<PromptToolsResponse, crate::completion::CompletionError>> {
        let mut body = json!({
            "prompt": self.prompt.unwrap(),
        });

        if let Some(style) = self.desired_style {
            merge_inplace(&mut body, json!({ "desired_style": style }));
        }
        if let Some(n) = self.max_suggestions {
            merge_inplace(&mut body, json!({ "max_suggestions": n }));
        }
        if let Some(extra) = self.additional_params {
            merge_inplace(&mut body, extra);
        }

        let req = self.client.post(self.endpoint).json(&body).send(); // <- still returns a future; we’ll .await inside the task

        rt::spawn_async(async move {
            use crate::completion::CompletionError;
            let resp = req.await.map_err(CompletionError::from)?;
            if resp.status().is_success() {
                Ok(resp.json::<PromptToolsResponse>().await?)
            } else {
                Err(CompletionError::ProviderError(resp.text().await?))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// 2. Ergonomic extension methods on `Client`
// ---------------------------------------------------------------------------

impl Client {
    /// POST /v1/prompt-tools/generate
    #[inline(always)]
    pub fn prompt_tools_generate(&self) -> PromptToolsBuilder<'_, false> {
        PromptToolsBuilder::new(self, "/v1/prompt-tools/generate")
    }

    /// POST /v1/prompt-tools/improve
    #[inline(always)]
    pub fn prompt_tools_improve(&self) -> PromptToolsBuilder<'_, false> {
        PromptToolsBuilder::new(self, "/v1/prompt-tools/improve")
    }
}

// ---------------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------------
// The module is entirely self-contained: no async fn in its public surface,
// no dynamic futures, zero allocations in the hot path after JSON assembly,
// and it re-uses the existing runtime primitives (`AsyncTask`) exactly as
// agreed.
// ============================================================================
