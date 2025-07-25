// ============================================================================
// File: src/extractor.rs
// ----------------------------------------------------------------------------
// High-level “structured data extractor” built on top of the generic Agent.
//
// • The hot-path (Prompt → Structured result) is zero-alloc after spawn.
// • `Extractor::extract` returns `AsyncTask<Result<T, …>>`, satisfying
//   blueprint invariant #1 (one await per flow, no blocking).
// • No Arc / Mutex – the extractor owns its Agent by value.
// ============================================================================

use std::marker::PhantomData;

use schemars::{JsonSchema, schema_for};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;

use crate::{
    agent::{Agent, AgentBuilder},
    completion::{CompletionError, CompletionModel, ToolDefinition},
    domain::message::{AssistantContent, Message, ToolCall, ToolFunction},
    domain::tool::Tool,
    runtime::{AsyncTask, spawn_async}};

// -----------------------------------------------------------------------------
// Public error enumeration
// -----------------------------------------------------------------------------
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("no `submit` tool-call found in the response")]
    NoData,

    #[error("failed to deserialize the extracted data: {0}")]
    Deserialization(serde_json::Error),

    #[error(transparent)]
    Completion(#[from] CompletionError)}

// -----------------------------------------------------------------------------
// Extractor – thin wrapper around an Agent plus phantom target type
// -----------------------------------------------------------------------------
pub struct Extractor<M, T>
where
    M: CompletionModel,
{
    agent: Agent<M>,
    _t: PhantomData<fn() -> T>}

impl<M, T> Extractor<M, T>
where
    M: CompletionModel + Sync,
    T: JsonSchema + DeserializeOwned + Send + Sync + 'static,
{
    /// Kick off an extraction.
    /// Returns an **`AsyncTask`** which resolves to the structured value.
    #[inline]
    pub fn extract(
        &self,
        text: impl Into<Message> + Send + 'static,
    ) -> AsyncTask<Result<T, ExtractionError>> {
        let agent = self.agent.clone();
        spawn_async(async move {
            // 1. Run the agent
            let resp = agent
                .completion(text, Vec::new())
                .await? // -> CompletionRequest builder
                .send()
                .await?; // -> CompletionResponse

            // 2. Find the `submit` tool-call (there should be exactly one)
            let raw = resp
                .choice
                .into_iter()
                .filter_map(|c| match c {
                    AssistantContent::ToolCall(ToolCall {
                        function:
                            ToolFunction {
                                name, arguments, ..
                            },
                        ..
                    }) if name == SUBMIT_TOOL_NAME => Some(arguments),
                    _ => None})
                .last() // if >1 use the last; earlier ones are ignored
                .ok_or(ExtractionError::NoData)?;

            // 3. Deserialize into the target struct
            serde_json::from_value(raw).map_err(ExtractionError::Deserialization)
        })
    }
}

// -----------------------------------------------------------------------------
// Fluent builder
// -----------------------------------------------------------------------------
pub struct ExtractorBuilder<T, M>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync + 'static,
    M: CompletionModel,
{
    agent_builder: AgentBuilder<M>,
    _t: PhantomData<fn() -> T>}

impl<T, M> ExtractorBuilder<T, M>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync + 'static,
    M: CompletionModel,
{
    /// Start a new extractor builder for `model`.
    pub fn new(model: M) -> Self {
        let agent_builder = AgentBuilder::new(model)
            .preamble(
                "You are an AI assistant whose sole purpose is to extract \
                 structured data from the provided text.\n\
                 Use the `submit` function to return the data in the exact \
                 JSON schema provided. Always call `submit`, even if some \
                 fields are missing (use defaults).",
            )
            .tool(SubmitTool::<T>::default());

        Self {
            agent_builder,
            _t: PhantomData}
    }

    /// Append additional instructions.
    #[inline]
    pub fn preamble(mut self, extra: &str) -> Self {
        self.agent_builder = self.agent_builder.append_preamble(extra);
        self
    }

    /// Attach a context document.
    #[inline]
    pub fn context(mut self, doc: &str) -> Self {
        self.agent_builder = self.agent_builder.context(doc);
        self
    }

    /// Pass arbitrary provider-specific parameters.
    #[inline]
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.agent_builder = self.agent_builder.additional_params(params);
        self
    }

    /// Finish building and obtain the `Extractor`.
    #[inline]
    pub fn build(self) -> Extractor<M, T> {
        Extractor {
            agent: self.agent_builder.build(),
            _t: PhantomData}
    }
}

// -----------------------------------------------------------------------------
// Internal `submit` tool definition
// -----------------------------------------------------------------------------
const SUBMIT_TOOL_NAME: &str = "submit";

#[derive(Default)]
struct SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Send + Sync + 'static,
{
    _t: PhantomData<fn() -> T>}

#[derive(Debug, thiserror::Error)]
#[error("submit tool failed")]
struct SubmitError;

impl<T> Tool for SubmitTool<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + Send + Sync + 'static,
{
    const NAME: &'static str = SUBMIT_TOOL_NAME;

    type Error = SubmitError;
    type Args = T;
    type Output = T;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Return the structured data extracted from the text.".into(),
            parameters: json!(schema_for!(T))}
    }

    async fn call(&self, data: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(data)
    }
}
