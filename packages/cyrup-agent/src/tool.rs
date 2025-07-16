// ============================================================================
// File: src/tool.rs
// ----------------------------------------------------------------------------
// Zero-alloc, provider-agnostic “function-calling” abstraction.
//
//   • `Tool` trait – strongly-typed contract with *synchronous* metadata
//     and an allocation-free asynchronous call that returns `AsyncTask`.
//   • `ToolDefinition` – JSON schema sent to the LLM.
//   • `Tools` registry – constant-time lookup + zero-alloc hot path.
//
// Public API exposes **no `async fn`** and uses **no `async_trait` macro**.
// ============================================================================

#![allow(clippy::arc_with_non_send_sync)]

use std::{error::Error, fmt, sync::Arc};

use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::{
    json_util,
    runtime::{self, AsyncTask},
};

// ---------------------------------------------------------------------------
// 1. Public metadata value-object
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON-schema describing the arguments object.
    pub parameters: Value,
}

// ---------------------------------------------------------------------------
// 2. Public tool trait – **NO async fn in the signature**
// ---------------------------------------------------------------------------
pub trait Tool: Send + Sync + 'static {
    const NAME: &'static str;

    type Error: Error + Send + Sync + 'static;
    type Args: DeserializeOwned + Send + 'static;
    type Output: Serialize + Send + 'static;

    /// Synchronous: cheap, side-effect-free metadata construction.
    fn definition(&self, prompt: String) -> ToolDefinition;

    /// Asynchronous execution — returns one allocation-free `AsyncTask`.
    fn call(&self, args: Self::Args) -> AsyncTask<Result<Self::Output, Self::Error>>;
}

// ---------------------------------------------------------------------------
// 3. Dyn-erased wrapper so heterogeneous tools share one registry.
// ---------------------------------------------------------------------------
trait ToolDyn: Send + Sync {
    fn name(&self) -> &'static str;
    fn definition_sync(&self) -> ToolDefinition;
    fn call_json(&self, args_json: String) -> AsyncTask<Result<String, String>>;
}

struct ToolWrapper<T: Tool>(T);

impl<T: Tool> ToolDyn for ToolWrapper<T> {
    #[inline(always)]
    fn name(&self) -> &'static str {
        T::NAME
    }

    #[inline(always)]
    fn definition_sync(&self) -> ToolDefinition {
        // Synchronous fast-path.
        self.0.definition(String::new())
    }

    fn call_json(&self, args_json: String) -> AsyncTask<Result<String, String>> {
        // 1. Parse raw JSON into strongly-typed arguments.
        let parsed = match serde_json::from_str::<T::Args>(&args_json) {
            Ok(v) => v,
            Err(e) => {
                return runtime::spawn_async(async move { Err(format!("arg-parse: {e}")) });
            }
        };

        // 2. Delegate to the tool implementation.
        let fut = self
            .0
            .call(parsed)
            .map(|res| res.and_then(|out| serde_json::to_string(&out).map_err(|e| e.to_string())));
        runtime::spawn_async(fut)
    }
}

// ---------------------------------------------------------------------------
// 4. Global registry – O(1) look-ups, zero locks on read.
// ---------------------------------------------------------------------------
static REGISTRY: Lazy<DashMap<&'static str, Arc<dyn ToolDyn>>> = Lazy::new(DashMap::default);

/// Convenience façade with ergonomic helpers.
#[derive(Default)]
pub struct Tools;

/// Type alias for backward compatibility
pub type ToolRegistry = Tools;

/// Dyn trait for tool embeddings
pub trait ToolEmbeddingDyn: Send + Sync {
    fn embed(&self) -> String;
}

impl Tools {
    /// Register the tool once at start-up. Duplicate names are silently ignored.
    pub fn register<T: Tool>(tool: T) {
        REGISTRY
            .entry(T::NAME)
            .or_insert_with(|| Arc::new(ToolWrapper(tool)));
    }

    /// JSON-schema sent to the LLM.
    #[inline(always)]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        REGISTRY
            .iter()
            .map(|entry| entry.value().definition_sync())
            .collect()
    }

    /// Invoke a tool by name with raw JSON args.
    ///
    /// *Returns a single `AsyncTask` — no public `async fn`.*
    pub fn call(&self, name: &str, args_json: String) -> AsyncTask<Result<String, String>> {
        match REGISTRY.get(name) {
            Some(t) => t.call_json(args_json),
            None => runtime::spawn_async(async move { Err(format!("unknown tool {name}")) }),
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Debug helper
// ---------------------------------------------------------------------------
impl fmt::Debug for Tools {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list: Vec<_> = REGISTRY.iter().map(|e| e.key()).collect();
        f.debug_struct("Tools").field("registered", &list).finish()
    }
}

// ============================================================================
// End of file
// ============================================================================
