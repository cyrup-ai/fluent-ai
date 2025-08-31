//=========================================================================
//  src/mcp/mod.rs
//  ──────────────────────────────────────────────────────────────────────
//  Core MCP types & shared helpers for JSON/TOML (zero-Serde).
//  Inlined and allocator-optimized for minimal heap churn.
//=========================================================================
#![allow(clippy::needless_return)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::upper_case_acronyms)]

use std::collections::HashMap;
use simd_json::{value::owned::Value as JsonValue, StaticNode};
use value_trait::prelude::*;

// Re-export format-specific modules:
pub mod json;
pub mod toml;

//─────────────────────────────────────────────────────────────────────────
//  Common Primitives & Domain Types
//─────────────────────────────────────────────────────────────────────────

/// A JSON-RPC request ID: either a number or a string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestId {
    Str(String),
    Num(i64),
}

/// A progress token (opaque identifier): either number or string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgressToken {
    Str(String),
    Num(i64),
}

/// Logging severity (RFC-5424 syslog levels).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}
impl LogLevel {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Notice => "notice",
            LogLevel::Warning => "warning",
            LogLevel::Error => "error",
            LogLevel::Critical => "critical",
            LogLevel::Alert => "alert",
            LogLevel::Emergency => "emergency",
        }
    }
    #[inline]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "debug" => Some(LogLevel::Debug),
            "info" => Some(LogLevel::Info),
            "notice" => Some(LogLevel::Notice),
            "warning" => Some(LogLevel::Warning),
            "error" => Some(LogLevel::Error),
            "critical" => Some(LogLevel::Critical),
            "alert" => Some(LogLevel::Alert),
            "emergency" => Some(LogLevel::Emergency),
            _ => None,
        }
    }
}

/// A role in a conversation: user or assistant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}
impl Role {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
    #[inline]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// Describes the name+version of an MCP implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct Implementation {
    pub name: String,
    pub version: String,
}

/// Logging capability: indicates server can send log message notifications.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggingCapability;

/// Prompt-related capability: server can send prompt-list change notifications.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptsCapability {
    pub list_changed: bool,
}

/// Resource capability: server can list/subscribe and send resource notifications.
#[derive(Clone, Debug, PartialEq)]
pub struct ResourcesCapability {
    pub subscribe: bool,
    pub list_changed: bool,
}

/// Tool capability: server can send tool-list change notifications.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolsCapability {
    pub list_changed: bool,
}

/// Completions capability: server can send completions list change notifications; optional max batch size.
#[derive(Clone, Debug, PartialEq)]
pub struct CompletionsCapability {
    pub list_changed: bool,
    pub max_batch: Option<u32>,
}

/// Combined server capabilities struct (advertised at initialize).
#[derive(Clone, Debug, PartialEq)]
pub struct ServerCapabilities {
    pub logging: Option<LoggingCapability>,
    pub prompts: Option<PromptsCapability>,
    pub resources: Option<ResourcesCapability>,
    pub tools: Option<ToolsCapability>,
    pub completions: Option<CompletionsCapability>,
    pub experimental: Option<HashMap<String, HashMap<String, JsonValue>>>,
}

/// A resource identifier (URI), name, and optional description.
#[derive(Clone, Debug, PartialEq)]
pub struct Resource {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
}

/// A prompt template’s metadata: name, optional description, optional arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct Prompt {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Option<Vec<PromptArgument>>,
}

/// A single argument of a prompt: name, optional description, and required flag.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptArgument {
    pub name: String,
    pub description: Option<String>,
    pub required: bool,
}

/// A message within a prompt: role + content (text, image, or embedded resource).
#[derive(Clone, Debug, PartialEq)]
pub struct PromptMessage {
    pub role: Role,
    pub content: PromptContent,
}

/// Possible content of a prompt or message.
#[derive(Clone, Debug, PartialEq)]
pub enum PromptContent {
    Text(String),
    Image(ImageContent),
    Embedded(EmbeddedResource),
}

/// Simple text content.
#[derive(Clone, Debug, PartialEq)]
pub struct TextContent {
    pub text: String,
}

/// Image content: base64-encoded data (optional mime type if you extend).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageContent {
    pub data: String,
}

/// Embedded resource content: either text or binary URI with optional name.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddedResource {
    pub text: Option<String>,
    pub data: Option<String>,
    pub name: Option<String>,
}

/// Data or text content returned by a tool call.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolContent {
    Text(String),
    // Extend here if tools can return images/audio/etc.
}

/// A tool’s metadata: name, optional description, and input schema (JSON Schema).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolInfo {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: JsonValue,
}

/// A simple reference to a prompt by name.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptReference {
    pub name: String,
}

/// A simple reference to a resource by URI.
#[derive(Clone, Debug, PartialEq)]
pub struct ResourceReference {
    pub uri: String,
}

/// The contents of a resource read: either text or binary (base64).
#[derive(Clone, Debug, PartialEq)]
pub enum ResourceContent {
    Text(String),
    Binary(String),
}

/// A sampling message (role + content) for server-initiated sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingMessage {
    pub role: Role,
    pub content: PromptContent,
}

/// Wrap everything into a single enum for any MCP envelope.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    Req(Request),
    Res(Response),
    Notif(Notification),
}

/// JSON-RPC 2.0 Request envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub id: RequestId,
    pub method: String,
    pub params: JsonValue,
    pub meta: Option<JsonValue>, // _meta for progress token or custom
}

/// JSON-RPC 2.0 Response envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    pub id: RequestId,
    pub result: Option<JsonValue>,
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 Notification envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct Notification {
    pub method: String,
    pub params: JsonValue,
}

/// JSON-RPC 2.0 Error object.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<JsonValue>,
}

/// Custom error for (de)serialization failures.
#[derive(Debug)]
pub enum McpError {
    Parse(String),
    BadTop,
    BadField(&'static str),
}

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpError::Parse(msg) => write!(f, "Parse error: {}", msg),
            McpError::BadTop => write!(f, "Invalid top-level structure"),
            McpError::BadField(field) => write!(f, "Invalid field: {}", field),
        }
    }
}

impl std::error::Error for McpError {}

//─────────────────────────────────────────────────────────────────────────────
//  Shared Helper Functions (inlined / optimized for minimal heap churn)
//─────────────────────────────────────────────────────────────────────────────

/// Escape a Rust string to a JSON string literal (no surrounding quotes).
#[inline(always)]
pub(crate) fn json_escape(src: &str, out: &mut String) {
    for c in src.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                // Use `write!` to format as \uXXXX
                use core::fmt::Write;
                write!(out, "\\u{:04x}", c as u32).unwrap();
            }
            _ => out.push(c),
        }
    }
}

/// Parse a JSON-RPC `"id"` field (string or number) into a `RequestId`.
#[inline(always)]
pub(crate) fn id_from_json(v: &JsonValue) -> Result<RequestId, McpError> {
    if let Some(n) = v.as_i64() {
        Ok(RequestId::Num(n))
    } else if let Some(s) = v.as_str() {
        Ok(RequestId::Str(s.to_owned()))
    } else {
        Err(McpError::BadField("id"))
    }
}

/// Serialize a `RequestId` into JSON (append to `String`).
#[inline(always)]
pub(crate) fn id_to_json(id: &RequestId, out: &mut String) {
    match id {
        RequestId::Num(n) => out.push_str(&n.to_string()),
        RequestId::Str(s) => {
            out.push('"');
            json_escape(s, out);
            out.push('"');
        }
    }
}

/// Parse a TOML `"id"` field (string or integer) into a `RequestId`.
#[inline(always)]
pub(crate) fn id_from_toml(item: &toml_edit::Item) -> Result<RequestId, McpError> {
    match item.as_value() {
        Some(toml_edit::Value::Integer(i)) => Ok(RequestId::Num(*i.value())),
        Some(toml_edit::Value::String(s)) => Ok(RequestId::Str(s.value().to_owned())),
        _ => Err(McpError::BadField("id")),
    }
}

/// Serialize a `RequestId` into a TOML `Item`.
#[inline(always)]
pub(crate) fn id_to_toml(id: &RequestId) -> toml_edit::Item {
    match id {
        RequestId::Num(n) => toml_edit::Item::Value(toml_edit::Value::Integer(toml_edit::Formatted::new(*n))),
        RequestId::Str(s) => toml_edit::Item::Value(toml_edit::Value::from(s.as_str())),
    }
}

/// Convert `toml_edit::Item` into `simd_json::OwnedValue`.
#[inline(always)]
pub(crate) fn toml_to_owned(
    item: toml_edit::Item,
) -> Result<JsonValue, McpError> {
    Ok(match item {
        toml_edit::Item::None => JsonValue::Static(StaticNode::Null),
        toml_edit::Item::Value(v) => match v {
            toml_edit::Value::String(s) => {
                if s.value() == "__SIMD_JSON_NULL__" {
                    JsonValue::Static(StaticNode::Null)
                } else {
                    s.value().into()
                }
            },
            toml_edit::Value::Integer(i) => (*i.value()).into(),
            toml_edit::Value::Float(f) => (*f.value()).into(),
            toml_edit::Value::Boolean(b) => (*b.value()).into(),
            toml_edit::Value::Datetime(dt) => dt.value().to_string().into(),
            toml_edit::Value::Array(arr) => {
                let mut vec = Vec::with_capacity(arr.len());
                for elem in arr.iter() {
                    vec.push(toml_to_owned(toml_edit::Item::Value(elem.clone()))?);
                }
                vec.into()
            }
            toml_edit::Value::InlineTable(t) => {
                let mut map = HashMap::new();
                for (k, v) in t.iter() {
                    map.insert(k.to_owned(), toml_to_owned(toml_edit::Item::Value(v.clone()))?);
                }
                map.into()
            }
        },
        toml_edit::Item::Table(t) => {
            let mut map = HashMap::new();
            for (k, v) in t.iter() {
                map.insert(k.to_owned(), toml_to_owned(v.clone())?);
            }
            map.into()
        }
        toml_edit::Item::ArrayOfTables(arr) => {
            let mut vec = Vec::with_capacity(arr.len());
            for t in arr.iter() {
                vec.push(toml_to_owned(toml_edit::Item::Table(t.clone()))?);
            }
            vec.into()
        }
    })
}

/// Convert `simd_json::OwnedValue` into `toml_edit::Item`.
#[inline(always)]
pub(crate) fn owned_to_toml(v: &JsonValue) -> toml_edit::Item {
    match v {
        JsonValue::Static(StaticNode::Null) => {
            toml_edit::Item::Value(toml_edit::Value::from("__SIMD_JSON_NULL__"))
        }
        JsonValue::String(s) => {
            toml_edit::Item::Value(toml_edit::Value::from(s.as_str()))
        }
        JsonValue::Static(StaticNode::Bool(b)) => {
            toml_edit::Item::Value(toml_edit::Value::Boolean(toml_edit::Formatted::new(*b)))
        }
        JsonValue::Static(StaticNode::I64(n)) => {
            toml_edit::Item::Value(toml_edit::Value::Integer(toml_edit::Formatted::new(*n)))
        }
        JsonValue::Static(StaticNode::U64(n)) => {
            toml_edit::Item::Value(toml_edit::Value::Integer(toml_edit::Formatted::new(*n as i64)))
        }
        JsonValue::Static(StaticNode::F64(f)) => {
            toml_edit::Item::Value(toml_edit::Value::Float(toml_edit::Formatted::new(*f)))
        }
        JsonValue::Array(arr) => {
            let mut a = toml_edit::Array::new();
            for el in arr.iter() {
                if let toml_edit::Item::Value(val) = owned_to_toml(el) {
                    a.push(val);
                }
            }
            toml_edit::Item::Value(toml_edit::Value::Array(a))
        }
        JsonValue::Object(map) => {
            let mut tbl = toml_edit::Table::new();
            for (k, v) in map.iter() {
                tbl[k.as_str()] = owned_to_toml(v);
            }
            toml_edit::Item::Table(tbl)
        }
    }
}
