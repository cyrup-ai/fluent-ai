//
//=========================================================================
//  src/mcp/json.rs   –   Zero-Serde JSON for MCP (simd-json 0.15.1)
//  * SIMD parse (mutates input) via simd_json::to_owned_value
//  * Stack-backed SmallVec => zero heap alloc for ≤ 4 KiB packets
//  * #[inline(always)] on hot paths
//=========================================================================

#![allow(clippy::inline_always)]

use smallvec::SmallVec;
use simd_json::{to_owned_value, value::owned::Value as JsonValue, StaticNode};
use value_trait::prelude::*;

use super::{
    id_from_json, id_to_json, json_escape, JsonRpcError, McpError, Message, Notification, Request,
    Response,
};

impl Message {
    //───────────────────────────────────────────────────────────────────
    //  Parse JSON → Message
    //───────────────────────────────────────────────────────────────────
    #[inline(always)]
    pub fn from_json(src: &str) -> Result<Self, McpError> {
        // 0. Copy src into a stack buffer (≤4 KiB stays on stack)
        const STACK_CAP: usize = 4096;
        let mut buf: SmallVec<[u8; STACK_CAP]> = SmallVec::new();
        buf.extend_from_slice(src.as_bytes());

        // 1. SIMD parse (mutates buffer in-place)
        let mut dom: JsonValue =
            to_owned_value(buf.as_mut_slice()).map_err(|e| McpError::Parse(e.to_string()))?;
        let obj = dom.as_object_mut().ok_or(McpError::BadTop)?;

        // 2. Validate JSON-RPC version
        if obj.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
            return Err(McpError::BadField("jsonrpc"));
        }

        let has_id = obj.get("id").is_some();
        let method_opt = obj.get("method").map(|m| m.as_str().ok_or(McpError::BadField("method")).map(|s| s.to_owned())).transpose()?;

        //──────────────── Request / Notification ────────────────
        if let Some(method) = method_opt {
            let params = obj.remove("params").unwrap_or(JsonValue::Static(StaticNode::Null));

            // Request (has id)
            if has_id {
                let id_json = obj.get("id").ok_or(McpError::BadField("id"))?;
                let id = id_from_json(id_json)?;
                let meta = obj.remove("_meta");
                return Ok(Message::Req(Request {
                    id,
                    method,
                    params,
                    meta,
                }));
            }

            // Notification (no id)
            return Ok(Message::Notif(Notification { method, params }));
        }

        //──────────────────────── Response ──────────────────────
        let id_json = obj.get("id").ok_or(McpError::BadField("id"))?;
        let id = id_from_json(id_json)?;

        let error = obj
            .remove("error")
            .map(|err_val| {
                let eobj = err_val.as_object().ok_or(McpError::BadField("error"))?;
                Ok(JsonRpcError {
                    code: eobj
                        .get("code")
                        .and_then(|v| v.as_i64())
                        .ok_or(McpError::BadField("error.code"))?,
                    message: eobj
                        .get("message")
                        .and_then(|v| v.as_str())
                        .ok_or(McpError::BadField("error.message"))?
                        .to_owned(),
                    // ---------- NO iterator Map / Cloned here ----------
                    data: if let Some(d) = eobj.get("data") {
                        Some(d.clone())
                    } else {
                        None
                    },
                })
            })
            .transpose()?;

        let result = obj.remove("result");

        Ok(Message::Res(Response { id, result, error }))
    }

    //───────────────────────────────────────────────────────────────────
    //  Serialize Message → JSON
    //───────────────────────────────────────────────────────────────────
    #[inline(always)]
    pub fn to_json(&self) -> String {
        // Reserve 256 B to avoid most reallocations.
        let mut out = String::with_capacity(256);
        out.push_str(r#"{"jsonrpc":"2.0","#);

        match self {
            &Message::Req(ref r) => {
                out.push_str(r#""id":"#);
                id_to_json(&r.id, &mut out);

                out.push_str(r#","method":""#);
                json_escape(&r.method, &mut out);
                out.push('"');

                out.push_str(r#","params":"#);
                out.push_str(&r.params.encode());

                if let Some(meta) = &r.meta {
                    out.push_str(r#","_meta":"#);
                    out.push_str(&meta.encode());
                }
                out.push('}');
            }

            &Message::Notif(ref n) => {
                out.push_str(r#""method":""#);
                json_escape(&n.method, &mut out);
                out.push('"');

                if !n.params.is_null() {
                    out.push_str(r#","params":"#);
                    out.push_str(&n.params.encode());
                }
                out.push('}');
            }

            &Message::Res(ref r) => {
                out.push_str(r#""id":"#);
                id_to_json(&r.id, &mut out);

                if let Some(err) = &r.error {
                    out.push_str(",\"error\":{\"code\":");
                    out.push_str(&err.code.to_string());
                    out.push_str(r#","message":""#);
                    json_escape(&err.message, &mut out);
                    out.push('"');

                    if let Some(d) = &err.data {
                        out.push_str(",\"data\":");
                        out.push_str(&d.encode());
                    }
                    out.push_str(r#"}}"#);
                } else if let Some(res) = &r.result {
                    out.push_str(",\"result\":");
                    out.push_str(&res.encode());
                    out.push('}');
                } else {
                    out.push_str(r#","result":null}"#);
                }
            }
        }

        out
    }
}