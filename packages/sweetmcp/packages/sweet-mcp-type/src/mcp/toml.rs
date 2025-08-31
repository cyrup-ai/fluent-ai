//=========================================================================
//  src/mcp/toml.rs
//  ----------------------------------------------------------------------
//  Zero-Serde TOML (de)serialization for MCP messages via toml_edit.
//  – #[inline(always)] on hot paths
//  – No `.cloned()`; uses map(|v| v.clone()) to avoid Cloned<Iter> types
//=========================================================================
use toml_edit::{DocumentMut, Item, Table, Value};

use super::{
    JsonRpcError, McpError, Message, Notification, Request, Response, id_from_toml,
    id_to_toml, owned_to_toml, toml_to_owned,
};
use simd_json::{value::owned::Value as JsonValue, StaticNode};

impl Message {
    //───────────────────────────────────────────────────────────────────
    //  Parse TOML text → Message
    //───────────────────────────────────────────────────────────────────
    #[inline(always)]
    pub fn from_toml(text: &str) -> Result<Self, McpError> {
        let doc: DocumentMut = text.parse().map_err(|e: toml_edit::TomlError| McpError::Parse(e.to_string()))?;

        if doc["jsonrpc"].as_str() != Some("2.0") {
            return Err(McpError::BadField("jsonrpc"));
        }

        let method_item = doc.get("method");
        let id_item = doc.get("id");

        if let Some(m_item) = method_item {
            // Request or Notification
            let method = m_item
                .as_str()
                .ok_or(McpError::BadField("method"))?
                .to_owned();

            let params = doc
                .get("params")
                .map(|v| toml_to_owned(v.clone()))
                .transpose()?
                .unwrap_or(JsonValue::Static(StaticNode::Null));

            if let Some(id_it) = id_item {
                // Request
                let req_id = id_from_toml(id_it)?;
                let meta = doc
                    .get("_meta")
                    .map(|v| toml_to_owned(v.clone()))
                    .transpose()?;
                return Ok(Message::Req(Request {
                    id: req_id,
                    method,
                    params,
                    meta,
                }));
            }

            // Notification
            return Ok(Message::Notif(Notification { method, params }));
        }

        // Response
        let id_it = id_item.ok_or(McpError::BadField("id"))?;
        let resp_id = id_from_toml(id_it)?;

        let error = doc
            .get("error")
            .map(|ei| {
                let eo = ei.as_table().ok_or(McpError::BadField("error"))?;
                Ok(JsonRpcError {
                    code: eo["code"]
                        .as_integer()
                        .ok_or(McpError::BadField("error.code"))? as i64,
                    message: eo["message"]
                        .as_str()
                        .ok_or(McpError::BadField("error.message"))?
                        .to_owned(),
                    data: eo
                        .get("data")
                        .map(|v| toml_to_owned(v.clone()))
                        .transpose()?,
                })
            })
            .transpose()?;

        let result = doc
            .get("result")
            .map(|v| toml_to_owned(v.clone()))
            .transpose()?;

        Ok(Message::Res(Response {
            id: resp_id,
            result,
            error,
        }))
    }

    //───────────────────────────────────────────────────────────────────
    //  Serialize Message → TOML text
    //───────────────────────────────────────────────────────────────────
    #[inline(always)]
    pub fn to_toml(&self) -> String {
        let mut doc = DocumentMut::new();
        doc["jsonrpc"] = Item::Value(Value::from("2.0"));

        match self {
            //---------------------------------------------------------
            Message::Req(r) => {
                doc["id"] = id_to_toml(&r.id);
                doc["method"] = Item::Value(Value::from(r.method.as_str()));
                doc["params"] = owned_to_toml(&r.params);
                if let Some(meta) = &r.meta {
                    doc["_meta"] = owned_to_toml(meta);
                }
            }

            //---------------------------------------------------------
            Message::Notif(n) => {
                doc["method"] = Item::Value(Value::from(n.method.as_str()));
                if !matches!(n.params, JsonValue::Static(StaticNode::Null)) {
                    doc["params"] = owned_to_toml(&n.params);
                }
            }

            //---------------------------------------------------------
            Message::Res(r) => {
                doc["id"] = id_to_toml(&r.id);
                if let Some(err) = &r.error {
                    let mut tbl = Table::new();
                    tbl["code"] = Item::Value(Value::Integer(toml_edit::Formatted::new(err.code)));
                    tbl["message"] = Item::Value(Value::from(err.message.as_str()));
                    if let Some(d) = &err.data {
                        tbl["data"] = owned_to_toml(d);
                    }
                    doc["error"] = Item::Table(tbl);
                } else if let Some(res) = &r.result {
                    doc["result"] = owned_to_toml(res);
                } else {
                    doc["result"] = Item::Value(Value::from("null"));
                }
            }
        }

        doc.to_string()
    }
}
