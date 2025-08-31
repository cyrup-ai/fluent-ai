//! tests/roundtrip.rs
//! ─────────────────────────
//! Unit tests to ensure JSON ↔ struct ↔ TOML ↔ struct ↔ JSON round-trip
//! fidelity for every MCP envelope type.

use sweet_mcp_type::mcp::{
    json::*, toml::*, JsonRpcError, Message, Notification, Request, RequestId, Response,
};
use simd_json::{value::owned::Value as JsonValue, StaticNode};

/// Sample Request: { "id": 1, "method": "ping", "params": null }
fn sample_request() -> Message {
    Message::Req(Request {
        id: RequestId::Num(1),
        method: "ping".into(),
        params: JsonValue::Static(StaticNode::Null),
        meta: None,
    })
}

/// Sample Notification: { "method": "notifications/initialized", "params": null }
fn sample_notification() -> Message {
    Message::Notif(Notification {
        method: "notifications/initialized".into(),
        params: JsonValue::Static(StaticNode::Null),
    })
}

/// Sample Response: { "id": 1, "result": "pong" }
fn sample_response() -> Message {
    Message::Res(Response {
        id: RequestId::Num(1),
        result: Some(JsonValue::from("pong")),
        error: None,
    })
}

/// Sample Error Response: { "id": "abc", "error": { "code": -32601, "message": "method not found" } }
fn sample_error_response() -> Message {
    Message::Res(Response {
        id: RequestId::Str("abc".into()),
        result: None,
        error: Some(JsonRpcError {
            code: -32601,
            message: "method not found".into(),
            data: None,
        }),
    })
}

/// Assert: struct → JSON → struct → JSON (bytes must match)
fn assert_json_roundtrip(orig: &Message) {
    let j1 = orig.to_json();
    println!("Generated JSON: {}", j1);
    let parsed = Message::from_json(&j1).unwrap();
    assert_eq!(orig, &parsed, "struct → JSON → struct mismatch");
    let j2 = parsed.to_json();
    assert_eq!(j1, j2, "JSON re-emission changed bytes");
}

/// Assert: struct → TOML → struct → TOML (strings must match)
fn assert_toml_roundtrip(orig: &Message) {
    let t1 = orig.to_toml();
    let parsed = Message::from_toml(&t1).unwrap();
    assert_eq!(orig, &parsed, "struct → TOML → struct mismatch");
    let t2 = parsed.to_toml();
    assert_eq!(t1, t2, "TOML re-emission changed string");
}

/// Assert: JSON → struct → TOML → struct → JSON yields identical JSON bytes.
fn assert_cross_roundtrip(orig: &Message) {
    let j1 = orig.to_json();
    let s1 = Message::from_json(&j1).unwrap();
    let t = s1.to_toml();
    let s2 = Message::from_toml(&t).unwrap();
    let j2 = s2.to_json();
    assert_eq!(j1, j2, "JSON↔TOML↔JSON cycle lost fidelity");
}

#[test]
fn request_roundtrip() {
    let m = sample_request();
    assert_json_roundtrip(&m);
    assert_toml_roundtrip(&m);
    assert_cross_roundtrip(&m);
}

#[test]
fn notification_roundtrip() {
    let m = sample_notification();
    assert_json_roundtrip(&m);
    assert_toml_roundtrip(&m);
    assert_cross_roundtrip(&m);
}

#[test]
fn response_roundtrip() {
    let m = sample_response();
    assert_json_roundtrip(&m);
    assert_toml_roundtrip(&m);
    assert_cross_roundtrip(&m);
}

#[test]
fn error_response_roundtrip() {
    let m = sample_error_response();
    assert_json_roundtrip(&m);
    assert_toml_roundtrip(&m);
    assert_cross_roundtrip(&m);
}
