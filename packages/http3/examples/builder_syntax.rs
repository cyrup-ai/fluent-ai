//! Example usage of Http3 builder with exact user syntax patterns

use std::collections::HashMap;

use axum::{
    Router, extract::Json, http::{StatusCode, HeaderMap}, response::Json as ResponseJson, routing::post,
};
use cyrup_sugars::ZeroOneOrMany;
use fluent_ai_http3::Http3;
use http::{HeaderValue, HeaderName};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

#[derive(Serialize, Deserialize, Debug)]
struct SerdeRequestType {
    message: String,
    data: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug)]
struct SerdeResponseType {
    result: String,
    count: u32,
}

// Handler for test server that logs received payload and headers
async fn handle_post(
    headers: HeaderMap,
    Json(payload): Json<SerdeRequestType>,
) -> Result<ResponseJson<SerdeResponseType>, StatusCode> {
    println!("🚀 Server received payload: {:#?}", payload);
    println!("📋 Server received headers:");
    for (name, value) in headers.iter() {
        if let Ok(value_str) = value.to_str() {
            println!("   {}: {}", name, value_str);
        } else {
            println!("   {}: <binary_data>", name);
        }
    }
    println!();

    let response = SerdeResponseType {
        result: format!("Processed: {}", payload.message),
        count: payload.data.len() as u32,
    };

    println!("📤 Server responding with: {:#?}", response);
    Ok(ResponseJson(response))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start local test server on random port
    let app = Router::new().route("/test", post(handle_post));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let local_addr = listener.local_addr()?;
    
    println!("🌍 Test server starting on {}", local_addr);
    
    // Spawn server in background
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    
    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Create test request
    let request = SerdeRequestType {
        message: "Hello HTTP3 Builder!".to_string(),
        data: vec!["test".to_string(), "data".to_string()],
    };

    let server_url = format!("http://{}/test", local_addr);

    println!("📡 Testing Http3 builder with local server...");

    println!("🧪 Testing Http3 builder with exact syntax patterns...\n");

    // Test 1: Stream of HttpChunk with headers using ZeroOneOrMany
    println!("📡 Test 1: POST with custom headers");
    let _stream1 = Http3::json()
        .headers(ZeroOneOrMany::One((HeaderName::from_static("x-api-key"), HeaderValue::from_static("test-key-123"))))
        .body(&request)
        .post(&server_url);
    println!("✅ Test 1 completed\n");

    // Test 2: API key shorthand method
    println!("📡 Test 2: POST with api_key shorthand");
    let _stream2 = Http3::json()
        .api_key("shorthand-api-key")
        .body(&request)
        .post(&server_url);
    println!("✅ Test 2 completed\n");

    // Test 3: Basic auth test  
    println!("📡 Test 3: POST with basic authentication");
    let _stream3 = Http3::form_urlencoded()
        .basic_auth("testuser", Some("testpass"))
        .body(&request)
        .post(&server_url);
    println!("✅ Test 3 completed\n");

    // Test 4: Multiple headers
    println!("📡 Test 4: POST with multiple headers");
    let _stream4 = Http3::json()
        .headers(ZeroOneOrMany::Many(vec![
            (HeaderName::from_static("x-api-key"), HeaderValue::from_static("multi-header-key")),
            (HeaderName::from_static("authorization"), HeaderValue::from_static("Bearer token123")),
            (HeaderName::from_static("user-agent"), HeaderValue::from_static("Http3-Test-Client/1.0"))
        ]))
        .body(&request)
        .post(&server_url);
    println!("✅ Test 4 completed\n");

    println!("🎉 All HTTP3 builder tests completed successfully!");

    // Give the server a moment to process all requests
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    println!("🎯 HTTP3 builder example completed!");
    Ok(())
}