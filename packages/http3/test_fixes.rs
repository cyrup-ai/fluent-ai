#!/usr/bin/env rust-script

//! Quick test to verify JSONPath fixes work

use bytes::Bytes;

fn main() {
    println!("Testing JSONPath fixes...");
    
    // Test 1: Double dot validation fix
    println!("\n1. Testing $.store..book (should be invalid):");
    match fluent_ai_http3::json_path::JsonPathParser::compile("$.store..book") {
        Ok(_) => println!("❌ FAILED: $.store..book was accepted (should be rejected)"),
        Err(e) => println!("✅ PASSED: $.store..book rejected - {}", e),
    }
    
    // Test 2: Null filter test
    println!("\n2. Testing null filter logic:");
    let json_data = r#"{
        "store": {
            "book": [
                {"metadata": null, "title": "Book 1"},
                {"title": "Book 2"},
                {"title": "Book 3"}
            ]
        }
    }"#;
    
    let mut stream = fluent_ai_http3::json_path::JsonArrayStream::<serde_json::Value>::new("$.store.book[?@.metadata == null]");
    let chunk = Bytes::from(json_data);
    let results: Vec<_> = stream.process_chunk(chunk).collect();
    
    if results.len() == 1 {
        println!("✅ PASSED: Null filter returned {} result (expected 1)", results.len());
    } else {
        println!("❌ FAILED: Null filter returned {} results (expected 1)", results.len());
    }
    
    println!("\nTest completed!");
}