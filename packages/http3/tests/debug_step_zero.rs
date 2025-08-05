//! Debug test for step=0 validation issue

use fluent_ai_http3::json_path::compiler::JsonPathParser;

#[test]
fn debug_step_zero_validation() {
    println!("Testing step=0 validation...");
    
    // This should fail because step cannot be zero
    let result = JsonPathParser::compile("$[1:5:0]");
    
    match result {
        Ok(_) => {
            println!("❌ PROBLEM: $[1:5:0] was accepted but should be rejected!");
            panic!("Step=0 should be invalid but was accepted");
        }
        Err(e) => {
            println!("✅ CORRECT: $[1:5:0] correctly rejected with error: {}", e);
            assert!(e.to_string().contains("step value cannot be zero"));
        }
    }
}

#[test]
fn debug_valid_step() {
    println!("Testing valid step...");
    
    // This should succeed
    let result = JsonPathParser::compile("$[1:5:2]");
    
    match result {
        Ok(_) => {
            println!("✅ CORRECT: $[1:5:2] correctly accepted");
        }
        Err(e) => {
            println!("❌ PROBLEM: $[1:5:2] was rejected but should be accepted: {}", e);
            panic!("Valid step should be accepted but was rejected");
        }
    }
}