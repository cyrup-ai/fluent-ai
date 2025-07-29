//! Test ARCHITECTURE.md builder pattern compilation
//!
//! This example tests the exact syntax from ARCHITECTURE.md to ensure
//! the Candle builder pattern compiles and works correctly.

use fluent_ai_candle::{CandleFluentAi, CandleKimiK2Provider};

fn main() {
    // Test the basic ARCHITECTURE.md syntax
    let _stream = CandleFluentAi::agent_role("rusty-squire")
        .completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))
        .temperature(1.0)
        .max_tokens(8000)
        .system_prompt("Act as a Rust developers 'right hand man'.")
        .into_agent();
        
    println!("ARCHITECTURE.md builder pattern compiled successfully!");
}