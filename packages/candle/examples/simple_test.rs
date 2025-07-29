//! Minimal test to verify CandleFluentAi entry point works

use fluent_ai_candle::CandleFluentAi;

fn main() {
    // Test that CandleFluentAi::agent_role() method exists and returns a builder
    let _builder = CandleFluentAi::agent_role("test-agent");
    
    println!("CandleFluentAi entry point works!");
}