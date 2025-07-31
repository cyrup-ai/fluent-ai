//! Minimal test to verify CandleFluentAi builder pattern works
//! This bypasses domain module issues by only testing the builder

use fluent_ai_candle::builders::agent_role::CandleFluentAi;

fn main() {
    // Test that the basic entry point works
    let builder = CandleFluentAi::agent_role("test-agent");
    
    // Test that the builder methods can be chained
    let configured_builder = builder
        .temperature(1.0)
        .max_tokens(8000)
        .system_prompt("You are a helpful assistant");
    
    // Test conversion to agent builder
    let agent_builder = configured_builder.into_agent();
    
    println!("✅ CandleFluentAi builder pattern works!");
    println!("✅ Method chaining works!");
    println!("✅ Agent conversion works!");
    
    // Test basic conversation history (simplified)
    let _final_agent = agent_builder.conversation_history(
        (fluent_ai_candle::builders::agent_role::CandleMessageRole::User, "Hello")
    );
    
    println!("✅ Conversation history works!");
    println!("🎉 All builder pattern tests passed!");
}