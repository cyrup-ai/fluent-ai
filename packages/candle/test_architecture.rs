// Test that ARCHITECTURE.md syntax still works after all our fixes
use crate::builders::agent_role::CandleFluentAi;

fn test_architecture_syntax() {
    // This should compile and demonstrate the ARCHITECTURE.md builder pattern works
    let _agent = CandleFluentAi::agent_role("test-agent")
        .temperature(0.7)
        .max_tokens(1000)
        .system_prompt("You are a helpful assistant")
        .into_agent();
        
    println!("✅ ARCHITECTURE.md syntax test passed!");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn architecture_builder_pattern_works() {
        test_architecture_syntax();
    }
}