//! Test the working CandleFluentAi builder pattern
//! This test verifies that the basic ARCHITECTURE.md syntax compiles

use fluent_ai_candle::builders::agent_role::{
    CandleFluentAi, CandleMessageRole, CandleChatLoop, CandleStdio,
};

fn main() {
    // Test basic builder pattern from ARCHITECTURE.md
    let _agent = CandleFluentAi::agent_role("test-agent")
        .temperature(1.0)
        .max_tokens(8000)
        .system_prompt("You are a helpful assistant")
        .additional_params([("beta", "true")])
        .metadata([("key", "val"), ("foo", "bar")])
        .on_tool_result(|_result| {
            // Handle tool result
        })
        .on_conversation_turn(|_conversation, _agent| {
            // Handle conversation turn
        })
        .into_agent()
        .conversation_history(
            (CandleMessageRole::User, "Hello world"),
            (CandleMessageRole::Assistant, "Hi there!")
        );

    println!("CandleFluentAi builder pattern works!");
    
    // Test chat loop pattern
    let _stream = _agent.chat_with_closure(|conversation| {
        let user_input = conversation.latest_user_message();
        
        if user_input.contains("finished") {
            CandleChatLoop::Break
        } else {
            CandleChatLoop::Reprompt("continue. use sequential thinking".to_string())
        }
    });
    
    println!("Chat loop pattern works!");
}