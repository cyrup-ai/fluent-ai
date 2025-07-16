use fluent_ai::prelude::*;
use fluent_ai::domain::context::Context;
use fluent_ai::domain::tool_v2::Tool;
use fluent_ai::domain::library::Library;
use fluent_ai::domain::agent_role::Stdio;
use fluent_ai_provider::Models;
use std::collections::HashMap;

// Mock provider types - these should be replaced with actual provider implementations
pub struct Mistral;
impl Mistral {
    pub const MagistralSmall: Models = Models::MistralSmallLatest;
}

pub struct Perplexity;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example demonstrating the pure ChatLoop pattern
    // All formatting, streaming, and I/O are handled automatically by the builder

    let stream = FluentAi::agent_role("rusty-squire")
        .completion_provider(Mistral::MagistralSmall)
        .temperature(1.0)
        .max_tokens(8000)
        .system_prompt("Act as a Rust developers 'right hand man'.
        You possess deep expertise in using tools to research rust, cargo doc and github libraries.
        You are a patient and thoughtful software artisan; a master of sequential thinking and step-by-step reasoning.
        You excel in compilation triage ...

        ...
        ...

        Today is {{ date }}

        ~ Be Useful, Not Thorough")
        .context((
            Context::file("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
            Context::files_glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
            Context::directory("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
            Context::github_glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
        ))
        .mcp_server::<Stdio>()
            .bin("/user/local/bin/sweetmcp")
            .init("cargo run -- --stdio")
        .tools((
            Tool::<Perplexity>::new(|| {
                let mut params = HashMap::new();
                params.insert("citations".to_string(), serde_json::json!("true"));
                params
            }),
            Tool::named("cargo")
                .bin("~/.cargo/bin")
                .description("cargo --help")
        ))
        .additional_params(|| {
            let mut params = HashMap::new();
            params.insert("beta".to_string(), serde_json::json!("true"));
            params
        })
        .memory(Library::named("obsidian_vault"))
        .metadata(|| {
            let mut metadata = HashMap::new();
            metadata.insert("key".to_string(), serde_json::json!("val"));
            metadata.insert("foo".to_string(), serde_json::json!("bar"));
            metadata
        })
        .on_tool_result(|_results| {
            // do stuff
        })
        .on_conversation_turn(|_conversation, _agent| {
            // log.info("Agent: " + conversation.last().message());
            // agent.chat(process_turn()) // your custom logic
        })
        .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
            println!("{:?}", chunk);   // stream response here or from the AsyncStream .chat() returns
            chunk                     // `.chat()` returns AsyncStream<MessageChunk> vs. AsyncStream<Result<MessageChunk>>
        })
        .into_agent() // Agent Now
        .conversation_history(conversation_history![
            MessageRole::User => "What time is it in Paris, France",
            MessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
            MessageRole::Assistant => "It's 1:45 AM CEST on July 7, 2025, in Paris, France. That's 9 hours ahead of your current time in Las Vegas."
        ])
        .chat("Hello"); // AsyncStream<MessageChunk>

    // Collect the stream to process results
    let _results: Vec<_> = stream.collect().await;

    Ok(())
}