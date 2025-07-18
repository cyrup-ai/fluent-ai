use fluent_ai::domain::agent_role::Stdio;
use fluent_ai::domain::context::{Context, Directory, File, Files, Github};
use fluent_ai::domain::library::Library;
use fluent_ai::domain::message::MessageRole;
use fluent_ai::domain::tool::{ExecToText, Tool};
use fluent_ai::json_map;
use fluent_ai::prelude::*;
use fluent_ai_provider::Models;
use futures::StreamExt;
// use cyrup_sugars::*;

// Mock provider types - these should be replaced with actual provider implementations
pub struct Mistral;
impl Mistral {
    pub const MAGISTRAL_SMALL: Models = Models::MistralSmallLatest;
}

pub struct Perplexity;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example demonstrating the pure ChatLoop pattern
    // All formatting, streaming, and I/O are handled automatically by the builder

    let stream = FluentAi::agent_role("rusty-squire")
        .completion_provider(Mistral::MAGISTRAL_SMALL)
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
            Context::<File>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
            Context::<Files>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
            Context::<Directory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
            Context::<Github>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
        ))
        .mcp_server::<Stdio>()
            .bin("/user/local/bin/sweetmcp")
            .init("cargo run -- --stdio")
        .tools((
            Tool::<Perplexity>::new(json_map!{"citations" => "true"}),
            Tool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
        ))
        .additional_params(json_map!{"beta" => "true"})
        .memory(Library::named("obsidian_vault"))
        .metadata(json_map!{"key" => "val", "foo" => "bar"})
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
        .conversation_history((
            (MessageRole::User, "What time is it in Paris, France"),
            (MessageRole::System, "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45"),
            (MessageRole::Assistant, "It's 1:45 AM CEST on July 7, 2025, in Paris, France. That's 9 hours ahead of your current time in Las Vegas.")
        ))
        .chat("Hello"); // AsyncStream<MessageChunk>

    // Process the stream
    let _results: Vec<_> = stream.collect().await;

    Ok(())
}
