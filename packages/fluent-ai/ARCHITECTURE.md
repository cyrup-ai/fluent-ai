# Fluent AI Architecture

## AgentRole

```rust
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
    .context( // trait Context
        Context<File>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
        Context<Files>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
        Context<Directory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
        Context<Github>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
    )
    .mcp_server<Stdio>::bin("/user/local/bin/sweetmcp").init("cargo run -- --stdio")
    .tools( // trait Tool
        Tool<Perplexity>::new({
            "citations" => "true"
        }),
        Tool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
    ) // ZeroOneOrMany `Tool` || `McpTool` || NamedTool (WASM)

    .additional_params({"beta" =>  "true"})
    .memory(Library::named("obsidian_vault"))
    .metadata({ "key" => "val", "foo" => "bar" })
    .on_tool_result(|results| {
        // do stuff
    })
    .on_conversation_turn(|conversation, agent| {
        log.info("Agent: " + conversation.last().message())
        agent.chat(process_turn()) // your custom logic
    })
    .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
        Ok => chunk.into()       // `.chat()` returns AsyncStream<MessageChunk> vs. AsyncStream<Result<MessageChunk>>
        println!("{}", chunk);   // stream response here or from the AsyncStream .chat() returns
    })
    .into_agent() // Agent Now
    .conversation_history(MessageRole::User => "What time is it in Paris, France",
            MessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
            MessageRole::Assistant => "It’s 1:45 AM CEST on July 7, 2025, in Paris, France. That’s 9 hours ahead of your current time in Las Vegas.")
    .chat(|conversation| {
        let user_input = conversation.latest_user_message();
        
        if user_input.contains("finished") {
            ChatLoop::Break
        } else {
            ChatLoop::Reprompt("continue. use sequential thinking")
        }
    })?

// Full Example with Pure ChatLoop Pattern:
FluentAi::agent_role("helpful assistant")
    .completion_provider(Providers::OpenAI)
    .model(Models::Gpt4OMini)
    .temperature(0.7)
    .on_chunk(|chunk| {
        // Real-time streaming - print each token as it arrives
        // All formatting and coloring happens automatically here
        print!("{}", chunk);
        io::stdout().flush().unwrap();
    })
    .chat(|conversation| {
        let user_input = conversation.latest_user_message();
        
        // Pure logic - no formatting, just conversation flow control
        match user_input.to_lowercase().as_str() {
            "quit" | "exit" | "bye" => {
                ChatLoop::Break
            },
            input if input.starts_with("/help") => {
                ChatLoop::Reprompt("Available commands: /help, quit/exit/bye, or just chat normally!".to_string())
            },
            input if input.contains("code") => {
                let response = format!(
                    "I see you mentioned code! Here's a Rust example: fn main() {{ println!(\"Hello!\"); }} Need help with a specific language?"
                );
                ChatLoop::Reprompt(response)
            },
            _ => {
                // Simple response - builder handles all formatting automatically
                let response = format!(
                    "I understand: '{}'. How can I help you further?", 
                    user_input
                );
                ChatLoop::Reprompt(response)
            }
        }
    })?
    .collect();
```

## Agent

```rust
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
    .context( // trait Context
        Context<File>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
        Context<Files>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
        Context<Directory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
        Context<Github>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
    )
    .mcp_server<Stdio>::bin("/user/local/bin/sweetmcp").init("cargo run -- --stdio")
    .tools( // trait Tool
        Tool<Perplexity>::new({
            "citations" => "true"
        }),
        Tool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
    ) // ZeroOneOrMany `Tool` || `McpTool` || NamedTool (WASM)

    .additional_params({"beta" =>  "true"})
    .memory(Library::named("obsidian_vault"))
    .metadata({ "key" => "val", "foo" => "bar" })
    .on_tool_result(|results| {
        // do stuff
    })
    .on_conversation_turn(|conversation, agent| {
        log.info("Agent: " + conversation.last().message())
        agent.chat(process_turn()) // your custom logic
    })
    .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
        Ok => chunk.into()       // `.chat()` returns AsyncStream<MessageChunk> vs. AsyncStream<Result<MessageChunk>>
        println!("{}", chunk);   // stream response here or from the AsyncStream .chat() returns
    })
    .into_agent() // Agent Now
    .conversation_history(MessageRole::User => "What time is it in Paris, France",
            MessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
            MessageRole::Assistant => "It’s 1:45 AM CEST on July 7, 2025, in Paris, France. That’s 9 hours ahead of your current time in Las Vegas.")
    .chat("Hello")? // AsyncStream<MessageChunk
    .collect()
```

## Agent

```rust
//  DO NOT MODIFY !!!  DO NOT MODIFY !!!
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
    .context( // trait Context
        Context<File>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
        Context<Files>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
        Context<Directory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
        Context<Github>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
    )
    .mcp_server<Stdio>::bin("/user/local/bin/sweetmcp").init("cargo run -- --stdio")
    .tools( // trait Tool
        Tool<Perplexity>::new({
            "citations" => "true"
        }),
        Tool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
    ) // ZeroOneOrMany `Tool` || `McpTool` || NamedTool (WASM)

    .additional_params({"beta" =>  "true"})
    .memory(Library::named("obsidian_vault"))
    .metadata({ "key" => "val", "foo" => "bar" })
    .on_tool_result(|results| {
        // do stuff
    })
    .on_conversation_turn(|conversation, agent| {
        log.info("Agent: " + conversation.last().message())
        agent.chat(process_turn()) // your custom logic
    })
    .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
        Ok => chunk.into()       // `.chat()` returns AsyncStream<MessageChunk> vs. AsyncStream<Result<MessageChunk>>
        println!("{}", chunk);   // stream response here or from the AsyncStream .chat() returns
    })
    .into_agent() // Agent Now
    .conversation_history(MessageRole::User => "What time is it in Paris, France",
            MessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
            MessageRole::Assistant => "It’s 1:45 AM CEST on July 7, 2025, in Paris, France. That’s 9 hours ahead of your current time in Las Vegas.")
    .chat("Hello")? // AsyncStream<MessageChunk
    .collect();
// DO NOT MODIFY !!!  DO NOT MODIFY !!!
```
