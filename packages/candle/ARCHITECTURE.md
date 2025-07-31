# Candle AI Architecture

## CandleAgentRole

```rust
let stream = CandleFluentAi::agent_role("rusty-squire")
    .completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))
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
    .context( // trait CandleContext
        CandleContext<CandleFile>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
        CandleContext<CandleFiles>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
        CandleContext<CandleDirectory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
        CandleContext<CandleGithub>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
    )
    .mcp_server<CandleStdio>().bin("/user/local/bin/sweetmcp").init("cargo run -- --stdio")
    .tools( // trait CandleTool
        CandleTool<CandlePerplexity>::new([
            ("citations", "true")
        ]),
        CandleTool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
    ) // CandleZeroOneOrMany `CandleTool` || `CandleMcpTool` || CandleNamedTool (WASM)

    .additional_params([("beta", "true")])
    .memory(CandleLibrary::named("obsidian_vault"))
    .metadata([("key", "val"), ("foo", "bar")])
    .on_tool_result(|results| {
        // do stuff
    })
    .on_conversation_turn(|conversation, agent| {
        log.info("Agent: " + conversation.last().message())
        agent.chat(process_turn()) // your custom logic
    })
    .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
        println!("{}", chunk);   // stream response here or from the AsyncStream .chat() returns
        chunk
    })
    .into_agent() // CandleAgent Now
    .conversation_history(
        CandleMessageRole::User => "What time is it in Paris, France",
        CandleMessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
        CandleMessageRole::Assistant => "It's 1:45 AM CEST on July 7, 2025, in Paris, France. That's 9 hours ahead of your current time in Las Vegas."
    )
    .chat(|conversation| {
        let user_input = conversation.latest_user_message();
        
        if user_input.contains("finished") {
            CandleChatLoop::Break
        } else {
            CandleChatLoop::Reprompt("continue. use sequential thinking")
        }
    })
    .collect()

// Full Example with Pure CandleChatLoop Pattern:
CandleFluentAi::agent_role("helpful assistant")
    .completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))
    .model(CandleModels::KimiK2)
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
                CandleChatLoop::Break
            },
            input if input.starts_with("/help") => {
                CandleChatLoop::Reprompt("Available commands: /help, quit/exit/bye, or just chat normally!".to_string())
            },
            input if input.contains("code") => {
                let response = format!(
                    "I see you mentioned code! Here's a Rust example: fn main() {{ println!(\"Hello!\"); }} Need help with a specific language?"
                );
                CandleChatLoop::Reprompt(response)
            },
            _ => {
                // Simple response - builder handles all formatting automatically
                let response = format!(
                    "I understand: '{}'. How can I help you further?", 
                    user_input
                );
                CandleChatLoop::Reprompt(response)
            }
        }
    })
    .collect();
```

## CandleAgent

```rust
let stream = CandleFluentAi::agent_role("rusty-squire")
    .completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))
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
    .context( // trait CandleContext
        CandleContext<CandleFile>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
        CandleContext<CandleFiles>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
        CandleContext<CandleDirectory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
        CandleContext<CandleGithub>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
    )
    .mcp_server<CandleStdio>().bin("/user/local/bin/sweetmcp").init("cargo run -- --stdio")
    .tools( // trait CandleTool
        CandleTool<CandlePerplexity>::new([
            ("citations", "true")
        ]),
        CandleTool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
    ) // CandleZeroOneOrMany `CandleTool` || `CandleMcpTool` || CandleNamedTool (WASM)

    .additional_params([("beta", "true")])
    .memory(CandleLibrary::named("obsidian_vault"))
    .metadata([("key", "val"), ("foo", "bar")])
    .on_tool_result(|results| {
        // do stuff
    })
    .on_conversation_turn(|conversation, agent| {
        log.info("Agent: " + conversation.last().message())
        agent.chat(process_turn()) // your custom logic
    })
    .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
        println!("{}", chunk);   // stream response here or from the AsyncStream .chat() returns
        chunk
    })
    .into_agent() // CandleAgent Now
    .conversation_history(
        CandleMessageRole::User => "What time is it in Paris, France",
        CandleMessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
        CandleMessageRole::Assistant => "It's 1:45 AM CEST on July 7, 2025, in Paris, France. That's 9 hours ahead of your current time in Las Vegas."
    )
    .chat("Hello") // AsyncStream<CandleMessageChunk>
    .collect()
```

## CandleAgent

```rust
//  DO NOT MODIFY !!!  DO NOT MODIFY !!!
let stream = CandleFluentAi::agent_role("rusty-squire")
    .completion_provider(CandleKimiK2Provider::new("./models/kimi-k2"))
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
    .context( // trait CandleContext
        CandleContext<CandleFile>::of("/home/kloudsamurai/ai_docs/mistral_agents.pdf"),
        CandleContext<CandleFiles>::glob("/home/kloudsamurai/cyrup-ai/**/*.{md,txt}"),
        CandleContext<CandleDirectory>::of("/home/kloudsamurai/cyrup-ai/agent-role/ambient-rust"),
        CandleContext<CandleGithub>::glob("/home/kloudsamurai/cyrup-ai/**/*.{rs,md}")
    )
    .mcp_server<CandleStdio>().bin("/user/local/bin/sweetmcp").init("cargo run -- --stdio")
    .tools( // trait CandleTool
        CandleTool<CandlePerplexity>::new([
            ("citations", "true")
        ]),
        CandleTool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
    ) // CandleZeroOneOrMany `CandleTool` || `CandleMcpTool` || CandleNamedTool (WASM)

    .additional_params([("beta", "true")])
    .memory(CandleLibrary::named("obsidian_vault"))
    .metadata([("key", "val"), ("foo", "bar")])
    .on_tool_result(|results| {
        // do stuff
    })
    .on_conversation_turn(|conversation, agent| {
        log.info("Agent: " + conversation.last().message())
        agent.chat(process_turn()) // your custom logic
    })
    .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
        println!("{}", chunk);   // stream response here or from the AsyncStream .chat() returns
        chunk
    })
    .into_agent() // CandleAgent Now
    .conversation_history(
        CandleMessageRole::User => "What time is it in Paris, France",
        CandleMessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
        CandleMessageRole::Assistant => "It's 1:45 AM CEST on July 7, 2025, in Paris, France. That's 9 hours ahead of your current time in Las Vegas."
    )
    .chat("Hello") // AsyncStream<CandleMessageChunk>
    .collect();
// DO NOT MODIFY !!!  DO NOT MODIFY !!!
```