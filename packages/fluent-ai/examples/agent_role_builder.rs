let stream = FluentAi::agent_role("rusty-squire")
    .completion_provider(Candle::KimiK2)
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
    .mcp_server<Stdio>().bin("/user/local/bin/sweetmcp").init("cargo run -- --stdio")
    .tools( // trait Tool
        Tool<Perplexity>::new({
            "citations" => "true"
        }),
        Tool::named("cargo").bin("~/.cargo/bin").description("cargo --help".exec_to_text())
    ) // ZeroOneOrMany `Tool` || `McpTool` || NamedTool (WASM)

    .additional_params([("beta", "true")])
    .memory(Library::named("obsidian_vault"))
    .metadata([("key", "val"), ("foo", "bar")])
    .on_tool_result(|results| {
        // do stuff
    })
    .on_conversation_turn(|conversation, agent| {
        log.info("Agent: " + conversation.last().message())
        agent.chat(process_turn()) // your custom logic
    })
    .on_chunk(|chunk| {          // unwrap chunk closure :: NOTE: THIS MUST PRECEDE .chat()
        Ok => chunk.into(),
        Err(error) => {
            eprintln!("Error: {}", error);
            BadChunk::from_err(error);
        }
    })
    .into_agent() // Agent Now
    .conversation_history(
        MessageRole::User => "What time is it in Paris, France",
        MessageRole::System => "The USER is inquiring about the time in Paris, France. Based on their IP address, I see they are currently in Las Vegas, Nevada, USA. The current local time is 16:45",
        MessageRole::Assistant => "It's 1:45 AM CEST on July 7, 2025, in Paris, France. That's 9 hours ahead of your current time in Las Vegas."
    )
    .chat(|conversation| {
        let user_input = conversation.latest_user_message();

        if user_input.contains("finished") {
            ChatLoop::Break
        } else {
            ChatLoop::Reprompt("continue. use sequential thinking")
        }
    })
    .collect();