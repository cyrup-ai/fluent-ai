use clap::Parser;
use fluent_ai_provider::{Model, Models, Provider, Providers};
use futures::StreamExt;
use std::io::{self, Write};
use tokio;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "fluent-ai-rig")]
#[command(about = "A CLI for interacting with FluentAI using various models")]
struct Cli {
    /// Provider to use (e.g., openai, anthropic, github, openrouter)
    #[arg(short, long, default_value = "mistral")]
    provider: String,

    /// Model to use for completions (e.g., gpt-4o-mini, claude-3-sonnet)
    #[arg(short, long, default_value = "magistral-medium-latest")]
    model: String,

    /// Temperature for generation (0.0 to 2.0)
    #[arg(short, long, default_value = "0.7")]
    temperature: f32,

    /// Agent role/persona to use
    #[arg(short, long, default_value = "assistant")]
    agent_role: String,

    /// Context files, directories, globs, or GitHub refs
    #[arg(short, long, value_delimiter = ',')]
    context: Vec<String>,

    /// Interactive mode (default: true if no prompt provided)
    #[arg(short, long)]
    interactive: bool,

    /// Single prompt to process (non-interactive mode)
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,
}

fn validate_provider_and_model(provider: &str, model: &str) -> Result<(Providers, Models), String> {
    // Use generated enum method for provider validation
    let provider_enum = Providers::from_name(provider)
        .ok_or_else(|| format!("Unsupported provider: {}", provider))?;

    // Validate that the model is supported by the provider
    if !is_model_supported_by_provider(&provider_enum, model) {
        return Err(format!(
            "Unsupported model '{}' for provider '{}'",
            model, provider
        ));
    }

    // Find the matching Models enum variant
    let model_enum =
        find_models_enum_by_name(model).ok_or_else(|| format!("Unknown model: {}", model))?;

    Ok((provider_enum, model_enum))
}

fn is_model_supported_by_provider(provider: &Providers, model_name: &str) -> bool {
    // Get all models for this provider and check if any match by name
    for model_box in provider.models() {
        if model_box.name().eq_ignore_ascii_case(model_name) {
            return true;
        }
    }
    false
}

fn find_models_enum_by_name(model_name: &str) -> Option<Models> {
    // Create instances of all Models variants and check their names
    let all_models = [
        Models::OpenaiGpt4o,
        Models::OpenaiGpt4oMini,
        Models::ClaudeClaude35Sonnet20241022,
        Models::MistralMistralMediumLatest,
        Models::MistralMistralSmallLatest,
        Models::MistralMagistralMediumLatest,
        Models::MistralMagistralSmallLatest,
        Models::MistralDevstralSmallLatest,
        Models::MistralCodestralLatest,
        Models::MistralMistralEmbed,
        // Add more as needed...
    ];

    for model in &all_models {
        if model.name().eq_ignore_ascii_case(model_name) {
            return Some(model.clone());
        }
    }
    None
}

fn validate_temperature(temp: f32) -> Result<f32, String> {
    if temp >= 0.0 && temp <= 2.0 {
        Ok(temp)
    } else {
        Err("Temperature must be between 0.0 and 2.0".to_string())
    }
}

async fn load_context(
    context_refs: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let mut context_data = Vec::new();

    for context_ref in context_refs {
        info!("Loading context from: {}", context_ref);

        if context_ref.starts_with("https://github.com/") {
            // Handle GitHub references
            context_data.push(format!("GitHub reference: {}", context_ref));
            // TODO: Implement actual GitHub content fetching
        } else if context_ref.contains('*') || context_ref.contains('?') {
            // Handle glob patterns
            context_data.push(format!("Glob pattern: {}", context_ref));
            // TODO: Implement glob pattern matching
        } else {
            // Handle regular files/directories
            match tokio::fs::metadata(context_ref).await {
                Ok(metadata) => {
                    if metadata.is_file() {
                        match tokio::fs::read_to_string(context_ref).await {
                            Ok(content) => {
                                context_data.push(format!("File {}: {}", context_ref, content));
                            }
                            Err(e) => {
                                error!("Failed to read file {}: {}", context_ref, e);
                            }
                        }
                    } else if metadata.is_dir() {
                        context_data.push(format!("Directory: {}", context_ref));
                        // TODO: Implement directory traversal
                    }
                }
                Err(e) => {
                    error!("Failed to access {}: {}", context_ref, e);
                }
            }
        }
    }

    Ok(context_data)
}

async fn interactive_mode(
    provider: &Providers,
    model: &Models,
    temperature: f32,
    agent_role: &str,
    context: &[String],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("🤖 FluentAI Interactive Mode");
    println!("Provider: {}", provider.name());
    println!("Model: {}", model.name());
    println!("Temperature: {}", temperature);
    println!("Agent Role: {}", agent_role);

    // Load context if provided
    let context_data = if !context.is_empty() {
        println!("Context loaded from: {:?}", context);
        let data = load_context(context).await?;
        println!("Context summary: {} items loaded", data.len());
        Some(data.join("\n"))
    } else {
        None
    };

    println!("Type 'quit' or 'exit' to end the session\n");

    // Use the existing FluentAI builder API instead of manual backend calls
    let mut agent_builder =
        fluent_ai::FluentAi::agent(model.clone()).temperature(temperature as f64);

    // Set agent role as system prompt if provided
    if !agent_role.is_empty() && agent_role != "assistant" {
        agent_builder = agent_builder.system_prompt(format!("You are a {}.", agent_role));
    }

    // Add context if available
    if let Some(ctx) = context_data {
        agent_builder = agent_builder.context_text(ctx);
    }

    // Use ergonomic ChatLoop pattern for interactive conversation
    let mut chat_stream = agent_builder
        .on_error(|err| eprintln!("❌ Error: {}", err))
        .chat(|conversation| {
            // Interactive mode: always prompt for user input after each assistant response
            match conversation.message_count() {
                0 => {
                    // First interaction - prompt for initial user input
                    fluent_ai::prelude::ChatLoop::UserPrompt(Some(
                        "Enter your message: ".to_string(),
                    ))
                }
                _ => {
                    // After assistant responses, continue asking for user input
                    fluent_ai::prelude::ChatLoop::UserPrompt(Some(
                        "Enter your message (or 'quit' to exit): ".to_string(),
                    ))
                }
            }
        });

    // Consume the stream to handle chat messages and user prompts
    while let Some(msg_chunk) = chat_stream.next().await {
        match msg_chunk.role {
            fluent_ai::MessageRole::System => {
                // Handle system messages (user prompts, etc.)
                if msg_chunk.content.contains("Enter your message") {
                    print!("{}", msg_chunk.content);
                    io::stdout().flush().unwrap_or(());

                    // Read user input
                    let mut input = String::new();
                    if io::stdin().read_line(&mut input).is_ok() {
                        let input = input.trim();
                        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit")
                        {
                            break;
                        }
                        // TODO: Send user input back to conversation
                        // This requires extending the ChatLoop pattern to handle dynamic input
                    }
                } else {
                    print!("{}", msg_chunk.content);
                    io::stdout().flush().unwrap_or(());
                }
            }
            _ => {
                // Handle assistant and user messages
                print!("{}", msg_chunk.content);
                io::stdout().flush().unwrap_or(());
            }
        }
    }

    println!("Goodbye! 👋");
    Ok(())
}

async fn single_prompt_mode(
    prompt: &str,
    provider: &Providers,
    model: &Models,
    temperature: f32,
    agent_role: &str,
    context: &[String],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("🤖 FluentAI Single Prompt Mode");
    println!("Provider: {}", provider.name());
    println!("Model: {}", model.name());
    println!("Agent Role: {}", agent_role);

    // Load context if provided
    let context_data = if !context.is_empty() {
        println!("Context loaded from: {:?}", context);
        let data = load_context(context).await?;
        println!("Context summary: {} items loaded", data.len());
        Some(data.join("\n"))
    } else {
        None
    };

    println!("\n👤 You: {}", prompt);
    print!("🤖 Assistant: ");
    io::stdout().flush()?;

    // Use the existing FluentAI builder API for single prompt
    let mut agent_builder =
        fluent_ai::FluentAi::agent(model.clone()).temperature(temperature as f64);

    // Set agent role as system prompt if provided
    if !agent_role.is_empty() && agent_role != "assistant" {
        agent_builder = agent_builder.system_prompt(format!("You are a {}.", agent_role));
    }

    // Add context if available
    if let Some(ctx) = context_data {
        agent_builder = agent_builder.context_text(ctx);
    }

    // Use streaming chat with ChatLoop closure pattern
    let prompt_text = prompt.to_string();
    let mut chat_stream = agent_builder
        .on_error(|err| eprintln!("❌ Error: {}", err))
        .chat(move |conversation| {
            if conversation.message_count() == 0 {
                // First interaction - use the provided prompt
                fluent_ai::prelude::ChatLoop::Reprompt(prompt_text.clone())
            } else {
                // For single prompt mode, end after first response
                fluent_ai::prelude::ChatLoop::Break
            }
        });

    // Consume the stream to handle the response and implement ChatLoop pattern
    while let Some(msg_chunk) = chat_stream.next().await {
        print!("{}", msg_chunk.content);
        io::stdout().flush().unwrap_or(());
    }

    println!(); // Add newline after response
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    // Validate inputs
    let (provider_enum, model_enum) = validate_provider_and_model(&cli.provider, &cli.model)?;
    let temperature = validate_temperature(cli.temperature)?;

    if let Some(prompt) = cli.prompt {
        // Single prompt mode
        single_prompt_mode(
            &prompt,
            &provider_enum,
            &model_enum,
            temperature,
            &cli.agent_role,
            &cli.context,
        )
        .await?
    } else {
        // Interactive mode (default)
        interactive_mode(
            &provider_enum,
            &model_enum,
            temperature,
            &cli.agent_role,
            &cli.context,
        )
        .await?
    }

    Ok(())
}
