use clap::Parser;
use fluent_ai_provider::{Model, Models, Provider, Providers};
use fluent_ai_rig::{create_fluent_engine_with_model, RigCompletionBackend};
use fluent_ai::domain::completion::CompletionBackend;
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
        return Err(format!("Unsupported model '{}' for provider '{}'", model, provider));
    }
    
    // Find the matching Models enum variant
    let model_enum = find_models_enum_by_name(model)
        .ok_or_else(|| format!("Unknown model: {}", model))?;
    
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

async fn load_context(context_refs: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
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

    if !context.is_empty() {
        println!("Context loaded from: {:?}", context);
        let context_data = load_context(context).await?;
        println!("Context summary: {} items loaded", context_data.len());
    }

    println!("Type 'quit' or 'exit' to end the session\n");

    let _engine = create_fluent_engine_with_model(provider.clone(), model.clone())?;

    loop {
        print!("👤 You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("Goodbye! 👋");
            break;
        }

        print!("🤖 Assistant: ");
        io::stdout().flush()?;

        // Create a prompt that incorporates agent role and context
        let mut full_prompt = String::new();

        if !agent_role.is_empty() && agent_role != "assistant" {
            full_prompt.push_str(&format!("You are a {}. ", agent_role));
        }

        if !context.is_empty() {
            full_prompt.push_str("Consider the following context: ");
            let context_data = load_context(context).await?;
            for ctx in context_data {
                full_prompt.push_str(&format!("{}\n", ctx));
            }
            full_prompt.push_str("\nNow respond to: ");
        }

        full_prompt.push_str(input);

        // Create backend and submit completion
        let backend = crate::RigCompletionBackend::new(provider.clone(), model.clone());
        println!("[Processing with model: {}, temp: {}]", model.name(), temperature);
        
        // Submit completion request
        let completion_task = backend.submit_completion(&full_prompt, &[]);
        match completion_task.await {
            Ok(response) => println!("Response: {}", response),
            Err(e) => println!("Error: {}", e),
        }
    }

    Ok(())
}

async fn single_prompt_mode(
    prompt: &str,
    provider: &Providers,
    model: &Models,
    _temperature: f32,
    agent_role: &str,
    context: &[String],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!(
        "🤖 Processing single prompt with provider: {} and model: {}",
        provider.name(),
        model.name()
    );

    let _engine = create_fluent_engine_with_model(provider.clone(), model.clone())?;

    let mut full_prompt = String::new();

    if !agent_role.is_empty() && agent_role != "assistant" {
        full_prompt.push_str(&format!("You are a {}. ", agent_role));
    }

    if !context.is_empty() {
        full_prompt.push_str("Consider the following context: ");
        let context_data = load_context(context).await?;
        for ctx in context_data {
            full_prompt.push_str(&format!("{}\n", ctx));
        }
        full_prompt.push_str("\nNow respond to: ");
    }

    full_prompt.push_str(prompt);

    // Create backend and submit completion
    let backend = crate::RigCompletionBackend::new(provider.clone(), model.clone());
    
    // Submit completion request
    let completion_task = backend.submit_completion(&full_prompt, &[]);
    match completion_task.await {
        Ok(response) => println!("Response: {}", response),
        Err(e) => println!("Error: {}", e),
    }

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
