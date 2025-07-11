use clap::{Arg, Command, Parser};
use fluent_ai_provider::{Model, Models, Provider, Providers};
use fluent_ai_rig::create_fluent_engine_with_model;
use std::error::Error;
use std::io::{self, Write};
use tokio;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "fluent-ai-rig")]
#[command(about = "A CLI for interacting with FluentAI using various models")]
struct Cli {
    /// Provider to use (e.g., openai, anthropic, github, openrouter)
    #[arg(short, long, default_value = "openai")]
    provider: String,

    /// Model to use for completions (e.g., gpt-4o-mini, claude-3-sonnet)
    #[arg(short, long, default_value = "gpt-4o-mini")]
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

fn validate_provider_and_model(
    provider: &str,
    model: &str,
) -> Result<(Providers, Models), String> {
    // Find provider enum by name matching
    let provider_enum = match provider.to_lowercase().as_str() {
        "openai" => Providers::Openai,
        "gemini" | "google" => Providers::Gemini,
        "claude" | "anthropic" => Providers::Claude,
        "mistral" => Providers::Mistral,
        "ai21" => Providers::Ai21,
        "cohere" => Providers::Cohere,
        "xai" => Providers::Xai,
        "perplexity" => Providers::Perplexity,
        "groq" => Providers::Groq,
        "vertexai" | "vertex" => Providers::Vertexai,
        "bedrock" => Providers::Bedrock,
        "cloudflare" => Providers::Cloudflare,
        "ernie" => Providers::Ernie,
        "qianwen" => Providers::Qianwen,
        "hunyuan" => Providers::Hunyuan,
        "moonshot" => Providers::Moonshot,
        "deepseek" => Providers::Deepseek,
        "zhipuai" => Providers::Zhipuai,
        "minimax" => Providers::Minimax,
        "openrouter" => Providers::Openrouter,
        "github" => Providers::Github,
        "deepinfra" => Providers::Deepinfra,
        "jina" => Providers::Jina,
        "voyageai" => Providers::Voyageai,
        _ => return Err(format!("Unsupported provider: {}", provider)),
    };

    // Find model enum by checking provider and model combination
    let model_enum = find_model_for_provider(&provider_enum, model)
        .ok_or_else(|| format!("Unsupported model '{}' for provider '{}'", model, provider))?;

    Ok((provider_enum, model_enum))
}

fn find_model_for_provider(provider: &Providers, model_name: &str) -> Option<Models> {
    let model_name_lower = model_name.to_lowercase().replace("-", "").replace("_", "");
    
    // Get all models for this provider and check if any match
    for model_box in provider.models() {
        let model_display_name = model_box.name().to_lowercase().replace("-", "").replace("_", "");
        if model_display_name == model_name_lower {
            // Convert Box<dyn Model> back to Models enum
            return model_box_to_enum(&**model_box);
        }
    }
    None
}

// Helper to convert Box<dyn Model> back to Models enum
fn model_box_to_enum(model: &dyn Model) -> Option<Models> {
    let model_name = model.name();
    
    // Match against all possible Models enum variants
    // This is auto-generated enum so we need to check all variants
    use Models::*;
    let all_models = [
        // OpenAI models
        OpenaiGpt41, OpenaiGpt41Mini, OpenaiGpt41Nano, OpenaiGpt4o, OpenaiGpt4oMini,
        OpenaiGpt4oSearchPreview, OpenaiGpt4oMiniSearchPreview, OpenaiChatgpt4oLatest,
        OpenaiO3, OpenaiO3Mini, OpenaiO3MiniHigh, OpenaiO4Mini, OpenaiO4MiniHigh,
        OpenaiGpt4Turbo, OpenaiGpt35Turbo, OpenaiTextEmbedding3Large, OpenaiTextEmbedding3Small,
        // Gemini models
        GeminiGemini25Flash, GeminiGemini25Pro, GeminiGemini25FlashLitePreview0617,
        GeminiGemini20Flash, GeminiGemini20FlashLite, GeminiGemma327bIt, GeminiTextEmbedding004,
        // Claude models
        ClaudeClaudeOpus420250514, ClaudeClaudeOpus420250514Thinking,
        ClaudeClaudeSonnet420250514, ClaudeClaudeSonnet420250514Thinking,
        ClaudeClaude37Sonnet20250219, ClaudeClaude37Sonnet20250219Thinking,
        ClaudeClaude35Sonnet20241022, ClaudeClaude35Haiku20241022,
        // Add more models as needed - this covers the most common ones
    ];
    
    for candidate in &all_models {
        if candidate.name() == model_name {
            return Some(*candidate);
        }
    }
    None
}

fn validate_provider(provider: &str) -> Result<String, String> {
    match provider.to_lowercase().as_str() {
        "openai" | "anthropic" | "github" | "openrouter" | "mistral" => Ok(provider.to_string()),
        _ => {
            // Allow custom providers
            Ok(provider.to_string())
        }
    }
}

fn validate_temperature(temp: f32) -> Result<f32, String> {
    if temp >= 0.0 && temp <= 2.0 {
        Ok(temp)
    } else {
        Err("Temperature must be between 0.0 and 2.0".to_string())
    }
}

async fn load_context(context_refs: &[String]) -> Result<Vec<String>, Box<dyn Error>> {
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
    provider: &ProviderImpl,
    model: &ModelImpl,
    temperature: f32,
    agent_role: &str,
    context: &[String],
) -> Result<(), Box<dyn Error>> {
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

    let engine = create_fluent_engine_with_model(model)?;

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

        // For now, echo the prompt since we need to implement the actual completion
        println!("[Processing with model: {}, temp: {}]", model, temperature);
        println!("Full prompt would be: {}", full_prompt);
        println!("[TODO: Implement actual completion call]\n");
    }

    Ok(())
}

async fn single_prompt_mode(
    prompt: &str,
    provider: &ProviderImpl,
    model: &ModelImpl,
    temperature: f32,
    agent_role: &str,
    context: &[String],
) -> Result<(), Box<dyn Error>> {
    println!(
        "🤖 Processing single prompt with provider: {} and model: {}",
        provider.name(),
        model.name()
    );

    let engine = create_fluent_engine_with_model(model)?;

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

    println!("Response: [TODO: Implement actual completion call]");
    println!("Full prompt would be: {}", full_prompt);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
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
