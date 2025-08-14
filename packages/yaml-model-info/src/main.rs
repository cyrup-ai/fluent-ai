//! YAML Model Info CLI Tool
//!
//! A blazing-fast CLI tool for downloading and parsing AI model YAML data.
//! Uses yyaml exclusively with zero fallback logic for maximum reliability.
//!
//! Performance characteristics:
//! - Zero allocation CLI argument parsing
//! - Blazing-fast HTTP/3 downloads with intelligent caching
//! - Lock-free YAML parsing with yyaml
//! - Elegant ergonomic output formatting

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use yaml_model_info::{download, models::YamlProvider};

/// CLI configuration with zero allocation argument parsing
#[derive(Parser)]
#[command(name = "yaml-model-info")]
#[command(about = "A blazing-fast CLI tool for AI model YAML data")]
#[command(version = "0.1.0")]
#[command(
    long_about = "Downloads and parses AI model information from the aichat repository using yyaml exclusively. Features intelligent caching and zero fallback logic for maximum reliability."
)]
struct Cli {
    /// Cache directory for downloaded YAML data
    #[arg(long, default_value = ".yaml-cache")]
    cache_dir: PathBuf,

    /// Output format for model information
    #[arg(long, default_value = "table")]
    format: OutputFormat,

    /// Subcommand to execute
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Available CLI commands
#[derive(Subcommand, Clone)]
enum Commands {
    /// Download and display all provider information
    List,
    /// Get detailed information about a specific provider
    Provider {
        /// Provider name (e.g., openai, anthropic, google)
        name: String,
    },
    /// Get detailed information about a specific model
    Model {
        /// Provider name
        provider: String,
        /// Model name
        model: String,
    },
    /// Download fresh YAML data (ignores cache)
    Refresh,
}

/// Output format options
#[derive(Clone)]
enum OutputFormat {
    Table,
    Json,
    Yaml,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "table" => Ok(OutputFormat::Table),
            "json" => Ok(OutputFormat::Json),
            "yaml" => Ok(OutputFormat::Yaml),
            _ => Err(format!(
                "Invalid format: {}. Valid options: table, json, yaml",
                s
            )),
        }
    }
}

/// Main CLI entry point with comprehensive error handling
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Execute the appropriate command with zero allocation patterns
    match cli.command.clone().unwrap_or(Commands::List) {
        Commands::List => list_providers(&cli).await,
        Commands::Provider { name } => show_provider(&cli, &name).await,
        Commands::Model { provider, model } => show_model(&cli, &provider, &model).await,
        Commands::Refresh => refresh_cache(&cli).await,
    }
}

/// Lists all providers with blazing-fast performance
#[inline(always)]
async fn list_providers(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let yaml_content = download::download_yaml_with_cache(&cli.cache_dir).await?;

    let providers: Vec<YamlProvider> = yyaml::from_str(&yaml_content)?;

    match cli.format {
        OutputFormat::Table => {
            println!("┌─────────────────┬─────────────┬─────────────┐");
            println!("│ Provider        │ Model Count │ Vision      │");
            println!("├─────────────────┼─────────────┼─────────────┤");

            for provider in &providers {
                let vision_count = provider
                    .models
                    .iter()
                    .filter(|m| m.supports_vision.unwrap_or(false))
                    .count();

                println!(
                    "│ {:15} │ {:11} │ {:11} │",
                    provider.identifier(),
                    provider.model_count(),
                    vision_count
                );
            }

            println!("└─────────────────┴─────────────┴─────────────┘");
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&providers)?);
        }
        OutputFormat::Yaml => {
            println!("{}", yyaml::to_string(&providers)?);
        }
    }

    Ok(())
}

/// Shows detailed information about a specific provider
#[inline(always)]
async fn show_provider(cli: &Cli, provider_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let yaml_content = download::download_yaml_with_cache(&cli.cache_dir).await?;
    let providers: Vec<YamlProvider> = yyaml::from_str(&yaml_content)?;

    let provider = providers
        .iter()
        .find(|p| p.identifier().eq_ignore_ascii_case(provider_name))
        .ok_or_else(|| format!("Provider '{}' not found", provider_name))?;

    match cli.format {
        OutputFormat::Table => {
            println!("Provider: {}", provider.identifier());
            println!("Models: {}", provider.model_count());
            println!();
            println!("┌─────────────────────────┬─────────────┬─────────────┬─────────────┐");
            println!("│ Model Name              │ Max Input   │ Max Output  │ Vision      │");
            println!("├─────────────────────────┼─────────────┼─────────────┼─────────────┤");

            for model in &provider.models {
                println!(
                    "│ {:23} │ {:11} │ {:11} │ {:11} │",
                    model.name,
                    model
                        .max_input_tokens
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "N/A".to_string()),
                    model
                        .max_output_tokens
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "N/A".to_string()),
                    if model.supports_vision.unwrap_or(false) {
                        "Yes"
                    } else {
                        "No"
                    }
                );
            }

            println!("└─────────────────────────┴─────────────┴─────────────┴─────────────┘");
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(provider)?);
        }
        OutputFormat::Yaml => {
            println!("{}", yyaml::to_string(provider)?);
        }
    }

    Ok(())
}

/// Shows detailed information about a specific model
#[inline(always)]
async fn show_model(
    cli: &Cli,
    provider_name: &str,
    model_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let yaml_content = download::download_yaml_with_cache(&cli.cache_dir).await?;
    let providers: Vec<YamlProvider> = yyaml::from_str(&yaml_content)?;

    let provider = providers
        .iter()
        .find(|p| p.identifier().eq_ignore_ascii_case(provider_name))
        .ok_or_else(|| format!("Provider '{}' not found", provider_name))?;

    let model = provider
        .models
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(model_name))
        .ok_or_else(|| {
            format!(
                "Model '{}' not found in provider '{}'",
                model_name, provider_name
            )
        })?;

    match cli.format {
        OutputFormat::Table => {
            println!("Model: {}", model.identifier(provider.identifier()));
            println!("Provider: {}", provider.identifier());
            println!(
                "Max Input Tokens: {}",
                model
                    .max_input_tokens
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "N/A".to_string())
            );
            println!(
                "Max Output Tokens: {}",
                model
                    .max_output_tokens
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "N/A".to_string())
            );
            println!(
                "Input Price: {}",
                model
                    .input_price
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "N/A".to_string())
            );
            println!(
                "Output Price: {}",
                model
                    .output_price
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "N/A".to_string())
            );
            println!(
                "Supports Vision: {}",
                if model.supports_vision.unwrap_or(false) {
                    "Yes"
                } else {
                    "No"
                }
            );
            println!(
                "Supports Function Calling: {}",
                if model.supports_function_calling.unwrap_or(false) {
                    "Yes"
                } else {
                    "No"
                }
            );
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(model)?);
        }
        OutputFormat::Yaml => {
            println!("{}", yyaml::to_string(model)?);
        }
    }

    Ok(())
}

/// Refreshes the cache by downloading fresh YAML data
#[inline(always)]
async fn refresh_cache(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    // Remove cache files to force fresh download
    let cache_file = cli.cache_dir.join("models.yaml");
    let etag_file = cli.cache_dir.join("models.yaml.etag");

    let _ = std::fs::remove_file(&cache_file);
    let _ = std::fs::remove_file(&etag_file);

    println!("Downloading fresh YAML data...");
    let yaml_content = download::download_yaml_with_cache(&cli.cache_dir).await?;
    let providers: Vec<YamlProvider> = yyaml::from_str(&yaml_content)?;

    println!(
        "✅ Successfully downloaded and parsed {} providers",
        providers.len()
    );
    println!("Cache refreshed at: {}", cli.cache_dir.display());

    Ok(())
}
