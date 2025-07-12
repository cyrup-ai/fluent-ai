//! Fluent AI Rig - CLI integration for fluent-ai
//!
//! This crate provides factory functions and utilities for using the fluent-ai
//! chat system from command-line applications.

use fluent_ai::async_task::{AsyncStream, AsyncTask};
use fluent_ai::domain::chunk::{CompletionChunk, FinishReason};
use fluent_ai::domain::completion::{CompletionBackend, CompletionModel};
use fluent_ai::domain::prompt::Prompt as FluentPrompt;
use fluent_ai::engine::{FluentEngine, engine_builder};
use fluent_ai_provider::{Model as ModelTrait, Models, Provider as ProviderTrait, Providers};
use rig::providers::{anthropic, mistral, openai};

use futures::StreamExt;
use rig::client::CompletionClient;
use rig::completion::message::AssistantContent;
use rig::streaming::StreamingPrompt;
use std::env;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error};

/// Rig-based CompletionBackend implementation
#[derive(Clone)]
pub struct RigCompletionBackend {
    provider: Providers,
    model: Models,
}

impl RigCompletionBackend {
    pub fn new(provider: Providers, model: Models) -> Self {
        Self { provider, model }
    }
}

impl CompletionModel for RigCompletionBackend {
    fn prompt(&self, prompt: FluentPrompt) -> AsyncStream<CompletionChunk> {
        let provider = self.provider.clone();
        let model = self.model.clone();
        let prompt_text = prompt.content;

        let (tx, rx) = mpsc::unbounded_channel();
        let stream = AsyncStream::new(rx);

        tokio::spawn(async move {
            let provider_name = provider.name();
            let model_name = model.name();

            debug!(
                "Starting streaming completion for provider={}, model={}",
                provider_name, model_name
            );

            match provider_name {
                "gpt" => {
                    let api_key = match env::var("OPENAI_API_KEY") {
                        Ok(key) => key,
                        Err(_) => {
                            let _ = tx.send(
                                CompletionChunk::new(
                                    "Error: OPENAI_API_KEY environment variable must be set",
                                )
                                .finished(FinishReason::Stop),
                            );
                            return;
                        }
                    };
                    let client = openai::Client::new(&api_key);
                    let agent = client.agent(model_name).build();
                    match agent.stream_prompt(&prompt_text).await {
                        Ok(mut stream) => {
                            debug!("Successfully created OpenAI stream, consuming chunks...");
                            while let Some(chunk) = stream.next().await {
                                match chunk {
                                    Ok(AssistantContent::Text(text)) => {
                                        debug!("Streaming chunk: {}", text.text);
                                        if tx.send(CompletionChunk::new(text.text)).is_err() {
                                            debug!("Receiver dropped, stopping stream");
                                            break;
                                        }
                                    }
                                    Ok(AssistantContent::ToolCall(_tool_call)) => {
                                        debug!("Tool call received but not yet supported");
                                    }
                                    Err(e) => {
                                        error!("Stream error: {}", e);
                                        let _ = tx.send(
                                            CompletionChunk::new(format!("Error: {}", e))
                                                .finished(FinishReason::Stop),
                                        );
                                        break;
                                    }
                                }
                            }
                            // Stream completed successfully
                            let _ = tx.send(CompletionChunk::new("").finished(FinishReason::Stop));
                        }
                        Err(e) => {
                            error!("OpenAI streaming failed: {}", e);
                            let _ = tx.send(
                                CompletionChunk::new(format!("Error: {}", e))
                                    .finished(FinishReason::Stop),
                            );
                        }
                    }
                }
                "claude" => {
                    let api_key = match env::var("ANTHROPIC_API_KEY") {
                        Ok(key) => key,
                        Err(_) => {
                            let _ = tx.send(
                                CompletionChunk::new(
                                    "Error: ANTHROPIC_API_KEY environment variable must be set",
                                )
                                .finished(FinishReason::Stop),
                            );
                            return;
                        }
                    };
                    let client = anthropic::ClientBuilder::new(&api_key).build();
                    let agent = client.agent(model_name).build();
                    match agent.stream_prompt(&prompt_text).await {
                        Ok(mut stream) => {
                            debug!("Successfully created Anthropic stream, consuming chunks...");
                            while let Some(chunk) = stream.next().await {
                                match chunk {
                                    Ok(AssistantContent::Text(text)) => {
                                        debug!("Streaming chunk: {}", text.text);
                                        if tx.send(CompletionChunk::new(text.text)).is_err() {
                                            debug!("Receiver dropped, stopping stream");
                                            break;
                                        }
                                    }
                                    Ok(AssistantContent::ToolCall(_tool_call)) => {
                                        debug!("Tool call received but not yet supported");
                                    }
                                    Err(e) => {
                                        error!("Stream error: {}", e);
                                        let _ = tx.send(
                                            CompletionChunk::new(format!("Error: {}", e))
                                                .finished(FinishReason::Stop),
                                        );
                                        break;
                                    }
                                }
                            }
                            // Stream completed successfully
                            let _ = tx.send(CompletionChunk::new("").finished(FinishReason::Stop));
                        }
                        Err(e) => {
                            error!("Anthropic streaming failed: {}", e);
                            let _ = tx.send(
                                CompletionChunk::new(format!("Error: {}", e))
                                    .finished(FinishReason::Stop),
                            );
                        }
                    }
                }
                "mistral" => {
                    let api_key = match env::var("MISTRAL_API_KEY") {
                        Ok(key) => key,
                        Err(_) => {
                            let _ = tx.send(
                                CompletionChunk::new(
                                    "Error: MISTRAL_API_KEY environment variable must be set",
                                )
                                .finished(FinishReason::Stop),
                            );
                            return;
                        }
                    };
                    let client = mistral::Client::new(&api_key);
                    let agent = client.agent(model_name).build();
                    match agent.stream_prompt(&prompt_text).await {
                        Ok(mut stream) => {
                            debug!("Successfully created Mistral stream, consuming chunks...");
                            while let Some(chunk) = stream.next().await {
                                match chunk {
                                    Ok(AssistantContent::Text(text)) => {
                                        debug!("Streaming chunk: {}", text.text);
                                        if tx.send(CompletionChunk::new(text.text)).is_err() {
                                            debug!("Receiver dropped, stopping stream");
                                            break;
                                        }
                                    }
                                    Ok(AssistantContent::ToolCall(_tool_call)) => {
                                        debug!("Tool call received but not yet supported");
                                    }
                                    Err(e) => {
                                        error!("Stream error: {}", e);
                                        let _ = tx.send(
                                            CompletionChunk::new(format!("Error: {}", e))
                                                .finished(FinishReason::Stop),
                                        );
                                        break;
                                    }
                                }
                            }
                            // Stream completed successfully
                            let _ = tx.send(CompletionChunk::new("").finished(FinishReason::Stop));
                        }
                        Err(e) => {
                            error!("Mistral streaming failed: {}", e);
                            let _ = tx.send(
                                CompletionChunk::new(format!("Error: {}", e))
                                    .finished(FinishReason::Stop),
                            );
                        }
                    }
                }
                _ => {
                    let _ = tx.send(
                        CompletionChunk::new(format!(
                            "Error: Unsupported provider: {}",
                            provider_name
                        ))
                        .finished(FinishReason::Stop),
                    );
                }
            }

            debug!(
                "Streaming completion finished for provider={}",
                provider_name
            );
        });

        stream
    }
}

// Legacy CompletionBackend support for backward compatibility
impl CompletionBackend for RigCompletionBackend {
    fn submit_completion(&self, prompt: &str, tools: &[String]) -> AsyncTask<String> {
        let fluent_prompt = FluentPrompt::new(prompt);
        let stream = self.prompt(fluent_prompt);

        AsyncTask::from_future(async move {
            let chunks: Vec<CompletionChunk> = stream.collect().await;
            chunks
                .into_iter()
                .map(|chunk| chunk.text)
                .collect::<Vec<_>>()
                .join("")
        })
    }
}

/// Create a FluentEngine with the specified provider and model
pub fn create_fluent_engine_with_model(
    provider: Providers,
    model: Models,
) -> Result<Arc<FluentEngine>, Box<dyn std::error::Error + Send + Sync>> {
    let backend = Arc::new(RigCompletionBackend::new(provider, model.clone()));
    let engine = Arc::new(FluentEngine::new(backend, model));

    engine_builder()
        .engine(engine.clone())
        .name("fluent-ai-rig-engine")
        .build_and_register()?;

    Ok(engine)
}

/// Utility functions for CLI integration
pub mod cli {
    use fluent_ai_provider::{Models, Providers};

    /// Validate that a provider supports a given model
    pub fn validate_provider_model_combination(
        provider: Providers,
        model: Models,
    ) -> Result<(), String> {
        use fluent_ai_provider::{Model, Provider};

        // Get all models supported by this provider
        let supported_models = provider.models();

        // Check if the requested model is supported
        let model_name = model.name();
        let is_supported = supported_models
            .iter()
            .any(|supported| supported.name().eq_ignore_ascii_case(model_name));

        if is_supported {
            Ok(())
        } else {
            Err(format!(
                "Model '{}' is not supported by provider '{}'",
                model_name,
                provider.name()
            ))
        }
    }

    /// Get the default model for a given provider (first model in the list)
    pub fn get_default_model_for_provider(provider: Providers) -> Option<Models> {
        use fluent_ai_provider::Provider;

        provider
            .models()
            .first()
            .and_then(|model| Models::from_name(model.name()))
    }
}
