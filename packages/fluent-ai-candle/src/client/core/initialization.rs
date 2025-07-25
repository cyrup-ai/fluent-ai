//! Client initialization methods for CandleCompletionClient
//!
//! This module contains the initialization logic extracted from the original
//! monolithic client.rs file, including constructors and HuggingFace Hub integration.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arc_swap::ArcSwap;
use fluent_ai_async::{AsyncStream, handle_error};

use super::client::CandleCompletionClient;
use super::super::config::CandleClientConfig;
use super::super::metrics::CANDLE_METRICS;
use crate::generator::CandleGenerator;
use crate::model::CandleModel;
use crate::tokenizer::CandleTokenizer;

impl CandleCompletionClient {
    /// Create a new CandleCompletionClient with zero-allocation patterns
    #[inline(always)]
    pub fn new(config: CandleClientConfig) -> AsyncStream<Self> {
        AsyncStream::with_channel(move |sender| {
            let device = match Self::create_device(config.device_type) {
                Ok(d) => Arc::new(d),
                Err(e) => {
                    handle_error!(e, "Failed to create device");
                }
            };

            // Load model - handling the async operation in synchronous context
            let model = Arc::new(CandleModel::new((*device).clone()));
            // TODO: Implement proper model loading for AsyncStream architecture
            // For now, assume model loading succeeds - proper implementation needed
            println!("Model loading: {}", config.model_path);

            // Load tokenizer with safe path handling
            let tokenizer_path = config.tokenizer_path.as_ref().unwrap_or(&config.model_path);
            let tokenizer =
                match CandleTokenizer::from_file(tokenizer_path, config.tokenizer_config.clone()) {
                    Ok(t) => Arc::new(t),
                    Err(e) => {
                        handle_error!(e, "Failed to load tokenizer");
                    }
                };

            // Create generator with sophisticated features if enabled
            let generator = if config.enable_sophisticated_sampling
                || config.enable_streaming_optimization
                || config.enable_kv_cache
            {
                match CandleGenerator::new_with_features(
                    config.generation_config.clone(),
                    config.sampling_config.clone(),
                    config.streaming_config.clone(),
                    config.kv_cache_config.clone(),
                ) {
                    Ok(g) => ArcSwap::new(Arc::new(g)),
                    Err(e) => {
                        handle_error!(e, "Failed to create sophisticated generator");
                    }
                }
            } else {
                match CandleGenerator::new(config.generation_config.clone()) {
                    Ok(g) => ArcSwap::new(Arc::new(g)),
                    Err(e) => {
                        handle_error!(e, "Failed to create basic generator");
                    }
                }
            };

            let client = Self {
                config,
                model,
                tokenizer,
                generator,
                device,
                metrics: &CANDLE_METRICS,
                is_initialized: std::sync::atomic::AtomicBool::new(true),
                max_concurrent_requests: 4, // Default value
            };

            let _ = sender.send(client);
        })
    }

    /// Create a new CandleCompletionClient from HuggingFace Hub
    #[inline(always)]
    pub fn from_hub(repo_id: String, config: CandleClientConfig) -> AsyncStream<Self> {
        AsyncStream::with_channel(move |sender| {
            Box::pin(async move {
                if !config.enable_hub_integration {
                    handle_error!(
                        crate::error::CandleError::configuration("Hub integration is disabled"),
                        "Hub integration disabled"
                    );
                    return Ok(());
                }

                // TODO: Implement HuggingFace Hub integration
                // For now, create a basic client with hub model path
                let mut hub_config = config.clone();
                hub_config.model_path = format!("hub://{}", repo_id);

                // Create client through normal initialization path
                let mut client_stream = Self::new(hub_config);
                
                // Forward the result
                // Note: In real implementation, this would handle Hub-specific loading
                if let Some(client) = client_stream.next().await {
                    let _ = sender.send(client).await;
                } else {
                    handle_error!(
                        crate::error::CandleError::model_loading("Failed to load model from Hub"),
                        "Hub model loading failed"
                    );
                }
                
                Ok(())
            })
        })
    }

    /// Initialize client with validation and setup
    pub(super) fn initialize(&self) -> Result<(), crate::error::CandleError> {
        // Validate configuration
        if self.config.model_path.is_empty() {
            return Err(crate::error::CandleError::configuration("Model path cannot be empty"));
        }

        // Validate concurrent request limits
        if self.config.max_concurrent_requests == 0 {
            return Err(crate::error::CandleError::configuration("Max concurrent requests must be > 0"));
        }

        // Set initialized flag
        self.is_initialized.store(true, Ordering::Release);

        Ok(())
    }
}