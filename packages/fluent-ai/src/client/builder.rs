// ============================================================================
// File: client/builder.rs            (REWRITE – zero-Rc/Arc implementation)
// ----------------------------------------------------------------------------
// Dynamic provider-agnostic client factory & model/agent dispatcher.
//
// * Absolutely no ref-counting or locking – even off the hot-path.
// * Factories are leaked once and stored as raw pointers.
// * Each `.build_*` call is allocation-free and branch-predictable.
// * Single authoritative error enum (`BuildErr`)
// ============================================================================

use std::{
    collections::HashMap,
    fmt::{self, Debug},
    sync::Arc,
};

use crossbeam::{
    channel::{Receiver, unbounded},
    thread,
};

use crate::{
    agent::{Agent, AgentBuilder},
    client::{
        ProviderClient,
        completion::{CompletionClientDyn, CompletionModelHandle},
        embeddings::EmbeddingsClientDyn,
        transcription::TranscriptionClientDyn,
    },
    completion::CompletionModelDyn,
    embedding::embedding::EmbeddingModelDyn,
    transcription::TranscriptionModelDyn,
};

/// Thread-safe factory function signature using crossbeam channels
type SafeFactoryFn =
    Arc<dyn Fn() -> Receiver<Result<Box<dyn ProviderClient>, BuildErr>> + Send + Sync>;

/// Utility so we can map/pipe inline without extra imports.
trait Pipe: Sized {
    #[inline(always)]
    fn pipe<T, F: FnOnce(Self) -> T>(self, f: F) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

/// -------------------------------------------------------------------------
/// Public error enumeration
/// -------------------------------------------------------------------------
#[derive(thiserror::Error, Debug)]
pub enum BuildErr {
    #[error("unknown provider: {0}")]
    UnknownProvider(String),

    #[error("unsupported feature {feature:?} for provider {provider:?}")]
    UnsupportedFeature {
        provider: String,
        feature: &'static str,
    },

    #[error("factory error: {0}")]
    Factory(String),

    #[error("invalid provider:model id: {0}")]
    InvalidId(String),
}

/// -------------------------------------------------------------------------
/// DynClientBuilder – central registry (zero Arc / Mutex)
/// -------------------------------------------------------------------------
#[derive(Default)]
pub struct DynClientBuilder {
    registry: HashMap<&'static str, SafeFactoryFn>,
}

impl DynClientBuilder {
    // ----- registration ----------------------------------------------------

    pub fn register<F, C>(mut self, name: &'static str, factory: F) -> Self
    where
        F: Fn() -> Result<C, anyhow::Error> + Send + Sync + 'static,
        C: ProviderClient + 'static,
    {
        // Lower-case once to store in registry
        let name_key = name.to_ascii_lowercase();

        // Create a thread-safe factory using Arc
        let safe_factory: SafeFactoryFn = Arc::new(move || {
            let (tx, rx) = unbounded();
            let factory_clone = Arc::new(factory);

            thread::spawn(move || {
                let result = factory_clone()
                    .map(|c| Box::new(c) as Box<dyn ProviderClient>)
                    .map_err(|e| BuildErr::Factory(e.to_string()));
                let _ = tx.send(result);
            });

            rx
        });

        self.registry
            .insert(Box::leak(name_key.into_boxed_str()), safe_factory);
        self
    }

    pub fn register_all<I, F, Fut, C>(self, providers: I) -> Self
    where
        I: IntoIterator<Item = (&'static str, F)>,
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<C, anyhow::Error>> + Send + 'static,
        C: ProviderClient + 'static,
    {
        providers
            .into_iter()
            .fold(self, |acc, (n, f)| acc.register(n, f))
    }

    // ----- low-level helper -------------------------------------------------

    fn build_raw(&self, provider: &str) -> Result<Box<dyn ProviderClient>, BuildErr> {
        let factory = self
            .registry
            .get(provider)
            .ok_or_else(|| BuildErr::UnknownProvider(provider.into()))?;

        // Use the safe factory function - no unsafe code needed
        let receiver = (factory)();
        match receiver.recv() {
            Ok(result) => result,
            Err(_) => Err(BuildErr::Factory(
                "Channel closed without result".to_string(),
            )),
        }
    }

    // ----- high-level helpers ----------------------------------------------

    pub fn completion<'a>(
        &'a self,
        provider: &str,
        model: &str,
    ) -> AsyncTask<Result<Box<dyn CompletionModelDyn + 'a>, BuildErr>> {
        let provider = provider.to_string();
        let model = model.to_string();
        crate::runtime::spawn_async(async move {
            self.build_raw(&provider)?
                .as_completion()
                .ok_or_else(|| BuildErr::UnsupportedFeature {
                    provider: provider.clone(),
                    feature: "completion",
                })?
                .completion_model(&model)
                .pipe(Ok)
        })
    }

    pub fn agent<'a>(
        &'a self,
        provider: &str,
        model: &str,
    ) -> AsyncTask<Result<AgentBuilder<CompletionModelHandle<'a>>, BuildErr>> {
        let provider = provider.to_string();
        let model = model.to_string();
        crate::runtime::spawn_async(async move {
            self.build_raw(&provider)?
                .as_completion()
                .ok_or_else(|| BuildErr::UnsupportedFeature {
                    provider: provider.clone(),
                    feature: "completion",
                })?
                .agent(&model)
                .pipe(Ok)
        })
    }

    pub fn embeddings<'a>(
        &'a self,
        provider: &str,
        model: &str,
    ) -> AsyncTask<Result<Box<dyn EmbeddingModelDyn + 'a>, BuildErr>> {
        let provider = provider.to_string();
        let model = model.to_string();
        crate::runtime::spawn_async(async move {
            self.build_raw(&provider)?
                .as_embeddings()
                .ok_or_else(|| BuildErr::UnsupportedFeature {
                    provider: provider.clone(),
                    feature: "embeddings",
                })?
                .embedding_model(&model)
                .pipe(Ok)
        })
    }

    pub fn transcription<'a>(
        &'a self,
        provider: &str,
        model: &str,
    ) -> AsyncTask<Result<Box<dyn TranscriptionModelDyn + 'a>, BuildErr>> {
        let provider = provider.to_string();
        let model = model.to_string();
        crate::runtime::spawn_async(async move {
            self.build_raw(&provider)?
                .as_transcription()
                .ok_or_else(|| BuildErr::UnsupportedFeature {
                    provider: provider.clone(),
                    feature: "transcription",
                })?
                .transcription_model(&model)
                .pipe(Ok)
        })
    }

    // ----- "provider:model" convenience ------------------------------------

    pub fn id<'builder, 'id>(
        &'builder self,
        fq_id: &'id str,
    ) -> Result<ProviderModelId<'builder, 'id>, BuildErr> {
        let (provider, model) = fq_id
            .split_once(':')
            .ok_or_else(|| BuildErr::InvalidId(fq_id.into()))?;
        Ok(ProviderModelId {
            builder: self,
            provider,
            model,
        })
    }
}

/// Proxy produced by `.id("provider:model")`
pub struct ProviderModelId<'builder, 'id> {
    builder: &'builder DynClientBuilder,
    provider: &'id str,
    model: &'id str,
}

impl<'b, 'i> ProviderModelId<'b, 'i> {
    pub fn completion(self) -> AsyncTask<Result<Box<dyn CompletionModelDyn + 'b>, BuildErr>> {
        self.builder.completion(self.provider, self.model)
    }
    pub fn agent(self) -> AsyncTask<Result<AgentBuilder<CompletionModelHandle<'b>>, BuildErr>> {
        self.builder.agent(self.provider, self.model)
    }
    pub fn embedding(self) -> AsyncTask<Result<Box<dyn EmbeddingModelDyn + 'b>, BuildErr>> {
        self.builder.embeddings(self.provider, self.model)
    }
    pub fn transcription(self) -> AsyncTask<Result<Box<dyn TranscriptionModelDyn + 'b>, BuildErr>> {
        self.builder.transcription(self.provider, self.model)
    }
}

// ----- pretty-printer --------------------------------------------------------
impl Debug for DynClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let providers: Vec<_> = self.registry.keys().cloned().collect();
        f.debug_struct("DynClientBuilder")
            .field("registered_providers", &providers)
            .finish()
    }
}
