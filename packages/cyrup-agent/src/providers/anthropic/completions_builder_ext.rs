// providers/anthropic/builder_ext.rs
use super::completion::CompletionModel as AnthropicModel;
use rig::completion::builder::{BuilderExt, CompletionBuilder};

pub trait AnthropicExt {
    fn beta(self, flag: &'static str) -> Self;
    fn prompt_cache(self, secs: u64) -> Self;
}

impl<S> AnthropicExt for CompletionBuilder<AnthropicModel, S> {
    fn beta(self, flag: &'static str) -> Self {
        self.provider_param("anthropic_beta", serde_json::json!(flag))
    }
    fn prompt_cache(self, secs: u64) -> Self {
        self.cache_control(serde_json::json!({ "max_age_secs": secs }))
    }
}
