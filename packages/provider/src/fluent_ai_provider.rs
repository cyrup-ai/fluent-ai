// This file is auto-generated. Do not edit manually.
use std::collections::HashMap;
use once_cell::sync::Lazy;

// AUTO-GENERATED START
/// Static mapping of provider names to client module names
pub static PROVIDER_CLIENT_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("openai", "openai");
    map.insert("gemini", "gemini");
    map.insert("claude", "claude");
    map.insert("mistral", "mistral");
    map.insert("ai21", "ai21");
    map.insert("cohere", "cohere");
    map.insert("xai", "xai");
    map.insert("perplexity", "perplexity");
    map.insert("groq", "groq");
    map.insert("vertexai", "vertexai");
    map.insert("bedrock", "bedrock");
    map.insert("deepseek", "deepseek");
    map.insert("openrouter", "openrouter");
    map
});

/// Get client module name for a provider
pub fn get_client_module(provider: &str) -> Option<&'static str> {
    PROVIDER_CLIENT_MAP.get(provider).copied()
}

/// Get provider name for a client module
pub fn get_provider_for_client(client: &str) -> Option<&'static str> {
    PROVIDER_CLIENT_MAP.iter()
        .find(|(_, &v)| v == client)
        .map(|(&k, _)| k)
}

/// Get all provider-to-client mappings
pub fn get_all_mappings() -> &'static HashMap<&'static str, &'static str> {
    &PROVIDER_CLIENT_MAP
}

// AUTO-GENERATED END
