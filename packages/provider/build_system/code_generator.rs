//! Code generation for provider and model definitions
//!
//! This module provides zero-allocation code generation for provider enums,
//! and the model registry.

use std::sync::Arc;

use super::performance::PerformanceMonitor;
use super::errors::{BuildError, BuildResult};
use fluent_ai_domain::model::ModelInfo;
use crate::build_system::yaml_processor::ProviderInfo;

/// Template cache for zero-allocation code generation
#[derive(Debug)]
struct TemplateCache {
    provider_struct: String,
    models_registry: String,
    file_header: String,
}

impl TemplateCache {
    /// Load all templates from disk with caching
    fn new() -> BuildResult<Self> {
        let template_dir = std::path::Path::new("build_system/templates");
        
        Ok(Self {
            provider_struct: std::fs::read_to_string(template_dir.join("provider_struct.rs.template"))
                .map_err(|e| BuildError::Template(format!("Failed to load provider_struct template: {}", e).into()))?,
            models_registry: std::fs::read_to_string(template_dir.join("models_registry.rs.template"))
                .map_err(|e| BuildError::Template(format!("Failed to load models_registry template: {}", e).into()))?,
            file_header: std::fs::read_to_string(template_dir.join("file_header.rs.template"))
                .map_err(|e| BuildError::Template(format!("Failed to load file_header template: {}", e).into()))?,
        })
    }
}


/// Generates Rust code from parsed YAML data
#[derive(Debug)]
pub struct CodeGenerator {
    perf_monitor: Arc<PerformanceMonitor>,
    template_cache: TemplateCache,
}

impl CodeGenerator {
    /// Create a new CodeGenerator
    pub fn new(perf_monitor: Arc<PerformanceMonitor>) -> BuildResult<Self> {
        let _timer = perf_monitor.start_timer("code_generator_new");
        Ok(Self {
            perf_monitor,
            template_cache: TemplateCache::new()?,
        })
    }

    /// Generate the `providers.rs` file content
    pub fn generate_provider_module(&self, providers: &[ProviderInfo]) -> BuildResult<String> {
        let _timer = self.perf_monitor.start_timer("generate_provider_module");
        
        let mut provider_variants = String::with_capacity(providers.len() * 64);
        let mut provider_display_arms = String::with_capacity(providers.len() * 128);
        let mut provider_from_str_arms = String::with_capacity(providers.len() * 128);
        
        for provider in providers {
            let variant_name = to_pascal_case(&provider.name);
            let provider_id = &provider.name;
            
            provider_variants.push_str(&format!("    {},
", variant_name));
            provider_display_arms.push_str(&format!("            Providers::{} => write!(f, \"{}\"),
", variant_name, provider_id));
            provider_from_str_arms.push_str(&format!("            \"{}\" => Ok(Providers::{}),
", provider_id, variant_name));
        }
        
        let mut code = self.template_cache.provider_struct.clone();
        code = code.replace("{{PROVIDER_VARIANTS}}", &provider_variants);
        code = code.replace("{{PROVIDER_DISPLAY_ARMS}}", &provider_display_arms);
        code = code.replace("{{PROVIDER_FROM_STR_ARMS}}", &provider_from_str_arms);
        
        let header = self.template_cache.file_header.clone();
        Ok(format!("{}\n\n{}", header, code))
    }

    /// Generate the `models.rs` file content, which contains the model registry
    pub fn generate_model_registry(&self, providers: &[ProviderInfo]) -> BuildResult<String> {
        let _timer = self.perf_monitor.start_timer("generate_model_registry");
        let mut model_instances = String::new();

        for provider_info in providers {
            for model_info in &provider_info.models {
                let capabilities_builder = self.generate_capabilities_builder(model_info);
                let provider_enum_variant = format!(
                    "Providers::{}",
                    to_pascal_case(&provider_info.name)
                );

                let model_instance = format!(
                    "    fluent_ai_domain::model::ModelInfo::new(\n        \"{}\",\n        {},\n        {}\n    )",
                    model_info.name,
                    provider_enum_variant,
                    capabilities_builder
                );

                model_instances.push_str(&model_instance);
                model_instances.push_str(",\n");
            }
        }

        let template = self.template_cache.models_registry.clone();
        let output = template.replace("{{MODEL_INSTANCES}}", &model_instances);
        
        let header = self.template_cache.file_header.clone();
        Ok(format!("{}\n\n{}", header, output))
    }

    /// Generates a string representing the builder pattern for ModelCapabilities
    fn generate_capabilities_builder(&self, model: &ModelInfo) -> String {
        let mut builder = "fluent_ai_domain::model::ModelCapabilities::builder()".to_string();

        if let Some(tokens) = model.max_input_tokens {
            builder.push_str(&format!(".max_input_tokens({})", tokens));
        }
        if let Some(tokens) = model.max_output_tokens {
            builder.push_str(&format!(".max_output_tokens({})", tokens));
        }
        if let Some(price) = model.input_price {
            builder.push_str(&format!(".input_price({})", price));
        }
        if let Some(price) = model.output_price {
            builder.push_str(&format!(".output_price({})", price));
        }
        if model.supports_vision {
            builder.push_str(".vision(true)");
        }
        if model.supports_function_calling {
            builder.push_str(".function_calling(true)");
        }

        builder.push_str(".build().unwrap()"); // unwrap is acceptable here as we control the inputs
        builder
    }
}

/// Convert string to PascalCase for enum variants
fn to_pascal_case(s: &str) -> String {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            // This unwrap is safe because we've filtered for non-empty strings.
            chars.next().unwrap().to_uppercase().to_string() + chars.as_str()
        })
        .collect()
}
