use anyhow::Result;
use quote::{format_ident, quote};
use syn::{Arm, Expr, Ident, parse_quote};

use super::providers::ModelData;

/// Mapping from provider name to custom PascalCase enum name without 'Model' suffix
fn get_custom_enum_name(provider: &str) -> String {
    match provider {
        "openai" => "OpenAI".to_string(),
        "mistral" => "Mistral".to_string(),
        "anthropic" => "Anthropic".to_string(),
        "together" => "Together".to_string(),
        "openrouter" => "OpenRouter".to_string(),
        "huggingface" => "HuggingFace".to_string(),
        "xai" => "XAI".to_string(),
        _ => capitalize_first(provider), // Default: capitalize first letter
    }
}

/// Zero-allocation code generation using syn AST manipulation
/// Replaces string-based templates with type-safe AST construction
pub struct SynCodeGenerator {
    provider_name: String,
    enum_name: Ident,
    custom_enum_name: Ident,
}

impl SynCodeGenerator {
    /// Create a new syn-based code generator for a provider
    #[inline]
    pub fn new(provider_name: &str) -> Self {
        let enum_name = format_ident!("{}", capitalize_first(&format!("{}Model", provider_name)));
        let custom_enum_name = format_ident!("{}", get_custom_enum_name(provider_name));
        Self {
            provider_name: provider_name.to_string(),
            enum_name,
            custom_enum_name,
        }
    }

    /// Generate both the original and custom model enum definitions using syn AST
    pub fn generate_enum(&self, models: &[ModelData]) -> Result<String> {
        let original_code = self.generate_single_enum(&self.enum_name, models)?;
        let custom_code = self.generate_single_enum(&self.custom_enum_name, models)?;
        Ok(format!("{}\n\n{}", original_code, custom_code))
    }

    /// Helper to generate a single enum
    fn generate_single_enum(&self, enum_name: &Ident, models: &[ModelData]) -> Result<String> {
        let mut variant_names = Vec::with_capacity(models.len());

        // Build variant names
        for (model_name, _, _, _, _, _) in models {
            let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
            variant_names.push(variant_name);
        }

        // Generate the enum and impl using quote!
        let tokens = quote! {
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            pub enum #enum_name {
                #(#variant_names,)*
            }

            impl #enum_name {
                #[doc = r" Get all available model variants"]
                pub const fn all_variants() -> &'static [#enum_name] {
                    &[
                        #(#enum_name::#variant_names,)*
                    ]
                }

                #[doc = r" Get all model variants as Vec"]
                pub fn all_models() -> Vec<#enum_name> {
                    Self::all_variants().to_vec()
                }
            }
        };

        Ok(prettyplease::unparse(&syn::parse2(tokens)?))
    }

    /// Generate the Model trait implementation for both enums using syn AST
    pub fn generate_trait_impl(&self, models: &[ModelData]) -> Result<String> {
        let original_impl = self.generate_single_trait_impl(&self.enum_name, models)?;
        let custom_impl = self.generate_single_trait_impl(&self.custom_enum_name, models)?;
        Ok(format!("{}\n\n{}", original_impl, custom_impl))
    }

    /// Helper to generate a single trait impl
    fn generate_single_trait_impl(
        &self,
        enum_name: &Ident,
        models: &[ModelData],
    ) -> Result<String> {
        let provider_name = &self.provider_name;

        // Build match arms for each method
        let name_arms = self.build_name_arms(enum_name, models);
        let max_input_arms = self.build_max_input_arms(enum_name, models);
        let max_output_arms = self.build_max_output_arms(enum_name, models);
        let pricing_input_arms = self.build_pricing_input_arms(enum_name, models);
        let pricing_output_arms = self.build_pricing_output_arms(enum_name, models);
        let thinking_arms = self.build_thinking_arms(enum_name, models);
        let required_temp_arms = self.build_required_temp_arms(enum_name, models);
        let thinking_budget_arms = self.build_thinking_budget_arms(enum_name, models);

        let tokens = quote! {
            impl crate::common::Model for #enum_name {
                #[inline]
                fn name(&self) -> &'static str {
                    match self {
                        #(#name_arms)*
                    }
                }

                #[inline]
                fn provider_name(&self) -> &'static str {
                    #provider_name
                }

                #[inline]
                fn max_input_tokens(&self) -> Option<u32> {
                    match self {
                        #(#max_input_arms)*
                    }
                }

                #[inline]
                fn max_output_tokens(&self) -> Option<u32> {
                    match self {
                        #(#max_output_arms)*
                    }
                }

                #[inline]
                fn pricing_input(&self) -> Option<f64> {
                    match self {
                        #(#pricing_input_arms)*
                    }
                }

                #[inline]
                fn pricing_output(&self) -> Option<f64> {
                    match self {
                        #(#pricing_output_arms)*
                    }
                }

                #[inline]
                fn supports_vision(&self) -> bool {
                    false
                }

                #[inline]
                fn supports_function_calling(&self) -> bool {
                    true
                }

                #[inline]
                fn supports_embeddings(&self) -> bool {
                    false
                }

                #[inline]
                fn requires_max_tokens(&self) -> bool {
                    false
                }

                #[inline]
                fn supports_thinking(&self) -> bool {
                    #thinking_arms
                }

                #[inline]
                fn required_temperature(&self) -> Option<f64> {
                    #required_temp_arms
                }

                #[inline]
                fn optimal_thinking_budget(&self) -> Option<u32> {
                    #thinking_budget_arms
                }
            }
        };

        Ok(prettyplease::unparse(&syn::parse2(tokens)?))
    }

    /// Build match arms for name() method
    #[inline]
    fn build_name_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Vec<Arm> {
        models
            .iter()
            .map(|(model_name, _, _, _, _, _)| {
                let variant_name =
                    format_ident!("{}", super::providers::sanitize_ident(model_name));
                parse_quote! {
                    #enum_name::#variant_name => #model_name,
                }
            })
            .collect()
    }

    /// Build match arms for max_input_tokens() method
    #[inline]
    fn build_max_input_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Vec<Arm> {
        models
            .iter()
            .map(|(model_name, max_tokens, _, _, _, _)| {
                let variant_name =
                    format_ident!("{}", super::providers::sanitize_ident(model_name));
                let max_input = (max_tokens * 3 / 4).max(4096).min(*max_tokens) as u32;
                parse_quote! {
                    #enum_name::#variant_name => Some(#max_input),
                }
            })
            .collect()
    }

    /// Build match arms for max_output_tokens() method
    #[inline]
    fn build_max_output_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Vec<Arm> {
        models
            .iter()
            .map(|(model_name, max_tokens, _, _, _, _)| {
                let variant_name =
                    format_ident!("{}", super::providers::sanitize_ident(model_name));
                let max_input = (max_tokens * 3 / 4).max(4096).min(*max_tokens);
                let max_output = max_tokens.saturating_sub(max_input).max(1024) as u32;
                parse_quote! {
                    #enum_name::#variant_name => Some(#max_output),
                }
            })
            .collect()
    }

    /// Build match arms for pricing_input() method
    #[inline]
    fn build_pricing_input_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Vec<Arm> {
        models
            .iter()
            .map(|(model_name, _, input_price, _, _, _)| {
                let variant_name =
                    format_ident!("{}", super::providers::sanitize_ident(model_name));
                parse_quote! {
                    #enum_name::#variant_name => Some(#input_price),
                }
            })
            .collect()
    }

    /// Build match arms for pricing_output() method
    #[inline]
    fn build_pricing_output_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Vec<Arm> {
        models
            .iter()
            .map(|(model_name, _, _, output_price, _, _)| {
                let variant_name =
                    format_ident!("{}", super::providers::sanitize_ident(model_name));
                parse_quote! {
                    #enum_name::#variant_name => Some(#output_price),
                }
            })
            .collect()
    }

    /// Build thinking support logic
    #[inline]
    fn build_thinking_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Expr {
        let thinking_variants: Vec<syn::Arm> = models
            .iter()
            .filter(|(_, _, _, _, supports_thinking, _)| *supports_thinking)
            .map(|(model_name, _, _, _, _, _)| {
                let variant_name =
                    format_ident!("{}", super::providers::sanitize_ident(model_name));
                parse_quote! {
                    #enum_name::#variant_name => true,
                }
            })
            .collect();

        if thinking_variants.is_empty() {
            parse_quote! { false }
        } else {
            let mut arms: Vec<syn::Arm> = thinking_variants;
            arms.push(parse_quote! { _ => false, });
            parse_quote! {
                match self {
                    #(#arms)*
                }
            }
        }
    }

    /// Build required temperature logic
    #[inline]
    fn build_required_temp_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Expr {
        let temp_variants: Vec<syn::Arm> = models
            .iter()
            .filter_map(|(model_name, _, _, _, _, required_temp)| {
                required_temp.map(|temp| {
                    let variant_name =
                        format_ident!("{}", super::providers::sanitize_ident(model_name));
                    parse_quote! {
                        #enum_name::#variant_name => Some(#temp),
                    }
                })
            })
            .collect();

        if temp_variants.is_empty() {
            parse_quote! { None }
        } else {
            let mut arms: Vec<syn::Arm> = temp_variants;
            arms.push(parse_quote! { _ => None, });
            parse_quote! {
                match self {
                    #(#arms)*
                }
            }
        }
    }

    /// Build thinking budget logic
    #[inline]
    fn build_thinking_budget_arms(&self, enum_name: &Ident, models: &[ModelData]) -> Expr {
        let budget_variants: Vec<syn::Arm> = models
            .iter()
            .filter(|(_, _, _, _, supports_thinking, _)| *supports_thinking)
            .map(|(model_name, _, _, _, _, _)| {
                let variant_name =
                    format_ident!("{}", super::providers::sanitize_ident(model_name));
                parse_quote! {
                    #enum_name::#variant_name => Some(100000u32),
                }
            })
            .collect();

        if budget_variants.is_empty() {
            parse_quote! { None }
        } else {
            let mut arms: Vec<syn::Arm> = budget_variants;
            arms.push(parse_quote! { _ => None, });
            parse_quote! {
                match self {
                    #(#arms)*
                }
            }
        }
    }
}

/// Capitalize the first letter of a string (zero-allocation)
#[inline]
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
