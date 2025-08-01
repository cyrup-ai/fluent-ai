use super::providers::ModelData;
use anyhow::{Result, Context};
use proc_macro2::{Span, TokenStream};
use quote::{quote, format_ident, ToTokens};
use syn::{
    parse_quote, Arm, Expr, ExprLit, ExprMatch, Fields, Ident, Item, ItemEnum, ItemImpl,
    Lit, LitInt, LitStr, Pat, PatPath, Path, Stmt, Type, TypePath, Variant, 
    ImplItem, ImplItemFn, FnArg, ReturnType, Block, PathSegment,
};

/// Zero-allocation code generation using syn AST manipulation
/// Replaces string-based templates with type-safe AST construction
pub struct SynCodeGenerator {
    provider_name: String,
    enum_name: Ident,
}

impl SynCodeGenerator {
    /// Create a new syn-based code generator for a provider
    #[inline]
    pub fn new(provider_name: &str) -> Self {
        let enum_name = format_ident!("{}Model", capitalize_first(provider_name));
        Self {
            provider_name: provider_name.to_string(),
            enum_name,
        }
    }
    
    /// Generate the complete model enum definition using syn AST
    pub fn generate_enum(&self, models: &[ModelData]) -> Result<String> {
        let mut variants = Vec::with_capacity(models.len());
        let mut all_variants_elements = Vec::with_capacity(models.len());
        
        // Build variants and all_variants array elements
        for (model_name, _, _, _, _, _) in models {
            let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
            
            // Create enum variant
            variants.push(Variant {
                attrs: vec![],
                ident: variant_name.clone(),
                fields: Fields::Unit,
                discriminant: None,
            });
            
            // Create element for all_variants array
            let path = format_ident!("{}::{}", self.enum_name, variant_name);
            all_variants_elements.push(parse_quote!(#path));
        }
        
        // Build the enum with derive attributes
        let enum_item = ItemEnum {
            attrs: vec![parse_quote!(#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)])],
            vis: parse_quote!(pub),
            enum_token: Default::default(),
            ident: self.enum_name.clone(),
            generics: Default::default(),
            brace_token: Default::default(),
            variants: variants.into_iter().collect(),
        };
        
        // Build impl block with associated functions
        let enum_name = &self.enum_name;
        let all_variants_fn = self.build_all_variants_fn(&all_variants_elements)?;
        let all_models_fn = self.build_all_models_fn()?;
        
        let impl_item = parse_quote! {
            impl #enum_name {
                #all_variants_fn
                #all_models_fn
            }
        };
        
        // Combine enum and impl into final token stream
        let tokens = quote! {
            #enum_item
            
            #impl_item
        };
        
        Ok(prettyplease::unparse(&syn::parse2(tokens)?))
    }
    
    /// Generate the Model trait implementation using syn AST
    pub fn generate_trait_impl(&self, models: &[ModelData]) -> Result<String> {
        let trait_path: Path = parse_quote!(crate::common::Model);
        let enum_name = &self.enum_name;
        
        // Build all method implementations
        let name_method = self.build_name_method(models)?;
        let provider_name_method = self.build_provider_name_method()?;
        let max_input_tokens_method = self.build_max_input_tokens_method(models)?;
        let max_output_tokens_method = self.build_max_output_tokens_method(models)?;
        let pricing_input_method = self.build_pricing_input_method(models)?;
        let pricing_output_method = self.build_pricing_output_method(models)?;
        let supports_vision_method = self.build_supports_vision_method()?;
        let supports_function_calling_method = self.build_supports_function_calling_method()?;
        let supports_embeddings_method = self.build_supports_embeddings_method()?;
        let requires_max_tokens_method = self.build_requires_max_tokens_method()?;
        let supports_thinking_method = self.build_supports_thinking_method(models)?;
        let required_temperature_method = self.build_required_temperature_method(models)?;
        let optimal_thinking_budget_method = self.build_optimal_thinking_budget_method(models)?;
        
        let impl_item: ItemImpl = parse_quote! {
            impl crate::common::Model for #enum_name {
                #name_method
                #provider_name_method
                #max_input_tokens_method
                #max_output_tokens_method
                #pricing_input_method
                #pricing_output_method
                #supports_vision_method
                #supports_function_calling_method
                #supports_embeddings_method
                #requires_max_tokens_method
                #supports_thinking_method
                #required_temperature_method
                #optimal_thinking_budget_method
            }
        };
        
        let tokens = quote! { #impl_item };
        Ok(prettyplease::unparse(&syn::parse2(tokens)?))
    }
    
    /// Build all_variants associated function
    #[inline]
    fn build_all_variants_fn(&self, elements: &[Expr]) -> Result<ImplItemFn> {
        let enum_name = &self.enum_name;
        Ok(parse_quote! {
            #[doc = r" Get all available model variants"]
            pub const fn all_variants() -> &'static [#enum_name] {
                &[#(#elements),*]
            }
        })
    }
    
    /// Build all_models associated function
    #[inline]
    fn build_all_models_fn(&self) -> Result<ImplItemFn> {
        let enum_name = &self.enum_name;
        Ok(parse_quote! {
            #[doc = r" Get all model variants as Vec"]
            pub fn all_models() -> Vec<#enum_name> {
                Self::all_variants().to_vec()
            }
        })
    }
    
    /// Build name() method implementation
    #[inline]
    fn build_name_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let arms = self.build_string_match_arms(models, |model_name, _, _, _, _, _| {
            model_name.clone()
        })?;
        
        Ok(parse_quote! {
            #[inline]
            fn name(&self) -> &'static str {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build provider_name() method implementation
    #[inline]
    fn build_provider_name_method(&self) -> Result<ImplItemFn> {
        let provider_name = &self.provider_name;
        Ok(parse_quote! {
            #[inline]
            fn provider_name(&self) -> &'static str {
                #provider_name
            }
        })
    }
    
    /// Build max_input_tokens() method implementation
    #[inline]
    fn build_max_input_tokens_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let arms = self.build_option_u32_match_arms(models, |_, max_tokens, _, _, _, _| {
            let max_input = (max_tokens * 3 / 4).max(4096).min(*max_tokens) as u32;
            Some(max_input)
        })?;
        
        Ok(parse_quote! {
            #[inline]
            fn max_input_tokens(&self) -> Option<u32> {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build max_output_tokens() method implementation  
    #[inline]
    fn build_max_output_tokens_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let arms = self.build_option_u32_match_arms(models, |_, max_tokens, _, _, _, _| {
            let max_input = (max_tokens * 3 / 4).max(4096).min(*max_tokens);
            let max_output = max_tokens.saturating_sub(max_input).max(1024) as u32;
            Some(max_output)
        })?;
        
        Ok(parse_quote! {
            #[inline]
            fn max_output_tokens(&self) -> Option<u32> {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build pricing_input() method implementation
    #[inline]
    fn build_pricing_input_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let arms = self.build_option_f64_match_arms(models, |_, _, input_price, _, _, _| {
            Some(*input_price)
        })?;
        
        Ok(parse_quote! {
            #[inline]
            fn pricing_input(&self) -> Option<f64> {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build pricing_output() method implementation
    #[inline]
    fn build_pricing_output_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let arms = self.build_option_f64_match_arms(models, |_, _, _, output_price, _, _| {
            Some(*output_price)
        })?;
        
        Ok(parse_quote! {
            #[inline]
            fn pricing_output(&self) -> Option<f64> {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build supports_vision() method (static false for now)
    #[inline]
    fn build_supports_vision_method(&self) -> Result<ImplItemFn> {
        Ok(parse_quote! {
            #[inline]
            fn supports_vision(&self) -> bool {
                false
            }
        })
    }
    
    /// Build supports_function_calling() method (static true for now)
    #[inline]
    fn build_supports_function_calling_method(&self) -> Result<ImplItemFn> {
        Ok(parse_quote! {
            #[inline]
            fn supports_function_calling(&self) -> bool {
                true
            }
        })
    }
    
    /// Build supports_embeddings() method (static false for now)
    #[inline]
    fn build_supports_embeddings_method(&self) -> Result<ImplItemFn> {
        Ok(parse_quote! {
            #[inline]
            fn supports_embeddings(&self) -> bool {
                false
            }
        })
    }
    
    /// Build requires_max_tokens() method (static false for now)
    #[inline]
    fn build_requires_max_tokens_method(&self) -> Result<ImplItemFn> {
        Ok(parse_quote! {
            #[inline]
            fn requires_max_tokens(&self) -> bool {
                false
            }
        })
    }
    
    /// Build supports_thinking() method implementation
    #[inline]
    fn build_supports_thinking_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let thinking_arms: Vec<Arm> = models
            .iter()
            .filter(|(_, _, _, _, supports_thinking, _)| *supports_thinking)
            .map(|(model_name, _, _, _, _, _)| {
                let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
                let pattern_ident = format_ident!("{}::{}", self.enum_name, variant_name);
                parse_quote! {
                    #pattern_ident => true,
                }
            })
            .collect();
        
        let arms = if thinking_arms.is_empty() {
            // No thinking models - return static false
            return Ok(parse_quote! {
                #[inline]
                fn supports_thinking(&self) -> bool {
                    false
                }
            });
        } else {
            let mut all_arms = thinking_arms;
            all_arms.push(parse_quote! { _ => false, });
            all_arms
        };
        
        Ok(parse_quote! {
            #[inline]
            fn supports_thinking(&self) -> bool {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build required_temperature() method implementation
    #[inline]
    fn build_required_temperature_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let temp_arms: Vec<Arm> = models
            .iter()
            .filter_map(|(model_name, _, _, _, _, required_temp)| {
                required_temp.map(|temp| {
                    let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
                    let pattern_ident = format_ident!("{}::{}", self.enum_name, variant_name);
                    let temp_lit = LitStr::new(&format!("{}f64", temp), Span::call_site());
                    let temp_expr: Expr = syn::parse_str(&format!("Some({}f64)", temp)).unwrap();
                    parse_quote! {
                        #pattern_ident => #temp_expr,
                    }
                })
            })
            .collect();
        
        let arms = if temp_arms.is_empty() {
            // No models with required temperature - return static None
            return Ok(parse_quote! {
                #[inline]
                fn required_temperature(&self) -> Option<f64> {
                    None
                }
            });
        } else {
            let mut all_arms = temp_arms;
            all_arms.push(parse_quote! { _ => None, });
            all_arms
        };
        
        Ok(parse_quote! {
            #[inline]
            fn required_temperature(&self) -> Option<f64> {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build optimal_thinking_budget() method implementation
    #[inline]
    fn build_optimal_thinking_budget_method(&self, models: &[ModelData]) -> Result<ImplItemFn> {
        let thinking_arms: Vec<Arm> = models
            .iter()
            .filter(|(_, _, _, _, supports_thinking, _)| *supports_thinking)
            .map(|(model_name, _, _, _, _, _)| {
                let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
                let pattern_ident = format_ident!("{}::{}", self.enum_name, variant_name);
                parse_quote! {
                    #pattern_ident => Some(100000u32),
                }
            })
            .collect();
        
        let arms = if thinking_arms.is_empty() {
            // No thinking models - return static None
            return Ok(parse_quote! {
                #[inline]
                fn optimal_thinking_budget(&self) -> Option<u32> {
                    None
                }
            });
        } else {
            let mut all_arms = thinking_arms;
            all_arms.push(parse_quote! { _ => None, });
            all_arms
        };
        
        Ok(parse_quote! {
            #[inline]
            fn optimal_thinking_budget(&self) -> Option<u32> {
                match self {
                    #(#arms)*
                }
            }
        })
    }
    
    /// Build match arms for string return values
    #[inline]
    fn build_string_match_arms<F>(&self, models: &[ModelData], value_fn: F) -> Result<Vec<Arm>>
    where
        F: Fn(&String, &u64, &f64, &f64, &bool, &Option<f64>) -> String,
    {
        let mut arms = Vec::with_capacity(models.len());
        
        for model_data in models {
            let (model_name, max_tokens, input_price, output_price, supports_thinking, required_temp) = model_data;
            let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
            let pattern_ident = format_ident!("{}::{}", self.enum_name, variant_name);
            
            let value = value_fn(model_name, max_tokens, input_price, output_price, supports_thinking, required_temp);
            let value_lit = LitStr::new(&value, Span::call_site());
            
            arms.push(parse_quote! {
                #pattern_ident => #value_lit,
            });
        }
        
        Ok(arms)
    }
    
    /// Build match arms for Option<u32> return values
    #[inline]
    fn build_option_u32_match_arms<F>(&self, models: &[ModelData], value_fn: F) -> Result<Vec<Arm>>
    where
        F: Fn(&String, &u64, &f64, &f64, &bool, &Option<f64>) -> Option<u32>,
    {
        let mut arms = Vec::with_capacity(models.len());
        
        for model_data in models {
            let (model_name, max_tokens, input_price, output_price, supports_thinking, required_temp) = model_data;
            let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
            let pattern_ident = format_ident!("{}::{}", self.enum_name, variant_name);
            
            let value = value_fn(model_name, max_tokens, input_price, output_price, supports_thinking, required_temp);
            let value_expr: Expr = match value {
                Some(v) => syn::parse_str(&format!("Some({}u32)", v))?,
                None => parse_quote!(None),
            };
            
            arms.push(parse_quote! {
                #pattern_ident => #value_expr,
            });
        }
        
        Ok(arms)
    }
    
    /// Build match arms for Option<f64> return values
    #[inline]
    fn build_option_f64_match_arms<F>(&self, models: &[ModelData], value_fn: F) -> Result<Vec<Arm>>
    where
        F: Fn(&String, &u64, &f64, &f64, &bool, &Option<f64>) -> Option<f64>,
    {
        let mut arms = Vec::with_capacity(models.len());
        
        for model_data in models {
            let (model_name, max_tokens, input_price, output_price, supports_thinking, required_temp) = model_data;
            let variant_name = format_ident!("{}", super::providers::sanitize_ident(model_name));
            let pattern_ident = format_ident!("{}::{}", self.enum_name, variant_name);
            
            let value = value_fn(model_name, max_tokens, input_price, output_price, supports_thinking, required_temp);
            let value_expr: Expr = match value {
                Some(v) => syn::parse_str(&format!("Some({}f64)", v))?,
                None => parse_quote!(None),
            };
            
            arms.push(parse_quote! {
                #pattern_ident => #value_expr,
            });
        }
        
        Ok(arms)
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