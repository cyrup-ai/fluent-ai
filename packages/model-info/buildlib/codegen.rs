use super::providers::ModelData;
use anyhow::Result;

/// Code generation utilities for model enums and trait implementations
/// Zero-allocation code generation with const string optimization
pub struct CodeGenerator {
    provider_name: String,
    enum_name: String,
}

impl CodeGenerator {
    /// Create a new code generator for a provider
    #[inline]
    pub fn new(provider_name: &str) -> Self {
        let enum_name = format!("{}Model", capitalize_first(provider_name));
        Self {
            provider_name: provider_name.to_string(),
            enum_name,
        }
    }
    
    /// Generate the complete model enum definition
    pub fn generate_enum(&self, models: &[ModelData]) -> Result<String> {
        let mut enum_variants = String::new();
        let mut all_variants = String::new();
        
        for (model_name, _, _, _, _, _) in models {
            let variant_name = super::providers::sanitize_ident(model_name);
            enum_variants.push_str(&format!("    {},\n", variant_name));
            all_variants.push_str(&format!("            {}Model::{},\n", 
                capitalize_first(&self.provider_name), variant_name));
        }
        
        let code = format!(
            r#"#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum {}Model {{
{}}}

impl {}Model {{
    #[doc = r" Get all available model variants"]
    pub const fn all_variants() -> &'static [{}Model] {{
        &[
{}        ]
    }}
    
    #[doc = r" Get all model variants as Vec"]
    pub fn all_models() -> Vec<{}Model> {{
        Self::all_variants().to_vec()
    }}
}}"#,
            capitalize_first(&self.provider_name),
            enum_variants,
            capitalize_first(&self.provider_name),
            capitalize_first(&self.provider_name),
            all_variants,
            capitalize_first(&self.provider_name)
        );
        
        Ok(code)
    }
    
    /// Generate the Model trait implementation
    pub fn generate_trait_impl(&self, models: &[ModelData]) -> Result<String> {
        let mut name_match = String::new();
        let mut max_input_match = String::new();
        let mut max_output_match = String::new();
        let mut pricing_input_match = String::new();
        let mut pricing_output_match = String::new();
        let mut supports_thinking_match = String::new();
        let mut required_temperature_match = String::new();
        let mut optimal_thinking_budget_match = String::new();
        
        for (model_name, max_tokens, input_price, output_price, supports_thinking, required_temp) in models {
            let variant_name = super::providers::sanitize_ident(model_name);
            let provider_enum = format!("{}Model", capitalize_first(&self.provider_name));
            
            // Calculate reasonable input/output split from max_tokens
            let max_input = (max_tokens * 3 / 4).max(4096).min(*max_tokens);
            let max_output = max_tokens.saturating_sub(max_input).max(1024);
            
            name_match.push_str(&format!(
                "            {}::{} => \"{}\",\n",
                provider_enum, variant_name, model_name
            ));
            
            max_input_match.push_str(&format!(
                "            {}::{} => Some({}u32),\n",
                provider_enum, variant_name, max_input
            ));
            
            max_output_match.push_str(&format!(
                "            {}::{} => Some({}u32),\n",
                provider_enum, variant_name, max_output
            ));
            
            pricing_input_match.push_str(&format!(
                "            {}::{} => Some({}f64),\n",
                provider_enum, variant_name, input_price
            ));
            
            pricing_output_match.push_str(&format!(
                "            {}::{} => Some({}f64),\n",
                provider_enum, variant_name, output_price
            ));
            
            if *supports_thinking {
                supports_thinking_match.push_str(&format!(
                    "            {}::{} => true,\n",
                    provider_enum, variant_name
                ));
                
                optimal_thinking_budget_match.push_str(&format!(
                    "            {}::{} => Some(100000u32),\n",
                    provider_enum, variant_name
                ));
            }
            
            if let Some(temp) = required_temp {
                required_temperature_match.push_str(&format!(
                    "            {}::{} => Some({}f64),\n",
                    provider_enum, variant_name, temp
                ));
            }
        }
        
        // Add default cases for thinking models
        if !supports_thinking_match.is_empty() {
            supports_thinking_match.push_str("            _ => false,\n");
            optimal_thinking_budget_match.push_str("            _ => None,\n");
        }
        
        if !required_temperature_match.is_empty() {
            required_temperature_match.push_str("            _ => None,\n");
        }
        
        let code = format!(
            r#"impl crate::common::Model for {}Model {{
    fn name(&self) -> &'static str {{
        match self {{
{}        }}
    }}
    
    fn provider_name(&self) -> &'static str {{
        "{}"
    }}
    
    fn max_input_tokens(&self) -> Option<u32> {{
        match self {{
{}        }}
    }}
    
    fn max_output_tokens(&self) -> Option<u32> {{
        match self {{
{}        }}
    }}
    
    fn pricing_input(&self) -> Option<f64> {{
        match self {{
{}        }}
    }}
    
    fn pricing_output(&self) -> Option<f64> {{
        match self {{
{}        }}
    }}
    
    fn supports_vision(&self) -> bool {{
        false
    }}
    
    fn supports_function_calling(&self) -> bool {{
        true
    }}
    
    fn supports_embeddings(&self) -> bool {{
        false
    }}
    
    fn requires_max_tokens(&self) -> bool {{
        false
    }}
    
    fn supports_thinking(&self) -> bool {{
{}    }}
    
    fn required_temperature(&self) -> Option<f64> {{
{}    }}
    
    fn optimal_thinking_budget(&self) -> Option<u32> {{
{}    }}
}}"#,
            capitalize_first(&self.provider_name),
            name_match,
            self.provider_name,
            max_input_match,
            max_output_match,
            pricing_input_match,
            pricing_output_match,
            if supports_thinking_match.is_empty() { "        false" } else { &supports_thinking_match },
            if required_temperature_match.is_empty() { "        None" } else { &required_temperature_match },
            if optimal_thinking_budget_match.is_empty() { "        None" } else { &optimal_thinking_budget_match }
        );
        
        Ok(code)
    }
}

/// Capitalize the first letter of a string
#[inline]
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}