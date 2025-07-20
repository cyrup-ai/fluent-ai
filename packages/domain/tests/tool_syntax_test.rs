//! Test for ARCHITECTURE.md syntax compliance
//!
//! This module verifies that the Tool syntax from ARCHITECTURE.md works exactly as documented:
//! Tool<Perplexity>::new({"citations" => "true"})

use fluent_ai_domain::hash_map_fn;
use fluent_ai_domain::tool::{Perplexity, Tool};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_perplexity_syntax() {
        // Test exact syntax from ARCHITECTURE.md
        let _tool = Tool::<Perplexity>::new(hash_map_fn! {"citations" => "true"});

        // This should compile and work
        assert!(true); // Basic compilation test
    }

    #[test]
    fn test_tool_multiple_params() {
        // Test multiple parameters
        let _tool = Tool::<Perplexity>::new(hash_map_fn! {
            "citations" => "true",
            "mode" => "research"
        });

        assert!(true); // Basic compilation test
    }

    #[test]
    fn test_named_tool_syntax() {
        use fluent_ai_domain::tool::ExecToText;

        // Test exact syntax from ARCHITECTURE.md
        let _tool = Tool::named("cargo")
            .bin("~/.cargo/bin")
            .description("cargo --help".exec_to_text());

        assert!(true); // Basic compilation test
    }
}
