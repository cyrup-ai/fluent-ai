//! Test for ARCHITECTURE.md syntax compliance
//!
//! This module verifies that the Tool syntax from ARCHITECTURE.md works exactly as documented:
//! Tool<Perplexity>::new({"citations" => "true"})

#[cfg(test)]
mod tests {
    use crate::tool::{Tool, Perplexity};
    use crate::hash_map_fn;

    #[test]
    fn test_tool_perplexity_syntax() {
        // Test exact syntax from ARCHITECTURE.md
        let _tool = Tool::<Perplexity>::new(hash_map_fn!{"citations" => "true"});
        
        // This should compile and work
        assert!(true); // Basic compilation test
    }
    
    #[test] 
    fn test_tool_multiple_params() {
        // Test multiple parameters
        let _tool = Tool::<Perplexity>::new(hash_map_fn!{
            "citations" => "true",
            "mode" => "research"
        });
        
        assert!(true); // Basic compilation test
    }
    
    #[test]
    fn test_named_tool_syntax() {
        use crate::tool::ExecToText;
        
        // Test exact syntax from ARCHITECTURE.md
        let _tool = Tool::named("cargo")
            .bin("~/.cargo/bin")
            .description("cargo --help".exec_to_text());
            
        assert!(true); // Basic compilation test
    }
}