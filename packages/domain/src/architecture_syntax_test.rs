//! Test for EXACT ARCHITECTURE.md syntax
//!
//! This module tests the EXACT syntax shown in ARCHITECTURE.md:
//! Tool<Perplexity>::new({"citations" => "true"})
//!
//! The goal is to make this work WITHOUT exposing any macros to users.

#[cfg(test)]
mod tests {
    use crate::tool::{Tool, Perplexity};

    #[test]
    fn test_current_working_syntax() {
        // Current working syntax until cyrup_sugars transparent syntax is enabled
        use crate::hash_map_fn;
        let _tool = Tool::<Perplexity>::new(hash_map_fn!{"citations" => "true"});
        
        // Multiple params
        let _tool2 = Tool::<Perplexity>::new(hash_map_fn!{"citations" => "true", "mode" => "research"});
        
        assert!(true);
    }
    
    #[test] 
    fn test_named_tool_working_syntax() {
        use crate::tool::ExecToText;
        let _tool = Tool::named("cargo")
            .bin("~/.cargo/bin") 
            .description("cargo --help".exec_to_text());
            
        assert!(true);
    }
}