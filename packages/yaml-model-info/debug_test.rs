use yaml_model_info::models::YamlProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let yaml_content = std::fs::read_to_string(".yaml-cache/models.yaml")?;
    println!("YAML content length: {}", yaml_content.len());
    println!("First 500 chars:\n{}", &yaml_content[..500.min(yaml_content.len())]);
    
    // Test parsing with yyaml
    match yyaml::from_str::<Vec<YamlProvider>>(&yaml_content) {
        Ok(providers) => {
            println!("Successfully parsed {} providers", providers.len());
            if let Some(first) = providers.first() {
                println!("First provider: {}", first.provider);
                println!("First provider has {} models", first.models.len());
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
            
            // Try parsing as raw serde_json::Value to see structure
            match yyaml::from_str::<serde_json::Value>(&yaml_content) {
                Ok(v) => println!("Raw parse successful: {:#?}", v),
                Err(e2) => println!("Even raw parse failed: {:?}", e2),
            }
        }
    }
    
    Ok(())
}