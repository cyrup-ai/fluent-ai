use std::fs;

fn main() {
    let yaml_content = fs::read_to_string("models.yaml").expect("Failed to read models.yaml");
    println!("YAML content ({} bytes):", yaml_content.len());
    println!("{}", yaml_content);
    
    println!("\n--- Testing yyaml::from_str ---");
    match yyaml::from_str::<serde_json::Value>(&yaml_content) {
        Ok(value) => {
            println!("✅ Successfully parsed with yyaml::from_str");
            println!("Value: {:?}", value);
        }
        Err(e) => {
            println!("❌ Error parsing with yyaml::from_str: {}", e);
        }
    }
}