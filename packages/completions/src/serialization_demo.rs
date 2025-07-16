{% if include_examples %}
//! Serialization examples - DELETE THIS MODULE IF NOT NEEDED
//!
//! Shows Serde integration with cyrup-sugars collection types.

use cyrup_sugars::{OneOrMany, ZeroOneOrMany};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    servers: OneOrMany<String>,
    middleware: ZeroOneOrMany<String>,
    timeout: u32,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Serialization Examples:");
    
    let config = Config {
        servers: OneOrMany::many(vec!["api1.example.com".to_string(), "api2.example.com".to_string()]).unwrap(),
        middleware: ZeroOneOrMany::many(vec!["auth".to_string(), "logging".to_string()]),
        timeout: 30,
    };
    
    // Serialize to JSON
    let json = serde_json::to_string_pretty(&config)?;
    println!("  Serialized config:\n{}", json);
    
    // Deserialize back
    let deserialized: Config = serde_json::from_str(&json)?;
    println!("  Deserialized: {:?}", deserialized);
    
    println!();
    Ok(())
}
{% endif %}