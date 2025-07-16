//! {{project-name}} - {{description}}
//!
//! This project uses cyrup-sugars with features: {{features}}
//! {% if include_examples %}Delete the examples you don't need and keep the ones you want to use.{% endif %}

use cyrup_sugars::{OneOrMany, ZeroOneOrMany, AsyncTask};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

{% if include_examples %}
mod one_or_many_demo;
mod zero_one_or_many_demo;
mod async_task_demo;
mod json_syntax_demo;
mod serialization_demo;
mod service_builder;
{% endif %}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== {{project-name}} ===\n");

    {% if include_examples %}
    // 🗑️ DELETE DEMO MODULES YOU DON'T NEED

    // OneOrMany - Non-empty collections
    one_or_many_demo::run();
    
    // ZeroOneOrMany - Flexible collections  
    zero_one_or_many_demo::run();
    
    // AsyncTask - Async patterns
    async_task_demo::run().await?;
    
    // JSON Syntax - Builder patterns  
    json_syntax_demo::run();
    
    // Serialization with Serde
    serialization_demo::run()?;

    // Service builder example
    service_builder::demo()?;
    {% else %}
    // TODO: Add your application logic here
    println!("Hello from {{project-name}}!");
    {% endif %}

    Ok(())
}