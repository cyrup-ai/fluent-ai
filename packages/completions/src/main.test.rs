//! Test template - Test project
//!
//! This project uses cyrup-sugars with features: tokio-async

use cyrup_sugars::{OneOrMany, ZeroOneOrMany, AsyncTask};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

mod one_or_many_demo;
mod zero_one_or_many_demo;
mod async_task_demo;
mod json_syntax_demo;
mod serialization_demo;
mod service_builder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Test Template ===\n");

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

    Ok(())
}