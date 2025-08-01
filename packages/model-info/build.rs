// Modular build script - replaces the 595-line monolithic build.rs
// Uses strategy pattern with provider modules for clean, maintainable code generation

use anyhow::Result;
use termcolor::{info, success_check, colored_println};

mod buildlib;

#[tokio::main]
async fn main() -> Result<()> {
    colored_println!("🚀 Starting modular model code generation...");
    info!("Using strategy pattern with dynamic provider modules");
    
    // Generate model code using the new modular architecture
    buildlib::generate_model_code().await?;
    
    success_check!("✅ Model code generation completed successfully!");
    
    // Note: Compilation verification will happen automatically during the build process
    
    Ok(())
}