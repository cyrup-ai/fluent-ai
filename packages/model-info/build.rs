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
    
    // Verify the generated code compiles
    info!("Verifying generated code compiles...");
    if std::process::Command::new("cargo")
        .args(&["check", "--quiet"])
        .status()
        .is_ok()
    {
        success_check!("✅ Generated code compiles successfully!");
    } else {
        anyhow::bail!("❌ Generated code has compilation errors. Build failed.");
    }
    
    Ok(())
}