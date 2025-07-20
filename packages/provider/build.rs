//! Simplified build script that avoids HTTP download to unblock compilation

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Re-run if any build files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build");
    println!("cargo:rerun-if-changed=providers");

    // Generate empty provider code to unblock compilation
    write_generated_code("providers.rs", "// Placeholder provider constants\n")?;
    write_generated_code("models.rs", "// Placeholder model registry\n")?;

    println!("cargo:warning=Build script simplified to unblock compilation");
    Ok(())
}

/// Helper to write generated code to file
fn write_generated_code(path: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| "Failed to get output directory")?;

    let dest_path = out_dir.join(path);
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&dest_path, content)?;
    println!("cargo:rerun-if-changed={}", dest_path.display());
    Ok(())
}
