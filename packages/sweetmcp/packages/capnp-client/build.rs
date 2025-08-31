fn main() {
    // Generate Rust code from Cap'n Proto schema
    if let Err(e) = ::capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/mcp_request.capnp")
        .run()
    {
        eprintln!("Failed to compile Cap'n Proto schema: {}", e);
        eprintln!("Make sure capnp compiler is installed and schema file exists");
        std::process::exit(1);
    }
    
    // Tell cargo to rerun this build script if the schema changes
    println!("cargo:rerun-if-changed=schema/mcp_request.capnp");
}