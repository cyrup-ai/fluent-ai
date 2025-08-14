// Example showing the SEXY fluent builder with semantic description builder

use extism_pdk::Error;
use serde_json::Value;
// Import required types from sweetmcp_plugin_builder
use sweetmcp_plugin_builder::{
    CallToolResult, ContentBuilder, DescriptionBuilder, McpPlugin, McpTool, Ready, SchemaBuilder,
    generate_mcp_functions, mcp_plugin,
};

// Example tool implementations
struct HashTool;
struct EncryptTool;
struct SignTool;

/// Example: Using the semantic description builder
fn build_hash_description() -> String {
    DescriptionBuilder::default()
        .does("Generate cryptographic hashes and encoded formats from input data")
        .when("Create SHA hashes for security verification (sha256, sha512, sha384, sha224, sha1)")
        .when("Generate MD5 checksums for file integrity")
        .when("Encode data in base64 format for transmission")
        .when("Encode data in base32 format for URLs or identifiers")
        .when("Verify data integrity before storage or transmission")
        .perfect_for("data integrity checks, password verification, API authentication, and encoding binary data for text protocols")
        .build()
}

/// Example: Multi-operation tool with semantic descriptions
struct TimeToolDescription;

impl TimeToolDescription {
    fn build() -> String {
        DescriptionBuilder::default()
            .does("Perform time operations and calculations")
            .operation("get_time_utc", "Returns current UTC time (no parameters)")
            .operation("parse_time", "Parse RFC2822 time strings to timestamps")
            .operation("time_offset", "Add/subtract time offsets from timestamps")
            .always_for("compute time operations, especially when working with time zone conversions, date calculations, or scheduling")
            .not_for("historical dates before 1970 (Unix epoch limitation)")
            .perfect_for("time zone conversions, date calculations, scheduling operations, and time-based comparisons")
            .build()
    }
}

/// Example: Browser tool with prerequisites
struct BrowserToolDescription;

impl BrowserToolDescription {
    fn build() -> String {
        DescriptionBuilder::default()
            .does("Automate browser interactions and web scraping")
            .when("Navigate to web pages and extract data")
            .when("Automate form submissions and user interactions")
            .when("Take screenshots of web pages")
            .requires("A running browser instance (Chrome/Firefox)")
            .requires("Internet connectivity")
            .not_for("Heavy web scraping that might violate terms of service")
            .perfect_for("web automation, testing, and light data extraction")
            .build()
    }
}

// Create the plugin instance
fn plugin() -> McpPlugin<Ready> {
    mcp_plugin("crypto-suite")
        .description("A collection of cryptographic operations including hashing, encryption, and digital signatures.")
        .tool::<HashTool>()
        .tool::<EncryptTool>()
        .tool::<SignTool>()
        .serve()
}

// Generate the standard MCP entry points for our plugin
generate_mcp_functions!(plugin);

// In a real plugin, this would be the main entry point
// but for this example, we'll just print a message
fn main() {
    println!("Crypto Suite Plugin");
    println!("Available tools:");
    println!("- HashTool: {}", HashTool::NAME);
    println!("- EncryptTool: {}", EncryptTool::NAME);
    println!("- SignTool: {}", SignTool::NAME);
}

// Tool implementations (structs already defined above)

impl McpTool for HashTool {
    const NAME: &'static str = "hash";

    fn description(builder: DescriptionBuilder) -> DescriptionBuilder {
        builder
            .does("Generate cryptographic hashes and encoded formats from input data")
            .when("Create SHA hashes for security verification (sha256, sha512, sha384, sha224, sha1)")
            .when("Generate MD5 checksums for file integrity")
            .when("Encode data in base64 format for transmission")
            .when("Encode data in base32 format for URLs or identifiers")
            .when("Verify data integrity before storage or transmission")
            .perfect_for("data integrity checks, password verification, API authentication, and encoding binary data for text protocols")
    }

    fn schema(builder: SchemaBuilder) -> Value {
        builder
            .required_string("data", "data to convert to hash or encoded format")
            .required_enum(
                "algorithm",
                "algorithm to use",
                &["sha256", "sha512", "md5", "base64"],
            )
            .build()
    }

    fn execute(_args: Value) -> Result<CallToolResult, Error> {
        // Business logic here
        Ok(ContentBuilder::text("hash_result"))
    }
}

impl McpTool for EncryptTool {
    const NAME: &'static str = "encrypt";

    fn description(builder: DescriptionBuilder) -> DescriptionBuilder {
        builder
            .does("Encrypt data using AES-256-GCM")
            .when("Protect sensitive data before storage")
            .when("Secure data for transmission")
            .when("Implement client-side encryption")
            .perfect_for("data protection and secure communication")
    }

    fn schema(builder: SchemaBuilder) -> Value {
        builder
            .required_string("data", "data to encrypt")
            .required_string("key", "encryption key (base64)")
            .build()
    }

    fn execute(_args: Value) -> Result<CallToolResult, Error> {
        Ok(ContentBuilder::text("encrypted_data"))
    }
}

impl McpTool for SignTool {
    const NAME: &'static str = "sign";

    fn description(builder: DescriptionBuilder) -> DescriptionBuilder {
        builder
            .does("Create digital signatures")
            .when("Sign documents or data")
            .when("Verify authenticity")
            .when("Implement non-repudiation")
            .perfect_for("digital signatures and authentication")
    }

    fn schema(builder: SchemaBuilder) -> Value {
        builder
            .required_string("data", "data to sign")
            .required_string("private_key", "private key for signing")
            .build()
    }

    fn execute(_args: Value) -> Result<CallToolResult, Error> {
        Ok(ContentBuilder::text("signature"))
    }
}
