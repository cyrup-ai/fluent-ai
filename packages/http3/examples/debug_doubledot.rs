use fluent_ai_http3::json_path::JsonPathParser;

fn main() {
    println!("Testing $..*");
    match JsonPathParser::compile("$..*") {
        Ok(_expr) => println!("✅ Successfully compiled: $..*"),
        Err(e) => println!("❌ Failed to compile: $..*\n   Error: {:?}", e),
    }
}