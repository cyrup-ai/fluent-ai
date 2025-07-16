# Cyrup Sugars Template

A cargo-generate template for projects using cyrup-sugars.

## Usage

```bash
# Generate a new project
cargo generate --git https://github.com/cyrup-ai/cyrup-sugars templates/

# Or locally
cargo generate --path path/to/sugars/templates
```

## Template Options

- **Project name**: Your project name
- **Features**: Cyrup sugars features to include (`tokio-async`, `std-async`, `crossbeam-async`, `hashbrown-json`, `all`)
- **Include examples**: Whether to include demonstration modules
- **Author**: Your name
- **Description**: Project description

## Generated Structure

With examples included:
```
src/
├── main.rs                    # Main entry point
├── one_or_many_demo.rs       # OneOrMany examples
├── zero_one_or_many_demo.rs  # ZeroOneOrMany examples  
├── async_task_demo.rs        # AsyncTask examples
├── json_syntax_demo.rs       # JSON syntax examples
├── serialization_demo.rs     # Serde integration examples
└── service_builder.rs        # Service builder pattern example
```

## Quick Start

1. Generate project with `cargo generate`
2. Run `cargo run` to see all examples
3. Delete demo modules you don't need
4. Keep the patterns you want to use
5. Add your application logic to `main.rs`

## Clean Template

Generate without examples for a minimal starting point:
- Choose `false` for "Include examples?"
- Get a clean `main.rs` ready for your code