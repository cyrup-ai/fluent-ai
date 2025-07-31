use model_info::generated_models::AnthropicModel;

fn main() {
    println!("=== OpenAI Models ===");
    // Let's just try to access some common variants that should exist based on build.rs
    
    println!("=== Anthropic Models ===");
    let anthropic = AnthropicModel::Claude35Sonnet20240620;
    println!("Anthropic variant: {:?}", anthropic);
    
    println!("=== Testing complete ===");
}