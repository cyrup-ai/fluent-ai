use model_info::generated_models::AnthropicModel;

fn main() {
    println!("=== Anthropic Models ===");
    let anthropic = AnthropicModel::Claude4Sonnet;
    println!("Anthropic variant: {:?}", anthropic);
    
    println!("=== Testing complete ===");
}