use model_info::common::Model;
use model_info::generated_models::{OpenAiModel, AnthropicModel};

fn main() {
    // Test that generated OpenAI models are accessible
    let openai_model = OpenAiModel::Gpt4o;
    println!("OpenAI Model: {}", openai_model.name());
    println!("Context Length: {}", openai_model.max_context_length());
    
    // Handle optional pricing properly
    if let Some(input_price) = openai_model.pricing_input() {
        println!("Input Price: ${:.6}", input_price);
    } else {
        println!("Input Price: Not available");
    }
    
    if let Some(output_price) = openai_model.pricing_output() {
        println!("Output Price: ${:.6}", output_price);
    } else {
        println!("Output Price: Not available");
    }
    
    println!("Supports Thinking: {}", openai_model.supports_thinking());
    
    // Test generated Anthropic models
    let anthropic_model = AnthropicModel::Claude35Sonnet20240620;
    println!("\nAnthropic Model: {}", anthropic_model.name());
    println!("Context Length: {}", anthropic_model.max_context_length());
    
    println!("\n✅ Generated models are working correctly!");
}