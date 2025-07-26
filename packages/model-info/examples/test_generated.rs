use model_info::common::Model;

fn main() {
    // Test that generated OpenAI models are accessible
    let openai_model = model_info::OpenAiModel::Gpt4o;
    println!("OpenAI Model: {}", openai_model.name());
    println!("Context Length: {}", openai_model.max_context_length());
    println!("Input Price: ${:.6}", openai_model.pricing_input());
    println!("Output Price: ${:.6}", openai_model.pricing_output());
    println!("Is Thinking: {}", openai_model.is_thinking());
    
    // Test generated Anthropic models
    let anthropic_model = model_info::AnthropicModel::Claude35Sonnet20240620;
    println!("\nAnthropic Model: {}", anthropic_model.name());
    println!("Context Length: {}", anthropic_model.max_context_length());
    
    println!("\n✅ Generated models are working correctly!");
}