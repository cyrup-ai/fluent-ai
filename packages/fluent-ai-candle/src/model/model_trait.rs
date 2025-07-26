//! Core Model trait for all AI models

use crate::types::CandleModelInfo as ModelInfo;

/// Core trait for all AI models with zero-allocation design
pub trait Model: Send + Sync + std::fmt::Debug + 'static {
    /// Get the model's comprehensive information with zero-allocation static access
    ///
    /// Returns a static reference to the model's complete information structure,
    /// providing access to capabilities, limitations, provider details, and
    /// configuration parameters. This method is the foundation for all other
    /// trait methods and enables zero-allocation introspection.
    ///
    /// # Returns
    ///
    /// `&'static ModelInfo` containing comprehensive model metadata:
    /// - **Identity**: Model name, provider, version information
    /// - **Capabilities**: Vision support, function calling, streaming abilities
    /// - **Limitations**: Token limits, context windows, rate limits
    /// - **Configuration**: Default parameters, supported formats
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time static reference
    /// - **Memory Usage**: Zero allocation - returns static reference
    /// - **Thread Safety**: Immutable static data, fully concurrent
    /// - **Lifetime**: Static lifetime ensures reference validity
    ///
    /// # Examples
    ///
    /// ## Basic Model Information Access
    /// ```rust
    /// use fluent_ai_candle::model::{Model, CandleModel};
    ///
    /// let model = CandleModel::load(&config)?;
    /// let info = model.info();
    ///
    /// println!("Model: {} by {}", info.name(), info.provider());
    /// println!("Max tokens: input={:?}, output={:?}", 
    ///          info.max_input_tokens, info.max_output_tokens);
    /// println!("Capabilities: vision={}, functions={}",
    ///          info.has_vision(), info.has_function_calling());
    /// ```
    ///
    /// ## Model Capability Detection
    /// ```rust
    /// fn analyze_model_capabilities(model: &dyn Model) {
    ///     let info = model.info();
    ///     
    ///     println!("Model Analysis: {}", info.name());
    ///     println!("Provider: {}", info.provider());
    ///     
    ///     if let Some(max_input) = info.max_input_tokens {
    ///         println!("Max input tokens: {}", max_input.get());
    ///     }
    ///     
    ///     if info.has_vision() {
    ///         println!("✓ Vision support enabled");
    ///     }
    ///     
    ///     if info.has_function_calling() {
    ///         println!("✓ Function calling supported");
    ///     }
    /// }
    /// ```
    ///
    /// ## Model Selection Logic
    /// ```rust
    /// fn select_optimal_model(candidates: &[&dyn Model], requirements: &Requirements) -> Option<&dyn Model> {
    ///     for &model in candidates {
    ///         let info = model.info();
    ///         
    ///         // Check token requirements
    ///         if let Some(required_input) = requirements.min_input_tokens {
    ///             if let Some(max_input) = info.max_input_tokens {
    ///                 if max_input.get() < required_input {
    ///                     continue; // Skip insufficient models
    ///                 }
    ///             }
    ///         }
    ///         
    ///         // Check capability requirements
    ///         if requirements.needs_vision && !info.has_vision() {
    ///             continue; // Skip models without vision
    ///         }
    ///         
    ///         if requirements.needs_functions && !info.has_function_calling() {
    ///             continue; // Skip models without function calling
    ///         }
    ///         
    ///         return Some(model); // Found suitable model
    ///     }
    ///     
    ///     None // No suitable model found
    /// }
    /// ```
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Static reference with no memory overhead
    /// - ✅ **Thread Safe**: Immutable static data supports concurrency
    /// - ✅ **Lifetime Safe**: Static lifetime prevents dangling references
    /// - ✅ **High Performance**: Direct static access with minimal overhead
    fn info(&self) -> &'static ModelInfo;

    /// Get the model's name with blazing-fast inline optimization and zero-allocation access
    ///
    /// Returns the model's canonical name as a static string reference, providing instant
    /// access to the model identifier for logging, selection, and display purposes. This
    /// method is inlined for maximum performance and leverages static string storage.
    ///
    /// # Returns
    ///
    /// `&'static str` containing the model's canonical name:
    /// - **Human-readable**: User-friendly model name for display
    /// - **Unique identifier**: Distinct name for model selection and comparison
    /// - **Static lifetime**: No allocation or memory management required
    /// - **UTF-8 safe**: Guaranteed valid Unicode string
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time with compiler inlining
    /// - **Memory Usage**: Zero allocation - static string reference
    /// - **Inlining**: Force-inlined for elimination of function call overhead
    /// - **Cache Friendly**: Static data resides in read-only memory segment
    ///
    /// # Examples
    ///
    /// ## Model Selection by Name
    /// ```rust
    /// use fluent_ai_candle::model::{Model, CandleModel};
    ///
    /// let model = CandleModel::load(&config)?;
    /// let model_name = model.name();
    ///
    /// println!("Using model: {}", model_name);
    /// 
    /// // Common model names
    /// match model_name {
    ///     "llama-2-7b" => println!("Meta's Llama 2 7B model"),
    ///     "mistral-7b" => println!("Mistral AI's 7B model"),
    ///     "codellama-7b" => println!("Code generation model"),
    ///     _ => println!("Custom or unknown model: {}", model_name),
    /// }
    /// ```
    ///
    /// ## Model Registry Integration
    /// ```rust
    /// use std::collections::HashMap;
    ///
    /// fn register_model(model: &dyn Model, registry: &mut HashMap<&'static str, String>) {
    ///     let name = model.name();
    ///     let description = format!("Model {} by {}", name, model.provider());
    ///     
    ///     registry.insert(name, description);
    ///     println!("Registered model: {}", name);
    /// }
    ///
    /// fn find_model_by_name(
    ///     models: &[&dyn Model], 
    ///     target_name: &str
    /// ) -> Option<&dyn Model> {
    ///     models.iter()
    ///         .find(|model| model.name() == target_name)
    ///         .copied()
    /// }
    /// ```
    ///
    /// ## Logging and Telemetry
    /// ```rust
    /// use log::info;
    ///
    /// fn log_model_usage(model: &dyn Model, operation: &str) {
    ///     info!(
    ///         "Model operation: {} using model '{}' from provider '{}'",
    ///         operation,
    ///         model.name(),
    ///         model.provider()
    ///     );
    /// }
    ///
    /// // Usage
    /// log_model_usage(&model, "text_generation");
    /// log_model_usage(&model, "completion_request");
    /// ```
    ///
    /// ## Configuration Validation
    /// ```rust
    /// fn validate_model_config(model: &dyn Model, expected_name: &str) -> Result<(), String> {
    ///     let actual_name = model.name();
    ///     
    ///     if actual_name != expected_name {
    ///         return Err(format!(
    ///             "Model name mismatch: expected '{}', got '{}'",
    ///             expected_name, actual_name
    ///         ));
    ///     }
    ///     
    ///     Ok(())
    /// }
    /// ```
    ///
    /// ## Performance Benchmarking
    /// ```rust
    /// fn benchmark_models(models: &[&dyn Model]) {
    ///     for model in models {
    ///         let name = model.name();
    ///         
    ///         let start = std::time::Instant::now();
    ///         for _ in 0..1000 {
    ///             let _ = model.name(); // Test inline performance
    ///         }
    ///         let duration = start.elapsed();
    ///         
    ///         println!("Model '{}': {} ns per name() call", 
    ///                 name, duration.as_nanos() / 1000);
    ///     }
    /// }
    /// ```
    ///
    /// # Common Model Names
    ///
    /// ## Language Models
    /// - `"llama-2-7b"`, `"llama-2-13b"` - Meta's Llama 2 series
    /// - `"mistral-7b-instruct"` - Mistral AI instruction-tuned models  
    /// - `"phi-2"` - Microsoft's small language model
    /// - `"gemma-7b"` - Google's Gemma model series
    ///
    /// ## Code Models
    /// - `"codellama-7b"` - Meta's code generation model
    /// - `"starcoder"` - BigCode's programming model
    /// - `"deepseek-coder"` - DeepSeek's code model
    ///
    /// ## Specialized Models
    /// - `"all-MiniLM-L6-v2"` - Sentence embeddings
    /// - `"whisper-base"` - OpenAI's speech recognition
    /// - `"stable-diffusion-v1-5"` - Image generation
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Static string reference with no heap usage
    /// - ✅ **Force Inlined**: Compiler eliminates function call overhead
    /// - ✅ **Thread Safe**: Immutable static data safe for concurrent access
    /// - ✅ **Cache Efficient**: Static strings in read-only memory segment
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.info().name()
    }

    /// Get the model's provider name with blazing-fast inline optimization and zero-allocation access
    ///
    /// Returns the name of the organization or company that created and maintains this model,
    /// providing essential metadata for attribution, licensing, and ecosystem integration.
    /// This method leverages static string storage and compiler inlining for maximum performance.
    ///
    /// # Returns
    ///
    /// `&'static str` containing the model's provider identifier:
    /// - **Organization name**: Company or research group that created the model
    /// - **Attribution ready**: Proper names for citations and acknowledgments
    /// - **Static lifetime**: No memory management or allocation overhead
    /// - **Standardized format**: Consistent naming across provider ecosystem
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time with aggressive inlining
    /// - **Memory Usage**: Zero allocation - static string reference
    /// - **Inlining**: Force-inlined for complete overhead elimination
    /// - **Cache Locality**: Provider data co-located with model metadata
    ///
    /// # Examples
    ///
    /// ## Provider-Based Model Selection
    /// ```rust
    /// use fluent_ai_candle::model::{Model, CandleModel};
    ///
    /// let model = CandleModel::load(&config)?;
    /// let provider = model.provider();
    ///
    /// println!("Model provider: {}", provider);
    ///
    /// // Common provider-based logic
    /// match provider {
    ///     "Meta" => {
    ///         println!("Using Meta's Llama model family");
    ///         // Apply Meta-specific optimizations
    ///     }
    ///     "Mistral AI" => {
    ///         println!("Using Mistral AI model");
    ///         // Apply Mistral-specific configurations
    ///     }
    ///     "OpenAI" => {
    ///         println!("Using OpenAI model");
    ///         // Handle OpenAI API compatibility
    ///     }
    ///     _ => {
    ///         println!("Using model from provider: {}", provider);
    ///         // Generic handling for other providers
    ///     }
    /// }
    /// ```
    ///
    /// ## License and Attribution Management
    /// ```rust
    /// fn generate_attribution(model: &dyn Model) -> String {
    ///     format!(
    ///         "This content was generated using '{}' by {} ({})",
    ///         model.name(),
    ///         model.provider(),
    ///         "Licensed under applicable terms"
    ///     )
    /// }
    ///
    /// fn check_commercial_usage(model: &dyn Model) -> bool {
    ///     match model.provider() {
    ///         "Meta" => {
    ///             // Check Meta's commercial license terms
    ///             true // Llama 2 allows commercial use
    ///         }
    ///         "Mistral AI" => {
    ///             // Mistral models generally allow commercial use
    ///             true
    ///         }
    ///         "Hugging Face" => {
    ///             // Depends on specific model license
    ///             false // Requires individual verification
    ///         }
    ///         _ => {
    ///             // Conservative approach for unknown providers
    ///             false
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Provider-Specific Optimizations
    /// ```rust
    /// fn apply_provider_optimizations(model: &dyn Model, config: &mut GenerationConfig) {
    ///     match model.provider() {
    ///         "Meta" => {
    ///             // Llama models work well with specific temperature ranges
    ///             if config.temperature > 1.2 {
    ///                 config.temperature = 1.0;
    ///                 println!("Adjusted temperature for Meta model");
    ///             }
    ///         }
    ///         "Mistral AI" => {
    ///             // Mistral models benefit from specific top-p settings
    ///             config.top_p = 0.9;
    ///             println!("Applied Mistral AI optimizations"); 
    ///         }
    ///         "Microsoft" => {
    ///             // Phi models are optimized for lower resource usage
    ///             config.max_tokens = config.max_tokens.min(2048);
    ///             println!("Applied Microsoft Phi optimizations");
    ///         }
    ///         _ => {
    ///             // Use conservative defaults
    ///             println!("Using generic optimizations for {}", model.provider());
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Multi-Provider Model Management
    /// ```rust
    /// use std::collections::HashMap;
    ///
    /// struct ModelRegistry {
    ///     models_by_provider: HashMap<&'static str, Vec<Box<dyn Model>>>,
    /// }
    ///
    /// impl ModelRegistry {
    ///     fn register_model(&mut self, model: Box<dyn Model>) {
    ///         let provider = model.provider();
    ///         
    ///         self.models_by_provider
    ///             .entry(provider)
    ///             .or_insert_with(Vec::new)
    ///             .push(model);
    ///             
    ///         println!("Registered model from provider: {}", provider);
    ///     }
    ///     
    ///     fn get_models_by_provider(&self, provider: &str) -> Option<&[Box<dyn Model>]> {
    ///         self.models_by_provider.get(provider).map(|v| v.as_slice())
    ///     }
    ///     
    ///     fn list_providers(&self) -> Vec<&'static str> {
    ///         self.models_by_provider.keys().copied().collect()
    ///     }
    /// }
    /// ```
    ///
    /// ## Provider Statistics and Analytics
    /// ```rust
    /// fn analyze_provider_usage(models: &[&dyn Model]) {
    ///     let mut provider_counts = HashMap::new();
    ///     
    ///     for model in models {
    ///         let provider = model.provider();
    ///         *provider_counts.entry(provider).or_insert(0) += 1;
    ///     }
    ///     
    ///     println!("Provider Usage Statistics:");
    ///     for (provider, count) in provider_counts {
    ///         let percentage = (count as f64 / models.len() as f64) * 100.0;
    ///         println!("  {}: {} models ({:.1}%)", provider, count, percentage);
    ///     }
    /// }
    /// ```
    ///
    /// ## Quality Assurance by Provider
    /// ```rust
    /// fn validate_provider_quality(model: &dyn Model) -> Result<(), String> {
    ///     let provider = model.provider();
    ///     
    ///     match provider {
    ///         "Meta" | "Mistral AI" | "Microsoft" | "Google" => {
    ///             // Trusted providers with established quality standards
    ///             Ok(())
    ///         }
    ///         "OpenAI" => {
    ///             // OpenAI models require API access validation
    ///             Ok(())
    ///         }
    ///         "Hugging Face" => {
    ///             // Community models require individual validation
    ///             println!("Warning: Community model from Hugging Face - validate independently");
    ///             Ok(())
    ///         }
    ///         _ => {
    ///             Err(format!("Unknown provider '{}' - manual validation required", provider))
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Common Provider Names
    ///
    /// ## Technology Companies
    /// - `"Meta"` - Meta AI (formerly Facebook AI)
    /// - `"Microsoft"` - Microsoft Research and Azure AI
    /// - `"Google"` - Google Research and DeepMind
    /// - `"OpenAI"` - OpenAI GPT and related models
    /// - `"Anthropic"` - Claude and constitutional AI models
    ///
    /// ## AI Research Organizations
    /// - `"Mistral AI"` - French AI company specializing in LLMs
    /// - `"Cohere"` - Enterprise-focused language models
    /// - `"AI21 Labs"` - Jurassic series language models
    /// - `"Stability AI"` - Stable Diffusion and related models
    ///
    /// ## Open Source Communities
    /// - `"Hugging Face"` - Community-contributed models
    /// - `"EleutherAI"` - Open source AI research collective
    /// - `"BigCode"` - Code-focused model development
    /// - `"Together"` - Collaborative AI model training
    ///
    /// # Provider Ecosystems
    ///
    /// ## Enterprise-Grade Providers
    /// - Established quality assurance processes
    /// - Professional support and documentation
    /// - Clear licensing and commercial usage terms
    /// - Regular security updates and patches
    ///
    /// ## Research Providers
    /// - Cutting-edge model architectures
    /// - Academic collaboration opportunities
    /// - Open research and reproducibility focus
    /// - Experimental features and capabilities
    ///
    /// ## Community Providers
    /// - Diverse model variations and fine-tunes
    /// - Rapid iteration and experimentation
    /// - Domain-specific specializations
    /// - Variable quality and support levels
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Static string reference with no memory overhead
    /// - ✅ **Force Inlined**: Compiler eliminates function call completely
    /// - ✅ **Thread Safe**: Immutable provider data safe for concurrent access
    /// - ✅ **Attribution Ready**: Proper provider names for legal compliance
    #[inline(always)]
    fn provider(&self) -> &'static str {
        self.info().provider()
    }

    /// Get the model's maximum input tokens with zero-allocation access and blazing-fast inline optimization
    ///
    /// Returns the maximum number of tokens this model can process in a single input context,
    /// providing essential information for prompt sizing, chunking strategies, and context
    /// window management. Uses zero-allocation Option mapping with compiler optimization.
    ///
    /// # Returns
    ///
    /// `Option<u32>` containing the input token limit:
    /// - `Some(limit)` - Model has a defined maximum input token count
    /// - `None` - Model has no explicit input token limit (effectively unlimited)
    ///
    /// Token counts represent the model's vocabulary units, not characters or words.
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time with aggressive inlining
    /// - **Memory Usage**: Zero allocation - Option operations optimized away
    /// - **Inlining**: Force-inlined for complete function call elimination
    /// - **Branch Prediction**: Option mapping optimized by compiler
    ///
    /// # Examples
    ///
    /// ## Context Window Management
    /// ```rust
    /// use fluent_ai_candle::model::{Model, CandleModel};
    ///
    /// let model = CandleModel::load(&config)?;
    ///
    /// match model.max_input_tokens() {
    ///     Some(limit) => {
    ///         println!("Model supports up to {} input tokens", limit);
    ///         
    ///         // Common context window sizes
    ///         match limit {
    ///             2048 => println!("Standard 2K context window"),
    ///             4096 => println!("Extended 4K context window"),
    ///             8192 => println!("Large 8K context window"),
    ///             16384 => println!("Very large 16K context window"),
    ///             32768 => println!("Huge 32K context window"),
    ///             _ => println!("Custom context window: {} tokens", limit),
    ///         }
    ///     }
    ///     None => {
    ///         println!("Model has unlimited input token capacity");
    ///     }
    /// }
    /// ```
    ///
    /// ## Prompt Chunking Strategy
    /// ```rust
    /// fn chunk_prompt_for_model(model: &dyn Model, prompt: &str, tokenizer: &Tokenizer) -> Vec<String> {
    ///     let tokens = tokenizer.encode(prompt);
    ///     
    ///     match model.max_input_tokens() {
    ///         Some(max_tokens) => {
    ///             let chunk_size = (max_tokens as usize).saturating_sub(100); // Reserve for special tokens
    ///             
    ///             tokens.chunks(chunk_size)
    ///                 .map(|chunk| tokenizer.decode(chunk))
    ///                 .collect()
    ///         }
    ///         None => {
    ///             // No chunking needed for unlimited models
    ///             vec![prompt.to_string()]
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Model Selection by Context Requirements
    /// ```rust
    /// fn select_model_for_context(
    ///     models: &[&dyn Model], 
    ///     required_tokens: u32
    /// ) -> Option<&dyn Model> {
    ///     for &model in models {
    ///         match model.max_input_tokens() {
    ///             Some(limit) if limit >= required_tokens => {
    ///                 println!("Selected model '{}' with {} token capacity", 
    ///                         model.name(), limit);
    ///                 return Some(model);
    ///             }
    ///             None => {
    ///                 // Unlimited models always satisfy requirements
    ///                 println!("Selected unlimited model '{}'", model.name());
    ///                 return Some(model);
    ///             }
    ///             Some(limit) => {
    ///                 println!("Model '{}' insufficient: {} < {} required", 
    ///                         model.name(), limit, required_tokens);
    ///             }
    ///         }
    ///     }
    ///     
    ///     None // No suitable model found
    /// }
    /// ```
    ///
    /// ## Dynamic Buffer Management
    /// ```rust
    /// struct ContextBuffer {
    ///     tokens: Vec<u32>,
    ///     max_capacity: Option<usize>,
    /// }
    ///
    /// impl ContextBuffer {
    ///     fn new_for_model(model: &dyn Model) -> Self {
    ///         let max_capacity = model.max_input_tokens().map(|n| n as usize);
    ///         
    ///         Self {
    ///             tokens: Vec::new(),
    ///             max_capacity,
    ///         }
    ///     }
    ///     
    ///     fn can_add_tokens(&self, count: usize) -> bool {
    ///         match self.max_capacity {
    ///             Some(limit) => self.tokens.len() + count <= limit,
    ///             None => true, // Unlimited capacity
    ///         }
    ///     }
    ///     
    ///     fn add_tokens(&mut self, new_tokens: &[u32]) -> Result<(), String> {
    ///         if !self.can_add_tokens(new_tokens.len()) {
    ///             return Err(format!(
    ///                 "Cannot add {} tokens: would exceed limit of {:?}",
    ///                 new_tokens.len(),
    ///                 self.max_capacity
    ///             ));
    ///         }
    ///         
    ///         self.tokens.extend_from_slice(new_tokens);
    ///         Ok(())
    ///     }
    /// }
    /// ```
    ///
    /// ## Memory Pre-allocation
    /// ```rust
    /// fn create_optimized_buffer(model: &dyn Model) -> Vec<u32> {
    ///     match model.max_input_tokens() {
    ///         Some(limit) => {
    ///             // Pre-allocate with known capacity
    ///             let capacity = limit as usize;
    ///             println!("Pre-allocating buffer for {} tokens", capacity);
    ///             Vec::with_capacity(capacity)
    ///         }
    ///         None => {
    ///             // Use default capacity for unlimited models
    ///             println!("Using default buffer capacity for unlimited model");
    ///             Vec::with_capacity(4096) // Reasonable default
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Performance Analysis
    /// ```rust
    /// fn analyze_context_efficiency(models: &[&dyn Model]) {
    ///     println!("Model Context Window Analysis:");
    ///     
    ///     for model in models {
    ///         let context_info = match model.max_input_tokens() {
    ///             Some(tokens) => {
    ///                 let kb_estimate = (tokens * 4) / 1024; // Rough memory estimate
    ///                 format!("{} tokens (~{} KB)", tokens, kb_estimate)
    ///             }
    ///             None => "Unlimited".to_string(),
    ///         };
    ///         
    ///         println!("  {}: {}", model.name(), context_info);
    ///     }
    /// }
    /// ```
    ///
    /// # Common Token Limits
    ///
    /// ## Traditional Models
    /// - **2,048 tokens**: Early transformer models (GPT-2, BERT base)
    /// - **4,096 tokens**: Standard large models (GPT-3.5, Llama 1)
    /// - **8,192 tokens**: Extended context models (GPT-4, Llama 2)
    ///
    /// ## Long Context Models  
    /// - **16,384 tokens**: Large context models (GPT-4 Turbo)
    /// - **32,768 tokens**: Very long context (Claude 2)
    /// - **100,000+ tokens**: Specialized long-context models
    ///
    /// ## Specialized Models
    /// - **None (unlimited)**: Streaming or chunk-based models
    /// - **Variable**: Dynamic context models that adapt to memory
    ///
    /// # Use Cases
    ///
    /// ## Document Processing
    /// - Determine if entire documents fit in context
    /// - Plan chunking strategies for large texts
    /// - Optimize memory allocation for document analysis
    ///
    /// ## Conversation Management
    /// - Track conversation token count
    /// - Implement context sliding windows
    /// - Manage multi-turn dialog history
    ///
    /// ## Code Analysis
    /// - Determine maximum file size for code review
    /// - Plan repository analysis strategies
    /// - Optimize code completion context
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Option mapping optimized away by compiler
    /// - ✅ **Force Inlined**: Complete function call elimination
    /// - ✅ **Thread Safe**: Atomic access to immutable model metadata
    /// - ✅ **Branch Optimized**: Option patterns optimized for common cases
    #[inline(always)]
    fn max_input_tokens(&self) -> Option<u32> {
        self.info().max_input_tokens.map(|n| n.get())
    }

    /// Get the model's maximum output tokens with zero-allocation access and blazing-fast inline optimization
    ///
    /// Returns the maximum number of tokens this model can generate in a single response,
    /// providing critical information for generation planning, resource allocation, and
    /// response length management. Uses zero-allocation Option mapping with aggressive optimization.
    ///
    /// # Returns
    ///
    /// `Option<u32>` containing the output token limit:
    /// - `Some(limit)` - Model has a defined maximum output token count
    /// - `None` - Model has no explicit output token limit (generation controlled by other factors)
    ///
    /// Output tokens represent the model's vocabulary units generated as response text.
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time with aggressive inlining
    /// - **Memory Usage**: Zero allocation - Option operations eliminated
    /// - **Inlining**: Force-inlined for complete overhead elimination
    /// - **Optimization**: Compiler optimizes Option mapping to direct field access
    ///
    /// # Examples
    ///
    /// ## Generation Length Planning
    /// ```rust
    /// use fluent_ai_candle::model::{Model, CandleModel};
    ///
    /// let model = CandleModel::load(&config)?;
    ///
    /// match model.max_output_tokens() {
    ///     Some(limit) => {
    ///         println!("Model can generate up to {} output tokens", limit);
    ///         
    ///         // Plan generation strategy based on limits
    ///         if limit < 512 {
    ///             println!("Short-form generation model (< 512 tokens)");
    ///         } else if limit < 2048 {
    ///             println!("Medium-form generation model (512-2K tokens)");
    ///         } else if limit < 8192 {
    ///             println!("Long-form generation model (2K-8K tokens)");
    ///         } else {
    ///             println!("Extended generation model ({}+ tokens)", limit);
    ///         }
    ///     }
    ///     None => {
    ///         println!("Model has unlimited output generation capacity");
    ///     }
    /// }
    /// ```
    ///
    /// ## Response Size Validation
    /// ```rust
    /// fn validate_generation_request(
    ///     model: &dyn Model, 
    ///     requested_tokens: u32
    /// ) -> Result<u32, String> {
    ///     match model.max_output_tokens() {
    ///         Some(limit) => {
    ///             if requested_tokens <= limit {
    ///                 Ok(requested_tokens)
    ///             } else {
    ///                 println!("Requested {} tokens exceeds model limit of {}", 
    ///                         requested_tokens, limit);
    ///                 Ok(limit) // Cap at model maximum
    ///             }
    ///         }
    ///         None => {
    ///             // No model-imposed limit
    ///             Ok(requested_tokens)
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Progressive Generation Planning
    /// ```rust
    /// fn plan_progressive_generation(
    ///     model: &dyn Model, 
    ///     total_needed: u32
    /// ) -> Vec<u32> {
    ///     match model.max_output_tokens() {
    ///         Some(chunk_size) => {
    ///             // Break into chunks that fit model limits
    ///             let mut chunks = Vec::new();
    ///             let mut remaining = total_needed;
    ///             
    ///             while remaining > 0 {
    ///                 let this_chunk = remaining.min(chunk_size);
    ///                 chunks.push(this_chunk);
    ///                 remaining -= this_chunk;
    ///             }
    ///             
    ///             println!("Planning {} chunks for {} total tokens", 
    ///                     chunks.len(), total_needed);
    ///             chunks
    ///         }
    ///         None => {
    ///             // Generate everything in one pass
    ///             vec![total_needed]
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Memory Allocation Optimization
    /// ```rust
    /// struct GenerationBuffer {
    ///     tokens: Vec<u32>,
    ///     max_size: Option<usize>,
    /// }
    ///
    /// impl GenerationBuffer {
    ///     fn new_for_model(model: &dyn Model) -> Self {
    ///         let max_size = model.max_output_tokens().map(|n| n as usize);
    ///         
    ///         let initial_capacity = match max_size {
    ///             Some(limit) => limit,
    ///             None => 4096, // Reasonable default for unlimited models
    ///         };
    ///         
    ///         Self {
    ///             tokens: Vec::with_capacity(initial_capacity),
    ///             max_size,
    ///         }
    ///     }
    ///     
    ///     fn can_generate_more(&self) -> bool {
    ///         match self.max_size {
    ///             Some(limit) => self.tokens.len() < limit,
    ///             None => true, // Always can generate more
    ///         }
    ///     }
    ///     
    ///     fn remaining_capacity(&self) -> Option<usize> {
    ///         self.max_size.map(|limit| limit.saturating_sub(self.tokens.len()))
    ///     }
    /// }
    /// ```
    ///
    /// ## Streaming Generation Control
    /// ```rust
    /// async fn controlled_streaming_generation(
    ///     model: &dyn Model,
    ///     prompt: &str
    /// ) -> Result<String, Box<dyn std::error::Error>> {
    ///     let mut generated_text = String::new();
    ///     let mut token_count = 0u32;
    ///     
    ///     let mut stream = model.generate_stream(prompt).await?;
    ///     
    ///     while let Some(chunk) = stream.next().await {
    ///         let chunk_tokens = chunk.token_count();
    ///         
    ///         // Check if we're approaching the limit
    ///         if let Some(limit) = model.max_output_tokens() {
    ///             if token_count + chunk_tokens > limit {
    ///                 println!("Approaching output limit, stopping generation");
    ///                 break;
    ///             }
    ///         }
    ///         
    ///         generated_text.push_str(&chunk.text);
    ///         token_count += chunk_tokens;
    ///         
    ///         // Progress reporting
    ///         if let Some(limit) = model.max_output_tokens() {
    ///             let percentage = (token_count as f64 / limit as f64) * 100.0;
    ///             println!("Generation progress: {:.1}% ({}/{})", 
    ///                     percentage, token_count, limit);
    ///         }
    ///     }
    ///     
    ///     Ok(generated_text)
    /// }
    /// ```
    ///
    /// ## Resource Planning
    /// ```rust
    /// fn estimate_generation_resources(model: &dyn Model) -> GenerationEstimate {
    ///     let max_tokens = model.max_output_tokens();
    ///     
    ///     GenerationEstimate {
    ///         max_response_tokens: max_tokens,
    ///         estimated_memory_mb: max_tokens.map(|n| (n * 4) / 1024 / 1024), // Rough estimate
    ///         generation_time_estimate: max_tokens.map(|n| {
    ///             // Assume 20 tokens/second average
    ///             std::time::Duration::from_millis((n as u64 * 50))
    ///         }),
    ///         parallel_capacity: match max_tokens {
    ///             Some(limit) if limit < 1024 => 8,  // Many short generations
    ///             Some(limit) if limit < 4096 => 4,  // Medium parallelism
    ///             Some(_) => 2,                       // Limited for long generations
    ///             None => 1,                          // Conservative for unlimited
    ///         },
    ///     }
    /// }
    ///
    /// struct GenerationEstimate {
    ///     max_response_tokens: Option<u32>,
    ///     estimated_memory_mb: Option<u32>,
    ///     generation_time_estimate: Option<std::time::Duration>,
    ///     parallel_capacity: usize,
    /// }
    /// ```
    ///
    /// # Common Output Limits
    ///
    /// ## Conversational Models
    /// - **512 tokens**: Short response models (chat assistants)
    /// - **1,024 tokens**: Standard conversation models
    /// - **2,048 tokens**: Extended conversation models
    ///
    /// ## Content Generation Models
    /// - **4,096 tokens**: Article and blog post generation
    /// - **8,192 tokens**: Long-form content generation
    /// - **16,384+ tokens**: Book chapter and document generation
    ///
    /// ## Specialized Models
    /// - **128-256 tokens**: Code completion models
    /// - **None (unlimited)**: Streaming or iterative generation models
    /// - **Variable**: Dynamic models that adapt based on context
    ///
    /// # Generation Strategies
    ///
    /// ## Fixed Length Models
    /// - Pre-allocate buffers based on known limits
    /// - Implement hard cutoffs at token boundaries
    /// - Plan multi-step generation for longer content
    ///
    /// ## Unlimited Models
    /// - Use dynamic buffer growth strategies
    /// - Implement alternative stopping criteria (time, cost, relevance)
    /// - Monitor resource usage during generation
    ///
    /// ## Adaptive Models
    /// - Query limits dynamically during generation
    /// - Implement flexible stopping and continuation logic
    /// - Balance between quality and resource constraints
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Option mapping completely optimized away
    /// - ✅ **Force Inlined**: Function call overhead eliminated
    /// - ✅ **Thread Safe**: Immutable model metadata safe for concurrent access
    /// - ✅ **Resource Aware**: Enables optimal memory and time planning
    #[inline(always)]
    fn max_output_tokens(&self) -> Option<u32> {
        self.info().max_output_tokens.map(|n| n.get())
    }

    /// Check if the model supports vision capabilities with blazing-fast inline optimization and zero-allocation access
    ///
    /// Returns whether this model can process and understand visual content such as images,
    /// videos, charts, and diagrams. This capability detection enables multimodal applications
    /// and determines appropriate input preprocessing strategies with maximum performance.
    ///
    /// # Returns
    ///
    /// `bool` indicating vision support status:
    /// - `true` - Model can process visual content alongside text
    /// - `false` - Model is text-only and cannot understand visual inputs
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time with aggressive inlining
    /// - **Memory Usage**: Zero allocation - direct boolean field access
    /// - **Inlining**: Force-inlined for complete function call elimination
    /// - **Branch Prediction**: Optimized for common vision/text-only patterns
    ///
    /// # Examples
    ///
    /// ## Multimodal Application Routing
    /// ```rust
    /// use fluent_ai_candle::model::{Model, CandleModel};
    ///
    /// let model = CandleModel::load(&config)?;
    ///
    /// if model.supports_vision() {
    ///     println!("✅ Vision-enabled model: can process images and text");
    ///     
    ///     // Enable multimodal features
    ///     enable_image_upload_ui();
    ///     enable_chart_analysis();
    ///     enable_document_ocr();
    ///     
    /// } else {
    ///     println!("📝 Text-only model: optimized for language tasks");
    ///     
    ///     // Optimize for text-only workflows
    ///     optimize_text_processing();
    ///     disable_visual_features();
    /// }
    /// ```
    ///
    /// ## Content Type Validation
    /// ```rust
    /// #[derive(Debug)]
    /// enum ContentType {
    ///     TextOnly(String),
    ///     TextWithImage { text: String, image_path: String },
    ///     ImageOnly(String),
    /// }
    ///
    /// fn validate_content_for_model(
    ///     model: &dyn Model, 
    ///     content: &ContentType
    /// ) -> Result<(), String> {
    ///     match content {
    ///         ContentType::TextOnly(_) => {
    ///             // All models support text
    ///             Ok(())
    ///         }
    ///         ContentType::TextWithImage { .. } | ContentType::ImageOnly(_) => {
    ///             if model.supports_vision() {
    ///                 Ok(())
    ///             } else {
    ///                 Err(format!(
    ///                     "Model '{}' does not support vision - cannot process visual content",
    ///                     model.name()
    ///                 ))
    ///             }
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Model Selection for Visual Tasks
    /// ```rust
    /// fn select_vision_model(models: &[&dyn Model]) -> Option<&dyn Model> {
    ///     for &model in models {
    ///         if model.supports_vision() {
    ///             println!("Selected vision-capable model: {} by {}", 
    ///                     model.name(), model.provider());
    ///             return Some(model);
    ///         }
    ///     }
    ///     
    ///     println!("Warning: No vision-capable models available");
    ///     None
    /// }
    ///
    /// fn get_best_model_for_task(models: &[&dyn Model], needs_vision: bool) -> Option<&dyn Model> {
    ///     if needs_vision {
    ///         // Must have vision capability
    ///         models.iter().find(|model| model.supports_vision()).copied()
    ///     } else {
    ///         // Prefer text-only models for better performance
    ///         models.iter()
    ///             .find(|model| !model.supports_vision())
    ///             .or_else(|| models.first()) // Fallback to any model
    ///             .copied()
    ///     }
    /// }
    /// ```
    ///
    /// ## Dynamic UI Adaptation
    /// ```rust
    /// struct ModelCapabilities {
    ///     supports_vision: bool,
    ///     supports_functions: bool,
    ///     max_input_tokens: Option<u32>,
    /// }
    ///
    /// fn get_model_capabilities(model: &dyn Model) -> ModelCapabilities {
    ///     ModelCapabilities {
    ///         supports_vision: model.supports_vision(),
    ///         supports_functions: model.supports_function_calling(),
    ///         max_input_tokens: model.max_input_tokens(),
    ///     }
    /// }
    ///
    /// fn configure_ui_for_model(capabilities: &ModelCapabilities) {
    ///     if capabilities.supports_vision {
    ///         show_image_upload_button();
    ///         show_camera_capture_option();
    ///         show_screen_analysis_tool();
    ///         
    ///         println!("🖼️  Visual analysis enabled");
    ///     } else {
    ///         hide_visual_features();
    ///         
    ///         println!("📝 Text-only mode optimized");
    ///     }
    /// }
    /// ```
    ///
    /// ## Performance Optimization
    /// ```rust
    /// fn optimize_for_model_type(model: &dyn Model) -> ModelOptimization {
    ///     if model.supports_vision() {
    ///         ModelOptimization {
    ///             enable_image_preprocessing: true,
    ///             image_resize_target: Some((224, 224)), // Common vision model input
    ///             batch_size: 4, // Lower batch size for vision models
    ///             memory_buffer_mb: 512, // Extra memory for image processing
    ///             preprocessing_threads: 2,
    ///         }
    ///     } else {
    ///         ModelOptimization {
    ///             enable_image_preprocessing: false,
    ///             image_resize_target: None,
    ///             batch_size: 16, // Higher batch size for text-only
    ///             memory_buffer_mb: 128, // Less memory needed
    ///             preprocessing_threads: 1,
    ///         }
    ///     }
    /// }
    ///
    /// struct ModelOptimization {
    ///     enable_image_preprocessing: bool,
    ///     image_resize_target: Option<(u32, u32)>,
    ///     batch_size: usize,
    ///     memory_buffer_mb: usize,
    ///     preprocessing_threads: usize,
    /// }
    /// ```
    ///
    /// ## Content Analysis Pipeline
    /// ```rust
    /// async fn analyze_content(
    ///     model: &dyn Model, 
    ///     text: &str, 
    ///     image_data: Option<&[u8]>
    /// ) -> Result<String, Box<dyn std::error::Error>> {
    ///     match (model.supports_vision(), image_data) {
    ///         (true, Some(image)) => {
    ///             // Multimodal analysis
    ///             println!("🔍 Analyzing text and image with vision model");
    ///             
    ///             let analysis = model.analyze_multimodal(text, image).await?;
    ///             Ok(format!("Multimodal Analysis: {}", analysis))
    ///         }
    ///         (true, None) => {
    ///             // Vision model with text-only input
    ///             println!("📝 Text analysis using vision-capable model");
    ///             
    ///             let analysis = model.analyze_text(text).await?;
    ///             Ok(format!("Text Analysis: {}", analysis))
    ///         }
    ///         (false, Some(_)) => {
    ///             // Cannot process image with text-only model
    ///             Err("Model does not support vision - cannot analyze image".into())
    ///         }
    ///         (false, None) => {
    ///             // Text-only analysis
    ///             println!("📖 Text analysis with optimized text-only model");
    ///             
    ///             let analysis = model.analyze_text(text).await?;
    ///             Ok(format!("Text Analysis: {}", analysis))
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// ## Model Registry and Statistics
    /// ```rust
    /// fn analyze_model_capabilities(models: &[&dyn Model]) {
    ///     let vision_models: Vec<_> = models.iter()
    ///         .filter(|model| model.supports_vision())
    ///         .collect();
    ///         
    ///     let text_only_models: Vec<_> = models.iter()
    ///         .filter(|model| !model.supports_vision())
    ///         .collect();
    ///     
    ///     println!("Model Capability Analysis:");
    ///     println!("  Vision-capable models: {} ({:.1}%)", 
    ///             vision_models.len(),
    ///             (vision_models.len() as f64 / models.len() as f64) * 100.0);
    ///             
    ///     println!("  Text-only models: {} ({:.1}%)",
    ///             text_only_models.len(),
    ///             (text_only_models.len() as f64 / models.len() as f64) * 100.0);
    ///     
    ///     println!("\\nVision-capable models:");
    ///     for model in &vision_models {
    ///         println!("  - {} by {}", model.name(), model.provider());
    ///     }
    /// }
    /// ```
    ///
    /// # Common Vision-Capable Models
    ///
    /// ## Multimodal Language Models
    /// - **GPT-4 Vision** - OpenAI's multimodal model
    /// - **Claude 3** - Anthropic's vision-enabled model
    /// - **Gemini Pro Vision** - Google's multimodal model
    /// - **LLaVA** - Open source vision-language model
    ///
    /// ## Specialized Vision Models
    /// - **CLIP** - Contrastive Language-Image Pre-training
    /// - **BLIP-2** - Bootstrapped vision-language pre-training
    /// - **Flamingo** - Few-shot learning with vision
    /// - **DALL-E** - Text-to-image generation (reverse vision)
    ///
    /// # Vision Use Cases
    ///
    /// ## Document Analysis
    /// - OCR and text extraction from images
    /// - Chart and graph interpretation
    /// - Form processing and data extraction
    /// - Scientific paper figure analysis
    ///
    /// ## Visual Content Understanding
    /// - Image captioning and description
    /// - Object detection and recognition
    /// - Scene understanding and context
    /// - Visual question answering
    ///
    /// ## Creative Applications
    /// - Image-based story generation
    /// - Visual art analysis and critique
    /// - Design feedback and suggestions
    /// - Meme and humor understanding
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Direct boolean field access with no memory overhead
    /// - ✅ **Force Inlined**: Function call completely eliminated by compiler
    /// - ✅ **Thread Safe**: Immutable capability data safe for concurrent access
    /// - ✅ **Branch Optimized**: Boolean operations optimized for predictable patterns
    #[inline(always)]
    fn supports_vision(&self) -> bool {
        self.info().has_vision()
    }

    /// Check if the model supports function calling capabilities with blazing-fast inline optimization and zero-allocation access
    ///
    /// Returns whether this model can invoke external functions and tools during generation,
    /// enabling agentic behaviors, API integrations, and structured data processing. This
    /// capability detection is essential for building AI agents and interactive applications.
    ///
    /// # Returns
    ///
    /// `bool` indicating function calling support status:
    /// - `true` - Model can call functions and use tools during generation
    /// - `false` - Model generates text only without external function capabilities
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) constant time with aggressive inlining
    /// - **Memory Usage**: Zero allocation - direct boolean field access
    /// - **Inlining**: Force-inlined for complete function call elimination
    /// - **Optimization**: Compiler optimizes to direct memory access
    ///
    /// # Examples
    ///
    /// ## Agent System Configuration
    /// ```rust
    /// use fluent_ai_candle::model::{Model, CandleModel};
    ///
    /// let model = CandleModel::load(&config)?;
    ///
    /// if model.supports_function_calling() {
    ///     println!("🤖 Agent-capable model: can use tools and functions");
    ///     
    ///     // Enable agent capabilities
    ///     register_tool_functions(&mut agent);
    ///     enable_api_integrations();
    ///     setup_function_schemas();
    ///     
    /// } else {
    ///     println!("💬 Conversational model: optimized for text generation");
    ///     
    ///     // Configure for text-only generation
    ///     optimize_text_generation();
    ///     disable_tool_integrations();
    /// }
    /// ```
    ///
    /// ## Function Registry Validation
    /// ```rust
    /// fn setup_agent_with_tools(model: &dyn Model) -> Result<Agent, String> {
    ///     if !model.supports_function_calling() {
    ///         return Err(format!(
    ///             "Model '{}' does not support function calling - cannot create agent",
    ///             model.name()
    ///         ));
    ///     }
    ///     
    ///     let mut agent = Agent::new(model);
    ///     
    ///     // Register available tools
    ///     agent.register_function("get_weather", get_weather_schema())?;
    ///     agent.register_function("send_email", send_email_schema())?;
    ///     agent.register_function("search_web", search_web_schema())?;
    ///     agent.register_function("calculate", calculate_schema())?;
    ///     
    ///     println!("✅ Agent configured with {} tools", agent.tool_count());
    ///     Ok(agent)
    /// }
    /// ```
    ///
    /// ## Model Selection for Agent Tasks
    /// ```rust
    /// fn select_agent_model(models: &[&dyn Model]) -> Option<&dyn Model> {
    ///     // Find models that support function calling
    ///     let function_capable: Vec<_> = models.iter()
    ///         .filter(|model| model.supports_function_calling())
    ///         .collect();
    ///     
    ///     if function_capable.is_empty() {
    ///         println!("❌ No function-calling capable models available");
    ///         return None;
    ///     }
    ///     
    ///     // Prefer models with larger context windows for complex tool usage
    ///     let best_model = function_capable.iter()
    ///         .max_by_key(|model| model.max_input_tokens().unwrap_or(0))
    ///         .copied();
    ///     
    ///     if let Some(model) = best_model {
    ///         println!("Selected agent model: {} by {} (context: {:?})", 
    ///                 model.name(), 
    ///                 model.provider(),
    ///                 model.max_input_tokens());
    ///     }
    ///     
    ///     best_model
    /// }
    /// ```
    ///
    /// ## Dynamic Tool Registration
    /// ```rust
    /// struct ToolRegistry {
    ///     available_tools: Vec<ToolDefinition>,
    ///     model_supports_functions: bool,
    /// }
    ///
    /// impl ToolRegistry {
    ///     fn new_for_model(model: &dyn Model) -> Self {
    ///         let supports_functions = model.supports_function_calling();
    ///         
    ///         Self {
    ///             available_tools: if supports_functions {
    ///                 load_all_tool_definitions()
    ///             } else {
    ///                 Vec::new() // No tools for non-function models
    ///             },
    ///             model_supports_functions: supports_functions,
    ///         }
    ///     }
    ///     
    ///     fn register_tool(&mut self, tool: ToolDefinition) -> Result<(), String> {
    ///         if !self.model_supports_functions {
    ///             return Err("Model does not support function calling".to_string());
    ///         }
    ///         
    ///         self.available_tools.push(tool);
    ///         Ok(())
    ///     }
    ///     
    ///     fn get_tools_for_context(&self, context: &str) -> Vec<&ToolDefinition> {
    ///         if !self.model_supports_functions {
    ///             return Vec::new();
    ///         }
    ///         
    ///         self.available_tools.iter()
    ///             .filter(|tool| tool.is_relevant_for_context(context))
    ///             .collect()
    ///     }
    /// }
    /// ```
    ///
    /// ## Conversation Flow Control
    /// ```rust
    /// async fn handle_user_message(
    ///     model: &dyn Model,
    ///     message: &str,
    ///     tools: &ToolRegistry
    /// ) -> Result<String, Box<dyn std::error::Error>> {
    ///     if model.supports_function_calling() {
    ///         // Enable function calling in the generation
    ///         let response = model.generate_with_functions(
    ///             message,
    ///             &tools.get_tools_for_context(message)
    ///         ).await?;
    ///         
    ///         // Check if model called any functions
    ///         if let Some(function_calls) = response.function_calls() {
    ///             println!("🔧 Model called {} functions", function_calls.len());
    ///             
    ///             // Execute function calls
    ///             let mut results = Vec::new();
    ///             for call in function_calls {
    ///                 let result = execute_function_call(call).await?;
    ///                 results.push(result);
    ///             }
    ///             
    ///             // Generate final response with function results
    ///             let final_response = model.generate_with_function_results(
    ///                 message,
    ///                 &results
    ///             ).await?;
    ///             
    ///             Ok(final_response.text())
    ///         } else {
    ///             // No functions called, return direct response
    ///             Ok(response.text())
    ///         }
    ///     } else {
    ///         // Simple text generation without function calling
    ///         let response = model.generate(message).await?;
    ///         Ok(response.text())
    ///     }
    /// }
    /// ```
    ///
    /// ## Performance Optimization
    /// ```rust
    /// fn optimize_for_function_support(model: &dyn Model) -> ProcessingConfig {
    ///     if model.supports_function_calling() {
    ///         ProcessingConfig {
    ///             enable_function_parsing: true,
    ///             context_buffer_size: 8192, // Larger buffer for tool context
    ///             max_function_calls_per_turn: 5,
    ///             function_timeout_ms: 30000,
    ///             tool_result_cache_size: 100,
    ///             parallel_function_execution: true,
    ///         }
    ///     } else {
    ///         ProcessingConfig {
    ///             enable_function_parsing: false,
    ///             context_buffer_size: 2048, // Smaller buffer for text-only
    ///             max_function_calls_per_turn: 0,
    ///             function_timeout_ms: 0,
    ///             tool_result_cache_size: 0,
    ///             parallel_function_execution: false,
    ///         }
    ///     }
    /// }
    ///
    /// struct ProcessingConfig {
    ///     enable_function_parsing: bool,
    ///     context_buffer_size: usize,
    ///     max_function_calls_per_turn: usize,
    ///     function_timeout_ms: u64,
    ///     tool_result_cache_size: usize,
    ///     parallel_function_execution: bool,
    /// }
    /// ```
    ///
    /// ## Agent Capability Analysis
    /// ```rust
    /// fn analyze_agent_potential(models: &[&dyn Model]) {
    ///     let function_models: Vec<_> = models.iter()
    ///         .filter(|model| model.supports_function_calling())
    ///         .collect();
    ///         
    ///     let text_only_models: Vec<_> = models.iter()
    ///         .filter(|model| !model.supports_function_calling())
    ///         .collect();
    ///     
    ///     println!("Agent Capability Analysis:");
    ///     println!("  Function-calling models: {} ({:.1}%)", 
    ///             function_models.len(),
    ///             (function_models.len() as f64 / models.len() as f64) * 100.0);
    ///             
    ///     println!("  Text-only models: {} ({:.1}%)",
    ///             text_only_models.len(),
    ///             (text_only_models.len() as f64 / models.len() as f64) * 100.0);
    ///     
    ///     println!("\\nAgent-capable models:");
    ///     for model in &function_models {
    ///         let context_info = match model.max_input_tokens() {
    ///             Some(tokens) => format!("{} token context", tokens),
    ///             None => "unlimited context".to_string(),
    ///         };
    ///         
    ///         println!("  - {} by {} ({})", 
    ///                 model.name(), 
    ///                 model.provider(),
    ///                 context_info);
    ///     }
    /// }
    /// ```
    ///
    /// # Common Function-Calling Models
    ///
    /// ## Commercial Models
    /// - **GPT-4** - OpenAI's function-calling capable model
    /// - **GPT-3.5-turbo** - OpenAI's efficient function model
    /// - **Claude 3** - Anthropic's tool-use capable model
    /// - **Gemini Pro** - Google's function-calling model
    ///
    /// ## Open Source Models
    /// - **Code Llama** - Meta's code-focused model with tool use
    /// - **Mistral 7B Instruct** - Function-calling capable Mistral model
    /// - **Yi-34B-Chat** - Large context function-calling model
    /// - **Qwen-72B-Chat** - Alibaba's function-enabled model
    ///
    /// # Function Calling Use Cases
    ///
    /// ## API Integration
    /// - Weather and news API calls
    /// - Database queries and updates
    /// - Email and messaging services
    /// - Cloud service management
    ///
    /// ## Data Processing
    /// - File operations and transformations
    /// - Mathematical calculations
    /// - Data analysis and visualization
    /// - Content generation and formatting
    ///
    /// ## Agent Behaviors
    /// - Multi-step task execution
    /// - Decision making with external data
    /// - Interactive problem solving
    /// - Autonomous workflow completion
    ///
    /// # Architecture Compliance
    ///
    /// - ✅ **Zero Allocation**: Direct boolean field access with no memory overhead
    /// - ✅ **Force Inlined**: Function call completely eliminated by compiler
    /// - ✅ **Thread Safe**: Immutable capability data safe for concurrent access
    /// - ✅ **Agent Ready**: Enables efficient agent and tool integration patterns
    #[inline(always)]
    fn supports_function_calling(&self) -> bool {
        self.info().has_function_calling()
    }
}
