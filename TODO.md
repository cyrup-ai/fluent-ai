# TOOLREGISTRY TYPESTATE BUILDER IMPLEMENTATION

*CRITICAL: Modern ergonomic tool registration API with zero allocation and compile-time type safety*

## Task A: Core Schema and Type System
**File:** `./packages/provider/src/clients/anthropic/tools.rs`
**Lines:** 1-100 (insert at beginning of file)
**Priority:** CRITICAL
**Architecture:** Foundation types for typestate builder pattern and schema system

**Technical Details:**
- Lines 1-20: SchemaType enum with variants: Serde, JsonSchema, Inline
- Lines 21-40: Event handler type aliases for zero-allocation closure storage
- Lines 41-60: Core builder state marker types (NamedState, DescribedState, WithDepsState, WithSchemasState)
- Lines 61-80: Error types for tool registration and execution (ToolRegistrationError, ToolExecutionError)
- Lines 81-100: Foundational trait definitions for tool execution pipeline

**Implementation Specifications:**
```rust
#[derive(Debug, Clone, Copy)]
pub enum SchemaType {
    Serde,     // Auto-generate schema from serde Serialize/Deserialize types
    JsonSchema, // Manual JSON schema definition
    Inline,    // Inline parameter definitions
}

// Zero-allocation closure storage types
type InvocationHandler<D, Req, Res> = Box<dyn Fn(&Conversation, &Emitter, Req, &D) -> BoxFuture<'_, AnthropicResult<()>> + Send + Sync>;
type ErrorHandler<D> = Box<dyn Fn(&Conversation, &ChainControl, AnthropicError, &D) + Send + Sync>;
type ResultHandler<D, Res> = Box<dyn Fn(&Conversation, &ChainControl, Res, &D) -> Res + Send + Sync>;

// Typestate marker types for compile-time safety
pub struct NamedState;
pub struct DescribedState;
pub struct WithDepsState<D>(PhantomData<D>);
pub struct WithSchemasState<D, Req, Res>(PhantomData<(D, Req, Res)>);
```

**Constraints:** All types must be zero-allocation with static dispatch. No unwrap() or expect() calls. Follow elegant ergonomic design principles.

---

## Task B: Typestate Builder Chain Implementation
**File:** `./packages/provider/src/clients/anthropic/tools.rs`
**Lines:** 101-300
**Priority:** CRITICAL
**Architecture:** Typestate builder pattern with compile-time state transitions

**Technical Details:**
- Lines 101-140: ToolBuilder entry point with named() static method
- Lines 141-180: NamedToolBuilder with description() method transitioning to DescribedToolBuilder
- Lines 181-220: DescribedToolBuilder with with() method for dependency injection
- Lines 221-260: ToolBuilderWithDeps with request_schema() and result_schema() methods
- Lines 261-300: ToolBuilderWithSchemas with event handler registration methods

**Implementation Specifications:**
```rust
pub struct ToolBuilder;

impl ToolBuilder {
    pub fn named(name: &'static str) -> NamedToolBuilder<NamedState> {
        NamedToolBuilder {
            name,
            state: PhantomData,
        }
    }
}

pub struct NamedToolBuilder<S> {
    name: &'static str,
    state: PhantomData<S>,
}

impl NamedToolBuilder<NamedState> {
    pub fn description(self, desc: &'static str) -> DescribedToolBuilder<DescribedState> {
        DescribedToolBuilder {
            name: self.name,
            description: desc,
            state: PhantomData,
        }
    }
}

pub struct DescribedToolBuilder<S> {
    name: &'static str,
    description: &'static str,
    state: PhantomData<S>,
}

impl DescribedToolBuilder<DescribedState> {
    pub fn with<D>(self, dependency: D) -> ToolBuilderWithDeps<D, WithDepsState<D>> 
    where D: Send + Sync + 'static {
        ToolBuilderWithDeps {
            name: self.name,
            description: self.description,
            dependency,
            state: PhantomData,
        }
    }
}
```

**Constraints:** Each builder step must transition to next type preventing invalid states. Zero allocations during builder chain construction. All strings must be &'static str for zero allocation.

---

## Task C: Event System Infrastructure  
**File:** `./packages/provider/src/clients/anthropic/tools.rs`
**Lines:** 301-450
**Priority:** CRITICAL
**Architecture:** Event handling objects for conversation, streaming, and chain control

**Technical Details:**
- Lines 301-340: Conversation struct with message history, context access, and zero-allocation iteration
- Lines 341-380: Emitter struct for real-time streaming with zero-copy chunk emission
- Lines 381-420: ChainControl struct for error handling with stop_propagation() and retry() methods
- Lines 421-450: Event handler registration and storage with type-safe closures

**Implementation Specifications:**
```rust
pub struct Conversation {
    messages: &'static [Message],
    context: &'static ToolExecutionContext,
    last_message: &'static Message,
}

impl Conversation {
    #[inline(always)]
    pub fn last_message(&self) -> &Message {
        self.last_message
    }
    
    #[inline(always)]
    pub fn messages(&self) -> &[Message] {
        self.messages
    }
    
    #[inline(always)]
    pub fn context(&self) -> &ToolExecutionContext {
        self.context
    }
}

pub struct Emitter {
    sender: tokio::sync::mpsc::UnboundedSender<ToolOutput>,
}

impl Emitter {
    #[inline(always)]
    pub fn emit(&self, chunk: impl Into<ToolOutput>) -> AnthropicResult<()> {
        self.sender.send(chunk.into())
            .map_err(|_| AnthropicError::StreamError("Failed to emit chunk".into()))
    }
}

pub struct ChainControl {
    should_stop: AtomicBool,
    retry_count: AtomicU32,
}

impl ChainControl {
    #[inline(always)]
    pub fn stop_propagation(&self) {
        self.should_stop.store(true, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn retry(&self) -> bool {
        let current = self.retry_count.load(Ordering::Relaxed);
        if current < 3 {
            self.retry_count.store(current + 1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}
```

**Constraints:** All objects must use zero-allocation patterns. Streaming must be lock-free with atomic operations. No unwrap() or expect() calls in error handling.

---

## Task D: Zero-Allocation Storage Engine
**File:** `./packages/provider/src/clients/anthropic/tools.rs`
**Lines:** 451-600
**Priority:** CRITICAL  
**Architecture:** Efficient tool storage without Box allocations using arena pattern

**Technical Details:**
- Lines 451-490: Arena-based tool storage with static dispatch and zero allocations
- Lines 491-530: TypedTool struct for storing tools with full type information
- Lines 531-570: Tool lookup and retrieval with compile-time type safety
- Lines 571-600: Memory management and cleanup for arena storage

**Implementation Specifications:**
```rust
use std::collections::HashMap;
use std::any::{Any, TypeId};

pub struct TypedToolStorage {
    tools: HashMap<&'static str, Box<dyn Any + Send + Sync>>,
    schemas: HashMap<&'static str, (Value, Value)>, // (request_schema, result_schema)
    handlers: HashMap<&'static str, ToolHandlers>,
}

pub struct TypedTool<D, Req, Res> {
    name: &'static str,
    description: &'static str,
    dependency: D,
    request_schema: &'static Value,
    result_schema: &'static Value,
    handlers: ToolHandlers<D, Req, Res>,
}

pub struct ToolHandlers<D, Req, Res> {
    invocation: InvocationHandler<D, Req, Res>,
    error: Option<ErrorHandler<D>>,
    result: Option<ResultHandler<D, Res>>,
}

impl TypedToolStorage {
    #[inline(always)]
    pub fn register<D, Req, Res>(&mut self, tool: TypedTool<D, Req, Res>) -> AnthropicResult<()>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        self.tools.insert(tool.name, Box::new(tool));
        Ok(())
    }
    
    #[inline(always)]
    pub fn get_tool<D, Req, Res>(&self, name: &str) -> Option<&TypedTool<D, Req, Res>>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        self.tools.get(name)
            .and_then(|any| any.downcast_ref::<TypedTool<D, Req, Res>>())
    }
}
```

**Constraints:** Zero Box allocations during normal operation. Use arena allocation for bulk storage. All type conversions must be zero-copy where possible. Static dispatch throughout.

---

## Task E: Type-Safe Execution Pipeline
**File:** `./packages/provider/src/clients/anthropic/tools.rs`
**Lines:** 601-800
**Priority:** CRITICAL
**Architecture:** Typed tool execution with automatic serde conversion and streaming support

**Technical Details:**
- Lines 601-650: Automatic JSON to typed request conversion using serde
- Lines 651-700: Tool invocation with dependency injection and typed parameters
- Lines 701-750: Streaming response handling with real-time chunk emission
- Lines 751-800: Error handling pipeline with chain control and retry logic

**Implementation Specifications:**
```rust
impl TypedToolStorage {
    pub async fn execute_typed_tool<D, Req, Res>(
        &self,
        name: &str,
        input: Value,
        context: &ToolExecutionContext,
    ) -> AnthropicResult<tokio::sync::mpsc::UnboundedReceiver<ToolOutput>>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        let tool = self.get_tool::<D, Req, Res>(name)
            .ok_or_else(|| AnthropicError::ToolExecutionError {
                tool_name: name.to_string(),
                error: "Tool not found".to_string(),
            })?;
        
        // Convert JSON input to typed request using serde (zero-copy where possible)
        let request: Req = serde_json::from_value(input)
            .map_err(|e| AnthropicError::InvalidRequest(format!("Invalid request schema: {}", e)))?;
        
        // Create streaming channel
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        let emitter = Emitter { sender };
        
        // Create conversation and chain control objects
        let conversation = Conversation {
            messages: &[], // TODO: Get from context
            context,
            last_message: &Message::default(), // TODO: Get actual last message
        };
        let chain_control = ChainControl {
            should_stop: AtomicBool::new(false),
            retry_count: AtomicU32::new(0),
        };
        
        // Execute tool with typed parameters
        tokio::spawn(async move {
            match (tool.handlers.invocation)(&conversation, &emitter, request, &tool.dependency).await {
                Ok(_) => {},
                Err(e) => {
                    if let Some(error_handler) = &tool.handlers.error {
                        error_handler(&conversation, &chain_control, e, &tool.dependency);
                    }
                }
            }
        });
        
        Ok(receiver)
    }
}
```

**Constraints:** All serde conversions must handle errors gracefully without unwrap(). Streaming must be non-blocking with backpressure handling. Type safety maintained throughout execution pipeline.

---

## Task F: Integration and Registry Updates
**File:** `./packages/provider/src/clients/anthropic/tools.rs`
**Lines:** 801-900
**Priority:** CRITICAL
**Architecture:** Update ToolRegistry to support both old and new patterns during migration

**Technical Details:**
- Lines 801-830: Update ToolRegistry::add() method to accept TypedTool instances
- Lines 831-860: Maintain backward compatibility with existing tools during transition
- Lines 861-890: Integration with existing tool execution pipeline
- Lines 891-900: Public API exposure and documentation

**Implementation Specifications:**
```rust
impl ToolRegistry {
    pub fn add<D, Req, Res>(mut self, builder_result: TypedTool<D, Req, Res>) -> AnthropicResult<Self>
    where
        D: Send + Sync + 'static,
        Req: serde::de::DeserializeOwned + Send + 'static,
        Res: serde::Serialize + Send + 'static,
    {
        self.typed_storage.register(builder_result)?;
        Ok(self)
    }
    
    // Backward compatibility for existing tools
    pub fn register_tool(&mut self, executor: Box<dyn ToolExecutor + Send + Sync>) {
        let definition = executor.definition();
        let name = definition.name.clone();
        
        self.tools.insert(name.clone(), definition);
        self.executors.insert(name, executor);
    }
    
    // Enhanced execution method supporting both patterns
    pub async fn execute_tool(
        &self,
        name: &str,
        input: Value,
        context: &ToolExecutionContext,
    ) -> AnthropicResult<ToolResult> {
        // Try typed execution first
        if let Some(_) = self.typed_storage.tools.get(name) {
            // Handle typed tool execution with streaming
            let receiver = self.typed_storage.execute_typed_tool::<(), Value, Value>(name, input, context).await?;
            // Convert streaming result to ToolResult
            // Implementation details...
        } else {
            // Fall back to legacy tool execution
            self.execute_legacy_tool(name, input, context).await
        }
    }
}
```

**Constraints:** Must maintain full backward compatibility with existing tools. Zero-allocation migration path. All new code must follow ergonomic design principles without unwrap() or expect().

---

# IMAGE GENERATION IMPLEMENTATION TODO

## PRODUCTION-QUALITY STABLE DIFFUSION 3 IMPLEMENTATION

### PHASE 0: PRODUCTION READINESS FIXES
*CRITICAL: Must be completed before other phases*

#### Task 1: Fix Critical Unwrap() in Generation.rs
**File:** `./packages/provider/src/image_processing/generation.rs`
**Lines:** 201
**Priority:** CRITICAL
**Architecture:** Replace dangerous unwrap() with proper error handling

**Technical Details:**
- Current code: `let model_manager = self.model_manager.as_ref().unwrap();`
- Violation: Can cause application panic if model_manager is None
- Solution: Replace with `let model_manager = self.model_manager.as_ref().ok_or_else(|| GenerationError::ModelLoadingError("Model manager not initialized".to_string()))?;`
- Ensure function returns Result type for proper error propagation
- Follow zero-allocation, blazing-fast, no unsafe, no unchecked, no locking, elegant ergonomic code constraints

**Constraints:** Never use unwrap() or expect() in src/* files. All operations must return Result types.

---

#### Task 2: Fix Time Approximation in Cache.rs
**File:** `./packages/http3/src/cache.rs`
**Lines:** 364-371
**Priority:** CRITICAL
**Architecture:** Replace time approximation with production-ready time handling

**Technical Details:**
- Current issue: Comment "in a real implementation you'd use a proper time library" with approximation code
- Lines 364-371: Replace approximation code with proper time handling
- Add dependency: `chrono = "0.4"` to Cargo.toml
- Solution implementation:
  ```rust
  use chrono::{DateTime, Utc};
  
  let unix_timestamp = DateTime::from_timestamp(total_seconds as i64, 0)
      .ok_or_else(|| "Invalid timestamp")?;
  let duration_since_epoch = unix_timestamp
      .signed_duration_since(DateTime::UNIX_EPOCH);
  Some(Instant::now() - Duration::from_secs(duration_since_epoch.num_seconds() as u64))
  ```
- Follow zero-allocation, blazing-fast, no unsafe, no unchecked, no locking, elegant ergonomic code constraints

**Constraints:** Must be production-ready with proper error handling. No approximations or "for now" implementations.

---

#### Task 3: Systematic Unwrap() Replacement
**File:** Multiple files in `./packages/provider/src/` and `./packages/http3/src/`
**Lines:** 100+ locations
**Priority:** CRITICAL
**Architecture:** Replace all unwrap() calls with proper error handling

**Technical Details:**
- Search pattern: `unwrap()` in all src/ directories
- Replace each instance with proper error handling using ? operator and Result types
- Ensure all functions return Result types instead of panicking
- Common patterns:
  - `some_operation().unwrap()` → `some_operation().map_err(|e| SpecificError::from(e))?`
  - `option.unwrap()` → `option.ok_or_else(|| SpecificError::new("description"))?`
  - `result.unwrap()` → `result.map_err(|e| SpecificError::from(e))?`
- Follow zero-allocation, lock-free patterns with elegant ergonomic error handling
- Prioritize files in image_processing module first as they're needed for Phase 1

**Constraints:** Never use unwrap() or expect() in src/* files. All operations must return Result types.

---

#### Task 4: Replace Anthropic Tools Placeholder Implementations
**File:** Multiple files containing "in production" comments
**Lines:** Various
**Priority:** MEDIUM
**Architecture:** Replace placeholder implementations with production-ready code

**Technical Details:**

**4a. Expression Calculator (Line 89)**
- **File:** `./packages/provider/src/clients/anthropic/tools.rs`
- **Lines:** 89 - Replace "Simple expression evaluation (in production, use a proper parser)"
- **Implementation:** Replace `evaluate_expression()` with proper mathematical expression parser using `pest` crate
- **Features:** Support arithmetic operations (+, -, *, /, %), parentheses, variables, mathematical functions (sin, cos, sqrt, etc.)
- **Error Handling:** Comprehensive parsing error messages, division by zero protection, overflow detection
- **Performance:** Zero-allocation parsing with stack-based evaluation, O(n) complexity

**4b. Web Search API Integration (Line 143)**
- **File:** `./packages/provider/src/clients/anthropic/tools.rs`
- **Lines:** 143-159 - Replace placeholder search results with actual API integration
- **Implementation:** Integrate with DuckDuckGo Instant Answer API or similar privacy-focused search
- **Features:** Query sanitization, result ranking, snippet extraction, URL validation
- **Rate Limiting:** Implement exponential backoff, request throttling, cache results
- **Security:** Input validation, XSS prevention, safe URL handling

**4c. Secure File Reading (Line 212)**
- **File:** `./packages/provider/src/clients/anthropic/tools.rs`
- **Lines:** 212-218 - Replace placeholder file reading with secure implementation
- **Implementation:** Path traversal prevention, file size limits, allowed directory restrictions
- **Features:** MIME type detection, binary file handling, encoding detection
- **Security:** Sandbox file access, symlink protection, permission validation
- **Performance:** Streaming reads for large files, memory-mapped access for performance

**4d. Directory Listing (Line 221)**
- **File:** `./packages/provider/src/clients/anthropic/tools.rs`
- **Lines:** 221-229 - Replace placeholder directory listing with secure implementation
- **Implementation:** Recursive directory traversal with depth limits, pattern matching
- **Features:** File metadata extraction, sorting options, filtering capabilities
- **Security:** Path validation, hidden file handling, permission checking
- **Performance:** Async directory traversal, lazy loading for large directories

**4e. CUDA Detection Enhancement (Line 200)**
- **File:** `./packages/provider/src/image_processing/factory.rs`
- **Lines:** 200-202 - Replace simple CUDA check with sophisticated detection
- **Implementation:** NVIDIA Management Library (NVML) integration, CUDA runtime detection
- **Features:** GPU capability detection, memory availability check, compute capability validation
- **Performance:** Cached detection results, lazy initialization, minimal overhead
- **Compatibility:** Support for different CUDA versions, multi-GPU systems

**Constraints:** All implementations must be production-ready. No placeholders or "in production" comments. Follow zero-allocation, blazing-fast, no unsafe, no unchecked, no locking, elegant ergonomic code constraints.

---

#### Task 5: Replace "For Now" Temporary Implementations
**File:** Multiple files containing "for now" comments
**Lines:** Various
**Priority:** MEDIUM
**Architecture:** Replace temporary implementations with production-ready code

**Technical Details:**

**5a. Groq Model Configuration (Line 413)**
- **File:** `./packages/provider/src/clients/groq/completion.rs`
- **Lines:** 413-424 - Replace "Get model config - for now using default values"
- **Implementation:** Dynamic model configuration based on specific model capabilities
- **Features:** Model-specific parameter optimization, context length detection, capability flags
- **Data Source:** Groq API model list endpoint, cached model specifications
- **Performance:** Lazy loading of model configs, zero-allocation parameter selection

**5b. Groq Streaming Implementation (Line 586)**
- **File:** `./packages/provider/src/clients/groq/completion.rs`
- **Lines:** 586 - Replace "For now, return a placeholder stream"
- **Implementation:** Full Server-Sent Events (SSE) streaming with Groq API
- **Features:** Real-time token streaming, partial response handling, connection recovery
- **Error Handling:** Stream interruption recovery, timeout handling, backpressure management
- **Performance:** Zero-copy streaming, async iterator pattern, minimal latency

**5c. Client Factory Implementations**
- **File:** `./packages/provider/src/client_factory.rs`
- **Lines:** 383, 396, 409, 422, 435 - Replace TODO client implementations
- **Implementation:** Complete client factory methods for all supported providers
- **Providers:** Gemini, Mistral, Groq, Perplexity, xAI
- **Features:** Authentication handling, configuration validation, client instantiation
- **Architecture:** Factory pattern with lazy initialization, connection pooling

**5d. OpenAI Vision Image Resizing (Line 274)**
- **File:** `./packages/provider/src/clients/openai/vision.rs`
- **Lines:** 274 - Replace "TODO: Implement actual image resizing"
- **Implementation:** High-performance image resizing using `image` crate
- **Features:** Aspect ratio preservation, quality optimization, format conversion
- **Performance:** SIMD-accelerated processing, memory-efficient resizing
- **Formats:** Support for JPEG, PNG, WebP, with automatic format detection

**5e. OpenAI Moderation Placeholders (Lines 621, 646)**
- **File:** `./packages/provider/src/clients/openai/moderation.rs`
- **Lines:** 621, 646 - Replace placeholder assessments and API simulation
- **Implementation:** Full OpenAI Moderation API integration
- **Features:** Content safety classification, confidence scoring, category detection
- **Categories:** Hate, harassment, self-harm, sexual content, violence
- **Performance:** Batch processing, caching, rate limiting

**Constraints:** All implementations must be production-ready. No temporary or "for now" implementations. Follow zero-allocation, blazing-fast, no unsafe, no unchecked, no locking, elegant ergonomic code constraints.

---

#### Task 6: Decompose Large Files for Maintainability
**File:** Multiple large files in `./packages/provider/src/`
**Lines:** Files >1000 lines
**Priority:** LOW
**Architecture:** Split large files into logical modules for better maintainability

**Technical Details:**

**6a. Decompose model_info.rs (2586 lines)**
- Split into: `model_info/definitions.rs`, `model_info/providers.rs`, `model_info/capabilities.rs`, `model_info/validation.rs`, `model_info/mod.rs`
- Ensure zero-allocation patterns and lock-free design throughout

**6b. Decompose gemini/completion.rs (1731 lines)**
- Split into: `gemini/completion/request.rs`, `gemini/completion/response.rs`, `gemini/completion/streaming.rs`, `gemini/completion/mod.rs`
- Follow zero-allocation, lock-free patterns with elegant ergonomic design

**6c. Decompose mistral/completion.rs (1284 lines)**
- Split into same pattern as gemini/completion.rs
- Ensure blazing-fast performance with zero-allocation patterns

**6d. Decompose workflow/prompt_enhancement.rs (1135 lines)**
- Split into: `workflow/prompt_enhancement/stages.rs`, `workflow/prompt_enhancement/pipeline.rs`, `workflow/prompt_enhancement/config.rs`, `workflow/prompt_enhancement/mod.rs`
- Follow lock-free concurrent programming patterns

**6e. Decompose domain/memory.rs (1088 lines)**
- Split into: `domain/memory/types.rs`, `domain/memory/management.rs`, `domain/memory/persistence.rs`, `domain/memory/mod.rs`
- Ensure zero-allocation memory management with elegant ergonomic design

**6f. Decompose embedding/image.rs (1020 lines)**
- Split into: `embedding/image/processing.rs`, `embedding/image/features.rs`, `embedding/image/backends.rs`, `embedding/image/mod.rs`
- Follow zero-allocation patterns with blazing-fast performance optimization

**Constraints:** All decomposed modules must follow zero-allocation, blazing-fast, no unsafe, no unchecked, no locking, elegant ergonomic code constraints.

---

### PHASE 1: FOUNDATION & CONFIGURATION

#### Task 7: Create Image Generation Foundation
**File:** `./packages/provider/src/image_processing/generation.rs`
**Lines:** 1-100
**Architecture:** Main implementation file with CandleImageGenerator struct

**Technical Details:**
- Lines 1-20: Imports (candle_transformers::models::mmdit, hf_hub, tokenizers, candle_nn::VarBuilder)
- Lines 21-40: CandleImageGenerator struct with fields: device, model_config, is_initialized, current_model
- Lines 41-60: GenerationError enum with variants: ModelLoadingError, TextEncodingError, SamplingError, VAEDecodingError, DeviceError, ConfigurationError
- Lines 61-80: Device management utilities (detect_optimal_device, estimate_memory_usage, configure_device)
- Lines 81-100: Constructor methods (new, with_device, with_config) with proper error handling

**Constraints:** Never use unwrap() or expect() in source code. All operations must return Result types.

---

#### Task 8: Create Generation Configuration
**File:** `./packages/provider/src/image_processing/generation/config.rs`
**Lines:** 1-150
**Architecture:** Configuration management for SD3 model variants and parameters

**Technical Details:**
- Lines 1-30: SD3ModelVariant enum with variants: ThreeMedium, ThreeFiveLarge, ThreeFiveLargeTurbo, ThreeFiveMedium
- Lines 31-70: GenerationConfig struct with fields: model_variant, num_inference_steps, cfg_scale, time_shift, use_flash_attn, use_slg, output_size, seed
- Lines 71-100: ModelLoadingConfig struct with model_id, revision, use_safetensors, cache_dir
- Lines 101-130: Configuration validation functions (validate_inference_steps, validate_cfg_scale, validate_output_size)
- Lines 131-150: Device configuration optimization (get_optimal_batch_size, calculate_memory_requirements)

**Constraints:** All configuration must match stable-diffusion-3 example patterns exactly.

---

#### Task 9: Create Text Encoder Implementation
**File:** `./packages/provider/src/image_processing/generation/text_encoder.rs`
**Lines:** 1-400
**Architecture:** Triple CLIP encoder following stable-diffusion-3/clip.rs patterns

**Technical Details:**
- Lines 1-50: Imports and ClipWithTokenizer struct definition
- Lines 51-120: CLIP-L implementation with tokenization and embedding generation
- Lines 121-190: CLIP-G implementation with proper padding and attention
- Lines 191-260: T5WithTokenizer implementation for T5-XXL long text understanding
- Lines 261-320: StableDiffusion3TripleClipWithTokenizer combining all three encoders
- Lines 321-370: encode_text_to_embedding method with context and y tensor generation
- Lines 371-400: Error handling and cleanup utilities

**Constraints:** Must follow stable-diffusion-3/clip.rs patterns exactly. No modifications to tokenization logic.

---

#### Task 10: Create Sampling Implementation
**File:** `./packages/provider/src/image_processing/generation/sampling.rs`
**Lines:** 1-200
**Architecture:** MMDiT sampling with Euler method following stable-diffusion-3/sampling.rs

**Technical Details:**
- Lines 1-30: Imports and SkipLayerGuidanceConfig struct definition
- Lines 31-80: euler_sample function with MMDiT integration, sigmas calculation, timestep scheduling
- Lines 81-120: CFG (Classifier-Free Guidance) implementation with apply_cfg function
- Lines 121-150: Skip Layer Guidance support for SD3.5 models with layer masking
- Lines 151-180: Noise generation using flux::sampling::get_noise patterns
- Lines 181-200: Time scheduling utilities (time_snr_shift function)

**Constraints:** Must follow stable-diffusion-3/sampling.rs patterns exactly. No modifications to sampling algorithms.

---

#### Task 11: Create VAE Decoder Implementation
**File:** `./packages/provider/src/image_processing/generation/vae.rs`
**Lines:** 1-150
**Architecture:** VAE decoder following stable-diffusion-3/vae.rs patterns

**Technical Details:**
- Lines 1-40: Imports and VAE configuration setup
- Lines 41-80: build_sd3_vae_autoencoder function with AutoEncoderKLConfig
- Lines 81-120: sd3_vae_vb_rename function for weight mapping and layer renaming
- Lines 121-150: Latent to image conversion with TAESD3 scale factor and post-processing

**Constraints:** Must follow stable-diffusion-3/vae.rs patterns exactly. Use exact scaling factors.

---

#### Task 12: Create Model Management
**File:** `./packages/provider/src/image_processing/generation/models.rs`
**Lines:** 1-250
**Architecture:** Model loading and management from HuggingFace Hub

**Technical Details:**
- Lines 1-50: Imports and ModelManager struct definition
- Lines 51-100: HuggingFace Hub integration with hf_hub::api for model downloading
- Lines 101-150: Weight management using candle_nn::VarBuilder::from_mmaped_safetensors
- Lines 151-200: Multi-model support infrastructure with model switching
- Lines 201-250: Memory optimization utilities and model cleanup

**Constraints:** Must handle all SD3 model variants. Efficient memory management required.

---

### PHASE 2: MAIN IMPLEMENTATION

#### Task 13: Implement Main Generation Logic
**File:** `./packages/provider/src/image_processing/generation.rs`
**Lines:** 101-550
**Architecture:** Complete ImageGenerationProvider trait implementation

**Technical Details:**
- Lines 101-150: ImageGenerationProvider trait implementation skeleton
- Lines 151-250: generate_image() method orchestrating text encoding, sampling, VAE decoding
- Lines 251-350: generate_image_batch() with efficient batch processing
- Lines 351-400: supported_models() returning all SD3 variants
- Lines 401-450: load_model() with proper error handling and model switching
- Lines 451-500: Device management and optimization utilities
- Lines 501-550: Cleanup and resource management methods

**Constraints:** Full pipeline orchestration required. All trait methods must be implemented.

---

#### Task 14: Create Module Integration
**File:** `./packages/provider/src/image_processing/generation/mod.rs`
**Lines:** 1-50
**Architecture:** Module declarations and public API

**Technical Details:**
- Lines 1-20: Module declarations for config, text_encoder, sampling, vae, models
- Lines 21-35: Public use statements for main types (CandleImageGenerator, GenerationConfig, SD3ModelVariant)
- Lines 36-50: Module-level documentation and visibility configuration

**Constraints:** Clean public API required. Proper encapsulation.

---

#### Task 15: Update Main Image Processing Module
**File:** `./packages/provider/src/image_processing/mod.rs`
**Lines:** Add generation module after line 14
**Architecture:** Integration with existing image processing module

**Technical Details:**
- Add: `#[cfg(feature = "generation")] pub mod generation;` after line 14
- Update pub use statements to include generation types
- Ensure feature flag compatibility

**Constraints:** Must maintain compatibility with existing module structure.

---

### PHASE 3: INTEGRATION & TESTING

#### Task 16: Update Provider Factory
**File:** `./packages/provider/src/image_processing/factory.rs`
**Lines:** 115-122 (update existing generation provider creation)
**Architecture:** Integration with factory pattern

**Technical Details:**
- Update create_candle_generation_provider function to use CandleImageGenerator
- Add proper error handling and configuration passing
- Ensure feature flag compatibility

**Constraints:** Must integrate seamlessly with existing factory pattern.

---

## ARCHITECTURAL NOTES

### Device Management Strategy
- Automatic device detection with fallback hierarchy: Metal → CUDA → CPU
- Memory estimation and optimization for large models
- Efficient batch processing with dynamic batch sizing

### Error Handling Architecture
- Comprehensive error types covering all failure modes
- Proper error propagation without unwrap/expect
- Rich error context for debugging and monitoring

### Memory Management
- Efficient model loading with memory mapping
- Proper cleanup and resource deallocation
- Batch processing optimization for memory usage

### Performance Optimizations
- Flash attention support for speed improvements
- Skip Layer Guidance for SD3.5 models
- Efficient tensor operations with proper device placement

## QUALITY REQUIREMENTS

1. **No unwrap() or expect() in source code** - All operations must return Result types
2. **Real ML operations only** - No mocking, simulation, or fake data
3. **Production-quality error handling** - Comprehensive error coverage
4. **Memory efficiency** - Proper resource management and cleanup
5. **Device optimization** - Automatic device selection and configuration
6. **Exact pattern matching** - Follow stable-diffusion-3 example patterns precisely

## CONSTRAINTS

- Must follow stable-diffusion-3 example patterns exactly
- Never use unwrap() or expect() in src/* files
- All operations must be real ML operations using Candle transformers
- Comprehensive error handling for all failure modes
- Memory-efficient implementation with proper cleanup
- Support for all SD3 model variants (3-medium, 3.5-large, 3.5-large-turbo, 3.5-medium)

---

# ULTRA-HIGH PERFORMANCE DOMAIN OPTIMIZATIONS

## Task 48: Lock-Free Message Processing Pipeline
**File:** `./packages/domain/src/message_processing.rs` (NEW FILE)
**Lines:** 1-400 (complete implementation)
**Priority:** CRITICAL
**Architecture:** High-performance message processing pipeline with crossbeam-queue, zero-allocation message handling, SIMD text processing, and atomic counters for statistics

**Performance Targets:**
- <1μs message routing latency
- 100K+ messages/second throughput  
- Zero allocation in steady state
- Lock-free operation under all conditions

**Technical Details:**

**Lines 1-50: Message Type Definitions**
```rust
use arrayvec::ArrayVec;
use smallvec::SmallVec;
use crossbeam_queue::{ArrayQueue, SegQueue};
use crossbeam_deque::{Injector, Stealer, Worker};
use atomic_counter::{AtomicCounter, RelaxedCounter};
use arc_swap::ArcSwap;
use packed_simd_2::f32x8;

// Zero-allocation message types with const generics
#[derive(Debug, Clone)]
pub struct Message<const N: usize = 256> {
    pub id: u64,
    pub message_type: MessageType,
    pub content: ArrayVec<u8, N>,
    pub metadata: SmallVec<[u8; 32]>,
    pub timestamp: std::time::Instant,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    AgentChat = 0,
    MemoryStore = 1,
    MemoryRecall = 2,
    ContextUpdate = 3,
    SystemControl = 4,
}
```

**Lines 51-150: Lock-Free Processing Pipeline**
```rust
pub struct MessageProcessor {
    // Lock-free MPMC queues for different message types
    chat_queue: ArrayQueue<Message>,
    memory_queue: ArrayQueue<Message>,
    control_queue: ArrayQueue<Message>,
    
    // Work-stealing deques for load balancing
    workers: Vec<Worker<Message>>,
    stealers: Vec<Stealer<Message>>,
    injector: Injector<Message>,
    
    // Atomic performance counters
    messages_processed: RelaxedCounter,
    processing_latency: RelaxedCounter,
    queue_depth: RelaxedCounter,
    
    // Copy-on-write shared state
    config: Arc<ArcSwap<ProcessingConfig>>,
}

#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub max_queue_depth: usize,
    pub worker_count: usize,
    pub batch_size: usize,
    pub timeout_micros: u64,
}
```

**Lines 151-250: SIMD Text Processing Integration**
```rust
// Integration with memory_ops.rs SIMD optimizations
use crate::memory_ops::{simd_cosine_similarity, generate_pooled_embedding};

impl MessageProcessor {
    #[inline(always)]
    pub fn process_message_with_simd(&self, message: &Message) -> Result<ProcessingResult, MessageError> {
        // Use SIMD for text pattern matching and classification
        let content_str = std::str::from_utf8(&message.content)
            .map_err(|_| MessageError::InvalidContent)?;
        
        // Generate embedding for content classification
        let embedding = generate_pooled_embedding(content_str);
        
        // Use SIMD similarity for routing decisions
        let route = self.classify_message_route(&embedding)?;
        
        // Process based on classification
        match message.message_type {
            MessageType::AgentChat => self.process_chat_message(message, route),
            MessageType::MemoryStore => self.process_memory_store(message, route),
            MessageType::MemoryRecall => self.process_memory_recall(message, route),
            MessageType::ContextUpdate => self.process_context_update(message, route),
            MessageType::SystemControl => self.process_system_control(message, route),
        }
    }
    
    #[inline(always)]
    fn classify_message_route(&self, embedding: &ArrayVec<f32, 64>) -> Result<RouteType, MessageError> {
        // Use SIMD operations for fast classification
        // Implementation uses SIMD cosine similarity against known patterns
    }
}
```

**Lines 251-350: Zero-Allocation Processing Workers**
```rust
pub struct ProcessingWorker {
    id: usize,
    worker: Worker<Message>,
    stealers: Vec<Stealer<Message>>,
    injector: Arc<Injector<Message>>,
    message_pool: ArrayQueue<Message<256>>,
    stats: WorkerStats,
}

#[derive(Debug, Default)]
pub struct WorkerStats {
    pub messages_processed: RelaxedCounter,
    pub steal_attempts: RelaxedCounter,
    pub successful_steals: RelaxedCounter,
    pub processing_time_nanos: RelaxedCounter,
}

impl ProcessingWorker {
    #[inline(always)]
    pub async fn run_worker_loop(&mut self) -> Result<(), MessageError> {
        loop {
            // Try to pop from local queue first (lock-free)
            if let Some(message) = self.worker.pop() {
                self.process_local_message(message).await?;
                continue;
            }
            
            // Try to steal from other workers (work-stealing algorithm)
            if let Some(message) = self.try_steal_work() {
                self.process_stolen_message(message).await?;
                continue;
            }
            
            // Try global injector as last resort
            if let Some(message) = self.injector.steal() {
                self.process_injected_message(message).await?;
                continue;
            }
            
            // No work available, yield briefly
            tokio::task::yield_now().await;
        }
    }
}
```

**Lines 351-400: Performance Monitoring and Error Handling**
```rust
#[derive(Debug, thiserror::Error)]
pub enum MessageError {
    #[error("Queue capacity exceeded: {0}")]
    QueueFull(usize),
    #[error("Invalid message content")]
    InvalidContent,
    #[error("Processing timeout")]
    ProcessingTimeout,
    #[error("Worker error: {0}")]
    WorkerError(String),
    #[error("SIMD processing error: {0}")]
    SimdError(String),
}

impl MessageProcessor {
    #[inline(always)]
    pub fn get_performance_stats(&self) -> ProcessingStats {
        ProcessingStats {
            messages_processed: self.messages_processed.get(),
            average_latency_nanos: self.processing_latency.get() / self.messages_processed.get().max(1),
            current_queue_depth: self.queue_depth.get(),
            throughput_per_second: self.calculate_throughput(),
        }
    }
}
```

**Integration Points:**
- `src/lib.rs` - Add module export: `pub mod message_processing;`
- `src/agent.rs` - Lines 150-180: Integrate agent chat with message pipeline
- `src/agent_role.rs` - Lines 247-273: Connect context-aware chat with message routing
- `src/memory_ops.rs` - Integration with SIMD text processing and embedding generation

**Dependencies Added:**
- crossbeam-queue = "0.3.12" (already present)
- crossbeam-deque = "0.8.6" (already present)
- atomic-counter = "1.0.1" (already present)
- packed_simd_2 = "0.3.8" (already present)

**Constraints:**
- Zero allocation using ArrayVec, SmallVec, object pooling
- No locking using crossbeam lock-free data structures
- Blazing fast with #[inline(always)] on hot paths
- No unsafe code except for properly justified SIMD operations
- No unchecked operations with comprehensive error handling
- Never use unwrap() or expect() in src/*
- Elegant ergonomic APIs with intuitive message processing patterns

---

## Task 49: High-Performance Context Management
**File:** `./packages/domain/src/context_management.rs` (NEW FILE)
**Lines:** 1-300 (complete implementation)
**Priority:** HIGH
**Architecture:** Optimize context switching and management with copy-on-write semantics, thread-local storage, and lock-free data structures

**Performance Targets:**
- <100ns context switching latency
- Zero-allocation context operations
- Lock-free concurrent context access
- SIMD-optimized context comparison

**Technical Details:**

**Lines 1-100: Context Types and Storage**
```rust
use arc_swap::ArcSwap;
use once_cell::sync::Lazy;
use crossbeam_skiplist::SkipMap;
use arrayvec::ArrayVec;
use smallvec::SmallVec;

// Thread-local context cache for zero-allocation access
thread_local! {
    static CONTEXT_CACHE: RefCell<ArrayVec<ContextSnapshot, 16>> = RefCell::new(ArrayVec::new());
}

#[derive(Debug, Clone)]
pub struct ContextManager {
    // Copy-on-write context storage
    current_context: Arc<ArcSwap<AgentContext>>,
    
    // Lock-free context history
    context_history: Arc<SkipMap<u64, ContextSnapshot>>,
    
    // Performance counters
    context_switches: RelaxedCounter,
    cache_hits: RelaxedCounter,
    cache_misses: RelaxedCounter,
}

#[derive(Debug, Clone)]
pub struct AgentContext {
    pub session_id: u64,
    pub conversation_history: SmallVec<[Message; 32]>,
    pub memory_context: SmallVec<[MemoryNode; 16]>,
    pub tool_state: SmallVec<[ToolState; 8]>,
    pub metadata: SmallVec<[u8; 64]>,
}
```

**Integration Points:**
- `src/agent.rs` - Context switching optimization
- `src/agent_role.rs` - Context-aware chat integration
- `src/memory_ops.rs` - Memory context integration

**Constraints:** Same ultra-strict constraints as Task 48

---

## Task 50: Zero-Allocation Error Handling System
**File:** `./packages/domain/src/error_handling.rs` (EXISTING FILE ENHANCEMENT)
**Lines:** 1-250 (complete rewrite)
**Priority:** HIGH  
**Architecture:** Create comprehensive error handling without heap allocation

**Already Completed** - This task has been implemented with zero-allocation error types using ArrayVec and SmallVec patterns, comprehensive error categories, and atomic error counters for statistics.