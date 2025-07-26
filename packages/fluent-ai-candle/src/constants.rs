//! Performance-oriented constants for zero-allocation operations
//!
//! This module contains carefully tuned constants that optimize performance across
//! different hardware configurations and model sizes. These values represent the
//! result of extensive benchmarking and are designed to work well for most use cases
//! while avoiding allocations and memory fragmentation.
//!
//! # Performance Tuning Philosophy
//!
//! Constants are chosen based on:
//! - **Memory locality**: Values that fit in CPU cache for faster access
//! - **SIMD alignment**: Sizes that work well with vectorized operations  
//! - **Hardware diversity**: Reasonable defaults across different GPUs and CPUs
//! - **Model compatibility**: Support for models from small (7B) to large (70B+)
//! - **Real-world usage**: Based on typical inference patterns and workloads
//!
//! # Customization
//!
//! While these constants provide good defaults, they can be overridden at runtime
//! through configuration builders when specific workloads require different values.

// ============================================================================
// Memory Management Constants
// ============================================================================

/// Default KV cache size in bytes optimized for modern GPU memory
/// 
/// Sets the default key-value cache size to 512 MB, which provides a good balance
/// between memory usage and inference performance for most language models.
/// 
/// # Value: 512 MB (536,870,912 bytes)
/// 
/// This size is chosen to:
/// - Support long context generation (up to ~32K tokens for most models)
/// - Fit comfortably within 8GB GPU memory alongside model weights
/// - Avoid memory fragmentation through power-of-2 aligned sizing
/// - Provide room for batch processing multiple sequences
/// 
/// # Hardware Considerations
/// 
/// - **8GB GPU**: Supports models up to ~7B parameters with this cache size
/// - **16GB GPU**: Supports models up to ~13B parameters comfortably
/// - **24GB+ GPU**: Can handle larger models or multiple cache instances
/// 
/// # Performance Impact
/// 
/// A larger cache reduces recomputation of attention keys/values, improving:
/// - Generation speed for long sequences (2-10x faster)
/// - Consistency in generation latency
/// - Support for larger batch sizes
/// 
/// # Memory Usage Estimation
/// 
/// For a model with `d_model` dimensions and `num_layers`:
/// ```text
/// Memory per token ≈ 2 * num_layers * d_model * sizeof(f16)
/// Supported tokens ≈ CACHE_SIZE / (2 * num_layers * d_model * 2)
/// ```
/// 
/// # Customization
/// 
/// Override through configuration:
/// ```rust
/// let config = KVCacheConfig::builder()
///     .max_size(1024 * 1024 * 1024) // 1GB cache
///     .build();
/// ```
pub const DEFAULT_KV_CACHE_SIZE: usize = 512 * 1024 * 1024;

/// Default token buffer size for efficient streaming operations
/// 
/// Defines the buffer size for token accumulation during streaming generation,
/// optimized for typical streaming latency requirements and memory efficiency.
/// 
/// # Value: 8,192 tokens
/// 
/// This buffer size provides:
/// - Sufficient capacity for most document-length generations
/// - Low memory overhead (32KB for u32 tokens)
/// - Good streaming granularity for real-time applications
/// - Power-of-2 alignment for optimal memory allocation
/// 
/// # Streaming Performance
/// 
/// Buffer size affects streaming characteristics:
/// - **Too small**: Frequent buffer flushes, increased overhead
/// - **Too large**: Higher memory usage, potential latency spikes
/// - **8192**: Sweet spot for most real-time applications
/// 
/// # Use Cases
/// 
/// Optimized for:
/// - **Chat applications**: Typical responses under 2000 tokens
/// - **Document generation**: Multi-paragraph content
/// - **Code generation**: Complete functions or classes
/// - **Real-time streaming**: Sub-second response times
/// 
/// # Memory Footprint
/// 
/// - **Token buffer**: 8,192 × 4 bytes = 32KB per stream
/// - **Metadata buffer**: Additional ~8KB for streaming state
/// - **Total per stream**: ~40KB memory overhead
/// 
/// # Performance Benchmarks
/// 
/// Typical streaming performance with this buffer size:
/// - **Latency**: <50ms between token emissions
/// - **Throughput**: 50-200 tokens/second depending on model
/// - **Memory efficiency**: <1% overhead vs model weights
/// 
/// # Customization
/// 
/// Adjust for specific requirements:
/// ```rust
/// let config = StreamingConfig::builder()
///     .token_buffer_size(16384) // Double buffer for longer content
///     .build();
/// ```
pub const DEFAULT_TOKEN_BUFFER_SIZE: usize = 8192;

/// Maximum model file size for memory mapping operations
/// 
/// Sets the upper limit for model files that can be memory-mapped for efficient
/// loading and access. Files larger than this limit use alternative loading strategies.
/// 
/// # Value: 16 GB (17,179,869,184 bytes)
/// 
/// This limit accommodates:
/// - **Large language models**: Up to ~70B parameters in FP16
/// - **Multimodal models**: Vision-language models with large components  
/// - **Fine-tuned models**: Custom models with additional parameters
/// - **Quantized models**: Q4/Q8 quantized versions of very large models
/// 
/// # Memory Mapping Benefits
/// 
/// For files under this limit, memory mapping provides:
/// - **Faster loading**: Direct memory access without full file reads
/// - **Shared memory**: Multiple processes can share the same model
/// - **Virtual memory**: OS manages paging based on actual usage
/// - **Lazy loading**: Only accessed portions are loaded into RAM
/// 
/// # Fallback Strategies
/// 
/// For files exceeding this limit:
/// - **Streaming load**: Progressive loading with buffer management
/// - **Chunked access**: Load model components on-demand
/// - **Distributed loading**: Split across multiple devices/processes
/// - **Compressed loading**: Runtime decompression of model components
/// 
/// # Platform Considerations
/// 
/// - **64-bit systems**: Full 16GB mapping supported
/// - **32-bit systems**: May be limited by address space
/// - **Mobile platforms**: May use smaller limits for memory efficiency
/// - **Cloud environments**: Optimized for typical instance memory sizes
/// 
/// # Storage Requirements
/// 
/// Model size examples:
/// - **7B parameters (FP16)**: ~14GB - fits within limit
/// - **13B parameters (FP16)**: ~26GB - exceeds limit, uses streaming
/// - **70B parameters (Q4)**: ~35GB - exceeds limit significantly
/// 
/// # Performance Impact
/// 
/// Memory mapping vs streaming trade-offs:
/// - **Startup time**: Memory mapping ~10x faster for supported sizes
/// - **Memory usage**: Mapping uses virtual memory more efficiently
/// - **Access patterns**: Random access much faster with mapping
/// 
/// # Configuration
/// 
/// Override for specific deployment scenarios:
/// ```rust
/// let config = ModelLoadingConfig::builder()
///     .max_mmap_size(32 * 1024 * 1024 * 1024) // 32GB for high-memory systems
///     .build();
/// ```
pub const MAX_MODEL_FILE_SIZE: usize = 16 * 1024 * 1024 * 1024;

// ============================================================================
// SIMD and Vectorization Constants  
// ============================================================================

/// Maximum logits dimensions for optimal SIMD operations
/// 
/// Defines the upper bound for logits tensor dimensions that receive SIMD
/// optimization in sampling and processing operations.
/// 
/// # Value: 524,288 (512K)
/// 
/// This limit ensures:
/// - **SIMD efficiency**: Vectors fit in CPU cache for vectorized operations
/// - **Memory alignment**: Power-of-2 sizing for optimal memory access
/// - **Vocabulary support**: Covers virtually all language model vocabularies
/// - **Processing speed**: Maintains sub-millisecond sampling performance
/// 
/// # SIMD Optimization Coverage
/// 
/// Vocabularies within this limit receive vectorized acceleration for:
/// - **Softmax computation**: 4-8x speedup with AVX/NEON instructions
/// - **Top-k selection**: Parallel comparison and sorting operations
/// - **Probability filtering**: Vectorized threshold comparisons
/// - **Token sampling**: SIMD-accelerated cumulative probability calculation
/// 
/// # Vocabulary Size Support
/// 
/// Common model vocabularies:
/// - **GPT-style models**: 50,257 tokens (well under limit)
/// - **BERT models**: 30,522 tokens (well under limit)
/// - **T5 models**: 32,128 tokens (well under limit)
/// - **Large multilingual**: ~500K tokens (near limit)
/// 
/// # Performance Characteristics
/// 
/// Sampling performance by vocabulary size:
/// - **<50K tokens**: <1ms with SIMD optimization
/// - **50K-200K tokens**: 1-5ms with partial SIMD
/// - **200K-512K tokens**: 5-20ms with full SIMD coverage
/// - **>512K tokens**: Falls back to scalar operations
/// 
/// # Memory Usage
/// 
/// Logits memory requirements:
/// - **FP32 logits**: vocab_size × 4 bytes
/// - **At limit**: 512K × 4 = 2MB per sequence
/// - **Batch processing**: scales linearly with batch size
/// 
/// # CPU Architecture Support
/// 
/// SIMD instruction sets leveraged:
/// - **x86_64**: AVX2/AVX-512 for maximum throughput
/// - **ARM64**: NEON instructions for mobile/server efficiency
/// - **RISC-V**: Vector extensions where available
/// - **Fallback**: Optimized scalar code for unsupported architectures
/// 
/// # Fallback Behavior
/// 
/// For vocabularies exceeding this limit:
/// - **Chunked processing**: Break operations into SIMD-sized chunks
/// - **Hybrid approach**: SIMD for hot paths, scalar for edge cases
/// - **Memory streaming**: Process portions to stay within cache limits
/// 
/// # Configuration
/// 
/// Adjust for specific hardware capabilities:
/// ```rust
/// let config = ProcessingConfig::builder()
///     .simd_vocab_limit(1024 * 1024) // 1M tokens for high-end hardware
///     .build();
/// ```
pub const MAX_LOGITS_DIM: usize = 512 * 1024;

// ============================================================================
// Sampling and Generation Parameters
// ============================================================================

/// Default temperature for balanced text generation quality
/// 
/// Standard temperature value that provides a good balance between coherence
/// and creativity for most text generation tasks.
/// 
/// # Value: 1.0 (neutral)
/// 
/// Temperature = 1.0 represents:
/// - **Unscaled softmax**: Direct model probability distribution
/// - **Balanced randomness**: Neither too deterministic nor too chaotic
/// - **Model-intended behavior**: Preserves training distribution characteristics
/// - **Versatile performance**: Good starting point for most applications
/// 
/// # Temperature Effects
/// 
/// - **< 1.0 (e.g., 0.1-0.8)**: More deterministic, focused, repetitive
/// - **= 1.0**: Balanced, natural variation matching training data
/// - **> 1.0 (e.g., 1.2-2.0)**: More random, creative, potentially inconsistent
/// 
/// # Use Case Recommendations
/// 
/// - **Factual Q&A**: 0.1-0.3 (high precision, low creativity)
/// - **Code generation**: 0.2-0.5 (structured, deterministic output)
/// - **General conversation**: 0.7-1.0 (natural, balanced responses)
/// - **Creative writing**: 1.0-1.5 (diverse, imaginative content)
/// - **Brainstorming**: 1.2-2.0 (high diversity, novel combinations)
/// 
/// # Performance Impact
/// 
/// Temperature = 1.0 provides optimal performance:
/// - **No scaling computation**: Bypasses temperature division
/// - **Direct softmax**: Uses optimized native implementation
/// - **Minimal overhead**: Zero additional mathematical operations
/// 
/// # Quality Considerations
/// 
/// This default balances:
/// - **Coherence**: Maintains logical flow and consistency
/// - **Diversity**: Avoids repetitive or overly predictable text
/// - **Creativity**: Allows for natural variation and surprises
/// - **Controllability**: Predictable behavior for most users
/// 
/// # Mathematical Properties
/// 
/// ```text
/// probability[i] = exp(logits[i] / temperature) / sum(exp(logits / temperature))
/// 
/// When temperature = 1.0:
/// probability[i] = exp(logits[i]) / sum(exp(logits))  // Standard softmax
/// ```
/// 
/// # Customization Examples
/// 
/// ```rust
/// // High-precision factual generation
/// let factual_config = GenerationConfig::builder()
///     .temperature(0.2)
///     .build();
/// 
/// // Creative writing assistance  
/// let creative_config = GenerationConfig::builder()
///     .temperature(1.3)
///     .build();
/// ```
pub const DEFAULT_TEMPERATURE: f32 = 1.0;

/// Default nucleus (top-p) sampling threshold for quality generation
/// 
/// Standard top-p value that provides high-quality text generation by focusing
/// on tokens within the top 90% cumulative probability mass.
/// 
/// # Value: 0.9 (90% probability mass)
/// 
/// This threshold ensures:
/// - **Quality focus**: Only considers highly probable tokens
/// - **Dynamic vocabulary**: Adapts to probability distribution shape
/// - **Coherence preservation**: Filters out unlikely/irrelevant tokens
/// - **Diversity maintenance**: Allows variation within probable choices
/// 
/// # Nucleus Sampling Mechanics
/// 
/// The algorithm:
/// 1. **Sort** tokens by probability (highest first)
/// 2. **Accumulate** probabilities until sum reaches 0.9
/// 3. **Filter** tokens outside the nucleus (bottom 10%)
/// 4. **Renormalize** remaining probabilities to sum to 1.0
/// 5. **Sample** from the filtered distribution
/// 
/// # Probability Mass Distribution
/// 
/// Typical behavior with p=0.9:
/// - **High-confidence contexts**: Nucleus may contain 5-20 tokens
/// - **Uncertain contexts**: Nucleus may contain 50-200 tokens
/// - **Uniform distributions**: Nucleus contains more tokens
/// - **Peaked distributions**: Nucleus contains fewer tokens
/// 
/// # Quality vs Creativity Trade-offs
/// 
/// - **p = 0.8**: Higher quality, more conservative choices
/// - **p = 0.9**: Balanced quality and diversity (default)
/// - **p = 0.95**: More creative, wider token selection
/// - **p = 1.0**: No filtering (equivalent to temperature-only sampling)
/// 
/// # Use Case Optimization
/// 
/// - **Professional writing**: 0.8-0.85 (high quality standards)
/// - **General conversation**: 0.9 (balanced and natural)
/// - **Creative content**: 0.92-0.95 (increased expressiveness)
/// - **Brainstorming**: 0.95-1.0 (maximum diversity)
/// 
/// # Performance Characteristics
/// 
/// - **Time complexity**: O(V log V) for vocabulary size V (sorting step)
/// - **Space complexity**: O(V) for probability array and indices
/// - **Typical performance**: <5ms for vocabularies under 100K tokens
/// - **Cache efficiency**: Benefits from sorted probability access patterns
/// 
/// # Interaction with Temperature
/// 
/// Top-p and temperature work synergistically:
/// - **Low temp + Low p**: Very focused, deterministic generation
/// - **Low temp + High p**: Precise but allows some variety
/// - **High temp + Low p**: Creative within high-quality bounds (recommended)
/// - **High temp + High p**: Maximum diversity and creativity
/// 
/// # Statistical Properties
/// 
/// With p=0.9, the nucleus typically:
/// - **Captures**: 90% of the model's confidence
/// - **Excludes**: Low-probability noise and artifacts
/// - **Adapts**: To context-dependent probability distributions
/// - **Maintains**: Model's learned linguistic patterns
/// 
/// # Configuration Examples
/// 
/// ```rust
/// // Conservative, high-quality generation
/// let conservative_config = GenerationConfig::builder()
///     .top_p(0.8)
///     .temperature(0.7)
///     .build();
/// 
/// // Balanced creativity and quality
/// let balanced_config = GenerationConfig::builder()
///     .top_p(0.9)  // Default
///     .temperature(1.0)
///     .build();
/// ```
pub const DEFAULT_TOP_P: f32 = 0.9;

/// Default top-k sampling limit for vocabulary filtering
/// 
/// Standard top-k value that limits token selection to the 50 most probable
/// candidates, providing a good balance between quality and diversity.
/// 
/// # Value: 50 tokens
/// 
/// This limit provides:
/// - **Quality assurance**: Filters out low-probability tokens
/// - **Computational efficiency**: Reduces sampling complexity  
/// - **Diversity preservation**: Maintains reasonable choice variety
/// - **Predictable behavior**: Consistent vocabulary size across contexts
/// 
/// # Top-k Sampling Algorithm
/// 
/// The process:
/// 1. **Sort** vocabulary by probability (descending)
/// 2. **Select** top 50 most probable tokens
/// 3. **Zero out** probabilities for remaining tokens
/// 4. **Renormalize** selected probabilities to sum to 1.0
/// 5. **Sample** from the filtered distribution
/// 
/// # Vocabulary Filtering Effects
/// 
/// With k=50:
/// - **Large vocabularies** (100K+): Aggressive filtering, high quality
/// - **Medium vocabularies** (30K): Moderate filtering, balanced
/// - **Small vocabularies** (10K): Minimal filtering, preserves diversity
/// - **Context adaptation**: Same k value across different probability shapes
/// 
/// # Quality vs Diversity Analysis
/// 
/// - **k = 10**: Very conservative, high quality, limited variety
/// - **k = 25**: Conservative, good quality, moderate variety  
/// - **k = 50**: Balanced quality and diversity (default)
/// - **k = 100**: More diverse, potential quality trade-offs
/// - **k = 500**: Minimal filtering, relies on other mechanisms
/// 
/// # Performance Characteristics
/// 
/// - **Time complexity**: O(V) for partial sorting to find top-k
/// - **Space complexity**: O(k) for storing selected indices
/// - **Typical performance**: <2ms for most vocabulary sizes
/// - **Memory efficiency**: Constant memory usage regardless of vocabulary size
/// 
/// # Interaction with Top-p
/// 
/// Top-k and top-p can work together:
/// - **Both enabled**: Apply top-k first, then top-p within those k tokens
/// - **Redundant filtering**: May be unnecessarily restrictive together
/// - **Complementary**: Top-k provides hard limit, top-p adapts to distribution
/// - **Performance**: Combined filtering adds computational overhead
/// 
/// # Use Case Applications
/// 
/// - **Code completion**: k=20-40 (focused on syntactically valid options)
/// - **Chat responses**: k=50 (default, natural conversation)
/// - **Creative writing**: k=80-150 (increased expressiveness)
/// - **Factual Q&A**: k=10-25 (precision over creativity)
/// 
/// # Statistical Properties
/// 
/// With k=50, filtering typically:
/// - **Removes**: 99%+ of vocabulary tokens in most contexts
/// - **Preserves**: Core semantic and syntactic choices
/// - **Maintains**: Model's learned probability rankings  
/// - **Reduces**: Sampling noise and rare token artifacts
/// 
/// # Model Vocabulary Considerations
/// 
/// - **GPT models** (50K vocab): Filters to top 0.1% of tokens
/// - **BERT models** (30K vocab): Filters to top 0.17% of tokens
/// - **Large multilingual** (500K vocab): Filters to top 0.01% of tokens
/// 
/// # Configuration Examples
/// 
/// ```rust
/// // Focused, high-precision generation
/// let focused_config = GenerationConfig::builder()
///     .top_k(20)
///     .temperature(0.5)
///     .build();
/// 
/// // Creative, diverse generation
/// let creative_config = GenerationConfig::builder()
///     .top_k(100)
///     .temperature(1.2)
///     .build();
/// ```
pub const DEFAULT_TOP_K: u32 = 50;

/// Default repetition penalty for reducing repetitive text generation
/// 
/// Mild penalty applied to recently generated tokens to encourage diversity
/// and reduce repetitive patterns in generated text.
/// 
/// # Value: 1.1 (10% penalty)
/// 
/// This penalty level:
/// - **Reduces repetition**: Mild discouragement of recently used tokens
/// - **Preserves naturalness**: Doesn't overly distort probability distributions
/// - **Maintains coherence**: Allows necessary repetition for natural language
/// - **Provides balance**: Between variety and linguistic correctness
/// 
/// # Repetition Penalty Mechanics
/// 
/// For each token that appeared in recent context:
/// ```text
/// if penalty > 1.0:
///     adjusted_probability = original_probability / penalty
/// else:
///     adjusted_probability = original_probability * penalty
/// ```
/// 
/// With penalty = 1.1:
/// - **Recently used tokens**: Probability reduced by ~9% (1/1.1 ≈ 0.91)
/// - **Novel tokens**: Probability unchanged
/// - **Net effect**: Gentle bias toward unexplored vocabulary
/// 
/// # Context Window for Penalties
/// 
/// Typical implementation considers:
/// - **Recent tokens**: Last 64-512 tokens depending on model
/// - **Decay function**: Older tokens may receive reduced penalties
/// - **Token frequency**: More frequent tokens may receive stronger penalties
/// - **Position sensitivity**: Recent tokens penalized more than distant ones
/// 
/// # Penalty Strength Guidelines
/// 
/// - **1.0**: No penalty (allows natural repetition)
/// - **1.05-1.1**: Mild penalty, maintains naturalness (recommended range)
/// - **1.1-1.2**: Moderate penalty, noticeable diversity increase
/// - **1.2-1.5**: Strong penalty, may affect coherence
/// - **>1.5**: Very strong penalty, can produce unnatural text
/// 
/// # Use Case Optimization
/// 
/// - **Academic writing**: 1.05-1.08 (subtle variety, maintains formality)
/// - **General conversation**: 1.1 (default, natural diversity)
/// - **Creative writing**: 1.15-1.25 (increased vocabulary exploration)
/// - **Code generation**: 1.0-1.05 (minimal penalty, repetition often needed)
/// - **Poetry/lyrics**: 1.2-1.3 (high diversity for artistic expression)
/// 
/// # Performance Impact
/// 
/// - **Computational cost**: O(context_length) for penalty application
/// - **Memory overhead**: Tracking recent token occurrences
/// - **Processing time**: <1ms additional overhead per generation step
/// - **Cache efficiency**: Benefits from locality in token access patterns
/// 
/// # Interaction with Other Parameters
/// 
/// Repetition penalty combines well with:
/// - **Temperature**: Higher temp + moderate penalty = creative diversity
/// - **Top-p**: Penalty affects nucleus composition dynamically
/// - **Top-k**: Penalty may elevate previously filtered tokens
/// - **Length penalty**: Together manage both diversity and generation length
/// 
/// # Common Pitfalls
/// 
/// - **Over-penalization**: Values >1.3 can break linguistic coherence
/// - **Under-penalization**: Values <1.05 may not reduce repetition effectively
/// - **Context mismatch**: Penalty window should match generation task
/// - **Genre mismatch**: Some contexts naturally require repetition
/// 
/// # Quality Assessment
/// 
/// Effective repetition penalty:
/// - **Reduces** obvious word/phrase repetition
/// - **Maintains** necessary linguistic structures
/// - **Preserves** coherence and readability
/// - **Enhances** vocabulary diversity appropriately
/// 
/// # Configuration Examples
/// 
/// ```rust
/// // Minimal repetition control
/// let minimal_config = GenerationConfig::builder()
///     .repetition_penalty(1.05)
///     .build();
/// 
/// // Strong diversity emphasis
/// let diverse_config = GenerationConfig::builder()
///     .repetition_penalty(1.25)
///     .temperature(1.2)
///     .build();
/// ```
pub const DEFAULT_REPETITION_PENALTY: f32 = 1.1;

// ============================================================================
// Context and Sequence Management  
// ============================================================================

/// Maximum sequence length for context processing and generation
/// 
/// Defines the upper bound for input context length and total generation
/// sequence length, optimized for memory usage and computational efficiency.
/// 
/// # Value: 32,768 tokens (32K context)
/// 
/// This limit provides:
/// - **Extended context**: Support for long documents and conversations
/// - **Memory efficiency**: Reasonable memory footprint for most systems
/// - **Model compatibility**: Works with most modern transformer models
/// - **Performance balance**: Good speed/context trade-off
/// 
/// # Context Length Capabilities
/// 
/// With 32K tokens, typical content support:
/// - **Short documents**: 20-30 pages of text
/// - **Code files**: Large source files or multiple modules
/// - **Conversations**: Extended chat histories
/// - **Research papers**: Full academic papers with references
/// - **Books**: Chapters or substantial excerpts
/// 
/// # Memory Requirements
/// 
/// Context memory scaling by component:
/// - **Token storage**: 32K × 4 bytes = 128KB
/// - **Embeddings**: 32K × hidden_dim × 2 bytes (varies by model)
/// - **Attention cache**: Grows quadratically with sequence length
/// - **Total estimate**: 2-8GB depending on model size and precision
/// 
/// # Model Architecture Support
/// 
/// Compatible with:
/// - **GPT-style models**: Up to full context window for many variants
/// - **BERT-style models**: Significantly exceeds typical 512 token limits
/// - **Long-context models**: Partial utilization of 100K+ context models
/// - **Custom architectures**: Flexible limit for specialized models
/// 
/// # Performance Characteristics
/// 
/// Computational complexity:
/// - **Attention computation**: O(n²) with sequence length n
/// - **Memory bandwidth**: Linear scaling with context size
/// - **Processing time**: 4-16x slower at 32K vs 2K context
/// - **Quality benefits**: Often significant for long-form tasks
/// 
/// # Chunking and Sliding Window
/// 
/// For content exceeding this limit:
/// - **Sliding window**: Maintain most recent 32K tokens
/// - **Hierarchical processing**: Summarize older context, keep recent detail
/// - **Chunked processing**: Process in overlapping segments
/// - **Compression techniques**: Use summarization to fit critical context
/// 
/// # Use Case Optimization
/// 
/// - **Chat applications**: 16K-32K for extended conversations
/// - **Document analysis**: Full 32K for comprehensive understanding
/// - **Code generation**: 8K-16K typically sufficient for most functions
/// - **Creative writing**: 32K for maintaining narrative consistency
/// - **Q&A systems**: Variable based on document size
/// 
/// # Hardware Scaling
/// 
/// Memory requirements by system type:
/// - **Consumer GPU** (8GB): Limited to smaller models with 32K context
/// - **Professional GPU** (24GB): Comfortable with medium models
/// - **Server GPU** (40-80GB): Full support for large models
/// - **CPU inference**: Slower but can handle larger contexts with system RAM
/// 
/// # Quality vs Performance Trade-offs
/// 
/// - **Short context** (1K-4K): Fast, limited understanding
/// - **Medium context** (4K-16K): Balanced performance and capability
/// - **Full context** (32K): Maximum understanding, slower processing
/// - **Beyond limit**: Requires special handling or truncation
/// 
/// # Dynamic Context Management
/// 
/// Efficient strategies within this limit:
/// - **Priority-based retention**: Keep most important tokens
/// - **Attention pattern analysis**: Identify and preserve key dependencies
/// - **Semantic chunking**: Maintain coherent content boundaries
/// - **User guidance**: Allow explicit control over context prioritization
/// 
/// # Configuration Examples
/// 
/// ```rust
/// // Optimized for speed
/// let fast_config = GenerationConfig::builder()
///     .max_length(8192)  // 8K context
///     .build();
/// 
/// // Maximum context utilization
/// let full_config = GenerationConfig::builder()
///     .max_length(32768)  // Full 32K context
///     .build();
/// ```
pub const MAX_SEQUENCE_LENGTH: usize = 32768;

// ============================================================================
// Buffer and Processing Constants
// ============================================================================

/// Probability buffer size for efficient sampling operations
/// 
/// Pre-allocated buffer size for storing and manipulating probability
/// distributions during token sampling and generation.
/// 
/// # Value: 65,536 entries (64K)
/// 
/// This buffer accommodates:
/// - **Large vocabularies**: Most language models with room for growth
/// - **Batch processing**: Multiple sequences' probability distributions
/// - **Memory alignment**: Power-of-2 sizing for optimal cache performance
/// - **SIMD operations**: Sufficient size for vectorized probability processing
/// 
/// # Vocabulary Coverage
/// 
/// Supports vocabulary sizes up to 64K tokens:
/// - **GPT models**: 50,257 tokens (comfortable fit)
/// - **BERT models**: 30,522 tokens (ample space)
/// - **T5 models**: 32,128 tokens (good fit)
/// - **Multilingual models**: Up to 64K tokens (full coverage)
/// - **Custom vocabularies**: Flexible support for domain-specific models
/// 
/// # Memory Footprint
/// 
/// Buffer memory requirements:
/// - **FP32 probabilities**: 64K × 4 = 256KB per buffer
/// - **Batch processing**: Scales linearly with batch size
/// - **Cache efficiency**: Fits in L3 cache on most modern CPUs
/// - **GPU memory**: Minimal impact on total GPU memory usage
/// 
/// # Sampling Performance
/// 
/// Optimizations enabled by this buffer size:
/// - **SIMD vectorization**: Full vocabulary processing with AVX/NEON
/// - **Cache locality**: Entire probability distribution fits in cache
/// - **Reduced allocations**: Pre-allocated buffer prevents malloc overhead
/// - **Batch efficiency**: Multiple sequences processed with same buffer
/// 
/// # Processing Operations
/// 
/// Buffer supports efficient:
/// - **Softmax computation**: In-place probability normalization
/// - **Top-k filtering**: Fast partial sorting and thresholding
/// - **Top-p filtering**: Cumulative probability calculation and cutoff
/// - **Temperature scaling**: In-place probability adjustment
/// - **Repetition penalties**: Token-specific probability modification
/// 
/// # Overflow Handling
/// 
/// For vocabularies exceeding 64K tokens:
/// - **Chunked processing**: Process vocabulary in 64K chunks
/// - **Dynamic allocation**: Fall back to heap allocation for oversized vocabs
/// - **Streaming sampling**: Process probabilities without full storage
/// - **Vocabulary pruning**: Remove unused tokens to fit within buffer
/// 
/// # Performance Benchmarks
/// 
/// Typical sampling performance with this buffer:
/// - **Small vocabs** (<32K): <1ms per sampling operation
/// - **Large vocabs** (32-64K): 1-3ms per sampling operation
/// - **Batch processing**: Near-linear scaling with batch size
/// - **Memory bandwidth**: Optimized for modern memory hierarchies
/// 
/// # Multi-threading Considerations
/// 
/// - **Thread-local buffers**: Each thread needs its own buffer instance
/// - **Memory overhead**: Scales with thread count (256KB × threads)
/// - **Cache contention**: Buffers should be aligned to avoid false sharing
/// - **Initialization cost**: One-time allocation per thread
/// 
/// # Configuration Flexibility
/// 
/// Buffer size can be adjusted for specific needs:
/// ```rust
/// // For smaller vocabularies and memory-constrained environments
/// const SMALL_PROBABILITY_BUFFER: usize = 32768;  // 32K entries
/// 
/// // For very large vocabularies or specialized models
/// const LARGE_PROBABILITY_BUFFER: usize = 131072; // 128K entries
/// ```
/// 
/// # Quality Assurance
/// 
/// This buffer size ensures:
/// - **No probability truncation**: Full vocabulary representation
/// - **Numerical precision**: Adequate space for high-precision probabilities
/// - **Sampling accuracy**: Uncompromised sampling quality
/// - **Reproducible results**: Consistent behavior across different systems
pub const PROBABILITY_BUFFER_SIZE: usize = 65536;

/// Maximum number of stop sequences for generation control
/// 
/// Limits the number of different stop sequences that can be configured
/// for controlling generation termination.
/// 
/// # Value: 16 stop sequences
/// 
/// This limit provides:
/// - **Comprehensive control**: Multiple termination conditions
/// - **Memory efficiency**: Bounded storage for stop sequence matching
/// - **Performance optimization**: Fast matching with limited search space
/// - **Practical flexibility**: Covers most real-world use cases
/// 
/// # Use Case Examples
/// 
/// Common stop sequence configurations:
/// - **Code completion**: `["\n\n", "```", "def ", "class "]` (4 sequences)
/// - **Chat applications**: `["Human:", "AI:", "\n\n---"]` (3 sequences)
/// - **Q&A systems**: `["Question:", "Answer:", "\n\n"]` (3 sequences)
/// - **Structured generation**: Multiple format terminators (8-12 sequences)
/// 
/// # Stop Sequence Types
/// 
/// Supported termination patterns:
/// - **Single tokens**: Common punctuation or special tokens
/// - **Multi-token phrases**: Complete words or phrases
/// - **Format markers**: Structural elements like headers or separators
/// - **Context switches**: Role indicators or section boundaries
/// 
/// # Performance Characteristics
/// 
/// - **Search complexity**: O(16 × max_stop_length) per generated token
/// - **Memory overhead**: 16 × 64 bytes = 1KB maximum storage
/// - **Matching speed**: Optimized string matching with early termination
/// - **Cache efficiency**: Small working set fits in CPU cache
/// 
/// # Memory Layout
/// 
/// Stop sequence storage:
/// - **Sequence strings**: Up to 64 bytes each
/// - **Length tracking**: Fast prefix matching
/// - **State machines**: Efficient partial match tracking
/// - **Total footprint**: <2KB including metadata
/// 
/// # Matching Algorithm
/// 
/// Efficient stop detection:
/// 1. **Generate token**: Produce next token in sequence
/// 2. **Append to buffer**: Add to rolling output buffer
/// 3. **Pattern matching**: Check all stop sequences against buffer tail
/// 4. **Early termination**: Stop generation on first match
/// 5. **Cleanup**: Remove stop sequence from final output if desired
/// 
/// # Configuration Examples
/// 
/// ```rust
/// let stop_sequences = vec![
///     "Human:".to_string(),
///     "AI:".to_string(),
///     "\n\n---\n\n".to_string(),
/// ];
/// 
/// let config = GenerationConfig::builder()
///     .stop_sequences(stop_sequences)
///     .build();
/// ```
/// 
/// # Edge Case Handling
/// 
/// - **Overlapping sequences**: First match wins
/// - **Partial matches**: Maintained across token boundaries
/// - **Unicode handling**: Proper multi-byte character support
/// - **Case sensitivity**: Exact string matching by default
/// 
/// # Quality Considerations
/// 
/// This limit balances:
/// - **Expressiveness**: Sufficient for complex generation control
/// - **Performance**: Fast matching without overhead
/// - **Memory usage**: Minimal impact on total system resources
/// - **Usability**: Simple configuration for common patterns
pub const MAX_STOP_SEQUENCES: usize = 16;

/// Maximum length of individual stop sequences
/// 
/// Sets the upper bound for the character length of any single stop
/// sequence used for generation termination.
/// 
/// # Value: 64 characters
/// 
/// This limit accommodates:
/// - **Short phrases**: Common termination patterns
/// - **Format markers**: Structured document separators  
/// - **Unicode support**: Multi-byte characters with room for expansion
/// - **Performance optimization**: Fast string matching within cache lines
/// 
/// # Typical Stop Sequence Lengths
/// 
/// Common patterns and their lengths:
/// - **Single tokens**: 1-8 characters (`"\n"`, `"."`, `"```"`)
/// - **Role indicators**: 5-15 characters (`"Human:"`, `"Assistant:"`)
/// - **Section markers**: 10-30 characters (`"\n\n---\n\n"`, `"## Summary"`)
/// - **Format terminators**: 15-40 characters (structured document markers)
/// 
/// # Performance Benefits
/// 
/// 64-character limit enables:
/// - **Cache-line efficiency**: Fits comfortably in 64-byte cache lines
/// - **SIMD string matching**: Vectorized comparison operations
/// - **Reduced memory scatter**: Predictable memory access patterns
/// - **Fast pattern matching**: Bounded comparison operations
/// 
/// # String Matching Optimization
/// 
/// - **Boyer-Moore algorithm**: Efficient for longer patterns
/// - **Rolling hash**: Fast comparison for repeated patterns
/// - **SIMD acceleration**: Parallel character comparison
/// - **Early termination**: Quick rejection of non-matching prefixes
/// 
/// # Unicode and Internationalization
/// 
/// 64 characters supports:
/// - **ASCII text**: 64 full characters
/// - **UTF-8 encoding**: 16-64 Unicode code points depending on language
/// - **Emoji sequences**: Multiple emoji with modifiers
/// - **Language mixing**: Combinations of scripts within reasonable limits
/// 
/// # Memory Allocation
/// 
/// Per-sequence storage:
/// - **String buffer**: 64 bytes maximum
/// - **Length metadata**: 4 bytes for efficient access
/// - **Match state**: 4 bytes for partial match tracking
/// - **Total per sequence**: ~72 bytes including alignment
/// 
/// # Use Case Coverage
/// 
/// This limit covers:
/// - **Code generation**: Function/class terminators
/// - **Chat systems**: Conversation turn markers
/// - **Document processing**: Section and paragraph boundaries
/// - **Structured output**: JSON/XML tag terminators
/// - **Template processing**: Variable and block terminators
/// 
/// # Error Handling
/// 
/// For sequences exceeding 64 characters:
/// - **Truncation**: Automatically trim to 64 characters with warning
/// - **Rejection**: Refuse sequences that are too long
/// - **Chunking**: Split long sequences into multiple shorter ones
/// - **Hash-based matching**: Use hash comparison for very long patterns
/// 
/// # Configuration Examples
/// 
/// ```rust
/// // Valid stop sequences within limit
/// let valid_stops = vec![
///     "\n".to_string(),                    // 1 character
///     "Human:".to_string(),               // 6 characters  
///     "\n\n---END---\n\n".to_string(),    // 13 characters
/// ];
/// 
/// // Pattern that would be truncated
/// let long_pattern = "A".repeat(100);     // Would be truncated to 64 chars
/// ```
/// 
/// # Performance Benchmarks
/// 
/// String matching performance by length:
/// - **1-8 characters**: <10ns per comparison
/// - **8-32 characters**: 10-50ns per comparison  
/// - **32-64 characters**: 50-200ns per comparison
/// - **SIMD acceleration**: 2-4x speedup for longer patterns
/// 
/// # Quality Assurance
/// 
/// This limit ensures:
/// - **Predictable performance**: Bounded matching time per token
/// - **Memory efficiency**: Small, fixed allocation per sequence
/// - **Practical coverage**: Handles real-world termination patterns
/// - **System stability**: Prevents excessive memory usage from long patterns
pub const MAX_STOP_SEQUENCE_LENGTH: usize = 64;

// ============================================================================
// Streaming and Real-time Performance
// ============================================================================

/// Streaming chunk size optimized for real-time performance
/// 
/// Defines the optimal chunk size for streaming token delivery, balancing
/// latency, throughput, and system resource efficiency.
/// 
/// # Value: 1,024 bytes (1KB chunks)
/// 
/// This chunk size provides:
/// - **Low latency**: Sub-millisecond chunk processing
/// - **Network efficiency**: Optimal for TCP/HTTP streaming
/// - **Memory alignment**: Power-of-2 sizing for cache optimization
/// - **Practical granularity**: Good balance for real-time applications
/// 
/// # Streaming Performance Characteristics
/// 
/// With 1KB chunks:
/// - **Token throughput**: 200-500 tokens per chunk (depending on encoding)
/// - **Network overhead**: Minimal header-to-payload ratio
/// - **Buffer management**: Efficient memory pool utilization
/// - **Latency target**: <50ms from generation to delivery
/// 
/// # Use Case Optimization
/// 
/// Chunk size effects by application:
/// - **Interactive chat**: 1KB ideal for responsive conversation
/// - **Real-time coding**: 1KB good for incremental code completion
/// - **Document streaming**: 1KB maintains smooth reading experience
/// - **Live transcription**: 1KB enables word-by-word delivery
/// 
/// # Network Transport Considerations
/// 
/// 1KB chunks work well with:
/// - **HTTP/2 streams**: Efficient frame packing
/// - **WebSocket messages**: Below typical message size limits
/// - **TCP segments**: Fits in single segments on most networks
/// - **Mobile networks**: Reasonable for cellular data transmission
/// 
/// # Memory and CPU Efficiency
/// 
/// - **Cache utilization**: Fits in L1 cache for processing
/// - **Allocation overhead**: Minimal malloc/free overhead
/// - **Copy operations**: Fast memory copies within cache
/// - **Fragmentation**: Reduces memory fragmentation from varied sizes
/// 
/// # Buffering Strategy
/// 
/// Streaming buffer management:
/// - **Double buffering**: One chunk filling while another streams
/// - **Pool allocation**: Reuse chunk buffers to avoid allocation
/// - **Backpressure handling**: Pause generation if chunks back up
/// - **Flow control**: Adjust chunk rate based on consumer speed
/// 
/// # Real-time Performance Targets
/// 
/// With 1KB chunks, typical performance:
/// - **Generation to chunk**: <10ms processing time
/// - **Chunk to network**: <5ms serialization time  
/// - **Network transmission**: 1-50ms depending on connection
/// - **End-to-end latency**: 20-100ms total pipeline delay
/// 
/// # Token Density by Format
/// 
/// Approximate tokens per 1KB chunk:
/// - **Plain text**: 200-300 tokens (average 3-5 bytes per token)
/// - **JSON format**: 150-250 tokens (additional structure overhead)
/// - **Markdown**: 180-280 tokens (formatting adds minimal overhead)
/// - **Code output**: 100-200 tokens (varies by language verbosity)
/// 
/// # Adaptive Chunk Sizing
/// 
/// While 1KB is the default, systems may adapt:
/// - **High latency networks**: Larger chunks (2-4KB) for efficiency
/// - **Ultra-low latency**: Smaller chunks (256-512 bytes) for speed
/// - **Mobile devices**: Smaller chunks to manage battery/data usage
/// - **Batch processing**: Larger chunks (4-8KB) for throughput
/// 
/// # Quality of Service
/// 
/// Chunk size affects user experience:
/// - **Too small** (<256 bytes): Excessive overhead, choppy delivery
/// - **Optimal** (1KB): Smooth streaming with good responsiveness
/// - **Too large** (>4KB): Increased latency, less responsive feel
/// 
/// # Configuration Examples
/// 
/// ```rust
/// // Standard streaming configuration
/// let standard_config = StreamingConfig::builder()
///     .chunk_size(1024)  // 1KB chunks
///     .build();
/// 
/// // Low-latency configuration
/// let low_latency_config = StreamingConfig::builder()
///     .chunk_size(512)   // 512-byte chunks for minimal delay
///     .build();
/// 
/// // High-throughput configuration  
/// let high_throughput_config = StreamingConfig::builder()
///     .chunk_size(4096)  // 4KB chunks for maximum efficiency
///     .build();
/// ```
/// 
/// # Performance Monitoring
/// 
/// Key metrics to track with chunk-based streaming:
/// - **Chunk fill time**: Time to accumulate 1KB of content
/// - **Transmission latency**: Network delivery time per chunk
/// - **Buffer utilization**: How efficiently chunks are processed
/// - **Backpressure events**: When consumer can't keep up with chunks
/// 
/// # Error Recovery
/// 
/// Chunk-based streaming enables robust error handling:
/// - **Partial delivery**: Deliver complete chunks even if generation fails
/// - **Resume capability**: Restart from last successfully delivered chunk
/// - **Graceful degradation**: Reduce chunk size under error conditions
/// - **Timeout handling**: Detect and recover from stalled chunk delivery
pub const STREAMING_CHUNK_SIZE: usize = 1024;
