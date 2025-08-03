//! Core JsonPathDeserializer struct and main API
//!
//! Contains the primary deserializer structure with JSONPath navigation capabilities
//! and the main public interface for streaming JSON deserialization.

use serde::de::DeserializeOwned;

use super::iterator::JsonPathIterator;
use crate::json_path::{
    buffer::StreamBuffer, parser::JsonPathExpression, state_machine::StreamStateMachine,
};

/// High-performance streaming JSON deserializer with JSONPath navigation
///
/// Combines JSONPath expression evaluation with incremental JSON parsing to extract
/// individual array elements from nested JSON structures during HTTP streaming.
/// Supports full JSONPath specification including recursive descent (..) operators.
pub struct JsonPathDeserializer<'a, T> {
    /// JSONPath expression for navigation and filtering
    pub(super) path_expression: &'a JsonPathExpression,
    /// Streaming buffer for efficient byte processing
    pub(super) buffer: &'a mut StreamBuffer,
    /// State machine for tracking parsing progress
    pub(super) state: &'a mut StreamStateMachine,
    /// Current parsing depth in JSON structure
    pub(super) current_depth: usize,
    /// Whether we've reached the target array location
    pub(super) in_target_array: bool,
    /// Current object nesting level within target array
    pub(super) object_nesting: usize,
    /// Buffer for accumulating complete JSON objects
    pub(super) object_buffer: Vec<u8>,
    /// Current selector index being evaluated in the JSONPath expression
    pub(super) current_selector_index: usize,
    /// Whether we're currently in recursive descent mode
    pub(super) in_recursive_descent: bool,
    /// Stack of depth levels where recursive descent should continue searching
    pub(super) recursive_descent_stack: Vec<usize>,
    /// Path breadcrumbs for backtracking during recursive descent
    pub(super) path_breadcrumbs: Vec<String>,
    /// Current array index for slice and index evaluation
    pub(super) current_array_index: i64,
    /// Array index stack for nested array processing
    pub(super) array_index_stack: Vec<i64>,
    /// Current position in the buffer for consistent reading
    pub(super) buffer_position: usize,
    /// Performance marker
    pub(super) _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> JsonPathDeserializer<'a, T>
where
    T: DeserializeOwned,
{
    /// Create new streaming deserializer instance
    ///
    /// # Arguments
    ///
    /// * `path_expression` - Compiled JSONPath expression for navigation
    /// * `buffer` - Streaming buffer containing JSON bytes
    /// * `state` - State machine for tracking parsing progress
    #[inline]
    pub fn new(
        path_expression: &'a JsonPathExpression,
        buffer: &'a mut StreamBuffer,
        state: &'a mut StreamStateMachine,
    ) -> Self {
        let has_recursive_descent = path_expression.has_recursive_descent();
        let initial_capacity = if has_recursive_descent { 256 } else { 32 };

        Self {
            path_expression,
            buffer,
            state,
            current_depth: 0,
            in_target_array: false,
            object_nesting: 0,
            object_buffer: Vec::with_capacity(1024), // Pre-allocate 1KB
            current_selector_index: 0,
            in_recursive_descent: false,
            recursive_descent_stack: Vec::with_capacity(initial_capacity),
            path_breadcrumbs: Vec::with_capacity(initial_capacity),
            current_array_index: 0,
            array_index_stack: Vec::with_capacity(16), // Support up to 16 nested arrays
            buffer_position: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process available bytes and yield deserialized objects
    ///
    /// Incrementally parses JSON while evaluating JSONPath expressions to identify
    /// and extract individual array elements for deserialization.
    ///
    /// # Returns
    ///
    /// Iterator over successfully deserialized objects of type `T`
    ///
    /// # Performance
    ///
    /// Uses zero-copy byte processing and pre-allocated buffers for optimal performance.
    /// Inlined hot paths minimize function call overhead during streaming.
    pub fn process_available(&mut self) -> JsonPathIterator<'_, 'a, T> {
        // Reset buffer position to start of current buffer when processing new chunks
        // This ensures we don't miss any data when new chunks are appended
        self.buffer_position = 0;
        JsonPathIterator::new(self)
    }
}
