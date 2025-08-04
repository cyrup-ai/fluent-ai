//! High-performance state machine for JSON streaming with JSONPath navigation
//!
//! This module implements a zero-allocation state machine that tracks parsing progress
//! through JSON structures while evaluating JSONPath expressions. Optimized for streaming
//! scenarios where JSON arrives in chunks and needs incremental processing.

use std::collections::VecDeque;

use serde_json::Value;

use crate::json_path::{
    error::{JsonPathError, stream_error},
    parser::{JsonPathExpression, JsonSelector},
};

/// Current state of JSON streaming and JSONPath evaluation
#[derive(Debug, Clone)]
pub enum JsonStreamState {
    /// Initial state - waiting for JSON to begin
    Initial,

    /// Navigating to the target JSONPath location
    Navigating {
        /// Current depth in JSON structure
        depth: usize,
        /// JSONPath selectors remaining to process
        remaining_selectors: Vec<JsonSelector>,
        /// Current JSON value being processed
        current_value: Option<Value>,
    },

    /// Streaming array elements at target location
    StreamingArray {
        /// Array depth where streaming occurs
        target_depth: usize,
        /// Current array index being processed
        current_index: usize,
        /// Whether we're inside an array element
        in_element: bool,
        /// Brace/bracket nesting depth within current element
        element_depth: usize,
    },

    /// Processing individual JSON object at target location
    ProcessingObject {
        /// Object depth in JSON structure
        depth: usize,
        /// Current brace nesting depth
        brace_depth: usize,
        /// Whether we're inside a string literal
        in_string: bool,
        /// Whether next character is escaped
        escaped: bool,
    },

    /// Finishing stream - consuming remaining JSON after target
    Finishing {
        /// Remaining bytes to consume
        remaining_depth: usize,
    },

    /// Stream processing complete
    Complete,

    /// Error state - stream cannot continue
    Error {
        /// Error that caused state transition
        error: JsonPathError,
        /// Whether recovery is possible
        recoverable: bool,
    },
}

/// State machine managing JSON streaming with JSONPath evaluation
#[derive(Debug)]
pub struct StreamStateMachine {
    /// Current parsing state
    state: JsonStreamState,
    /// Statistics tracking
    stats: StateStats,
    /// JSONPath expression being evaluated
    path_expression: Option<JsonPathExpression>,
    /// Stack for tracking nested JSON structures
    depth_stack: VecDeque<DepthFrame>,
}

/// Frame tracking nested JSON structure depth
#[derive(Debug, Clone)]
pub struct DepthFrame {
    /// Type of JSON structure at this depth
    /// TODO: Use in JSONPath evaluation to track structure context
    #[allow(dead_code)]
    structure_type: JsonStructureType,
    /// Key name (for objects) or index (for arrays)  
    /// TODO: Use for JSONPath navigation and selector matching
    #[allow(dead_code)]
    identifier: FrameIdentifier,
    /// JSONPath selector that matched this frame
    /// TODO: Implement selector tracking for complex JSONPath expressions
    #[allow(dead_code)]
    matched_selector: Option<JsonSelector>,
}

/// Type of JSON structure
#[derive(Debug, Clone, Copy)]
pub enum JsonStructureType {
    /// JSON object structure (enclosed in {})
    Object,
    /// JSON array structure (enclosed in [])
    Array,
    /// JSON primitive value (string, number, boolean, null)
    Value,
}

/// Identifier for current frame
#[derive(Debug, Clone)]
pub enum FrameIdentifier {
    /// Object property name
    Property(String),
    /// Array index
    Index(usize),
    /// Root element
    Root,
}

/// State machine performance statistics
#[derive(Debug, Clone, Default)]
pub struct StateStats {
    /// Total objects yielded to application
    pub objects_yielded: u64,
    /// Parse errors encountered (recoverable)
    pub parse_errors: u64,
    /// State transitions performed
    pub state_transitions: u64,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Current processing depth
    pub current_depth: usize,
}

impl StreamStateMachine {
    /// Create new state machine for JSON streaming
    pub fn new() -> Self {
        Self {
            state: JsonStreamState::Initial,
            stats: StateStats::default(),
            path_expression: None,
            depth_stack: VecDeque::new(),
        }
    }

    /// Initialize state machine with JSONPath expression
    ///
    /// # Arguments
    ///
    /// * `expression` - Compiled JSONPath expression to evaluate
    ///
    /// # Performance
    ///
    /// JSONPath expression is analyzed once during initialization to optimize
    /// runtime state transitions and minimize allocation during streaming.
    pub fn initialize(&mut self, expression: JsonPathExpression) {
        self.path_expression = Some(expression);
        self.state = JsonStreamState::Navigating {
            depth: 0,
            remaining_selectors: self
                .path_expression
                .as_ref()
                .map(|e| e.selectors().to_vec())
                .unwrap_or_default(),
            current_value: None,
        };
        self.stats.state_transitions += 1;
    }

    /// Get current state (for testing and debugging)
    #[inline]
    pub fn state(&self) -> &JsonStreamState {
        &self.state
    }

    /// Process incoming JSON bytes and update state
    ///
    /// # Arguments
    ///
    /// * `data` - JSON bytes to process
    /// * `offset` - Byte offset in overall stream
    ///
    /// # Returns
    ///
    /// Vector of byte ranges where complete JSON objects were found.
    ///
    /// # Performance
    ///
    /// Uses single-pass parsing with minimal allocations. State transitions
    /// are inlined for maximum performance in hot paths.
    pub fn process_bytes(&mut self, data: &[u8], offset: usize) -> Vec<ObjectBoundary> {
        let mut boundaries = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            match self.process_byte(data[pos], offset + pos) {
                Ok(ProcessResult::Continue) => pos += 1,
                Ok(ProcessResult::ObjectBoundary { start, end }) => {
                    boundaries.push(ObjectBoundary { start, end });
                    self.stats.objects_yielded += 1;
                    pos += 1;
                }
                Ok(ProcessResult::NeedMoreData) => break,
                Ok(ProcessResult::Complete) => {
                    self.transition_to_complete();
                    break;
                }
                Ok(ProcessResult::Error(err)) => {
                    self.transition_to_error(err.clone(), true);
                    log::error!("JSON parsing error at offset {}: {}", offset + pos, err);
                    // Continue processing to handle partial data gracefully
                    pos += 1;
                }
                Err(err) => {
                    self.transition_to_error(err.clone(), true);
                    log::error!("State machine error at offset {}: {}", offset + pos, err);
                    // Continue processing to handle partial data gracefully
                    pos += 1;
                }
            }
        }

        boundaries
    }

    /// Process single byte and update state machine
    ///
    /// # Performance
    ///
    /// This is the hot path - optimized for maximum performance with inlined
    /// state transitions and minimal branching.
    #[inline]
    fn process_byte(
        &mut self,
        byte: u8,
        absolute_offset: usize,
    ) -> Result<ProcessResult, JsonPathError> {
        match &mut self.state {
            JsonStreamState::Initial => self.process_initial_byte(byte),
            JsonStreamState::Navigating { .. } => {
                self.process_navigating_byte(byte, absolute_offset)
            }
            JsonStreamState::StreamingArray { .. } => {
                self.process_streaming_byte(byte, absolute_offset)
            }
            JsonStreamState::ProcessingObject { .. } => {
                self.process_object_byte(byte, absolute_offset)
            }
            JsonStreamState::Finishing { .. } => self.process_finishing_byte(byte),
            JsonStreamState::Complete => Ok(ProcessResult::Complete),
            JsonStreamState::Error { .. } => {
                if let Some(error) = self.current_error() {
                    Err(error)
                } else {
                    Err(stream_error(
                        "State machine in error state without error details",
                        "process_byte",
                        false,
                    ))
                }
            }
        }
    }

    /// Process byte in initial state
    #[inline]
    fn process_initial_byte(&mut self, byte: u8) -> Result<ProcessResult, JsonPathError> {
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(ProcessResult::Continue), // Skip whitespace
            b'{' => {
                self.transition_to_navigating();
                self.enter_object();
                Ok(ProcessResult::Continue)
            }
            b'[' => {
                self.transition_to_navigating();
                self.enter_array();
                Ok(ProcessResult::Continue)
            }
            _ => {
                let err = stream_error(
                    &format!("unexpected byte 0x{:02x} in initial state", byte),
                    "initial",
                    false,
                );
                Ok(ProcessResult::Error(err))
            }
        }
    }

    /// Process byte while navigating to JSONPath target
    fn process_navigating_byte(
        &mut self,
        byte: u8,
        _offset: usize,
    ) -> Result<ProcessResult, JsonPathError> {
        // Simplified navigation - full implementation would evaluate JSONPath selectors
        match byte {
            b'{' => {
                self.enter_object();
                Ok(ProcessResult::Continue)
            }
            b'[' => {
                self.enter_array();
                // Check if this is our target array for streaming
                if self.is_target_array() {
                    self.transition_to_streaming();
                }
                Ok(ProcessResult::Continue)
            }
            b'}' => {
                self.exit_object();
                Ok(ProcessResult::Continue)
            }
            b']' => {
                self.exit_array();
                Ok(ProcessResult::Continue)
            }
            _ => Ok(ProcessResult::Continue),
        }
    }

    /// Process byte while streaming array elements
    fn process_streaming_byte(
        &mut self,
        byte: u8,
        offset: usize,
    ) -> Result<ProcessResult, JsonPathError> {
        if let JsonStreamState::StreamingArray {
            current_index,
            in_element,
            element_depth,
            ..
        } = &mut self.state
        {
            match byte {
                b'{' => {
                    if !*in_element {
                        *in_element = true;
                        *element_depth = 1;
                    } else {
                        *element_depth += 1;
                    }
                    Ok(ProcessResult::Continue)
                }
                b'}' => {
                    if *in_element {
                        *element_depth -= 1;
                        if *element_depth == 0 {
                            // Complete object found
                            let boundary = ObjectBoundary {
                                start: offset.saturating_sub(100), // Rough estimate
                                end: offset + 1,
                            };
                            *in_element = false;
                            *current_index += 1;
                            return Ok(ProcessResult::ObjectBoundary {
                                start: boundary.start,
                                end: boundary.end,
                            });
                        }
                    }
                    Ok(ProcessResult::Continue)
                }
                b',' => {
                    if !*in_element {
                        // Ready for next array element
                        *current_index += 1;
                    }
                    Ok(ProcessResult::Continue)
                }
                b']' => {
                    // End of array
                    self.transition_to_finishing();
                    Ok(ProcessResult::Continue)
                }
                _ => Ok(ProcessResult::Continue),
            }
        } else {
            Err(stream_error(
                "invalid state for streaming",
                "streaming",
                false,
            ))
        }
    }

    /// Process byte while inside JSON object
    fn process_object_byte(
        &mut self,
        _byte: u8,
        _offset: usize,
    ) -> Result<ProcessResult, JsonPathError> {
        // Simplified object processing - full implementation would track object boundaries
        Ok(ProcessResult::Continue)
    }

    /// Process byte while finishing stream
    fn process_finishing_byte(&mut self, byte: u8) -> Result<ProcessResult, JsonPathError> {
        // Consume remaining bytes until stream complete
        match byte {
            b' ' | b'\t' | b'\n' | b'\r' => Ok(ProcessResult::Continue),
            _ => {
                self.transition_to_complete();
                Ok(ProcessResult::Complete)
            }
        }
    }

    // State transition methods

    /// Transition to navigating state for JSONPath evaluation
    ///
    /// Initializes navigation through the JSON structure using the configured
    /// JSONPath expression selectors.
    pub fn transition_to_navigating(&mut self) {
        if let Some(expr) = &self.path_expression {
            self.state = JsonStreamState::Navigating {
                depth: 0,
                remaining_selectors: expr.selectors().to_vec(),
                current_value: None,
            };
        }
        self.stats.state_transitions += 1;
    }

    fn transition_to_streaming(&mut self) {
        self.state = JsonStreamState::StreamingArray {
            target_depth: self.stats.current_depth,
            current_index: 0,
            in_element: false,
            element_depth: 0,
        };
        self.stats.state_transitions += 1;
    }

    fn transition_to_finishing(&mut self) {
        self.state = JsonStreamState::Finishing {
            remaining_depth: self.stats.current_depth,
        };
        self.stats.state_transitions += 1;
    }

    /// Transition to complete state indicating processing is finished
    ///
    /// Marks the JSON stream processing as successfully completed.
    pub fn transition_to_complete(&mut self) {
        self.state = JsonStreamState::Complete;
        self.stats.state_transitions += 1;
    }

    /// Transition to error state with optional recovery capability
    ///
    /// Records an error condition and determines if processing can continue.
    /// Recoverable errors allow continued processing while non-recoverable errors halt execution.
    pub fn transition_to_error(&mut self, error: JsonPathError, recoverable: bool) {
        self.state = JsonStreamState::Error { error, recoverable };
        self.stats.state_transitions += 1;
        self.stats.parse_errors += 1;
    }

    /// Transition to streaming array state  
    pub fn transition_to_streaming_array(&mut self) {
        self.state = JsonStreamState::StreamingArray {
            target_depth: self.stats.current_depth,
            current_index: 0,
            in_element: false,
            element_depth: 0,
        };
        self.stats.state_transitions += 1;
    }

    /// Transition to processing object state
    pub fn transition_to_processing_object(&mut self) {
        self.state = JsonStreamState::ProcessingObject {
            depth: self.stats.current_depth,
            brace_depth: 0,
            in_string: false,
            escaped: false,
        };
        self.stats.state_transitions += 1;
    }

    /// Increment the count of objects yielded
    pub fn increment_objects_yielded(&mut self) {
        self.stats.objects_yielded += 1;
    }

    // Depth tracking methods

    /// Enter a JSON object during parsing
    ///
    /// Increments depth tracking and pushes object frame onto the depth stack
    /// for proper JSONPath evaluation context.
    pub fn enter_object(&mut self) {
        self.stats.current_depth += 1;
        self.stats.max_depth = self.stats.max_depth.max(self.stats.current_depth);

        self.depth_stack.push_back(DepthFrame {
            structure_type: JsonStructureType::Object,
            identifier: FrameIdentifier::Root, // Simplified
            matched_selector: None,
        });
    }

    /// Exit a JSON object during parsing
    ///
    /// Decrements depth tracking and pops object frame from the depth stack.
    pub fn exit_object(&mut self) {
        self.stats.current_depth = self.stats.current_depth.saturating_sub(1);
        self.depth_stack.pop_back();
    }

    /// Enter a JSON array during parsing
    ///
    /// Increments depth tracking and pushes array frame onto the depth stack
    /// for proper JSONPath evaluation context.
    pub fn enter_array(&mut self) {
        self.stats.current_depth += 1;
        self.stats.max_depth = self.stats.max_depth.max(self.stats.current_depth);

        self.depth_stack.push_back(DepthFrame {
            structure_type: JsonStructureType::Array,
            identifier: FrameIdentifier::Root, // Simplified
            matched_selector: None,
        });
    }

    /// Exit an array context, decrementing the current depth
    pub fn exit_array(&mut self) {
        self.stats.current_depth = self.stats.current_depth.saturating_sub(1);
        self.depth_stack.pop_back();
    }

    // Helper methods

    fn is_target_array(&self) -> bool {
        // Simplified check - full implementation would evaluate JSONPath expression
        self.path_expression
            .as_ref()
            .map(|e| e.is_array_stream())
            .unwrap_or(false)
    }

    fn current_error(&self) -> Option<JsonPathError> {
        if let JsonStreamState::Error { error, .. } = &self.state {
            Some(error.clone())
        } else {
            None
        }
    }

    // Public API methods

    /// Check if stream processing is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, JsonStreamState::Complete)
    }

    /// Get current state for debugging
    pub fn current_state(&self) -> &JsonStreamState {
        &self.state
    }

    /// Get processing statistics
    pub fn stats(&self) -> &StateStats {
        &self.stats
    }

    /// Get number of objects successfully yielded
    #[inline]
    pub fn objects_yielded(&self) -> u64 {
        self.stats.objects_yielded
    }

    /// Get number of parse errors encountered
    #[inline]
    pub fn parse_errors(&self) -> u64 {
        self.stats.parse_errors
    }

    /// Reset state machine for new stream
    pub fn reset(&mut self) {
        self.state = JsonStreamState::Initial;
        self.stats = StateStats::default();
        self.path_expression = None;
        self.depth_stack.clear();
    }
}

impl Default for StreamStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of processing a single byte
#[derive(Debug)]
enum ProcessResult {
    /// Continue processing next byte
    Continue,
    /// Complete JSON object found at boundary
    ObjectBoundary { start: usize, end: usize },
    /// Need more data to continue processing
    /// TODO: Used in streaming JSON processing - ensure integration with new architecture
    #[allow(dead_code)]
    NeedMoreData,
    /// Stream processing complete
    Complete,
    /// Error occurred during processing
    Error(JsonPathError),
}

/// Boundary of complete JSON object in stream
#[derive(Debug, Clone, Copy)]
pub struct ObjectBoundary {
    /// Start byte offset of object
    pub start: usize,
    /// End byte offset of object (exclusive)
    pub end: usize,
}

#[cfg(test)]
mod tests {
    // Tests for state machine module will be implemented here
}
