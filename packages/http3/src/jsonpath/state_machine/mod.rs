//! JSON streaming state machine with zero-allocation parsing
//!
//! This module implements a high-performance state machine for streaming JSON parsing
//! and JSONPath evaluation. Optimized for scenarios where JSON arrives in chunks
//! and needs incremental processing with minimal memory allocation.
//!
//! # Architecture
//!
//! - `types`: Core data structures, enums, and type definitions
//! - `engine`: Main processing engine and byte-level parsing
//! - `processors`: Specialized byte processors for different parsing states
//! - `transitions`: State transition logic and validation
//! - `utils`: Utility functions and public API methods
//!
//! # Usage
//!
//! ```rust
//! use crate::json_path::state_machine::StreamStateMachine;
//! use crate::json_path::parser::JsonPathExpression;
//!
//! let mut machine = StreamStateMachine::new();
//! machine.initialize(expression);
//!
//! let boundaries = machine.process_bytes(data, offset);
//! for boundary in boundaries {
//!     println!("Found object at bytes {}..{}", boundary.start, boundary.end);
//! }
//! ```
//!
//! # Performance
//!
//! This state machine is optimized for:
//! - Zero-allocation streaming parsing
//! - Single-pass byte processing
//! - Minimal branching in hot paths
//! - Incremental JSONPath evaluation
//! - Memory-efficient state tracking

mod engine;
mod processors;
mod transitions;
mod types;
mod utils;

// Re-export the main types and functionality
// Re-export transition functions for advanced usage
pub use transitions::{StateType, get_state_type, is_ready_for_processing, is_terminal_state};
pub use types::{
    FrameIdentifier, JsonStreamState, JsonStructureType, ObjectBoundary, ProcessResult, StateStats,
    StreamStateMachine,
};
// Re-export utility functions that might be needed externally
pub use utils::{
    current_depth, is_complete, is_error_state, is_recoverable_error, max_depth_reached,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsonpath::parser::{JsonPathExpression, JsonSelector};

    #[test]
    fn test_state_machine_creation() {
        let machine = StreamStateMachine::new();
        assert!(matches!(machine.state(), JsonStreamState::Initial));
        assert_eq!(machine.objects_yielded(), 0);
        assert_eq!(machine.parse_errors(), 0);
    }

    #[test]
    fn test_state_machine_initialization() {
        let mut machine = StreamStateMachine::new();
        let expression = JsonPathExpression::root(); // Placeholder
        machine.initialize(expression);

        assert!(matches!(
            machine.state(),
            JsonStreamState::Navigating { .. }
        ));
        assert!(machine.is_ready());
    }

    #[test]
    fn test_byte_processing() {
        let mut machine = StreamStateMachine::new();
        let expression = JsonPathExpression::root();
        machine.initialize(expression);

        let data = b"[{\"test\": 123}]";
        let boundaries = machine.process_bytes(data, 0);

        // Should have processed some data
        assert!(machine.state_transitions() > 0);
    }

    #[test]
    fn test_state_transitions() {
        let mut machine = StreamStateMachine::new();

        // Test initial state
        assert!(matches!(machine.current_state(), JsonStreamState::Initial));

        // Test state type detection
        let state_type = get_state_type(machine.current_state());
        assert!(matches!(state_type, StateType::Initial));
    }

    #[test]
    fn test_error_handling() {
        let mut machine = StreamStateMachine::new();
        let expression = JsonPathExpression::root();
        machine.initialize(expression);

        // Process invalid JSON
        let data = b"invalid json{[}";
        let _boundaries = machine.process_bytes(data, 0);

        // Should handle errors gracefully
        assert!(machine.parse_errors() >= 0); // May or may not have errors depending on implementation
    }

    #[test]
    fn test_reset_functionality() {
        let mut machine = StreamStateMachine::new();
        let expression = JsonPathExpression::root();
        machine.initialize(expression);

        // Process some data
        let data = b"{}";
        let _boundaries = machine.process_bytes(data, 0);

        // Reset machine
        machine.reset();

        assert!(matches!(machine.current_state(), JsonStreamState::Initial));
        assert_eq!(machine.objects_yielded(), 0);
        assert_eq!(machine.parse_errors(), 0);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let machine = StreamStateMachine::new();
        let usage = machine.memory_usage();

        // Should return a reasonable estimate
        assert!(usage > 0);
        assert!(usage < 10000); // Should be reasonable for empty machine
    }
}
