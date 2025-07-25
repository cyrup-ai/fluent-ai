//! Decoder state management for streaming UTF-8 decoding

use std::fmt;

/// State of the incremental decoder
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderState {
    /// Ready to decode new bytes
    Ready,
    /// Waiting for more bytes to complete a multi-byte sequence
    Partial { pending_bytes: Vec<u8> },
    /// Error state requiring reset
    Error { error: String },
}

impl Default for DecoderState {
    fn default() -> Self {
        DecoderState::Ready
    }
}

impl fmt::Display for DecoderState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "Ready"),
            Self::Partial { pending_bytes } => write!(
                f,
                "Partial[{} bytes: {:?}]",
                pending_bytes.len(),
                pending_bytes
            ),
            Self::Error { error } => write!(f, "Error: {}", error),
        }
    }
}

impl DecoderState {
    /// Create a new Ready state
    pub fn ready() -> Self {
        Self::Ready
    }

    /// Create a new Partial state with the given pending bytes
    pub fn partial(pending_bytes: Vec<u8>) -> Self {
        Self::Partial { pending_bytes }
    }

    /// Create a new Error state with the given error message
    pub fn error<S: Into<String>>(error: S) -> Self {
        Self::Error {
            error: error.into(),
        }
    }

    /// Check if the state is Ready
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if the state is Partial
    pub fn is_partial(&self) -> bool {
        matches!(self, Self::Partial { .. })
    }

    /// Check if the state is Error
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the pending bytes if in Partial state
    pub fn pending_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Partial { pending_bytes } => Some(pending_bytes.as_slice()),
            _ => None,
        }
    }

    /// Get the error message if in Error state
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { error } => Some(error.as_str()),
            _ => None,
        }
    }

    /// Reset the state to Ready
    pub fn reset(&mut self) {
        *self = Self::Ready;
    }

    /// Transition to a new state based on the result of an operation
    pub fn transition<T, E: ToString>(&mut self, result: Result<T, E>) -> Option<T> {
        match result {
            Ok(value) => {
                if self.is_error() {
                    *self = Self::Ready;
                }
                Some(value)
            }
            Err(e) => {
                *self = Self::Error {
                    error: e.to_string(),
                };
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_transitions() {
        let mut state = DecoderState::ready();
        assert!(state.is_ready());

        state = DecoderState::partial(vec![0xC3]);
        assert!(state.is_partial());
        assert_eq!(state.pending_bytes(), Some(&[0xC3][..]));

        state = DecoderState::error("test error");
        assert!(state.is_error());
        assert_eq!(state.error_message(), Some("test error"));

        state.reset();
        assert!(state.is_ready());
    }

    #[test]
    fn test_transition() {
        let mut state = DecoderState::ready();
        
        // Successful transition
        let result: Result<&str, &str> = Ok("success");
        assert_eq!(state.transition(result), Some("success"));
        assert!(!state.is_error());
        
        // Error transition
        let result: Result<&str, &str> = Err("failure");
        assert_eq!(state.transition(result), None);
        assert!(state.is_error());
        
        // Reset on success after error
        let result: Result<&str, &str> = Ok("recovered");
        assert_eq!(state.transition(result), Some("recovered"));
        assert!(!state.is_error());
    }
}
