//! Optimized JSON Constraint Implementation
//!
//! This module provides the user's optimized zero-allocation JSON constraint
//! with direct tokenizer integration and finite state machine parsing.

use super::GenerationConstraint;
use crate::error::CandleResult;

/// Maximum nesting depth for JSON structures to prevent stack overflow
const MAX_DEPTH: usize = 32;

/// JSON stack item types for tracking nested structures
#[derive(Clone, Copy, PartialEq)]
pub enum JsonStackItem {
    Object,
    Array,
}

/// Number parsing states for JSON number validation
#[derive(Clone, Copy, PartialEq)]
pub enum NumberState {
    AfterSign,
    AfterZero,
    AfterIntDigit,
    AfterDot,
    AfterFracDigit,
    AfterE,
    AfterExpSign,
    AfterExpDigit,
}

/// Current JSON parsing state
#[derive(Clone, Copy, PartialEq)]
pub enum JsonCurrentState {
    ExpectValue,
    ExpectObjectKey,
    ExpectColon,
    ExpectCommaOrObjectEnd,
    ExpectCommaOrArrayEnd,
    InString { escape: bool, is_key: bool },
    InNumber { state: NumberState },
    InTrue { pos: u8 },
    InFalse { pos: u8 },
    InNull { pos: u8 },
}

/// Zero-allocation JSON parsing state
#[derive(Clone)]
pub struct JsonState {
    stack: [Option<JsonStackItem>; MAX_DEPTH],
    stack_len: usize,
    current: JsonCurrentState,
}

impl JsonState {
    fn new() -> Self {
        JsonState {
            stack: [None; MAX_DEPTH],
            stack_len: 0,
            current: JsonCurrentState::ExpectValue,
        }
    }

    fn push_stack(&mut self, item: JsonStackItem) -> CandleResult<()> {
        if self.stack_len >= MAX_DEPTH {
            return Err(crate::error::CandleError::ProcessingError(
                "JSON depth exceeds maximum",
            ));
        }
        self.stack[self.stack_len] = Some(item);
        self.stack_len += 1;
        Ok(())
    }

    fn pop_stack(&mut self) -> Option<JsonStackItem> {
        if self.stack_len == 0 {
            return None;
        }
        self.stack_len -= 1;
        self.stack[self.stack_len]
    }

    fn top_stack(&self) -> Option<JsonStackItem> {
        if self.stack_len == 0 {
            return None;
        }
        self.stack[self.stack_len - 1]
    }

    fn set_after_value(&mut self) {
        self.current = match self.top_stack() {
            Some(JsonStackItem::Object) => JsonCurrentState::ExpectCommaOrObjectEnd,
            Some(JsonStackItem::Array) => JsonCurrentState::ExpectCommaOrArrayEnd,
            None => JsonCurrentState::ExpectValue,
        };
    }

    fn is_end_char(b: u8) -> bool {
        matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b',' | b']' | b'}')
    }

    fn advance(&mut self, b: u8) -> CandleResult<()> {
        use JsonCurrentState as S;
        self.current = match self.current {
            S::ExpectValue => match b {
                b' ' | b'\t' | b'\n' | b'\r' => S::ExpectValue,
                b'{' => {
                    self.push_stack(JsonStackItem::Object)?;
                    S::ExpectObjectKey
                }
                b'[' => {
                    self.push_stack(JsonStackItem::Array)?;
                    S::ExpectValue
                }
                b'"' => S::InString {
                    escape: false,
                    is_key: false,
                },
                b't' => S::InTrue { pos: 1 },
                b'f' => S::InFalse { pos: 1 },
                b'n' => S::InNull { pos: 1 },
                b'-' => S::InNumber {
                    state: NumberState::AfterSign,
                },
                b'0' => S::InNumber {
                    state: NumberState::AfterZero,
                },
                b'1'..=b'9' => S::InNumber {
                    state: NumberState::AfterIntDigit,
                },
                _ => {
                    return Err(crate::error::CandleError::ValidationError(format!(
                        "Invalid value start: {}",
                        b as char
                    )));
                }
            },
            S::ExpectObjectKey => match b {
                b' ' | b'\t' | b'\n' | b'\r' => S::ExpectObjectKey,
                b'"' => S::InString {
                    escape: false,
                    is_key: true,
                },
                b'}' => {
                    if self.pop_stack() != Some(JsonStackItem::Object) {
                        return Err(crate::error::CandleError::ProcessingError(
                            "Mismatched object close",
                        ));
                    }
                    match self.top_stack() {
                        Some(JsonStackItem::Object) => S::ExpectCommaOrObjectEnd,
                        Some(JsonStackItem::Array) => S::ExpectCommaOrArrayEnd,
                        None => S::ExpectValue,
                    }
                }
                _ => {
                    return Err(crate::error::CandleError::ValidationError(format!(
                        "Invalid key start: {}",
                        b as char
                    )));
                }
            },
            S::ExpectColon => match b {
                b' ' | b'\t' | b'\n' | b'\r' => S::ExpectColon,
                b':' => S::ExpectValue,
                _ => {
                    return Err(crate::error::CandleError::ValidationError(format!(
                        "Expected colon, got: {}",
                        b as char
                    )));
                }
            },
            S::ExpectCommaOrObjectEnd => match b {
                b' ' | b'\t' | b'\n' | b'\r' => S::ExpectCommaOrObjectEnd,
                b',' => S::ExpectObjectKey,
                b'}' => {
                    if self.pop_stack() != Some(JsonStackItem::Object) {
                        return Err(crate::error::CandleError::ProcessingError(
                            "Mismatched object close",
                        ));
                    }
                    match self.top_stack() {
                        Some(JsonStackItem::Object) => S::ExpectCommaOrObjectEnd,
                        Some(JsonStackItem::Array) => S::ExpectCommaOrArrayEnd,
                        None => S::ExpectValue,
                    }
                }
                _ => {
                    return Err(crate::error::CandleError::ValidationError(format!(
                        "Expected comma or object end: {}",
                        b as char
                    )));
                }
            },
            S::ExpectCommaOrArrayEnd => match b {
                b' ' | b'\t' | b'\n' | b'\r' => S::ExpectCommaOrArrayEnd,
                b',' => S::ExpectValue,
                b']' => {
                    if self.pop_stack() != Some(JsonStackItem::Array) {
                        return Err(crate::error::CandleError::ProcessingError(
                            "Mismatched array close",
                        ));
                    }
                    match self.top_stack() {
                        Some(JsonStackItem::Object) => S::ExpectCommaOrObjectEnd,
                        Some(JsonStackItem::Array) => S::ExpectCommaOrArrayEnd,
                        None => S::ExpectValue,
                    }
                }
                _ => {
                    return Err(crate::error::CandleError::ValidationError(format!(
                        "Expected comma or array end: {}",
                        b as char
                    )));
                }
            },
            S::InString { escape, is_key } => {
                if escape {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u' => {
                            S::InString {
                                escape: false,
                                is_key,
                            }
                        }
                        _ => {
                            return Err(crate::error::CandleError::ValidationError(format!(
                                "Invalid escape char: {}",
                                b as char
                            )));
                        }
                    }
                } else {
                    match b {
                        b'\\' => S::InString {
                            escape: true,
                            is_key,
                        },
                        b'"' => {
                            if is_key {
                                S::ExpectColon
                            } else {
                                self.set_after_value();
                                self.current
                            }
                        }
                        b if b >= 32 && b <= 126 => S::InString {
                            escape: false,
                            is_key,
                        },
                        _ => {
                            return Err(crate::error::CandleError::ValidationError(format!(
                                "Invalid string char: {}",
                                b as char
                            )));
                        }
                    }
                }
            }
            S::InNumber { state } => match state {
                NumberState::AfterSign => match b {
                    b'0' => S::InNumber {
                        state: NumberState::AfterZero,
                    },
                    b'1'..=b'9' => S::InNumber {
                        state: NumberState::AfterIntDigit,
                    },
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(format!(
                            "Expected digit after sign: {}",
                            b as char
                        )));
                    }
                },
                NumberState::AfterZero => match b {
                    b'0'..=b'9' => {
                        return Err(crate::error::CandleError::ProcessingError(
                            "No leading zeros",
                        ));
                    }
                    b'.' => S::InNumber {
                        state: NumberState::AfterDot,
                    },
                    b'e' | b'E' => S::InNumber {
                        state: NumberState::AfterE,
                    },
                    _ if Self::is_end_char(b) => {
                        self.set_after_value();
                        self.advance(b)?;
                        self.current
                    }
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(format!(
                            "Invalid after zero: {}",
                            b as char
                        )));
                    }
                },
                NumberState::AfterIntDigit => match b {
                    b'0'..=b'9' => S::InNumber {
                        state: NumberState::AfterIntDigit,
                    },
                    b'.' => S::InNumber {
                        state: NumberState::AfterDot,
                    },
                    b'e' | b'E' => S::InNumber {
                        state: NumberState::AfterE,
                    },
                    _ if Self::is_end_char(b) => {
                        self.set_after_value();
                        self.advance(b)?;
                        self.current
                    }
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(format!(
                            "Invalid after int digit: {}",
                            b as char
                        )));
                    }
                },
                NumberState::AfterDot => match b {
                    b'0'..=b'9' => S::InNumber {
                        state: NumberState::AfterFracDigit,
                    },
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(format!(
                            "Expected digit after dot: {}",
                            b as char
                        )));
                    }
                },
                NumberState::AfterFracDigit => match b {
                    b'0'..=b'9' => S::InNumber {
                        state: NumberState::AfterFracDigit,
                    },
                    b'e' | b'E' => S::InNumber {
                        state: NumberState::AfterE,
                    },
                    _ if Self::is_end_char(b) => {
                        self.set_after_value();
                        self.advance(b)?;
                        self.current
                    }
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(format!(
                            "Invalid after frac digit: {}",
                            b as char
                        )));
                    }
                },
                NumberState::AfterE => match b {
                    b'+' | b'-' => S::InNumber {
                        state: NumberState::AfterExpSign,
                    },
                    b'0'..=b'9' => S::InNumber {
                        state: NumberState::AfterExpDigit,
                    },
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(format!(
                            "Expected exp sign or digit: {}",
                            b as char
                        )));
                    }
                },
                NumberState::AfterExpSign => match b {
                    b'0'..=b'9' => S::InNumber {
                        state: NumberState::AfterExpDigit,
                    },
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(format!(
                            "Expected exp digit: {}",
                            b as char
                        )));
                    }
                },
                NumberState::AfterExpDigit => match b {
                    b'0'..=b'9' => S::InNumber {
                        state: NumberState::AfterExpDigit,
                    },
                    _ if Self::is_end_char(b) => {
                        self.set_after_value();
                        self.advance(b)?;
                        self.current
                    }
                    _ => {
                        return Err(crate::error::CandleError::ValidationError(
                            "Invalid character after exponential digit in number".to_string(),
                        ));
                    }
                },
            },
            S::InTrue { pos } => {
                let expected = b"true"[pos as usize];
                if b == expected {
                    if pos == 3 {
                        self.set_after_value();
                        self.current
                    } else {
                        S::InTrue { pos: pos + 1 }
                    }
                } else {
                    return Err(crate::error::CandleError::ProcessingError(
                        "Invalid 'true' sequence",
                    ));
                }
            }
            S::InFalse { pos } => {
                let expected = b"false"[pos as usize];
                if b == expected {
                    if pos == 4 {
                        self.set_after_value();
                        self.current
                    } else {
                        S::InFalse { pos: pos + 1 }
                    }
                } else {
                    return Err(crate::error::CandleError::ProcessingError(
                        "Invalid 'false' sequence",
                    ));
                }
            }
            S::InNull { pos } => {
                let expected = b"null"[pos as usize];
                if b == expected {
                    if pos == 3 {
                        self.set_after_value();
                        self.current
                    } else {
                        S::InNull { pos: pos + 1 }
                    }
                } else {
                    return Err(crate::error::CandleError::ProcessingError(
                        "Invalid 'null' sequence",
                    ));
                }
            }
        };
        Ok(())
    }
}

/// Optimized JSON constraint with tokenizer integration
pub struct JsonConstraint {
    vocab_size: usize,
    token_bytes: Vec<Vec<u8>>,
    tokens_per_start_byte: [Vec<u32>; 256],
}

impl JsonConstraint {
    /// Create a new JSON constraint from tokenizer vocabulary
    ///
    /// Note: In the user's original code, this takes a tokenizer reference,
    /// but we adapt it to work with pre-computed token mappings for fluent_ai_candle
    pub fn new() -> Self {
        Self {
            vocab_size: 0,
            token_bytes: Vec::new(),
            tokens_per_start_byte: std::array::from_fn(|_| Vec::new()),
        }
    }

    /// Initialize with tokenizer vocabulary
    pub fn initialize_with_tokenizer(
        &mut self,
        tokenizer_tokens: &[(u32, Vec<u8>)],
    ) -> CandleResult<()> {
        self.vocab_size = tokenizer_tokens.len();
        self.token_bytes = vec![Vec::new(); self.vocab_size];

        // Build token->bytes mapping
        for (token_id, bytes) in tokenizer_tokens {
            if (*token_id as usize) < self.vocab_size {
                self.token_bytes[*token_id as usize] = bytes.clone();

                if !bytes.is_empty() {
                    let first_byte = bytes[0] as usize;
                    self.tokens_per_start_byte[first_byte].push(*token_id);
                }
            }
        }

        Ok(())
    }

    fn possible_next_bytes(&self, state: &JsonState) -> [bool; 256] {
        let mut poss = [false; 256];
        for b in 0u8..=255 {
            let mut s = state.clone();
            if s.advance(b).is_ok() {
                poss[b as usize] = true;
            }
        }
        poss
    }
}

impl GenerationConstraint for JsonConstraint {
    type State = JsonState;

    fn new_state(&self) -> Self::State {
        JsonState::new()
    }

    fn update(&self, state: &mut Self::State, token: u32) -> CandleResult<bool> {
        if (token as usize) >= self.token_bytes.len() {
            return Err(crate::error::CandleError::ProcessingError(
                "Token ID exceeds vocabulary size",
            ));
        }

        let bytes = &self.token_bytes[token as usize];
        for &byte in bytes {
            state.advance(byte)?;
        }
        Ok(self.is_done(state))
    }

    fn try_next(&self, state: &Self::State, token: u32) -> CandleResult<bool> {
        if (token as usize) >= self.token_bytes.len() {
            return Ok(false);
        }

        let mut s = state.clone();
        let bytes = &self.token_bytes[token as usize];
        for &byte in bytes {
            if s.advance(byte).is_err() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn is_done(&self, state: &Self::State) -> bool {
        state.stack_len == 0 && matches!(state.current, JsonCurrentState::ExpectValue)
    }

    fn get_deterministic_sequence(&self, state: &Self::State) -> CandleResult<Vec<u32>> {
        let mut seq = Vec::with_capacity(32);
        let mut current = state.clone();

        loop {
            let poss_bytes = self.possible_next_bytes(&current);
            let mut count = 0;
            let mut the_token: Option<u32> = None;

            'outer: for byte_idx in 0..256 {
                if !poss_bytes[byte_idx] {
                    continue;
                }
                for &t in &self.tokens_per_start_byte[byte_idx] {
                    let mut s = current.clone();
                    let bytes = &self.token_bytes[t as usize];
                    let mut valid = true;
                    for &b in bytes {
                        if s.advance(b).is_err() {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        if count == 0 {
                            the_token = Some(t);
                        }
                        count += 1;
                        if count > 1 {
                            break 'outer;
                        }
                    }
                }
            }

            if count == 1 {
                let t = the_token.ok_or_else(|| {
                    crate::error::CandleError::ProcessingError("No token found despite count 1")
                })?;
                seq.push(t);
                let bytes = &self.token_bytes[t as usize];
                for &b in bytes {
                    current.advance(b)?;
                }
                if self.is_done(&current) {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(seq)
    }
}

impl Default for JsonConstraint {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create a JSON constraint with tokenizer data
pub fn create_json_constraint_for_tokenizer(
    tokenizer_tokens: &[(u32, Vec<u8>)],
) -> CandleResult<JsonConstraint> {
    let mut constraint = JsonConstraint::new();
    constraint.initialize_with_tokenizer(tokenizer_tokens)?;
    Ok(constraint)
}
