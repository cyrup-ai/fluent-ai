//! Zero-allocation string utilities for the build system

use std::borrow::Cow;

/// Sanitize an identifier by replacing invalid characters with underscores
pub fn sanitize_identifier(input: &str) -> Cow<'_, str> {
    if input.chars().all(|c| c.is_alphanumeric() || c == '_') 
        && input.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_') {
        Cow::Borrowed(input)
    } else {
        let sanitized: String = input.chars()
            .enumerate()
            .map(|(i, c)| {
                if i == 0 && (c.is_alphabetic() || c == '_') {
                    c
                } else if i == 0 {
                    '_'
                } else if c.is_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        Cow::Owned(sanitized)
    }
}

// Removed unused function: to_pascal_case

// Removed unused function: to_snake_case