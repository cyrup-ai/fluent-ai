//! ZeroOneOrMany utility type

use serde::{Deserialize, Serialize};

/// Represents zero, one, or many values of type T
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ZeroOneOrMany<T> {
    /// No values
    None,
    /// Single value
    One(T),
    /// Multiple values
    Many(Vec<T>),
}

impl<T> Default for ZeroOneOrMany<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> ZeroOneOrMany<T> {
    /// Create a new empty ZeroOneOrMany
    pub fn none() -> Self {
        Self::None
    }

    /// Create a new ZeroOneOrMany with a single value
    pub fn one(value: T) -> Self {
        Self::One(value)
    }

    /// Create a new ZeroOneOrMany with multiple values
    pub fn many(values: Vec<T>) -> Self {
        Self::Many(values)
    }

    /// Check if this contains no values
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Get the length of values
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Many(vec) => vec.len(),
        }
    }

    /// Convert to a Vec
    pub fn to_vec(self) -> Vec<T> {
        match self {
            Self::None => Vec::new(),
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }

    /// Get an iterator over the values
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        match self {
            Self::None => Box::new(std::iter::empty()) as Box<dyn Iterator<Item = &T>>,
            Self::One(value) => Box::new(std::iter::once(value)) as Box<dyn Iterator<Item = &T>>,
            Self::Many(values) => Box::new(values.iter()) as Box<dyn Iterator<Item = &T>>,
        }
    }
}

impl<T> From<T> for ZeroOneOrMany<T> {
    fn from(value: T) -> Self {
        Self::One(value)
    }
}

impl<T> From<Vec<T>> for ZeroOneOrMany<T> {
    fn from(values: Vec<T>) -> Self {
        if values.is_empty() {
            Self::None
        } else {
            Self::Many(values)
        }
    }
}

impl<T> From<Option<T>> for ZeroOneOrMany<T> {
    fn from(option: Option<T>) -> Self {
        match option {
            Some(value) => Self::One(value),
            None => Self::None,
        }
    }
}