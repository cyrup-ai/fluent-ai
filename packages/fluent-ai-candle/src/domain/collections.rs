//! Candle Collections - Standalone collection types

use serde::{Deserialize, Serialize};

/// Candle-specific collection type supporting zero, one, or many items
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CandleZeroOneOrMany<T> {
    /// No items
    None,
    /// Exactly one item
    One(T),
    /// Multiple items
    Many(Vec<T>),
}

impl<T> Default for CandleZeroOneOrMany<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> CandleZeroOneOrMany<T> {
    /// Create a new collection with one item
    pub fn one(item: T) -> Self {
        Self::One(item)
    }

    /// Create a new collection with many items
    pub fn many(items: Vec<T>) -> Self {
        if items.is_empty() {
            Self::None
        } else if items.len() == 1 {
            // Safe: we verified len() == 1 above
            match items.into_iter().next() {
                Some(item) => Self::One(item),
                None => Self::None, // Fallback in impossible case
            }
        } else {
            Self::Many(items)
        }
    }

    /// Push an item, returning a new collection
    pub fn with_pushed(self, item: T) -> Self {
        match self {
            Self::None => Self::One(item),
            Self::One(existing) => Self::Many(vec![existing, item]),
            Self::Many(mut items) => {
                items.push(item);
                Self::Many(items)
            }
        }
    }

    /// Convert to Vec, always returning a Vec regardless of variant
    pub fn to_vec(self) -> Vec<T> {
        match self {
            Self::None => vec![],
            Self::One(item) => vec![item],
            Self::Many(items) => items,
        }
    }

    /// Get the length/count of items
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Many(items) => items.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Iterate over the items
    pub fn iter(&self) -> CandleZeroOneOrManyIter<&T> {
        match self {
            Self::None => CandleZeroOneOrManyIter::None,
            Self::One(item) => CandleZeroOneOrManyIter::One(Some(item)),
            Self::Many(items) => CandleZeroOneOrManyIter::Many(items.iter()),
        }
    }
}

impl<T> IntoIterator for CandleZeroOneOrMany<T> {
    type Item = T;
    type IntoIter = CandleZeroOneOrManyIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::None => CandleZeroOneOrManyIntoIter::None,
            Self::One(item) => CandleZeroOneOrManyIntoIter::One(Some(item)),
            Self::Many(items) => CandleZeroOneOrManyIntoIter::Many(items.into_iter()),
        }
    }
}

/// Iterator for CandleZeroOneOrMany references
pub enum CandleZeroOneOrManyIter<T> {
    None,
    One(Option<T>),
    Many(std::slice::Iter<'static, T>),
}

impl<T> Iterator for CandleZeroOneOrManyIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::None => None,
            Self::One(ref mut item) => item.take(),
            Self::Many(ref mut iter) => iter.next().copied(),
        }
    }
}

/// IntoIterator for CandleZeroOneOrMany
pub enum CandleZeroOneOrManyIntoIter<T> {
    None,
    One(Option<T>),
    Many(std::vec::IntoIter<T>),
}

impl<T> Iterator for CandleZeroOneOrManyIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::None => None,
            Self::One(ref mut item) => item.take(),
            Self::Many(ref mut iter) => iter.next(),
        }
    }
}

impl<T> From<T> for CandleZeroOneOrMany<T> {
    fn from(item: T) -> Self {
        Self::One(item)
    }
}

impl<T> From<Vec<T>> for CandleZeroOneOrMany<T> {
    fn from(items: Vec<T>) -> Self {
        Self::many(items)
    }
}

impl<T> From<Option<T>> for CandleZeroOneOrMany<T> {
    fn from(option: Option<T>) -> Self {
        match option {
            Some(item) => Self::One(item),
            None => Self::None,
        }
    }
}