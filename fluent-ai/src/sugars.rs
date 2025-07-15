//! Utility types and traits previously provided by the sugars crate

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use crate::async_task::error_handlers::BadTraitImpl;

/// A type that can hold zero, one, or many items
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroOneOrMany<T> {
    None,
    One(T),
    Many(Vec<T>),
}

impl<T> ZeroOneOrMany<T> {
    /// Create a new ZeroOneOrMany with a single item
    pub fn one(item: T) -> Self {
        Self::One(item)
    }

    /// Create a new ZeroOneOrMany with multiple items
    pub fn many(items: Vec<T>) -> Self {
        if items.is_empty() {
            Self::None
        } else if items.len() == 1 {
            if let Some(item) = items.into_iter().next() {
                Self::One(item)
            } else {
                Self::None
            }
        } else {
            Self::Many(items)
        }
    }

    /// Create a new ZeroOneOrMany from a vector
    pub fn from_vec(items: Vec<T>) -> Self {
        Self::many(items)
    }

    /// Add an item to this ZeroOneOrMany
    pub fn push(&mut self, item: T) {
        match self {
            Self::None => *self = Self::One(item),
            Self::One(_) => {
                let old_item = std::mem::replace(self, Self::None);
                if let Self::One(old_item) = old_item {
                    *self = Self::Many(vec![old_item, item]);
                }
            }
            Self::Many(items) => items.push(item),
        }
    }

    /// Check if this contains any items
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Get the number of items
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::One(_) => 1,
            Self::Many(items) => items.len(),
        }
    }

    /// Iterator over items
    pub fn iter(&self) -> ZeroOneOrManyIter<T> {
        match self {
            Self::None => ZeroOneOrManyIter::None,
            Self::One(item) => ZeroOneOrManyIter::One(std::iter::once(item)),
            Self::Many(items) => ZeroOneOrManyIter::Many(items.iter()),
        }
    }
}

impl<T> Default for ZeroOneOrMany<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> From<T> for ZeroOneOrMany<T> {
    fn from(item: T) -> Self {
        Self::One(item)
    }
}

impl<T> From<Vec<T>> for ZeroOneOrMany<T> {
    fn from(items: Vec<T>) -> Self {
        Self::many(items)
    }
}

impl<T> BadTraitImpl for ZeroOneOrMany<T> 
where
    T: Send + Sync + std::fmt::Debug + Clone,
{
    fn bad_impl(error: String) -> Self {
        // Return None as the default "bad" implementation for error states
        eprintln!("ZeroOneOrMany BadTraitImpl: {}", error);
        ZeroOneOrMany::None
    }
}

pub enum ZeroOneOrManyIter<'a, T> {
    None,
    One(std::iter::Once<&'a T>),
    Many(std::slice::Iter<'a, T>),
}

impl<'a, T> Iterator for ZeroOneOrManyIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::None => None,
            Self::One(iter) => iter.next(),
            Self::Many(iter) => iter.next(),
        }
    }
}

impl<T> IntoIterator for ZeroOneOrMany<T> {
    type Item = T;
    type IntoIter = ZeroOneOrManyIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::None => ZeroOneOrManyIntoIter::None,
            Self::One(item) => ZeroOneOrManyIntoIter::One(std::iter::once(item)),
            Self::Many(items) => ZeroOneOrManyIntoIter::Many(items.into_iter()),
        }
    }
}

pub enum ZeroOneOrManyIntoIter<T> {
    None,
    One(std::iter::Once<T>),
    Many(std::vec::IntoIter<T>),
}

impl<T> Iterator for ZeroOneOrManyIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::None => None,
            Self::One(iter) => iter.next(),
            Self::Many(iter) => iter.next(),
        }
    }
}

/// A type representing a size in bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteSize(pub u64);

impl ByteSize {
    /// Create a new ByteSize from bytes
    pub fn bytes(n: u64) -> Self {
        Self(n)
    }

    /// Create a new ByteSize from kilobytes
    pub fn kilobytes(n: u64) -> Self {
        Self(n * 1024)
    }

    /// Create a new ByteSize from megabytes
    pub fn megabytes(n: u64) -> Self {
        Self(n * 1024 * 1024)
    }

    /// Get the size in bytes
    pub fn as_bytes(&self) -> u64 {
        self.0
    }
}

impl From<u64> for ByteSize {
    fn from(bytes: u64) -> Self {
        Self(bytes)
    }
}

/// Extension trait for creating ByteSize from numbers
pub trait ByteSizeExt {
    fn bytes(self) -> ByteSize;
    fn kilobytes(self) -> ByteSize;
    fn megabytes(self) -> ByteSize;
}

impl ByteSizeExt for u64 {
    fn bytes(self) -> ByteSize {
        ByteSize::bytes(self)
    }

    fn kilobytes(self) -> ByteSize {
        ByteSize::kilobytes(self)
    }

    fn megabytes(self) -> ByteSize {
        ByteSize::megabytes(self)
    }
}

impl ByteSizeExt for u32 {
    fn bytes(self) -> ByteSize {
        ByteSize::bytes(self as u64)
    }

    fn kilobytes(self) -> ByteSize {
        ByteSize::kilobytes(self as u64)
    }

    fn megabytes(self) -> ByteSize {
        ByteSize::megabytes(self as u64)
    }
}

impl ByteSizeExt for usize {
    fn bytes(self) -> ByteSize {
        ByteSize::bytes(self as u64)
    }

    fn kilobytes(self) -> ByteSize {
        ByteSize::kilobytes(self as u64)
    }

    fn megabytes(self) -> ByteSize {
        ByteSize::megabytes(self as u64)
    }
}

impl ByteSizeExt for i32 {
    fn bytes(self) -> ByteSize {
        ByteSize::bytes(self as u64)
    }

    fn kilobytes(self) -> ByteSize {
        ByteSize::kilobytes(self as u64)
    }

    fn megabytes(self) -> ByteSize {
        ByteSize::megabytes(self as u64)
    }
}

/// Extension trait for futures
pub trait FutureExt: Future {
    /// Map the output of this future
    fn map<U, F>(self, f: F) -> Map<Self, F>
    where
        F: FnOnce(Self::Output) -> U,
        Self: Sized,
    {
        Map::new(self, f)
    }
}

impl<T: Future> FutureExt for T {}

/// Extension trait for streams
pub trait StreamExt {
    type Item;

    /// Map each item in the stream
    fn map<U, F>(self, f: F) -> StreamMap<Self, F>
    where
        F: FnMut(Self::Item) -> U,
        Self: Sized;

    /// Filter items in the stream
    fn filter<F>(self, f: F) -> StreamFilter<Self, F>
    where
        F: FnMut(&Self::Item) -> bool,
        Self: Sized;
}

// Future combinator for map
pub struct Map<Fut, F> {
    future: Option<Fut>,
    f: Option<F>,
}

impl<Fut, F> Map<Fut, F> {
    fn new(future: Fut, f: F) -> Self {
        Self {
            future: Some(future),
            f: Some(f),
        }
    }
}

impl<Fut, F, U> Future for Map<Fut, F>
where
    Fut: Future,
    F: FnOnce(Fut::Output) -> U,
{
    type Output = U;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };

        if let Some(future) = this.future.as_mut() {
            let future = unsafe { Pin::new_unchecked(future) };
            match future.poll(cx) {
                Poll::Ready(output) => {
                    this.future = None;
                    if let Some(f) = this.f.take() {
                        Poll::Ready(f(output))
                    } else {
                        panic!("Map future f function already taken")
                    }
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            panic!("Map future polled after completion")
        }
    }
}

// Stream combinators (basic implementations)
pub struct StreamMap<S, F> {
    stream: S,
    f: F,
}

pub struct StreamFilter<S, F> {
    stream: S,
    f: F,
}
