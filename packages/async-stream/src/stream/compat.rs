//! Compatibility layer for AsyncStream
//!
//! Iterator implementation and other compatibility traits.

use std::sync::Arc;

use crossbeam_utils::sync::Parker;
use cyrup_sugars::prelude::*;

use super::core::Inner;
use super::receiver::AsyncStream;

/// Iterator implementation for AsyncStream
pub struct AsyncStreamIterator<T, const CAP: usize = 1024> {
    stream: AsyncStream<T, CAP>,
}

impl<T, const CAP: usize> AsyncStreamIterator<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    pub fn new(inner: Arc<Inner<T, CAP>>, parker: Parker) -> Self {
        Self {
            stream: AsyncStream::new(inner, parker),
        }
    }
}

impl<T, const CAP: usize> Iterator for AsyncStreamIterator<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // Use try_next for non-blocking iteration
        // For blocking iteration, users should use AsyncStream::next() directly
        self.stream.try_next()
    }
}

impl<T, const CAP: usize> IntoIterator for AsyncStream<T, CAP>
where
    T: MessageChunk + Send + Default + 'static,
{
    type Item = T;
    type IntoIter = AsyncStreamIterator<T, CAP>;

    fn into_iter(self) -> Self::IntoIter {
        AsyncStreamIterator { stream: self }
    }
}
