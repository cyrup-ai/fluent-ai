//! Collection and utility wrappers for implementing MessageChunk trait
//! Includes Vec, Option, Tuple, Result, and other utility wrappers

use std::ops::{Deref, DerefMut};

use fluent_ai_async::prelude::MessageChunk;

/// Wrapper for Result types to implement MessageChunk + Default  
#[derive(Debug, Clone)]
pub struct ResultWrapper<T, E>(pub Result<T, E>);

impl<T, E> Default for ResultWrapper<T, E>
where
    T: Default,
    E: Default,
{
    fn default() -> Self {
        Self(Ok(T::default()))
    }
}

impl<T, E> MessageChunk for ResultWrapper<T, E>
where
    T: Clone,
    E: std::fmt::Display,
{
    fn bad_chunk(_error_message: String) -> Self {
        // This requires concrete error type - will be implemented per use case
        panic!("ResultWrapper::bad_chunk requires concrete error type")
    }

    fn is_error(&self) -> bool {
        self.0.is_err()
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some("Result contains error")
        } else {
            None
        }
    }
}

/// Wrapper for Vec<SocketAddr> to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct SocketAddrListWrapper(pub Vec<std::net::SocketAddr>);

impl MessageChunk for SocketAddrListWrapper {
    fn bad_chunk(_error: String) -> Self {
        Self(Vec::new())
    }

    fn error(&self) -> Option<&str> {
        if self.0.is_empty() {
            Some("Empty socket address list")
        } else {
            None
        }
    }
}

impl From<Vec<std::net::SocketAddr>> for SocketAddrListWrapper {
    fn from(addrs: Vec<std::net::SocketAddr>) -> Self {
        Self(addrs)
    }
}

impl From<SocketAddrListWrapper> for Vec<std::net::SocketAddr> {
    fn from(wrapper: SocketAddrListWrapper) -> Self {
        wrapper.0
    }
}

impl Deref for SocketAddrListWrapper {
    type Target = Vec<std::net::SocketAddr>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SocketAddrListWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Wrapper for tuple types used in H3 connections
#[derive(Debug, Clone)]
pub struct TupleWrapper<T, U>(pub T, pub U);

impl<T, U> MessageChunk for TupleWrapper<T, U>
where
    T: Default,
    U: Default,
{
    fn bad_chunk(_error: String) -> Self {
        Self(T::default(), U::default())
    }

    fn error(&self) -> Option<&str> {
        None
    }
}

impl<T, U> From<(T, U)> for TupleWrapper<T, U> {
    fn from((t, u): (T, U)) -> Self {
        Self(t, u)
    }
}

impl<T, U> From<TupleWrapper<T, U>> for (T, U) {
    fn from(wrapper: TupleWrapper<T, U>) -> Self {
        (wrapper.0, wrapper.1)
    }
}

impl<T, U> Default for TupleWrapper<T, U>
where
    T: Default,
    U: Default,
{
    fn default() -> Self {
        Self(T::default(), U::default())
    }
}

/// Generic wrapper for Option<T> to implement MessageChunk
#[derive(Debug, Clone)]
pub struct OptionWrapper<T>(pub Option<T>);

impl<T> Default for OptionWrapper<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T> MessageChunk for OptionWrapper<T> {
    fn bad_chunk(_error: String) -> Self {
        Self(None)
    }

    fn error(&self) -> Option<&str> {
        if self.0.is_none() {
            Some("No value available")
        } else {
            None
        }
    }
}

impl<T> From<Option<T>> for OptionWrapper<T> {
    fn from(option: Option<T>) -> Self {
        Self(option)
    }
}

impl<T> From<OptionWrapper<T>> for Option<T> {
    fn from(wrapper: OptionWrapper<T>) -> Self {
        wrapper.0
    }
}

impl<T> Deref for OptionWrapper<T> {
    type Target = Option<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for OptionWrapper<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
