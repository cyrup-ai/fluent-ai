//! Basic type wrappers for implementing MessageChunk trait
//! Includes wrappers for unit, string, bytes, and generic types

use std::ops::{Deref, DerefMut};

use bytes::Bytes;
use fluent_ai_async::prelude::MessageChunk;

/// Wrapper for unit type () to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct UnitWrapper {
    pub error_message: Option<String>,
}

impl MessageChunk for UnitWrapper {
    fn bad_chunk(error: String) -> Self {
        UnitWrapper {
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl From<()> for UnitWrapper {
    fn from(_: ()) -> Self {
        UnitWrapper {
            error_message: None,
        }
    }
}

impl From<UnitWrapper> for () {
    fn from(_: UnitWrapper) -> Self {
        ()
    }
}

/// Wrapper for String to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct StringWrapper {
    pub data: String,
    pub error_message: Option<String>,
}

impl MessageChunk for StringWrapper {
    fn bad_chunk(error: String) -> Self {
        StringWrapper {
            data: String::new(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl From<String> for StringWrapper {
    fn from(data: String) -> Self {
        StringWrapper {
            data,
            error_message: None,
        }
    }
}

impl From<StringWrapper> for String {
    fn from(wrapper: StringWrapper) -> Self {
        wrapper.data
    }
}

impl Deref for StringWrapper {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for StringWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Wrapper for Bytes to implement MessageChunk
#[derive(Debug, Clone, Default)]
pub struct BytesWrapper {
    pub data: Bytes,
    pub error_message: Option<String>,
}

impl MessageChunk for BytesWrapper {
    fn bad_chunk(error: String) -> Self {
        BytesWrapper {
            data: Bytes::new(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl BytesWrapper {
    pub fn into_bytes(self) -> Bytes {
        self.data
    }
}

impl From<Bytes> for BytesWrapper {
    fn from(data: Bytes) -> Self {
        BytesWrapper {
            data,
            error_message: None,
        }
    }
}

impl From<BytesWrapper> for Bytes {
    fn from(wrapper: BytesWrapper) -> Self {
        wrapper.data
    }
}

impl Deref for BytesWrapper {
    type Target = Bytes;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for BytesWrapper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Generic wrapper for any type to implement MessageChunk
#[derive(Debug, Clone)]
pub struct GenericWrapper<T> {
    pub data: Option<T>,
    pub error_message: Option<String>,
}

impl<T> Default for GenericWrapper<T> {
    fn default() -> Self {
        GenericWrapper {
            data: None,
            error_message: None,
        }
    }
}

impl<T> MessageChunk for GenericWrapper<T> {
    fn bad_chunk(error: String) -> Self {
        GenericWrapper {
            data: None,
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl<T> From<T> for GenericWrapper<T> {
    fn from(data: T) -> Self {
        GenericWrapper {
            data: Some(data),
            error_message: None,
        }
    }
}
