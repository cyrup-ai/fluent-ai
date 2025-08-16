//! Part implementation for multipart/form-data fields
//!
//! Zero-allocation part creation with comprehensive file handling and metadata management.

use std::borrow::Cow;
use std::fmt;
use std::path::Path;

use bytes::Bytes;
use mime_guess::Mime;

use super::types::{Part, PartMetadata, PartProps};
use crate::header::{HeaderMap, HeaderName, HeaderValue};
use crate::hyper::async_impl::Body;

impl Part {
    /// Makes a text field.
    pub fn text<T>(value: T) -> Part
    where
        T: Into<Cow<'static, str>>,
    {
        let body = match value.into() {
            Cow::Borrowed(slice) => Body::from(slice),
            Cow::Owned(string) => Body::from(string),
        };
        Part::new(body, None)
    }

    /// Makes a file field.
    pub fn file<T>(path: T) -> crate::Result<Part>
    where
        T: AsRef<Path>,
    {
        let path = path.as_ref();
        let file_name = path
            .file_name()
            .and_then(|filename| Some(Cow::Owned(filename.to_string_lossy().into_owned())));
        let ext = path.extension().and_then(|ext| ext.to_str());
        let mime = mime_guess::from_ext(ext.unwrap_or("")).first_or_octet_stream();
        let body = Body::from_file(path)?;

        let mut part = Part::new(body, None);
        part.meta.file_name = file_name;
        part.meta.mime = Some(mime);
        Ok(part)
    }

    /// Makes a field from arbitrary bytes.
    pub fn bytes<T>(value: T) -> Part
    where
        T: Into<Cow<'static, [u8]>>,
    {
        let body = match value.into() {
            Cow::Borrowed(slice) => Body::from(slice),
            Cow::Owned(vec) => Body::from(vec),
        };
        Part::new(body, None)
    }

    /// Makes a field from a stream of bytes.
    pub fn stream<T>(value: T) -> Part
    where
        T: Into<Body>,
    {
        Part::new(value.into(), None)
    }

    /// Makes a field from a stream of bytes with known length.
    pub fn stream_with_length<T>(value: T, length: u64) -> Part
    where
        T: Into<Body>,
    {
        Part::new(value.into(), Some(length))
    }

    /// Add a custom header.
    pub fn headers(mut self, headers: HeaderMap) -> Part {
        self.meta.headers = headers;
        self
    }

    /// Set custom mime type.
    pub fn mime_str(self, mime: &str) -> crate::Result<Part> {
        Ok(self.mime(mime.parse().map_err(crate::Error::from)?))
    }

    /// Set custom mime type.
    pub fn mime(mut self, mime: Mime) -> Part {
        self.meta.mime = Some(mime);
        self
    }

    /// Set custom filename.
    pub fn file_name<T>(mut self, filename: T) -> Part
    where
        T: Into<Cow<'static, str>>,
    {
        self.meta.file_name = Some(filename.into());
        self
    }

    /// Get the value length of this part.
    pub fn value_len(&self) -> crate::Result<u64> {
        if let Some(len) = self.body_length {
            Ok(len)
        } else {
            self.value
                .content_length()
                .ok_or_else(|| crate::Error::from("cannot determine part length"))
        }
    }

    fn new(value: Body, length: Option<u64>) -> Part {
        Part {
            meta: PartMetadata::new(),
            value,
            body_length: length,
        }
    }
}

impl PartProps for Part {
    fn value_len(&self) -> Option<u64> {
        if let Some(len) = self.body_length {
            Some(len)
        } else {
            self.value.content_length()
        }
    }

    fn metadata(&self) -> &PartMetadata {
        &self.meta
    }
}

impl fmt::Debug for Part {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = f.debug_struct("Part");
        self.meta.fmt_fields(&mut builder);
        builder.finish()
    }
}
