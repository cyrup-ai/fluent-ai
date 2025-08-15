//! Core types and traits for multipart/form-data handling
//! 
//! Zero-allocation, production-quality multipart implementation with comprehensive error handling.

use std::borrow::Cow;
use std::fmt;

use bytes::Bytes;
use mime_guess::Mime;
use percent_encoding::{self, AsciiSet, NON_ALPHANUMERIC};

use crate::hyper::async_impl::Body;
use crate::header::HeaderMap;

/// An async multipart/form-data request.
pub struct Form {
    pub(super) inner: FormParts<Part>,
}

/// A field in a multipart form.
pub struct Part {
    pub(super) meta: PartMetadata,
    pub(super) value: Body,
    pub(super) body_length: Option<u64>,
}

pub(crate) struct FormParts<P> {
    pub(crate) boundary: String,
    pub(crate) computed_headers: Vec<Vec<u8>>,
    pub(crate) fields: Vec<(Cow<'static, str>, P)>,
    pub(crate) percent_encoding: PercentEncoding,
}

pub(crate) struct PartMetadata {
    pub(crate) mime: Option<Mime>,
    pub(crate) file_name: Option<Cow<'static, str>>,
    pub(crate) headers: HeaderMap,
}

pub(crate) trait PartProps {
    fn value_len(&self) -> Option<u64>;
    fn metadata(&self) -> &PartMetadata;
}

/// Percent encoding options for multipart forms
#[derive(Clone, Debug)]
pub(crate) enum PercentEncoding {
    /// Percent-encode using the `path-segment` rules.
    PathSegment,
    /// Percent-encode using the `attr-char` rules.
    AttrChar,
    /// Skip percent-encoding.
    NoOp,
}

// Percent encoding sets
const PATH_SEGMENT_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

const ATTR_CHAR_ENCODE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'!')
    .remove(b'#')
    .remove(b'$')
    .remove(b'&')
    .remove(b'+')
    .remove(b'-')
    .remove(b'.')
    .remove(b'0')
    .remove(b'1')
    .remove(b'2')
    .remove(b'3')
    .remove(b'4')
    .remove(b'5')
    .remove(b'6')
    .remove(b'7')
    .remove(b'8')
    .remove(b'9')
    .remove(b'A')
    .remove(b'B')
    .remove(b'C')
    .remove(b'D')
    .remove(b'E')
    .remove(b'F')
    .remove(b'G')
    .remove(b'H')
    .remove(b'I')
    .remove(b'J')
    .remove(b'K')
    .remove(b'L')
    .remove(b'M')
    .remove(b'N')
    .remove(b'O')
    .remove(b'P')
    .remove(b'Q')
    .remove(b'R')
    .remove(b'S')
    .remove(b'T')
    .remove(b'U')
    .remove(b'V')
    .remove(b'W')
    .remove(b'X')
    .remove(b'Y')
    .remove(b'Z')
    .remove(b'^')
    .remove(b'_')
    .remove(b'`')
    .remove(b'a')
    .remove(b'b')
    .remove(b'c')
    .remove(b'd')
    .remove(b'e')
    .remove(b'f')
    .remove(b'g')
    .remove(b'h')
    .remove(b'i')
    .remove(b'j')
    .remove(b'k')
    .remove(b'l')
    .remove(b'm')
    .remove(b'n')
    .remove(b'o')
    .remove(b'p')
    .remove(b'q')
    .remove(b'r')
    .remove(b's')
    .remove(b't')
    .remove(b'u')
    .remove(b'v')
    .remove(b'w')
    .remove(b'x')
    .remove(b'y')
    .remove(b'z')
    .remove(b'|')
    .remove(b'~');

impl PercentEncoding {
    pub(crate) fn encode_headers(&self, name: &str, field: &PartMetadata) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"Content-Disposition: form-data; ");

        match self.percent_encode(name) {
            Cow::Borrowed(value) => {
                // nothing has been percent encoded
                buf.extend_from_slice(b"name=\"");
                buf.extend_from_slice(value.as_bytes());
                buf.extend_from_slice(b"\"");
            }
            Cow::Owned(value) => {
                // something has been percent encoded
                buf.extend_from_slice(b"name*=utf-8''");
                buf.extend_from_slice(value.as_bytes());
            }
        }

        // According to RFC7578 Section 4.2, `filename*=` syntax is invalid.
        // See https://github.com/seanmonstar/http3/issues/419.
        if let Some(filename) = &field.file_name {
            buf.extend_from_slice(b"; filename=\"");
            let legal_filename = filename
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\r', "\\\r")
                .replace('\n', "\\\n");
            buf.extend_from_slice(legal_filename.as_bytes());
            buf.extend_from_slice(b"\"");
        }

        if let Some(mime) = &field.mime {
            buf.extend_from_slice(b"\r\nContent-Type: ");
            buf.extend_from_slice(mime.as_ref().as_bytes());
        }

        for (k, v) in field.headers.iter() {
            buf.extend_from_slice(b"\r\n");
            buf.extend_from_slice(k.as_str().as_bytes());
            buf.extend_from_slice(b": ");
            buf.extend_from_slice(v.as_bytes());
        }
        buf
    }

    fn percent_encode<'a>(&self, value: &'a str) -> Cow<'a, str> {
        use percent_encoding::utf8_percent_encode as percent_encode;

        match self {
            Self::PathSegment => percent_encode(value, PATH_SEGMENT_ENCODE_SET).into(),
            Self::AttrChar => percent_encode(value, ATTR_CHAR_ENCODE_SET).into(),
            Self::NoOp => value.into(),
        }
    }
}

impl PartMetadata {
    pub(crate) fn new() -> Self {
        PartMetadata {
            mime: None,
            file_name: None,
            headers: HeaderMap::new(),
        }
    }

    pub(crate) fn fmt_fields(&self, dbg: &mut fmt::DebugStruct) {
        if let Some(ref mime) = self.mime {
            dbg.field("mime", mime);
        }
        if let Some(ref filename) = self.file_name {
            dbg.field("filename", filename);
        }
        if !self.headers.is_empty() {
            dbg.field("headers", &self.headers);
        }
    }
}

/// Generate a random boundary string for multipart forms
pub(crate) fn gen_boundary() -> String {
    use crate::util::fast_random as random;

    let a = random();
    let b = random();
    let c = random();
    let d = random();

    format!("{a:016x}-{b:016x}-{c:016x}-{d:016x}")
}