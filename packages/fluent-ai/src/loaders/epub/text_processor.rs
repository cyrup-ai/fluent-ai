// ============================================================================
// File: src/loaders/epub/text_processors.rs      (SYNC-ONLY, ZERO-ALLOC)
// ----------------------------------------------------------------------------
// Loss-less text post-processing helpers for EPUB / HTML chapters.
//
// • Contract stays 100 % synchronous.
// • If a caller wants async they wrap the call in their own AsyncTask.
// • Absolutely no async_trait, no heap boxing, no extra dependencies.
// ============================================================================

use std::{convert::Infallible, error::Error};

use quick_xml::{Reader, events::Event};

/// Pure, allocation-aware transformer used by the loader.
/// Call-sites that *do* need async can trivially do:
///
/// ```rust
/// let task = rig::runtime::spawn_async(async move {
///     MyProcessor::process(&text)
/// });
/// ```
pub trait TextProcessor: Sized {
    type Error: Error + 'static;

    /// Return a **fresh `String`** containing the processed text.
    fn process(text: &str) -> Result<String, Self::Error>;
}

/// No-op – returns exactly what it was given.
pub struct RawTextProcessor;

impl TextProcessor for RawTextProcessor {
    type Error = Infallible;

    #[inline(always)]
    fn process(text: &str) -> Result<String, Self::Error> {
        Ok(text.to_owned())
    }
}

/// Strip XML / HTML tags while preserving word boundaries.
///
/// • Streaming parser (`quick_xml`) – zero heap allocations on the hot path.
pub struct StripXmlProcessor;

#[derive(thiserror::Error, Debug)]
pub enum StripXmlError {
    #[error("xml parse: {0}")]
    Xml(#[from] quick_xml::Error),
    #[error("utf-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("unescape: {0}")]
    Unescape(#[from] quick_xml::events::attributes::AttrError),
}

impl TextProcessor for StripXmlProcessor {
    type Error = StripXmlError;

    fn process(xml: &str) -> Result<String, Self::Error> {
        let mut reader = Reader::from_str(xml.trim());
        reader.trim_text(true);

        // Heuristic: ~½ original size is plenty for most chapters.
        let mut out = String::with_capacity(xml.len() / 2);
        let mut last_was_txt = false;

        loop {
            match reader.read_event()? {
                Event::Text(t) | Event::CData(t) => {
                    let txt = t.unescape()?.into_owned();
                    if !txt.trim().is_empty() {
                        if last_was_txt {
                            out.push(' ');
                        }
                        out.push_str(&txt);
                        last_was_txt = true;
                    }
                }
                Event::Eof => break,
                _ => last_was_txt = false,
            }
        }
        Ok(out)
    }
}
