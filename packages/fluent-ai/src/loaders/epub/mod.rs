// ============================================================================
// File: src/loaders/epub/mod.rs
// ----------------------------------------------------------------------------
// Public façade – re-exports and glue so downstream crates only need
// `use rig::loaders::epub::*;`
// ============================================================================

mod errors;
mod loader;
mod text_processors;

pub use errors::EpubLoaderError;
pub use loader::{EpubFileLoader, LoaderIter};
pub use text_processors::{RawTextProcessor, StripXmlProcessor, TextProcessor};

// ---------------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------------
