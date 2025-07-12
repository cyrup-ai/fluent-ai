//! This module provides utility structs for loading and preprocessing files.
//!
//! File loading is now handled through the Document API which provides a unified interface
//! for loading from files, URLs, GitHub, and glob patterns.
//!
//! The [PdfFileLoader] is specifically designed to load PDF files. This loader provides
//! PDF-specific preprocessing methods for splitting the PDF into pages and keeping track
//! of the page numbers along with their contents.
//!
//! Note: The [PdfFileLoader] requires the `pdf` feature to be enabled in the `Cargo.toml` file.
//!
//! The [EpubFileLoader] is specifically designed to load EPUB files. This loader provides
//! EPUB-specific preprocessing methods for splitting the EPUB into chapters and keeping track
//! of the chapter numbers along with their contents.
//!
//! Note: The [EpubFileLoader] requires the `epub` feature to be enabled in the `Cargo.toml` file.

pub mod file;
pub mod pdf;
pub use pdf::PdfFileLoader;

pub mod epub;
pub use epub::{EpubFileLoader, RawTextProcessor, StripXmlProcessor, TextProcessor};
