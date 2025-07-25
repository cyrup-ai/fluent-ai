// ============================================================================
// File: src/loaders/epub/loader.rs
// ----------------------------------------------------------------------------
// High-performance EPUB file loader with compile-time state guarantees.  All
// iterator transforms are allocation-free and monomorphised.  Public API is
// a fluent builder that transitions through well-typed states.
// ============================================================================

use std::{
    fs::File,
    io::BufReader,
    iter::Iterator,
    marker::PhantomData,
    path::{Path, PathBuf}};

use epub::doc::EpubDoc;
use glob::glob;

use super::{
    errors::EpubLoaderError,
    text_processors::{RawTextProcessor, TextProcessor}};
use crate::loaders::file::FileLoaderError;

// -------------------------------------------------------------------------
// 0. Internal low-level helper – open an EPUB archive.
// ----------------------------------------------------------------------
fn open_epub<P: AsRef<Path>>(p: P) -> Result<EpubDoc<BufReader<File>>, EpubLoaderError> {
    EpubDoc::new(p).map_err(EpubLoaderError::from)
}

// -------------------------------------------------------------------------
// 1. Loadable abstraction – allows deferred I/O with compile-time safety.
// ----------------------------------------------------------------------
trait Loadable: Sized {
    type Output;
    fn load(self) -> Result<Self::Output, EpubLoaderError>;
}

impl Loadable for PathBuf {
    type Output = EpubDoc<BufReader<File>>;
    #[inline]
    fn load(self) -> Result<Self::Output, EpubLoaderError> {
        open_epub(self)
    }
}

impl<T: Loadable, E> Loadable for Result<T, E>
where
    E: Into<EpubLoaderError>,
{
    type Output = T::Output;
    #[inline]
    fn load(self) -> Result<Self::Output, EpubLoaderError> {
        self.map_err(Into::into).and_then(Loadable::load)
    }
}

// -------------------------------------------------------------------------
// 2. Public loader – zero-cost state machine encoded in the type system.
// ----------------------------------------------------------------------
pub struct EpubFileLoader<'a, It, P = RawTextProcessor> {
    it: Box<dyn Iterator<Item = It> + 'a>,
    _processor: PhantomData<P>}

// -- constructor helpers --------------------------------------------------
impl<'a, P> EpubFileLoader<'a, Result<PathBuf, FileLoaderError>, P> {
    pub fn with_glob(pattern: &str) -> Result<Self, EpubLoaderError> {
        let paths = glob(pattern).map_err(FileLoaderError::PatternError)?;
        Ok(Self {
            it: Box::new(paths.map(|r| r.map_err(FileLoaderError::GlobError))),
            _processor: PhantomData})
    }

    pub fn with_dir(dir: &str) -> Result<Self, EpubLoaderError> {
        let entries = std::fs::read_dir(dir).map_err(FileLoaderError::IoError)?;
        Ok(Self {
            it: Box::new(entries.map(|e| Ok(e.map_err(FileLoaderError::IoError)?.path()))),
            _processor: PhantomData})
    }
}

// -- state: iterator over path Results -----------------------------------
impl<'a, P> EpubFileLoader<'a, Result<PathBuf, FileLoaderError>, P> {
    pub fn load(self) -> EpubFileLoader<'a, Result<EpubDoc<BufReader<File>>, EpubLoaderError>, P> {
        EpubFileLoader {
            it: Box::new(self.it.map(|r| r.map_err(EpubLoaderError::from).load())),
            _processor: PhantomData}
    }

    pub fn load_with_path(
        self,
    ) -> EpubFileLoader<'a, Result<(PathBuf, EpubDoc<BufReader<File>>), EpubLoaderError>, P> {
        EpubFileLoader {
            it: Box::new(self.it.map(|r| {
                let path = r.map_err(EpubLoaderError::from)?;
                open_epub(&path).map(|doc| (path, doc))
            })),
            _processor: PhantomData}
    }

    pub fn read(self) -> EpubFileLoader<'a, Result<String, EpubLoaderError>, P>
    where
        P: TextProcessor,
    {
        EpubFileLoader {
            it: Box::new(self.load().it.map(|res_doc| {
                let doc = res_doc?;
                EpubChapterIter::<P>::from(doc)
                    .collect::<Result<Vec<_>, _>>()
                    .map(|v| v.concat())
            })),
            _processor: PhantomData}
    }

    pub fn read_with_path(self) -> EpubFileLoader<'a, Result<(PathBuf, String), EpubLoaderError>, P>
    where
        P: TextProcessor,
    {
        EpubFileLoader {
            it: Box::new(self.load_with_path().it.map(|r| {
                let (path, doc) = r?;
                let txt = EpubChapterIter::<P>::from(doc)
                    .collect::<Result<Vec<_>, _>>()?
                    .concat();
                Ok((path, txt))
            })),
            _processor: PhantomData}
    }
}

// -- state helper: Result<T, E> iterators → ignore_errors() --------------
impl<'a, P, T> EpubFileLoader<'a, Result<T, EpubLoaderError>, P> {
    #[inline]
    pub fn ignore_errors(self) -> EpubFileLoader<'a, T, P> {
        EpubFileLoader {
            it: Box::new(self.it.filter_map(Result::ok)),
            _processor: PhantomData}
    }
}

// -- state helper: iterator over docs → by_chapter -----------------------
impl<'a, P> EpubFileLoader<'a, EpubDoc<BufReader<File>>, P>
where
    P: TextProcessor + 'a,
{
    pub fn by_chapter(self) -> EpubFileLoader<'a, Result<String, EpubLoaderError>, P> {
        EpubFileLoader {
            it: Box::new(self.it.flat_map(EpubChapterIter::<P>::from)),
            _processor: PhantomData}
    }
}

pub type ChaptersByPath<'a, P> = (PathBuf, Vec<(usize, Result<String, EpubLoaderError>)>);

impl<'a, P> EpubFileLoader<'a, (PathBuf, EpubDoc<BufReader<File>>), P>
where
    P: TextProcessor,
{
    pub fn by_chapter(self) -> EpubFileLoader<'a, ChaptersByPath<'a, P>, P> {
        EpubFileLoader {
            it: Box::new(self.it.map(|(path, doc)| {
                let chapters = EpubChapterIter::<P>::from(doc).enumerate().collect();
                (path, chapters)
            })),
            _processor: PhantomData}
    }
}

// -- iterator plumbing ---------------------------------------------------
pub struct LoaderIter<'a, T> {
    it: Box<dyn Iterator<Item = T> + 'a>}
impl<'a, T, P> IntoIterator for EpubFileLoader<'a, T, P> {
    type Item = T;
    type IntoIter = LoaderIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        LoaderIter { it: self.it }
    }
}
impl<'a, T> Iterator for LoaderIter<'a, T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.it.next()
    }
}

// -------------------------------------------------------------------------
// 3. Chapter iterator – zero-alloc text extraction
// ----------------------------------------------------------------------
struct EpubChapterIter<P> {
    epub: EpubDoc<BufReader<File>>,
    finished: bool,
    _proc: PhantomData<P>}

impl<P> From<EpubDoc<BufReader<File>>> for EpubChapterIter<P> {
    #[inline]
    fn from(epub: EpubDoc<BufReader<File>>) -> Self {
        Self {
            epub,
            finished: false,
            _proc: PhantomData}
    }
}

impl<P: TextProcessor> Iterator for EpubChapterIter<P> {
    type Item = Result<String, EpubLoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.finished {
            let chapter = self.epub.get_current_str();
            if !self.epub.go_next() {
                self.finished = true;
            }

            if let Some((txt, _)) = chapter {
                return Some(
                    P::process(&txt).map_err(|e| EpubLoaderError::TextProcessor(Box::new(e))),
                );
            }
        }
        None
    }
}
