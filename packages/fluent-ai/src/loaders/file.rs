// ============================================================================
// File: src/loaders/file.rs                     (ASYNC-ENABLED REVISION)
// ----------------------------------------------------------------------------
// Text-file ingestion with two parallel surfaces:
//
//   • Iterator API  – zero-cost for quick scripts & sync pipelines.
//   • Async  API    – non-blocking, back-pressured `AsyncStream` that fits
//                     the Better-RIG contract (only AsyncTask / AsyncStream
//                     leave the module).
// ----------------------------------------------------------------------------

use std::{fs, path::PathBuf};

use glob::glob;
use thiserror::Error;

use crate::runtime::{AsyncStream, spawn_async};

// ring size for the producer → consumer channel
const STREAM_CAP: usize = 256;

// ---------------------------------------------------------------------------
// 0. Exhaustive error tier
// ---------------------------------------------------------------------------
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum FileLoaderError {
    #[error("glob pattern: {0}")]
    Pattern(#[from] glob::PatternError),

    #[error("glob walk: {0}")]
    Glob(#[from] glob::GlobError),

    #[error("io: {0}")]
    Io(#[from] std::io::Error)}

// ---------------------------------------------------------------------------
// 1. Helper trait – path-ish → String
// ---------------------------------------------------------------------------
trait Readable {
    fn read(self) -> Result<String, FileLoaderError>;
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError>;
}

impl Readable for PathBuf {
    #[inline(always)]
    fn read(self) -> Result<String, FileLoaderError> {
        fs::read_to_string(&self).map_err(Into::into)
    }
    #[inline(always)]
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError> {
        self.read().map(|s| (self, s))
    }
}

impl<T: Readable> Readable for Result<T, FileLoaderError> {
    #[inline(always)]
    fn read(self) -> Result<String, FileLoaderError> {
        self.and_then(Readable::read)
    }
    #[inline(always)]
    fn read_with_path(self) -> Result<(PathBuf, String), FileLoaderError> {
        self.and_then(Readable::read_with_path)
    }
}

// ---------------------------------------------------------------------------
// 2. Public façade (iterator + async flavours)
// ---------------------------------------------------------------------------
pub struct FileLoader<'a, T> {
    iterator: Box<dyn Iterator<Item = T> + 'a>}

// -------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------
impl FileLoader<'_, Result<PathBuf, FileLoaderError>> {
    pub fn with_glob(pattern: &str) -> Result<Self, FileLoaderError> {
        let paths = glob(pattern)?;
        Ok(Self {
            iterator: Box::new(paths.map(|p| p.map_err(Into::into)))})
    }

    pub fn with_dir(dir: &str) -> Result<Self, FileLoaderError> {
        Ok(Self {
            iterator: Box::new(
                fs::read_dir(dir)?
                    .filter_map(|e| e.ok())
                    .map(|e| Ok(e.path())),
            )})
    }
}

// -------------------------------------------------------------------------
// Stage 1 – SYNC read (iterator)
// ----------------------------------------------------------------------
impl<'a> FileLoader<'a, Result<PathBuf, FileLoaderError>> {
    #[inline(always)]
    pub fn read(self) -> FileLoader<'a, Result<String, FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(Readable::read))}
    }

    #[inline(always)]
    pub fn read_with_path(self) -> FileLoader<'a, Result<(PathBuf, String), FileLoaderError>> {
        FileLoader {
            iterator: Box::new(self.iterator.map(Readable::read_with_path))}
    }
}

// -------------------------------------------------------------------------
// Stage 2 – ASYNC read (stream)
// ----------------------------------------------------------------------
impl<'a> FileLoader<'a, Result<PathBuf, FileLoaderError>> {
    /// Non-blocking stream of plain file contents.
    pub fn read_async(self) -> AsyncStream<Result<String, FileLoaderError>, STREAM_CAP> {
        self.spawn_stream::<String>(|pb| pb.read())
    }

    /// Non-blocking stream of `(PathBuf, contents)`.
    pub fn read_with_path_async(
        self,
    ) -> AsyncStream<Result<(PathBuf, String), FileLoaderError>, STREAM_CAP> {
        self.spawn_stream::<(PathBuf, String)>(|pb| pb.read_with_path())
    }

    /// Internal fan-out helper: spawn one blocking read per file and push the
    /// result into a bounded ring.
    fn spawn_stream<R, F>(self, f: F) -> AsyncStream<Result<R, FileLoaderError>, STREAM_CAP>
    where
        R: Send + 'static,
        F: Fn(PathBuf) -> Result<R, FileLoaderError> + Send + Sync + 'static,
    {
        use crate::runtime::AsyncStream;
        let (tx, stream) = AsyncStream::<Result<R, FileLoaderError>, STREAM_CAP>::channel();

        for path_res in self.iterator {
            match path_res {
                Ok(path) => {
                    let tx = tx.clone();
                    // Off-thread blocking read, then push into ring
                    spawn_async(async move {
                        let res = f(path);
                        let _ = tx.try_send(res);
                    });
                }
                Err(e) => {
                    let _ = tx.try_send(Err(e));
                }
            }
        }

        // Last sender drop ⇒ consumer sees EOF
        drop(tx);
        stream
    }
}

// -------------------------------------------------------------------------
// Error-skipping helper – iterator variant
// ----------------------------------------------------------------------
impl<'a, T> FileLoader<'a, Result<T, FileLoaderError>> {
    #[inline(always)]
    pub fn ignore_errors(self) -> FileLoader<'a, T> {
        FileLoader {
            iterator: Box::new(self.iterator.filter_map(Result::ok))}
    }
}

// -------------------------------------------------------------------------
// Thin iterator wrapper
// ----------------------------------------------------------------------
pub struct LoaderIter<'a, T> {
    inner: Box<dyn Iterator<Item = T> + 'a>}

impl<'a, T> IntoIterator for FileLoader<'a, T> {
    type Item = T;
    type IntoIter = LoaderIter<'a, T>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        LoaderIter {
            inner: self.iterator}
    }
}

impl<T> Iterator for LoaderIter<'_, T> {
    type Item = T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

// ---------------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------------
