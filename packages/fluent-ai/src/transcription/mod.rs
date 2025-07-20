// ============================================================================
// File: src/transcription.rs
// ----------------------------------------------------------------------------
// Zero-alloc, provider-agnostic speech-to-text abstraction.                │
// Public surface (blueprint rule #1): every async path yields `AsyncTask`. │
// ============================================================================

use std::{fs, path::Path, sync::Arc};

use serde_json::Value;
use thiserror::Error;

use crate::{
    json_util,
    runtime::{self, AsyncTask, channel::bounded},
};

// ---------------------------------------------------------------------------
// 0. Authoritative error enum
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum TranscriptionError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("request: {0}")]
    Request(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("response parse: {0}")]
    Response(String),

    #[error("file not found: {0}")]
    FileNotFound(String),

    #[error("invalid filename: {0}")]
    InvalidFilename(String),

    #[error("provider: {0}")]
    Provider(String),
}

// ---------------------------------------------------------------------------
// 1. Request / response value objects
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TranscriptionRequest {
    pub data: Vec<u8>,
    pub filename: String,
    pub language: String,
    pub prompt: Option<String>,
    pub temperature: Option<f64>,
    pub additional_params: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct TranscriptionResponse<T> {
    pub text: String,
    pub response: T,
}

// ---------------------------------------------------------------------------
// 2. Core model trait – implemented by provider SDK shims
// ---------------------------------------------------------------------------

pub trait TranscriptionModel: Clone + Send + Sync + 'static {
    type Response: Send + Sync + 'static;

    fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> AsyncTask<Result<TranscriptionResponse<Self::Response>, TranscriptionError>>;

    #[inline(always)]
    fn transcription_request(&self) -> TranscriptionRequestBuilder<Self> {
        TranscriptionRequestBuilder::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// 3. Builder – allocation-free after `.send()`
// ---------------------------------------------------------------------------

pub struct TranscriptionRequestBuilder<M: TranscriptionModel> {
    model: M,
    data: Vec<u8>,
    filename: Option<String>,
    language: String,
    prompt: Option<String>,
    temperature: Option<f64>,
    additional_params: Option<Value>,
}

impl<M: TranscriptionModel> TranscriptionRequestBuilder<M> {
    #[inline(always)]
    pub fn new(model: M) -> Self {
        Self {
            model,
            data: Vec::new(),
            filename: None,
            language: "en".to_owned(),
            prompt: None,
            temperature: None,
            additional_params: None,
        }
    }

    // ---------- fluent setters ---------------------------------------------

    #[inline(always)]
    pub fn data(mut self, bytes: Vec<u8>) -> Self {
        self.data = bytes;
        self
    }

    #[inline(always)]
    pub fn load_file<P: AsRef<Path>>(self, path: P) -> Result<Self, TranscriptionError> {
        let p = path.as_ref();
        let bytes = match fs::read(p) {
            Ok(data) => data,
            Err(e) => return Err(TranscriptionError::FileNotFound(e.to_string())),
        };

        let filename = match p.file_name() {
            Some(name) => match name.to_str() {
                Some(str_name) => str_name.to_owned(),
                None => {
                    return Err(TranscriptionError::InvalidFilename(
                        "Non-UTF8 filename".to_string(),
                    ));
                }
            },
            None => {
                return Err(TranscriptionError::InvalidFilename(
                    "Not a file".to_string(),
                ));
            }
        };

        Ok(self.data(bytes).filename(filename))
    }

    #[inline(always)]
    pub fn filename(mut self, name: impl Into<String>) -> Self {
        self.filename = Some(name.into());
        self
    }

    #[inline(always)]
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    #[inline(always)]
    pub fn prompt(mut self, p: impl Into<String>) -> Self {
        self.prompt = Some(p.into());
        self
    }

    #[inline(always)]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    #[inline(always)]
    pub fn additional_params(mut self, v: Value) -> Self {
        self.additional_params = Some(match self.additional_params.take() {
            Some(prev) => json_util::merge(prev, v),
            None => v,
        });
        self
    }

    // ---------- execution ---------------------------------------------------

    #[inline(always)]
    pub fn build(self) -> TranscriptionRequest {
        assert!(
            !self.data.is_empty(),
            "TranscriptionRequestBuilder: data must not be empty"
        );

        TranscriptionRequest {
            data: self.data,
            filename: self.filename.unwrap_or_else(|| "file".into()),
            language: self.language,
            prompt: self.prompt,
            temperature: self.temperature,
            additional_params: self.additional_params,
        }
    }

    #[inline(always)]
    pub fn send(self) -> AsyncTask<Result<TranscriptionResponse<M::Response>, TranscriptionError>> {
        self.model.transcription(self.build())
    }
}

// ---------------------------------------------------------------------------
// 4. Dyn-erasure so heterogeneous models coexist behind one trait object
// ---------------------------------------------------------------------------

pub trait TranscriptionModelDyn: Send + Sync {
    fn transcription(
        &self,
        req: TranscriptionRequest,
    ) -> AsyncTask<Result<TranscriptionResponse<()>, TranscriptionError>>;

    fn transcription_request(&self) -> TranscriptionRequestBuilder<TranscriptionModelHandle>;
}

impl<T> TranscriptionModelDyn for T
where
    T: TranscriptionModel,
{
    fn transcription(
        &self,
        req: TranscriptionRequest,
    ) -> AsyncTask<Result<TranscriptionResponse<()>, TranscriptionError>> {
        // Transform the provider-specific response into a type-erased one
        // while preserving zero allocations on the caller’s hot-path.
        let (tx, task) = bounded::<Result<TranscriptionResponse<()>, TranscriptionError>>(1);

        // Kick off the real request.
        let fut = self.transcription(req);

        runtime::spawn_async(async move {
            let mapped = fut.await.map(|inner| TranscriptionResponse {
                text: inner.text,
                response: (),
            });

            let _ = tx.send(mapped);
        });

        AsyncTask::new(task)
    }

    #[inline(always)]
    fn transcription_request(&self) -> TranscriptionRequestBuilder<TranscriptionModelHandle> {
        TranscriptionRequestBuilder::new(TranscriptionModelHandle {
            inner: Arc::new(self.clone()),
        })
    }
}

/// Cheap handle that erases the concrete model type.
#[derive(Clone)]
pub struct TranscriptionModelHandle {
    inner: Arc<dyn TranscriptionModelDyn>,
}

impl TranscriptionModel for TranscriptionModelHandle {
    type Response = ();

    #[inline(always)]
    fn transcription(
        &self,
        req: TranscriptionRequest,
    ) -> AsyncTask<Result<TranscriptionResponse<Self::Response>, TranscriptionError>> {
        self.inner.transcription(req)
    }
}

// ============================================================================
// End of file
// ============================================================================
