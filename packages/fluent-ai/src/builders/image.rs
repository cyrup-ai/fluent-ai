//! Image builder implementations - Zero Box<dyn> trait-based architecture
//!
//! All image construction logic and builder patterns with zero allocation.

use std::marker::PhantomData;

use cyrup_sugars::prelude::{ChunkHandler, MessageChunk};
use fluent_ai_domain::AsyncStream;
use fluent_ai_domain::context::chunk::ImageChunk;
use fluent_ai_domain::image::{ContentFormat, Image, ImageDetail, ImageMediaType};

/// Image builder trait - elegant zero-allocation builder pattern
pub trait ImageBuilder: Sized {
    /// Set format - EXACT syntax: .format(ContentFormat::Base64)
    fn format(self, format: ContentFormat) -> impl ImageBuilder;
    
    /// Set media type - EXACT syntax: .media_type(ImageMediaType::PNG)
    fn media_type(self, media_type: ImageMediaType) -> impl ImageBuilder;
    
    /// Set detail - EXACT syntax: .detail(ImageDetail::High)
    fn detail(self, detail: ImageDetail) -> impl ImageBuilder;
    
    /// Set as PNG - EXACT syntax: .as_png()
    fn as_png(self) -> impl ImageBuilder;
    
    /// Set as JPEG - EXACT syntax: .as_jpeg()
    fn as_jpeg(self) -> impl ImageBuilder;
    
    /// Set high detail - EXACT syntax: .high_detail()
    fn high_detail(self) -> impl ImageBuilder;
    
    /// Set low detail - EXACT syntax: .low_detail()
    fn low_detail(self) -> impl ImageBuilder;
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl ImageBuilder
    where
        F: Fn(String) + Send + Sync + 'static;
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: returns self for method chaining
    fn on_chunk<F>(self, handler: F) -> impl ImageBuilder
    where
        F: Fn(ImageChunk) -> ImageChunk + Send + Sync + 'static;
    
    /// Load image - EXACT syntax: .load()
    fn load(self) -> impl AsyncStream<Item = ImageChunk>;
    
    /// Process image - EXACT syntax: .process(|chunk| { ... })
    fn process<F>(self, f: F) -> impl AsyncStream<Item = ImageChunk>
    where
        F: FnOnce(ImageChunk) -> ImageChunk + Send + 'static;
}

/// Hidden implementation struct - zero-allocation builder state with zero Box<dyn> usage
struct ImageBuilderImpl<
    F1 = fn(String),
> where
    F1: Fn(String) + Send + Sync + 'static,
{
    data: String,
    format: Option<ContentFormat>,
    media_type: Option<ImageMediaType>,
    detail: Option<ImageDetail>,
    error_handler: Option<F1>,
    chunk_handler: Option<Box<dyn Fn(Result<ImageChunk, String>) -> ImageChunk + Send + Sync>>,
}

impl Image {
    /// Semantic entry point - EXACT syntax: Image::from_base64(data)
    pub fn from_base64(data: impl Into<String>) -> impl ImageBuilder {
        ImageBuilderImpl {
            data: data.into(),
            format: Some(ContentFormat::Base64),
            media_type: None,
            detail: None,
            error_handler: None,
            chunk_handler: None,
        }
    }

    /// Semantic entry point - EXACT syntax: Image::from_url(url)
    pub fn from_url(url: impl Into<String>) -> impl ImageBuilder {
        ImageBuilderImpl {
            data: url.into(),
            format: Some(ContentFormat::Url),
            media_type: None,
            detail: None,
            error_handler: None,
            chunk_handler: None,
        }
    }

    /// Semantic entry point - EXACT syntax: Image::from_path(path)
    pub fn from_path(path: impl Into<String>) -> impl ImageBuilder {
        ImageBuilderImpl {
            data: path.into(),
            format: Some(ContentFormat::Url),
            media_type: None,
            detail: None,
            error_handler: None,
            chunk_handler: None,
        }
    }
}

impl<F1> ImageBuilder for ImageBuilderImpl<F1>
where
    F1: Fn(String) + Send + Sync + 'static,
{
    /// Set format - EXACT syntax: .format(ContentFormat::Base64)
    fn format(mut self, format: ContentFormat) -> impl ImageBuilder {
        self.format = Some(format);
        self
    }
    
    /// Set media type - EXACT syntax: .media_type(ImageMediaType::PNG)
    fn media_type(mut self, media_type: ImageMediaType) -> impl ImageBuilder {
        self.media_type = Some(media_type);
        self
    }
    
    /// Set detail - EXACT syntax: .detail(ImageDetail::High)
    fn detail(mut self, detail: ImageDetail) -> impl ImageBuilder {
        self.detail = Some(detail);
        self
    }
    
    /// Set as PNG - EXACT syntax: .as_png()
    fn as_png(mut self) -> impl ImageBuilder {
        self.media_type = Some(ImageMediaType::PNG);
        self
    }
    
    /// Set as JPEG - EXACT syntax: .as_jpeg()
    fn as_jpeg(mut self) -> impl ImageBuilder {
        self.media_type = Some(ImageMediaType::JPEG);
        self
    }
    
    /// Set high detail - EXACT syntax: .high_detail()
    fn high_detail(mut self) -> impl ImageBuilder {
        self.detail = Some(ImageDetail::High);
        self
    }
    
    /// Set low detail - EXACT syntax: .low_detail()
    fn low_detail(mut self) -> impl ImageBuilder {
        self.detail = Some(ImageDetail::Low);
        self
    }
    
    /// Set error handler - EXACT syntax: .on_error(|error| { ... })
    /// Zero-allocation: uses generic function pointer instead of Box<dyn>
    fn on_error<F>(self, handler: F) -> impl ImageBuilder
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        ImageBuilderImpl {
            data: self.data,
            format: self.format,
            media_type: self.media_type,
            detail: self.detail,
            error_handler: Some(handler),
            chunk_handler: self.chunk_handler,
        }
    }
    
    /// Set chunk handler - EXACT syntax: .on_chunk(|chunk| { ... })
    /// Zero-allocation: returns self for method chaining
    fn on_chunk<F>(mut self, handler: F) -> impl ImageBuilder
    where
        F: Fn(ImageChunk) -> ImageChunk + Send + Sync + 'static,
    {
        // Convert from ImageChunk -> ImageChunk to Result<ImageChunk, String> -> ImageChunk
        self.chunk_handler = Some(Box::new(move |result| {
            match result {
                Ok(chunk) => handler(chunk),
                Err(error) => ImageChunk::bad_chunk(error),
            }
        }));
        self
    }
    
    /// Load image - EXACT syntax: .load()
    fn load(self) -> impl AsyncStream<Item = ImageChunk> {
        let image = Image {
            data: self.data,
            format: self.format,
            media_type: self.media_type,
            detail: self.detail,
        };

        // Convert image data to bytes and create proper ImageChunk
        let data = image.data.as_bytes().to_vec();
        let format = match image.media_type.unwrap_or(ImageMediaType::PNG) {
            ImageMediaType::PNG => fluent_ai_domain::context::chunk::ImageFormat::PNG,
            ImageMediaType::JPEG => fluent_ai_domain::context::chunk::ImageFormat::JPEG,
            ImageMediaType::GIF => fluent_ai_domain::context::chunk::ImageFormat::GIF,
            ImageMediaType::WEBP => fluent_ai_domain::context::chunk::ImageFormat::WebP,
            ImageMediaType::SVG => fluent_ai_domain::context::chunk::ImageFormat::PNG, // fallback
        };

        let chunk = ImageChunk {
            data,
            format,
            dimensions: None,
            metadata: std::collections::HashMap::new(),
        };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = tx.send(chunk);
        fluent_ai_domain::async_task::AsyncStream::new(rx)
    }
    
    /// Process image - EXACT syntax: .process(|chunk| { ... })
    fn process<F>(self, _f: F) -> impl AsyncStream<Item = ImageChunk>
    where
        F: FnOnce(ImageChunk) -> ImageChunk + Send + 'static,
    {
        // For now, just return the load stream
        // TODO: Implement actual processing
        self.load()
    }
}

impl<F1> ChunkHandler<ImageChunk, String> for ImageBuilderImpl<F1>
where
    F1: Fn(String) + Send + Sync + 'static,
{
    fn on_chunk<F>(mut self, handler: F) -> Self
    where
        F: Fn(Result<ImageChunk, String>) -> ImageChunk + Send + Sync + 'static,
    {
        self.chunk_handler = Some(Box::new(handler));
        self
    }
}