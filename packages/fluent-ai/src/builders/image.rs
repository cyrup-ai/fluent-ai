//! Image builder implementations
//!
//! All image construction logic and builder patterns.

use fluent_ai_domain::AsyncStream;
use fluent_ai_domain::chunk::ImageChunk;
use fluent_ai_domain::image::{ContentFormat, Image, ImageDetail, ImageMediaType};

pub struct ImageBuilder {
    data: String,
    format: Option<ContentFormat>,
    media_type: Option<ImageMediaType>,
    detail: Option<ImageDetail>,
}

pub struct ImageBuilderWithHandler {
    #[allow(dead_code)] // TODO: Use for image data content (base64, URL, file path)
    data: String,
    #[allow(dead_code)] // TODO: Use for image content format specification (Base64, URL, Raw)
    format: Option<ContentFormat>,
    #[allow(dead_code)] // TODO: Use for image media type specification (PNG, JPEG, GIF, WEBP, SVG)
    media_type: Option<ImageMediaType>,
    #[allow(dead_code)] // TODO: Use for image detail level specification (Low, High, Auto)
    detail: Option<ImageDetail>,
    #[allow(dead_code)] // TODO: Use for polymorphic error handling during image operations
    error_handler: Box<dyn Fn(String) + Send + Sync>,
    #[allow(dead_code)] // TODO: Use for image streaming chunk processing
    chunk_handler: Option<Box<dyn FnMut(ImageChunk) -> ImageChunk + Send + 'static>>,
}

impl Image {
    // Semantic entry points
    pub fn from_base64(data: impl Into<String>) -> ImageBuilder {
        ImageBuilder {
            data: data.into(),
            format: Some(ContentFormat::Base64),
            media_type: None,
            detail: None,
        }
    }

    pub fn from_url(url: impl Into<String>) -> ImageBuilder {
        ImageBuilder {
            data: url.into(),
            format: Some(ContentFormat::Url),
            media_type: None,
            detail: None,
        }
    }

    pub fn from_path(path: impl Into<String>) -> ImageBuilder {
        ImageBuilder {
            data: path.into(),
            format: Some(ContentFormat::Url),
            media_type: None,
            detail: None,
        }
    }
}

impl ImageBuilder {
    pub fn format(mut self, format: ContentFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub fn media_type(mut self, media_type: ImageMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }

    pub fn detail(mut self, detail: ImageDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    pub fn as_png(mut self) -> Self {
        self.media_type = Some(ImageMediaType::PNG);
        self
    }

    pub fn as_jpeg(mut self) -> Self {
        self.media_type = Some(ImageMediaType::JPEG);
        self
    }

    pub fn high_detail(mut self) -> Self {
        self.detail = Some(ImageDetail::High);
        self
    }

    pub fn low_detail(mut self) -> Self {
        self.detail = Some(ImageDetail::Low);
        self
    }

    // Error handling - required before terminal methods
    pub fn on_error<F>(self, handler: F) -> ImageBuilderWithHandler
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        ImageBuilderWithHandler {
            data: self.data,
            format: self.format,
            media_type: self.media_type,
            detail: self.detail,
            error_handler: Box::new(handler),
            chunk_handler: None,
        }
    }

    pub fn on_chunk<F>(self, handler: F) -> ImageBuilderWithHandler
    where
        F: FnMut(ImageChunk) -> ImageChunk + Send + 'static,
    {
        ImageBuilderWithHandler {
            data: self.data,
            format: self.format,
            media_type: self.media_type,
            detail: self.detail,
            error_handler: Box::new(|e| eprintln!("Image chunk error: {}", e)),
            chunk_handler: Some(Box::new(handler)),
        }
    }
}

impl ImageBuilderWithHandler {
    // Terminal method - returns AsyncStream<ImageChunk>
    pub fn load(self) -> impl AsyncStream<Item = ImageChunk> {
        let image = Image {
            data: self.data,
            format: self.format,
            media_type: self.media_type,
            detail: self.detail,
        };

        // Convert image data to bytes and create proper ImageChunk
        let data = image.data.as_bytes().to_vec();
        let format = match image.media_type.unwrap_or(ImageMediaType::PNG) {
            ImageMediaType::PNG => fluent_ai_domain::chunk::ImageFormat::PNG,
            ImageMediaType::JPEG => fluent_ai_domain::chunk::ImageFormat::JPEG,
            ImageMediaType::GIF => fluent_ai_domain::chunk::ImageFormat::GIF,
            ImageMediaType::WEBP => fluent_ai_domain::chunk::ImageFormat::WebP,
            ImageMediaType::SVG => fluent_ai_domain::chunk::ImageFormat::PNG, // fallback
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

    // Terminal method - async load with processing
    pub fn process<F>(self, _f: F) -> impl AsyncStream<Item = ImageChunk>
    where
        F: FnOnce(ImageChunk) -> ImageChunk + Send + 'static,
    {
        // For now, just return the load stream
        // TODO: Implement actual processing
        self.load()
    }
}
