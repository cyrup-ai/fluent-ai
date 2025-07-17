use crate::domain::image::{Image, ImageMediaType, ContentFormat as ImageContentFormat, ImageDetail};
use crate::domain::async_task::error_handlers::BadTraitImpl;

/// Builder for Image objects
pub struct ImageBuilder {
    data: Vec<u8>,
    media_type: Option<ImageMediaType>,
    format: Option<ImageContentFormat>,
    detail: Option<ImageDetail>,
}

impl ImageBuilder {
    /// Create a new ImageBuilder with image data
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            media_type: None,
            format: None,
            detail: None,
        }
    }

    /// Set the media type
    pub fn media_type(mut self, media_type: ImageMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }

    /// Set the content format
    pub fn format(mut self, format: ImageContentFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the image detail level
    pub fn detail(mut self, detail: ImageDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Build the Image object
    pub fn build(self) -> Image {
        Image {
            data: self.data,
            media_type: self.media_type,
            format: self.format,
            detail: self.detail,
            additional_props: Default::default(),
        }
    }
}

/// Builder with error handler for Image objects
pub struct ImageBuilderWithHandler<F> {
    builder: ImageBuilder,
    error_handler: F,
}

impl<F> ImageBuilderWithHandler<F>
where
    F: Fn(&str) + Send + Sync + 'static,
{
    /// Create a new ImageBuilderWithHandler
    pub fn new(data: Vec<u8>, error_handler: F) -> Self {
        Self {
            builder: ImageBuilder::new(data),
            error_handler,
        }
    }

    /// Set the media type
    pub fn media_type(mut self, media_type: ImageMediaType) -> Self {
        self.builder = self.builder.media_type(media_type);
        self
    }

    /// Set the content format
    pub fn format(mut self, format: ImageContentFormat) -> Self {
        self.builder = self.builder.format(format);
        self
    }

    /// Set the image detail level
    pub fn detail(mut self, detail: ImageDetail) -> Self {
        self.builder = self.builder.detail(detail);
        self
    }

    /// Build the Image object with error handling
    pub fn build(self) -> Image {
        self.builder.build()
    }
}

impl Image {
    /// Create a new ImageBuilder from raw image data
    pub fn from_data(data: Vec<u8>) -> ImageBuilder {
        ImageBuilder::new(data)
    }

    /// Create image with error handler
    pub fn from_data_with_handler<F>(data: Vec<u8>, error_handler: F) -> ImageBuilderWithHandler<F>
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        ImageBuilderWithHandler::new(data, error_handler)
    }
}

impl BadTraitImpl for Image {
    fn bad_impl(error: &str) -> Self {
        eprintln!("Image BadTraitImpl: {}", error);
        Image {
            data: vec![],
            media_type: None,
            format: Some(ImageContentFormat::Base64),
            detail: Some(ImageDetail::Auto),
            additional_props: Default::default(),
        }
    }
}