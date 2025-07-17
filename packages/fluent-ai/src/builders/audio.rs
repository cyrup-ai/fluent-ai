use crate::domain::audio::{Audio, AudioMediaType, ContentFormat as AudioContentFormat};
use crate::domain::async_task::error_handlers::BadTraitImpl;

/// Builder for Audio objects
pub struct AudioBuilder {
    data: Vec<u8>,
    media_type: Option<AudioMediaType>,
    format: Option<AudioContentFormat>,
}

impl AudioBuilder {
    /// Create a new AudioBuilder with audio data
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            media_type: None,
            format: None,
        }
    }

    /// Set the media type
    pub fn media_type(mut self, media_type: AudioMediaType) -> Self {
        self.media_type = Some(media_type);
        self
    }

    /// Set the content format
    pub fn format(mut self, format: AudioContentFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Build the Audio object
    pub fn build(self) -> Audio {
        Audio {
            data: self.data,
            media_type: self.media_type,
            format: self.format,
            additional_props: Default::default(),
        }
    }
}

/// Builder with error handler for Audio objects
pub struct AudioBuilderWithHandler<F> {
    builder: AudioBuilder,
    error_handler: F,
}

impl<F> AudioBuilderWithHandler<F>
where
    F: Fn(&str) + Send + Sync + 'static,
{
    /// Create a new AudioBuilderWithHandler
    pub fn new(data: Vec<u8>, error_handler: F) -> Self {
        Self {
            builder: AudioBuilder::new(data),
            error_handler,
        }
    }

    /// Set the media type
    pub fn media_type(mut self, media_type: AudioMediaType) -> Self {
        self.builder = self.builder.media_type(media_type);
        self
    }

    /// Set the content format
    pub fn format(mut self, format: AudioContentFormat) -> Self {
        self.builder = self.builder.format(format);
        self
    }

    /// Build the Audio object with error handling
    pub fn build(self) -> Audio {
        self.builder.build()
    }
}

impl Audio {
    /// Create a new AudioBuilder from raw audio data
    pub fn from_data(data: Vec<u8>) -> AudioBuilder {
        AudioBuilder::new(data)
    }

    /// Create audio with error handler
    pub fn from_data_with_handler<F>(data: Vec<u8>, error_handler: F) -> AudioBuilderWithHandler<F>
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        AudioBuilderWithHandler::new(data, error_handler)
    }
}

impl BadTraitImpl for Audio {
    fn bad_impl(error: &str) -> Self {
        eprintln!("Audio BadTraitImpl: {}", error);
        Audio {
            data: vec![],
            media_type: None,
            format: Some(AudioContentFormat::Raw),
            additional_props: Default::default(),
        }
    }
}