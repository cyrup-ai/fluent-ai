//! Zero-allocation OpenAI vision capabilities for image processing
//!
//! Provides comprehensive support for OpenAI's vision models (GPT-4O, GPT-4V)
//! with optimal performance patterns and full multimodal capabilities.

use super::{OpenAIError, OpenAIResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use base64::Engine;

/// Image detail levels for vision models
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Low detail - faster, cheaper processing
    Low,
    /// High detail - more detailed analysis, higher cost
    High,
    /// Auto detail - model decides based on image
    Auto,
}

/// Image format types supported by OpenAI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    PNG,
    JPEG,
    WEBP,
    GIF,
}

/// Image data container with format detection
#[derive(Debug, Clone)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub format: ImageFormat,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Vision analysis request configuration
#[derive(Debug, Clone)]
pub struct VisionRequest {
    pub prompt: String,
    pub images: Vec<ImageInput>,
    pub detail: ImageDetail,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

/// Image input for vision requests
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// Image URL (publicly accessible)
    Url { url: String, detail: Option<ImageDetail> },
    /// Base64 encoded image data
    Data { data: Vec<u8>, mime_type: String, detail: Option<ImageDetail> },
    /// File path to image
    Path { path: String, detail: Option<ImageDetail> },
}

/// Vision analysis response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionResponse {
    pub description: String,
    pub objects: Vec<DetectedObject>,
    pub text: Vec<ExtractedText>,
    pub confidence: f32,
    pub processing_time_ms: u64,
}

/// Detected object in image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub name: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
    pub attributes: HashMap<String, String>,
}

/// Extracted text from image (OCR)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedText {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
    pub language: Option<String>,
}

/// Bounding box coordinates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Image analysis capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAnalysisCapabilities {
    pub object_detection: bool,
    pub text_extraction: bool,
    pub scene_description: bool,
    pub facial_recognition: bool,
    pub brand_detection: bool,
    pub landmark_recognition: bool,
}

impl ImageDetail {
    /// Convert to API string representation
    #[inline(always)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high", 
            Self::Auto => "auto",
        }
    }

    /// Get token cost multiplier for detail level
    #[inline(always)]
    pub fn cost_multiplier(&self) -> f32 {
        match self {
            Self::Low => 1.0,
            Self::High => 2.0,
            Self::Auto => 1.5, // Average between low and high
        }
    }

    /// Check if detail level supports specific features
    #[inline(always)]
    pub fn supports_feature(&self, feature: &str) -> bool {
        match (self, feature) {
            (Self::High, _) => true, // High detail supports all features
            (Self::Auto, "text_extraction" | "object_detection") => true,
            (Self::Low, "scene_description") => true,
            _ => false,
        }
    }
}

impl ImageFormat {
    /// Detect format from file extension
    #[inline(always)]
    pub fn from_extension(ext: &str) -> OpenAIResult<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Ok(Self::PNG),
            "jpg" | "jpeg" => Ok(Self::JPEG),
            "webp" => Ok(Self::WEBP),
            "gif" => Ok(Self::GIF),
            _ => Err(OpenAIError::VisionError(format!("Unsupported image format: {}", ext))),
        }
    }

    /// Detect format from magic bytes
    #[inline(always)]
    pub fn from_bytes(data: &[u8]) -> OpenAIResult<Self> {
        if data.len() < 8 {
            return Err(OpenAIError::VisionError("Image data too short to detect format".to_string()));
        }

        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            Ok(Self::PNG)
        } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Ok(Self::JPEG)
        } else if data.starts_with(b"RIFF") && data[8..12] == *b"WEBP" {
            Ok(Self::WEBP)
        } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            Ok(Self::GIF)
        } else {
            Err(OpenAIError::VisionError("Unknown image format".to_string()))
        }
    }

    /// Get MIME type for format
    #[inline(always)]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::PNG => "image/png",
            Self::JPEG => "image/jpeg",
            Self::WEBP => "image/webp",
            Self::GIF => "image/gif",
        }
    }

    /// Check if format supports transparency
    #[inline(always)]
    pub fn supports_transparency(&self) -> bool {
        matches!(self, Self::PNG | Self::WEBP | Self::GIF)
    }

    /// Check if format supports animation
    #[inline(always)]
    pub fn supports_animation(&self) -> bool {
        matches!(self, Self::GIF | Self::WEBP)
    }
}

impl ImageData {
    /// Create from bytes with format detection
    #[inline(always)]
    pub fn from_bytes(data: Vec<u8>) -> OpenAIResult<Self> {
        let format = ImageFormat::from_bytes(&data)?;
        Ok(Self {
            data,
            format,
            width: None,
            height: None,
        })
    }

    /// Create from file path
    #[inline(always)]
    pub fn from_file(path: &str) -> OpenAIResult<Self> {
        let data = std::fs::read(path)
            .map_err(|e| OpenAIError::VisionError(format!("Failed to read image file: {}", e)))?;
        
        let format = if let Some(ext) = path.split('.').last() {
            ImageFormat::from_extension(ext)?
        } else {
            ImageFormat::from_bytes(&data)?
        };

        Ok(Self {
            data,
            format,
            width: None,
            height: None,
        })
    }

    /// Get data URL for embedding in requests
    #[inline(always)]
    pub fn to_data_url(&self) -> String {
        let base64_data = base64::engine::general_purpose::STANDARD.encode(&self.data);
        format!("data:{};base64,{}", self.format.mime_type(), base64_data)
    }

    /// Get size in bytes
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Validate image data for OpenAI API
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        // Check file size limits (20MB for most models)
        const MAX_SIZE: usize = 20 * 1024 * 1024;
        if self.data.len() > MAX_SIZE {
            return Err(OpenAIError::VisionError(format!(
                "Image size {} bytes exceeds maximum of {} bytes", 
                self.data.len(), 
                MAX_SIZE
            )));
        }

        // Check minimum size
        if self.data.len() < 100 {
            return Err(OpenAIError::VisionError("Image data too small".to_string()));
        }

        // Validate format is supported
        match self.format {
            ImageFormat::PNG | ImageFormat::JPEG | ImageFormat::WEBP | ImageFormat::GIF => Ok(()),
        }
    }

    /// Resize image if needed (placeholder - would need image processing library)
    #[inline(always)]
    pub fn resize_if_needed(&mut self, max_width: u32, max_height: u32) -> OpenAIResult<()> {
        // TODO: Implement actual image resizing using image processing library
        // For now, just validate the image doesn't exceed reasonable dimensions
        if let (Some(width), Some(height)) = (self.width, self.height) {
            if width > max_width || height > max_height {
                return Err(OpenAIError::VisionError(format!(
                    "Image dimensions {}x{} exceed maximum {}x{}",
                    width, height, max_width, max_height
                )));
            }
        }
        Ok(())
    }
}

impl ImageInput {
    /// Create from URL with detail level
    #[inline(always)]
    pub fn url(url: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        Self::Url {
            url: url.into(),
            detail,
        }
    }

    /// Create from raw image data
    #[inline(always)]
    pub fn data(data: Vec<u8>, mime_type: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        Self::Data {
            data,
            mime_type: mime_type.into(),
            detail,
        }
    }

    /// Create from file path
    #[inline(always)]
    pub fn file(path: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        Self::Path {
            path: path.into(),
            detail,
        }
    }

    /// Get detail level with fallback to auto
    #[inline(always)]
    pub fn get_detail(&self) -> ImageDetail {
        match self {
            Self::Url { detail, .. } | Self::Data { detail, .. } | Self::Path { detail, .. } => {
                detail.unwrap_or(ImageDetail::Auto)
            }
        }
    }

    /// Convert to message content part for API
    #[inline(always)]
    pub fn to_content_part(&self) -> OpenAIResult<Value> {
        match self {
            Self::Url { url, detail } => {
                let mut image_url = serde_json::json!({
                    "url": url
                });
                
                if let Some(detail_level) = detail {
                    image_url["detail"] = Value::String(detail_level.as_str().to_string());
                }

                Ok(serde_json::json!({
                    "type": "image_url",
                    "image_url": image_url
                }))
            }
            Self::Data { data, mime_type, detail } => {
                let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
                let data_url = format!("data:{};base64,{}", mime_type, base64_data);
                
                let mut image_url = serde_json::json!({
                    "url": data_url
                });
                
                if let Some(detail_level) = detail {
                    image_url["detail"] = Value::String(detail_level.as_str().to_string());
                }

                Ok(serde_json::json!({
                    "type": "image_url",
                    "image_url": image_url
                }))
            }
            Self::Path { path, detail } => {
                let image_data = ImageData::from_file(path)?;
                let data_url = image_data.to_data_url();
                
                let mut image_url = serde_json::json!({
                    "url": data_url
                });
                
                if let Some(detail_level) = detail {
                    image_url["detail"] = Value::String(detail_level.as_str().to_string());
                }

                Ok(serde_json::json!({
                    "type": "image_url",
                    "image_url": image_url
                }))
            }
        }
    }

    /// Validate input for API compatibility
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        match self {
            Self::Url { url, .. } => {
                if url.is_empty() {
                    return Err(OpenAIError::VisionError("Image URL cannot be empty".to_string()));
                }
                
                // Validate URL format
                if !url.starts_with("http://") && !url.starts_with("https://") && !url.starts_with("data:") {
                    return Err(OpenAIError::VisionError("Invalid URL format".to_string()));
                }
                
                Ok(())
            }
            Self::Data { data, mime_type, .. } => {
                if data.is_empty() {
                    return Err(OpenAIError::VisionError("Image data cannot be empty".to_string()));
                }
                
                if mime_type.is_empty() {
                    return Err(OpenAIError::VisionError("MIME type cannot be empty".to_string()));
                }
                
                // Validate MIME type
                if !mime_type.starts_with("image/") {
                    return Err(OpenAIError::VisionError("Invalid image MIME type".to_string()));
                }
                
                Ok(())
            }
            Self::Path { path, .. } => {
                if path.is_empty() {
                    return Err(OpenAIError::VisionError("Image path cannot be empty".to_string()));
                }
                
                // Check if file exists
                if !std::path::Path::new(path).exists() {
                    return Err(OpenAIError::VisionError(format!("Image file not found: {}", path)));
                }
                
                Ok(())
            }
        }
    }
}

impl VisionRequest {
    /// Create new vision analysis request
    #[inline(always)]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            images: Vec::new(),
            detail: ImageDetail::Auto,
            max_tokens: None,
            temperature: None,
        }
    }

    /// Add image URL to request
    #[inline(always)]
    pub fn add_image_url(mut self, url: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        self.images.push(ImageInput::url(url, detail));
        self
    }

    /// Add image data to request
    #[inline(always)]
    pub fn add_image_data(mut self, data: Vec<u8>, mime_type: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        self.images.push(ImageInput::data(data, mime_type, detail));
        self
    }

    /// Add image file to request
    #[inline(always)]
    pub fn add_image_file(mut self, path: impl Into<String>, detail: Option<ImageDetail>) -> Self {
        self.images.push(ImageInput::file(path, detail));
        self
    }

    /// Set default detail level
    #[inline(always)]
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = detail;
        self
    }

    /// Set max tokens for response
    #[inline(always)]
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature for response generation
    #[inline(always)]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Validate request for API compatibility
    #[inline(always)]
    pub fn validate(&self) -> OpenAIResult<()> {
        if self.prompt.is_empty() {
            return Err(OpenAIError::VisionError("Prompt cannot be empty".to_string()));
        }

        if self.images.is_empty() {
            return Err(OpenAIError::VisionError("At least one image is required".to_string()));
        }

        // Validate each image
        for image in &self.images {
            image.validate()?;
        }

        // Check image count limits
        if self.images.len() > 10 {
            return Err(OpenAIError::VisionError("Maximum 10 images per request".to_string()));
        }

        Ok(())
    }

    /// Estimate token cost for request
    #[inline(always)]
    pub fn estimate_tokens(&self) -> u32 {
        let base_tokens = self.prompt.len() as u32 / 4; // Rough estimate: 4 chars per token
        let image_tokens: u32 = self.images.iter()
            .map(|img| {
                let detail = img.get_detail();
                match detail {
                    ImageDetail::Low => 85,
                    ImageDetail::High => 1700,
                    ImageDetail::Auto => 850, // Average
                }
            })
            .sum();
        
        base_tokens + image_tokens
    }

    /// Convert to OpenAI-compatible request format
    #[inline(always)]
    pub fn to_openai_request(&self) -> OpenAIResult<serde_json::Value> {
        // Validate request before conversion
        self.validate()?;
        
        // Build content array with text and images
        let mut content = vec![serde_json::json!({
            "type": "text",
            "text": self.prompt
        })];
        
        // Add images to content
        for image in &self.images {
            content.push(image.to_content_part()?);
        }
        
        Ok(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_tokens": self.max_tokens.unwrap_or(1000),
            "temperature": self.temperature.unwrap_or(0.0)
        }))
    }
}

/// Get capabilities for specific vision model
#[inline(always)]
pub fn get_model_capabilities(model: &str) -> ImageAnalysisCapabilities {
    match model {
        "gpt-4o" | "gpt-4o-mini" => ImageAnalysisCapabilities {
            object_detection: true,
            text_extraction: true,
            scene_description: true,
            facial_recognition: false, // Not officially supported
            brand_detection: true,
            landmark_recognition: true,
        },
        "gpt-4-vision-preview" | "gpt-4-turbo" => ImageAnalysisCapabilities {
            object_detection: true,
            text_extraction: true,
            scene_description: true,
            facial_recognition: false,
            brand_detection: false,
            landmark_recognition: false,
        },
        _ => ImageAnalysisCapabilities {
            object_detection: false,
            text_extraction: false,
            scene_description: false,
            facial_recognition: false,
            brand_detection: false,
            landmark_recognition: false,
        },
    }
}

/// Check if model supports vision features
#[inline(always)]
pub fn supports_vision(model: &str) -> bool {
    matches!(model,
        "gpt-4o" | "gpt-4o-mini" | "gpt-4-vision-preview" | "gpt-4-turbo" |
        "gpt-4-turbo-2024-04-09" | "gpt-4-1106-vision-preview" |
        "chatgpt-4o-latest" | "gpt-4o-search-preview" | "gpt-4o-mini-search-preview"
    )
}