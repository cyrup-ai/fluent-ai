use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub data: String,
    pub format: Option<ContentFormat>,
    pub media_type: Option<ImageMediaType>,
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ContentFormat {
    Base64,
    Url,
    Raw,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImageMediaType {
    PNG,
    JPEG,
    GIF,
    WEBP,
    SVG,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
}

// Builder implementations moved to fluent_ai/src/builders/image.rs
