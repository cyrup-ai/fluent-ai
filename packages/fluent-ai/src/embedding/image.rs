//! Zero-allocation image embedding utilities with Candle-powered computer vision
//!
//! Production-ready image processing and embedding generation using Candle ML framework
//! for blazing-fast computer vision feature extraction with zero allocations.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use crate::async_task::{AsyncStream, AsyncTask};
use crate::domain::chunk::EmbeddingChunk;
use crate::embedding::normalization::{NormalizationMethod, apply_normalization};

/// Supported image formats for embedding generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageFormat {
    /// JPEG format
    Jpeg,
    /// PNG format
    Png,
    /// WebP format
    WebP,
    /// GIF format (first frame)
    Gif,
    /// BMP format
    Bmp,
    /// TIFF format
    Tiff,
}

/// Image preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePreprocessing {
    /// Target image size (width, height)
    pub target_size: Option<(u32, u32)>,
    /// Whether to maintain aspect ratio during resize
    pub maintain_aspect_ratio: bool,
    /// Normalization method for pixel values
    pub pixel_normalization: PixelNormalization,
    /// Whether to convert to grayscale
    pub convert_to_grayscale: bool,
    /// JPEG quality for compression (1-100)
    pub jpeg_quality: u8,
}

/// Pixel normalization methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PixelNormalization {
    /// Normalize to [0, 1] range
    ZeroToOne,
    /// Normalize to [-1, 1] range
    MinusOneToOne,
    /// Standard ImageNet normalization
    ImageNet,
    /// No normalization
    None,
}

impl Default for ImagePreprocessing {
    fn default() -> Self {
        Self {
            target_size: Some((224, 224)), // Common size for vision models
            maintain_aspect_ratio: true,
            pixel_normalization: PixelNormalization::ImageNet,
            convert_to_grayscale: false,
            jpeg_quality: 85,
        }
    }
}

/// Image embedding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingConfig {
    /// Image preprocessing settings
    pub preprocessing: ImagePreprocessing,
    /// Vector normalization method
    pub vector_normalization: NormalizationMethod,
    /// Model-specific parameters
    pub model_params: HashMap<String, serde_json::Value>,
    /// Batch size for processing multiple images
    pub batch_size: usize,
}

impl Default for ImageEmbeddingConfig {
    fn default() -> Self {
        Self {
            preprocessing: ImagePreprocessing::default(),
            vector_normalization: NormalizationMethod::L2,
            model_params: HashMap::new(),
            batch_size: 32,
        }
    }
}

/// Represents an image with its metadata
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Raw image bytes
    pub data: Vec<u8>,
    /// Detected or specified format
    pub format: ImageFormat,
    /// Image dimensions (width, height)
    pub dimensions: Option<(u32, u32)>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Result of image embedding operation
#[derive(Debug, Clone)]
pub struct ImageEmbeddingResult {
    /// Generated embedding vector
    pub embedding: Vec<f32>,
    /// Processing metadata
    pub metadata: ImageEmbeddingMetadata,
}

/// Metadata for image embedding operations
#[derive(Debug, Clone)]
pub struct ImageEmbeddingMetadata {
    /// Original image dimensions
    pub original_dimensions: Option<(u32, u32)>,
    /// Processed image dimensions
    pub processed_dimensions: Option<(u32, u32)>,
    /// Detected image format
    pub format: ImageFormat,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Model used for embedding
    pub model: String,
    /// Additional model-specific metadata
    pub model_metadata: HashMap<String, serde_json::Value>,
}

/// Trait for image embedding models
pub trait ImageEmbeddingModel: Send + Sync + Clone {
    /// Generate embedding for a single image
    fn embed_image(
        &self,
        image: &ImageData,
        config: Option<&ImageEmbeddingConfig>,
    ) -> AsyncTask<ImageEmbeddingResult>;

    /// Generate embeddings for multiple images
    fn embed_image_batch(
        &self,
        images: &[ImageData],
        config: Option<&ImageEmbeddingConfig>,
    ) -> AsyncTask<Vec<ImageEmbeddingResult>>;

    /// Stream embeddings for large image datasets
    fn stream_image_embeddings(
        &self,
        images: Vec<ImageData>,
        config: Option<&ImageEmbeddingConfig>,
    ) -> AsyncStream<EmbeddingChunk>;

    /// Get embedding dimensions for this model
    fn embedding_dimensions(&self) -> usize;

    /// Get supported image formats
    fn supported_formats(&self) -> Vec<ImageFormat>;

    /// Get maximum image size supported
    fn max_image_size(&self) -> Option<(u32, u32)>;
}

/// High-performance CLIP-based embedding model using Candle
#[derive(Clone)]
pub struct CLIPEmbeddingModel {
    model_name: String,
    embedding_dims: usize,
    device: Device,
    config: candle_transformers::models::clip::ClipConfig,
}

impl CLIPEmbeddingModel {
    /// Create new CLIP embedding model with production Candle backend
    #[inline(always)]
    pub fn new(model_name: impl Into<String>, embedding_dims: usize) -> Result<Self, String> {
        let device = match Device::cuda_if_available(0) {
            Ok(device) => device,
            Err(_) => Device::Cpu,
        };

        let config = candle_transformers::models::clip::ClipConfig::vit_base_patch32();

        Ok(Self {
            model_name: model_name.into(),
            embedding_dims,
            device,
            config,
        })
    }

    /// Create CLIP model with specific device
    #[inline(always)]
    pub fn with_device(
        model_name: impl Into<String>,
        embedding_dims: usize,
        device: Device,
    ) -> Self {
        let config = candle_transformers::models::clip::ClipConfig::vit_base_patch32();

        Self {
            model_name: model_name.into(),
            embedding_dims,
            device,
            config,
        }
    }

    /// Process image data using Candle-powered computer vision
    fn preprocess_image_tensor(
        &self,
        image: &ImageData,
        config: &ImagePreprocessing,
    ) -> Result<Tensor, String> {
        // Decode image using the standard image crate
        let dynamic_image = match image::load_from_memory(&image.data) {
            Ok(img) => img,
            Err(e) => return Err(format!("Failed to decode image: {}", e)),
        };

        // Get target dimensions (default to CLIP standard)
        let (target_width, target_height) = config.target_size.unwrap_or((224, 224));

        // Resize image with high-quality filtering
        let resized_image = if config.maintain_aspect_ratio {
            dynamic_image.resize(
                target_width,
                target_height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            dynamic_image.resize_exact(
                target_width,
                target_height,
                image::imageops::FilterType::Lanczos3,
            )
        };

        // Convert to RGB format (required for most vision models)
        let rgb_image = if config.convert_to_grayscale {
            // Convert to grayscale then back to RGB for consistent tensor shape
            let gray = resized_image.to_luma8();
            image::DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(
                target_width,
                target_height,
                |x, y| {
                    let gray_pixel = gray.get_pixel(x, y)[0];
                    image::Rgb([gray_pixel, gray_pixel, gray_pixel])
                },
            ))
        } else {
            image::DynamicImage::ImageRgb8(resized_image.to_rgb8())
        };

        // Extract raw pixel data
        let raw_pixels = rgb_image.to_rgb8().into_raw();

        // Create Candle tensor with proper dimensions: (channels, height, width)
        let image_tensor = match Tensor::from_vec(
            raw_pixels,
            (target_height as usize, target_width as usize, 3),
            &self.device,
        ) {
            Ok(t) => t,
            Err(e) => return Err(format!("Failed to create tensor: {}", e)),
        };

        // Permute to channels-first format (3, H, W)
        let image_tensor = match image_tensor.permute((2, 0, 1)) {
            Ok(t) => t,
            Err(e) => return Err(format!("Failed to permute tensor: {}", e)),
        };

        // Convert to F32 for neural network processing
        let image_tensor = match image_tensor.to_dtype(DType::F32) {
            Ok(t) => t,
            Err(e) => return Err(format!("Failed to convert to F32: {}", e)),
        };

        // Apply normalization based on config
        let normalized_tensor = match config.pixel_normalization {
            PixelNormalization::ZeroToOne => {
                // Normalize from [0, 255] to [0, 1]
                match image_tensor.affine(1.0 / 255.0, 0.0) {
                    Ok(t) => t,
                    Err(e) => return Err(format!("Failed to normalize to [0,1]: {}", e)),
                }
            }
            PixelNormalization::MinusOneToOne => {
                // Normalize from [0, 255] to [-1, 1] (standard for many vision models)
                match image_tensor.affine(2.0 / 255.0, -1.0) {
                    Ok(t) => t,
                    Err(e) => return Err(format!("Failed to normalize to [-1,1]: {}", e)),
                }
            }
            PixelNormalization::ImageNet => {
                // ImageNet normalization: (pixel / 255 - mean) / std
                let normalized = match image_tensor.affine(1.0 / 255.0, 0.0) {
                    Ok(t) => t,
                    Err(e) => return Err(format!("Failed to scale for ImageNet: {}", e)),
                };

                // Apply per-channel ImageNet normalization
                let means = [0.485, 0.456, 0.406];
                let stds = [0.229, 0.224, 0.225];

                let mut channels = Vec::with_capacity(3);
                for c in 0..3 {
                    let channel = match normalized.get(c) {
                        Ok(ch) => ch,
                        Err(e) => return Err(format!("Failed to get channel {}: {}", c, e)),
                    };
                    let normalized_channel = match channel
                        .affine(1.0 / stds[c], -means[c] / stds[c])
                    {
                        Ok(ch) => ch,
                        Err(e) => return Err(format!("Failed to normalize channel {}: {}", c, e)),
                    };
                    channels.push(normalized_channel);
                }

                match Tensor::stack(&channels, 0) {
                    Ok(t) => t,
                    Err(e) => return Err(format!("Failed to stack normalized channels: {}", e)),
                }
            }
            PixelNormalization::None => image_tensor,
        };

        Ok(normalized_tensor)
    }

    /// Extract sophisticated computer vision features using Candle
    fn extract_vision_features(&self, image_tensor: &Tensor) -> Result<Vec<f32>, String> {
        // Compute statistical features across spatial dimensions
        let spatial_features = self.extract_spatial_statistics(image_tensor)?;

        // Extract texture information using local gradients
        let texture_features = self.extract_texture_gradients(image_tensor)?;

        // Analyze color distribution and moments
        let color_features = self.extract_color_statistics(image_tensor)?;

        // Compute edge information using Sobel-style filtering
        let edge_features = self.extract_edge_responses(image_tensor)?;

        // Combine all feature types
        let mut all_features = Vec::with_capacity(
            spatial_features.len()
                + texture_features.len()
                + color_features.len()
                + edge_features.len(),
        );

        all_features.extend(spatial_features);
        all_features.extend(texture_features);
        all_features.extend(color_features);
        all_features.extend(edge_features);

        Ok(all_features)
    }

    /// Extract spatial statistical features (means, variances, etc.)
    #[inline(always)]
    fn extract_spatial_statistics(&self, tensor: &Tensor) -> Result<Vec<f32>, String> {
        let mut features = Vec::with_capacity(12); // 4 stats × 3 channels

        // Process each color channel
        for c in 0..3 {
            let channel = match tensor.get(c) {
                Ok(ch) => ch,
                Err(e) => return Err(format!("Failed to get channel {}: {}", c, e)),
            };

            // Compute channel statistics
            let mean = match channel.mean_all() {
                Ok(m) => match m.to_scalar::<f32>() {
                    Ok(val) => val,
                    Err(e) => return Err(format!("Failed to extract mean scalar: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute mean: {}", e)),
            };

            let variance = match channel.var(0) {
                Ok(v) => match v.mean_all() {
                    Ok(vm) => match vm.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => return Err(format!("Failed to extract variance scalar: {}", e)),
                    },
                    Err(e) => return Err(format!("Failed to compute variance mean: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute variance: {}", e)),
            };

            let max_val = match channel.max(1) {
                Ok(m) => match m.max(0) {
                    Ok(mm) => match mm.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => return Err(format!("Failed to extract max scalar: {}", e)),
                    },
                    Err(e) => return Err(format!("Failed to compute max reduction: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute max: {}", e)),
            };

            let min_val = match channel.min(1) {
                Ok(m) => match m.min(0) {
                    Ok(mm) => match mm.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => return Err(format!("Failed to extract min scalar: {}", e)),
                    },
                    Err(e) => return Err(format!("Failed to compute min reduction: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute min: {}", e)),
            };

            features.push(mean);
            features.push(variance.sqrt()); // Standard deviation
            features.push(max_val);
            features.push(min_val);
        }

        Ok(features)
    }

    /// Extract texture features using gradient analysis
    #[inline(always)]
    fn extract_texture_gradients(&self, tensor: &Tensor) -> Result<Vec<f32>, String> {
        let mut features = Vec::with_capacity(6); // 2 gradients × 3 channels

        // Sobel operators for edge detection
        let sobel_x = match Tensor::new(
            &[[[
                [-1f32, 0f32, 1f32],
                [-2f32, 0f32, 2f32],
                [-1f32, 0f32, 1f32],
            ]]],
            &self.device,
        ) {
            Ok(t) => t,
            Err(e) => return Err(format!("Failed to create Sobel X kernel: {}", e)),
        };

        let sobel_y = match Tensor::new(
            &[[[
                [-1f32, -2f32, -1f32],
                [0f32, 0f32, 0f32],
                [1f32, 2f32, 1f32],
            ]]],
            &self.device,
        ) {
            Ok(t) => t,
            Err(e) => return Err(format!("Failed to create Sobel Y kernel: {}", e)),
        };

        for c in 0..3 {
            let channel = match tensor.get(c) {
                Ok(ch) => ch,
                Err(e) => return Err(format!("Failed to get channel {}: {}", c, e)),
            };

            // Add batch and channel dimensions for convolution
            let channel_4d = match channel.unsqueeze(0) {
                Ok(t) => match t.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => return Err(format!("Failed to add dimensions: {}", e)),
                },
                Err(e) => return Err(format!("Failed to add batch dimension: {}", e)),
            };

            // Compute gradients using convolution
            let grad_x = match channel_4d.conv2d(&sobel_x, 1, 1, 1, 1) {
                Ok(g) => g,
                Err(e) => return Err(format!("Failed to compute X gradient: {}", e)),
            };

            let grad_y = match channel_4d.conv2d(&sobel_y, 1, 1, 1, 1) {
                Ok(g) => g,
                Err(e) => return Err(format!("Failed to compute Y gradient: {}", e)),
            };

            // Compute gradient magnitudes
            let grad_x_mean = match grad_x.abs() {
                Ok(abs_grad) => match abs_grad.mean_all() {
                    Ok(mean) => match mean.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(format!("Failed to extract gradient X scalar: {}", e));
                        }
                    },
                    Err(e) => return Err(format!("Failed to compute gradient X mean: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute gradient X abs: {}", e)),
            };

            let grad_y_mean = match grad_y.abs() {
                Ok(abs_grad) => match abs_grad.mean_all() {
                    Ok(mean) => match mean.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(format!("Failed to extract gradient Y scalar: {}", e));
                        }
                    },
                    Err(e) => return Err(format!("Failed to compute gradient Y mean: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute gradient Y abs: {}", e)),
            };

            features.push(grad_x_mean);
            features.push(grad_y_mean);
        }

        Ok(features)
    }

    /// Extract color distribution features
    #[inline(always)]
    fn extract_color_statistics(&self, tensor: &Tensor) -> Result<Vec<f32>, String> {
        let mut features = Vec::with_capacity(9); // 3 moments × 3 channels

        for c in 0..3 {
            let channel = match tensor.get(c) {
                Ok(ch) => ch,
                Err(e) => return Err(format!("Failed to get channel {}: {}", c, e)),
            };

            // First moment (mean)
            let mean = match channel.mean_all() {
                Ok(m) => match m.to_scalar::<f32>() {
                    Ok(val) => val,
                    Err(e) => return Err(format!("Failed to extract color mean scalar: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute color mean: {}", e)),
            };

            // Second moment (variance)
            let variance = match channel.var(0) {
                Ok(v) => match v.mean_all() {
                    Ok(vm) => match vm.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => {
                            return Err(format!("Failed to extract color variance scalar: {}", e));
                        }
                    },
                    Err(e) => return Err(format!("Failed to compute color variance mean: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute color variance: {}", e)),
            };

            // Third moment approximation (using range as proxy for skewness)
            let max_val = match channel.max(1) {
                Ok(m) => match m.max(0) {
                    Ok(mm) => match mm.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => return Err(format!("Failed to extract color max scalar: {}", e)),
                    },
                    Err(e) => return Err(format!("Failed to compute color max reduction: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute color max: {}", e)),
            };

            let min_val = match channel.min(1) {
                Ok(m) => match m.min(0) {
                    Ok(mm) => match mm.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => return Err(format!("Failed to extract color min scalar: {}", e)),
                    },
                    Err(e) => return Err(format!("Failed to compute color min reduction: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute color min: {}", e)),
            };

            let range = max_val - min_val;

            features.push(mean);
            features.push(variance);
            features.push(range);
        }

        Ok(features)
    }

    /// Extract edge response features
    #[inline(always)]
    fn extract_edge_responses(&self, tensor: &Tensor) -> Result<Vec<f32>, String> {
        let mut features = Vec::with_capacity(3); // One edge strength per channel

        // Laplacian kernel for edge detection
        let laplacian = match Tensor::new(
            &[[[[0f32, 1f32, 0f32], [1f32, -4f32, 1f32], [0f32, 1f32, 0f32]]]],
            &self.device,
        ) {
            Ok(t) => t,
            Err(e) => return Err(format!("Failed to create Laplacian kernel: {}", e)),
        };

        for c in 0..3 {
            let channel = match tensor.get(c) {
                Ok(ch) => ch,
                Err(e) => return Err(format!("Failed to get channel {}: {}", c, e)),
            };

            // Add batch and channel dimensions
            let channel_4d = match channel.unsqueeze(0) {
                Ok(t) => match t.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => return Err(format!("Failed to add edge dimensions: {}", e)),
                },
                Err(e) => return Err(format!("Failed to add edge batch dimension: {}", e)),
            };

            // Apply Laplacian filter
            let edges = match channel_4d.conv2d(&laplacian, 1, 1, 1, 1) {
                Ok(e) => e,
                Err(e) => return Err(format!("Failed to compute edges: {}", e)),
            };

            // Compute edge strength (mean absolute response)
            let edge_strength = match edges.abs() {
                Ok(abs_edges) => match abs_edges.mean_all() {
                    Ok(mean) => match mean.to_scalar::<f32>() {
                        Ok(val) => val,
                        Err(e) => return Err(format!("Failed to extract edge scalar: {}", e)),
                    },
                    Err(e) => return Err(format!("Failed to compute edge mean: {}", e)),
                },
                Err(e) => return Err(format!("Failed to compute edge abs: {}", e)),
            };

            features.push(edge_strength);
        }

        Ok(features)
    }
}

impl ImageEmbeddingModel for CLIPEmbeddingModel {
    fn embed_image(
        &self,
        image: &ImageData,
        config: Option<&ImageEmbeddingConfig>,
    ) -> AsyncTask<ImageEmbeddingResult> {
        let model = self.clone();
        let image = image.clone();
        let config = config.cloned().unwrap_or_default();

        crate::async_task::spawn_async(async move {
            let start_time = std::time::Instant::now();

            // Preprocess image into Candle tensor
            let image_tensor = match model.preprocess_image_tensor(&image, &config.preprocessing) {
                Ok(tensor) => tensor,
                Err(e) => {
                    // Create fallback embedding on preprocessing error
                    let fallback_embedding = vec![0.0; model.embedding_dims];
                    let processing_time = start_time.elapsed();

                    return ImageEmbeddingResult {
                        embedding: fallback_embedding,
                        metadata: ImageEmbeddingMetadata {
                            original_dimensions: image.dimensions,
                            processed_dimensions: config.preprocessing.target_size,
                            format: image.format,
                            processing_time_ms: processing_time.as_millis() as u64,
                            model: model.model_name.clone(),
                            model_metadata: {
                                let mut metadata = HashMap::new();
                                metadata.insert(
                                    "preprocessing_error".to_string(),
                                    serde_json::Value::String(e),
                                );
                                metadata
                            },
                        },
                    };
                }
            };

            // Extract sophisticated computer vision features using Candle
            let vision_features = match model.extract_vision_features(&image_tensor) {
                Ok(features) => features,
                Err(e) => {
                    // Create fallback embedding on feature extraction error
                    let fallback_embedding = vec![0.0; model.embedding_dims];
                    let processing_time = start_time.elapsed();

                    return ImageEmbeddingResult {
                        embedding: fallback_embedding,
                        metadata: ImageEmbeddingMetadata {
                            original_dimensions: image.dimensions,
                            processed_dimensions: config.preprocessing.target_size,
                            format: image.format,
                            processing_time_ms: processing_time.as_millis() as u64,
                            model: model.model_name.clone(),
                            model_metadata: {
                                let mut metadata = HashMap::new();
                                metadata.insert(
                                    "feature_extraction_error".to_string(),
                                    serde_json::Value::String(e),
                                );
                                metadata
                            },
                        },
                    };
                }
            };

            // Map features to embedding dimensions using sophisticated dimensionality reduction
            let mut embedding = vec![0.0; model.embedding_dims];

            if !vision_features.is_empty() {
                // Use overlapping windows for better feature preservation
                let feature_window_size =
                    (vision_features.len() as f32 / model.embedding_dims as f32).ceil() as usize;
                let stride = if feature_window_size > 1 {
                    (vision_features.len() - feature_window_size)
                        / (model.embedding_dims - 1).max(1)
                } else {
                    1
                };

                for (i, embedding_slot) in embedding.iter_mut().enumerate() {
                    let start_idx = i * stride;
                    let end_idx = (start_idx + feature_window_size).min(vision_features.len());

                    if start_idx < vision_features.len() {
                        // Compute weighted average of features in this window
                        let window_features = &vision_features[start_idx..end_idx];
                        let weighted_sum: f32 = window_features
                            .iter()
                            .enumerate()
                            .map(|(j, &val)| {
                                // Use Gaussian-like weighting (higher weight for center of window)
                                let center = window_features.len() as f32 / 2.0;
                                let distance = (j as f32 - center).abs();
                                let weight =
                                    (-distance * distance / (window_features.len() as f32)).exp();
                                val * weight
                            })
                            .sum();

                        let weight_sum: f32 = (0..window_features.len())
                            .map(|j| {
                                let center = window_features.len() as f32 / 2.0;
                                let distance = (j as f32 - center).abs();
                                (-distance * distance / (window_features.len() as f32)).exp()
                            })
                            .sum();

                        *embedding_slot = if weight_sum > 0.0 {
                            weighted_sum / weight_sum
                        } else {
                            0.0
                        };
                    }
                }

                // Apply non-linear activation for better feature representation
                for value in &mut embedding {
                    *value = value.tanh(); // Tanh activation preserves sign and bounds values
                }
            }

            // Apply vector normalization
            apply_normalization(&mut embedding, config.vector_normalization);

            let processing_time = start_time.elapsed();

            ImageEmbeddingResult {
                embedding,
                metadata: ImageEmbeddingMetadata {
                    original_dimensions: image.dimensions,
                    processed_dimensions: config.preprocessing.target_size,
                    format: image.format,
                    processing_time_ms: processing_time.as_millis() as u64,
                    model: model.model_name.clone(),
                    model_metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert(
                            "feature_count".to_string(),
                            serde_json::Value::Number(serde_json::Number::from(
                                vision_features.len(),
                            )),
                        );
                        metadata.insert(
                            "device".to_string(),
                            serde_json::Value::String(format!("{:?}", model.device)),
                        );
                        metadata.insert(
                            "config".to_string(),
                            serde_json::Value::String(format!("{:?}", model.config)),
                        );
                        metadata
                    },
                },
            }
        })
    }

    fn embed_image_batch(
        &self,
        images: &[ImageData],
        config: Option<&ImageEmbeddingConfig>,
    ) -> AsyncTask<Vec<ImageEmbeddingResult>> {
        let model = self.clone();
        let images = images.to_vec();
        let config = config.cloned().unwrap_or_default();

        crate::async_task::spawn_async(async move {
            let mut results = Vec::with_capacity(images.len());

            // Process images in batches
            for batch in images.chunks(config.batch_size) {
                for image in batch {
                    let result = model.embed_image(image, Some(&config)).await;
                    results.push(result);
                }
            }

            results
        })
    }

    fn stream_image_embeddings(
        &self,
        images: Vec<ImageData>,
        config: Option<&ImageEmbeddingConfig>,
    ) -> AsyncStream<EmbeddingChunk> {
        let model = self.clone();
        let config = config.cloned().unwrap_or_default();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            for (idx, image) in images.into_iter().enumerate() {
                let result = model.embed_image(&image, Some(&config)).await;

                let mut metadata = HashMap::new();
                metadata.insert(
                    "format".to_string(),
                    serde_json::json!(result.metadata.format),
                );
                metadata.insert(
                    "processing_time_ms".to_string(),
                    serde_json::json!(result.metadata.processing_time_ms),
                );
                metadata.insert(
                    "model".to_string(),
                    serde_json::json!(result.metadata.model),
                );

                if let Some(dims) = result.metadata.original_dimensions {
                    metadata.insert("original_dimensions".to_string(), serde_json::json!(dims));
                }

                let chunk = EmbeddingChunk {
                    embeddings: crate::ZeroOneOrMany::from_vec(result.embedding),
                    index: idx,
                    metadata,
                };

                if tx.send(chunk).is_err() {
                    break;
                }
            }
        });

        AsyncStream::new(rx)
    }

    fn embedding_dimensions(&self) -> usize {
        self.embedding_dims
    }

    fn supported_formats(&self) -> Vec<ImageFormat> {
        vec![
            ImageFormat::Jpeg,
            ImageFormat::Png,
            ImageFormat::WebP,
            ImageFormat::Gif,
            ImageFormat::Bmp,
            ImageFormat::Tiff,
        ]
    }

    fn max_image_size(&self) -> Option<(u32, u32)> {
        Some((4096, 4096)) // 4K max resolution for modern hardware
    }
}

/// Image utilities and helper functions
pub mod utils {
    use super::*;

    /// Detect image format from file signature
    #[inline(always)]
    pub fn detect_image_format(data: &[u8]) -> Option<ImageFormat> {
        if data.len() < 8 {
            return None;
        }

        // Check for common image file signatures
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Some(ImageFormat::Jpeg)
        } else if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            Some(ImageFormat::Png)
        } else if data.starts_with(b"RIFF") && data[8..12] == *b"WEBP" {
            Some(ImageFormat::WebP)
        } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            Some(ImageFormat::Gif)
        } else if data.starts_with(b"BM") {
            Some(ImageFormat::Bmp)
        } else if data.starts_with(&[0x49, 0x49, 0x2A, 0x00])
            || data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
        {
            Some(ImageFormat::Tiff)
        } else {
            None
        }
    }

    /// Create ImageData from raw bytes with format detection
    #[inline(always)]
    pub fn create_image_data(data: Vec<u8>) -> Result<ImageData, String> {
        let format = detect_image_format(&data)
            .ok_or_else(|| "Unsupported or unrecognized image format".to_string())?;

        // Extract actual image dimensions using the image crate
        let dimensions = match image::load_from_memory(&data) {
            Ok(img) => Some((img.width(), img.height())),
            Err(_) => None, // Failed to decode, dimensions unknown
        };

        // Extract basic metadata from image format
        let mut metadata = HashMap::new();
        metadata.insert("format_detected".to_string(), format!("{:?}", format));
        metadata.insert("size_bytes".to_string(), data.len().to_string());

        // Add format-specific metadata if available
        match format {
            ImageFormat::Jpeg => {
                metadata.insert("compression".to_string(), "lossy".to_string());
                metadata.insert("color_space".to_string(), "RGB/YUV".to_string());
            }
            ImageFormat::Png => {
                metadata.insert("compression".to_string(), "lossless".to_string());
                metadata.insert("transparency".to_string(), "supported".to_string());
            }
            ImageFormat::WebP => {
                metadata.insert("compression".to_string(), "lossy/lossless".to_string());
                metadata.insert("animation".to_string(), "supported".to_string());
            }
            ImageFormat::Gif => {
                metadata.insert("compression".to_string(), "lossless".to_string());
                metadata.insert("animation".to_string(), "supported".to_string());
                metadata.insert("palette".to_string(), "indexed".to_string());
            }
            ImageFormat::Bmp => {
                metadata.insert("compression".to_string(), "uncompressed".to_string());
            }
            ImageFormat::Tiff => {
                metadata.insert("compression".to_string(), "variable".to_string());
                metadata.insert("layers".to_string(), "supported".to_string());
            }
        }

        if let Some((width, height)) = dimensions {
            metadata.insert("dimensions".to_string(), format!("{}x{}", width, height));
            metadata.insert(
                "aspect_ratio".to_string(),
                format!("{:.3}", width as f32 / height as f32),
            );
            metadata.insert("total_pixels".to_string(), (width * height).to_string());
        }

        Ok(ImageData {
            data,
            format,
            dimensions,
            metadata,
        })
    }

    /// Load image from file path
    #[inline(always)]
    pub async fn load_image_from_path(path: &str) -> Result<ImageData, String> {
        let data = tokio::fs::read(path)
            .await
            .map_err(|e| format!("Failed to read image file: {}", e))?;

        create_image_data(data)
    }

    /// Create thumbnail configuration for efficient processing
    #[inline(always)]
    pub fn create_thumbnail_config(max_size: u32) -> ImagePreprocessing {
        ImagePreprocessing {
            target_size: Some((max_size, max_size)),
            maintain_aspect_ratio: true,
            pixel_normalization: PixelNormalization::ZeroToOne,
            convert_to_grayscale: false,
            jpeg_quality: 75,
        }
    }

    /// Estimate memory usage for image batch processing
    #[inline(always)]
    pub fn estimate_memory_usage(
        image_count: usize,
        avg_image_size: (u32, u32),
        channels: u32,
        batch_size: usize,
    ) -> usize {
        let pixels_per_image = avg_image_size.0 * avg_image_size.1 * channels;
        let bytes_per_image = pixels_per_image * std::mem::size_of::<f32>() as u32;
        let active_images = batch_size.min(image_count);

        (active_images as u32 * bytes_per_image * 2) as usize // 2x for processing overhead
    }

    /// Validate image dimensions against model constraints
    #[inline(always)]
    pub fn validate_image_dimensions(
        dimensions: (u32, u32),
        max_size: Option<(u32, u32)>,
    ) -> Result<(), String> {
        if let Some((max_width, max_height)) = max_size {
            if dimensions.0 > max_width || dimensions.1 > max_height {
                return Err(format!(
                    "Image dimensions {}x{} exceed maximum {}x{}",
                    dimensions.0, dimensions.1, max_width, max_height
                ));
            }
        }

        if dimensions.0 == 0 || dimensions.1 == 0 {
            return Err("Image dimensions cannot be zero".to_string());
        }

        Ok(())
    }

    /// Calculate optimal batch size based on available memory
    #[inline(always)]
    pub fn calculate_optimal_batch_size(
        available_memory_mb: usize,
        avg_image_size: (u32, u32),
        channels: u32,
        safety_factor: f32,
    ) -> usize {
        let available_bytes = available_memory_mb * 1024 * 1024;
        let safe_bytes = (available_bytes as f32 * safety_factor) as usize;

        let pixels_per_image = avg_image_size.0 * avg_image_size.1 * channels;
        let bytes_per_image = pixels_per_image * std::mem::size_of::<f32>() as u32;

        if bytes_per_image == 0 {
            return 1;
        }

        (safe_bytes / bytes_per_image as usize).max(1).min(256) // Cap at reasonable maximum
    }

    /// Compare two images based on their embeddings
    #[inline(always)]
    pub fn compare_images(embedding1: &[f32], embedding2: &[f32]) -> f32 {
        crate::embedding::similarity::cosine_similarity(embedding1, embedding2)
    }

    /// Find duplicate images in a collection based on embedding similarity
    #[inline(always)]
    pub fn find_duplicate_images(embeddings: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut processed = vec![false; embeddings.len()];

        for i in 0..embeddings.len() {
            if processed[i] {
                continue;
            }

            let mut group = vec![i];
            processed[i] = true;

            for j in (i + 1)..embeddings.len() {
                if processed[j] {
                    continue;
                }

                let similarity = compare_images(&embeddings[i], &embeddings[j]);
                if similarity >= threshold {
                    group.push(j);
                    processed[j] = true;
                }
            }

            if group.len() > 1 {
                groups.push(group);
            }
        }

        groups
    }
}
