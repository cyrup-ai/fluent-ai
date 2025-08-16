//! Decoder type enumeration and content encoding mapping
//! Zero-allocation, blazing-fast decoder type detection with const optimizations

/// Decoder type enumeration for content encoding with feature gates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderType {
    #[cfg(feature = "gzip")]
    Gzip,
    #[cfg(feature = "brotli")]
    Brotli,
    #[cfg(feature = "zstd")]
    Zstd,
    #[cfg(feature = "deflate")]
    Deflate,
    Identity,
}

impl DecoderType {
    /// Map content-encoding header to decoder type with zero-allocation string matching
    #[inline]
    pub fn from_content_encoding(content_encoding: &str) -> Self {
        // Use byte-level comparison for maximum performance
        let bytes = content_encoding.as_bytes();

        match bytes {
            #[cfg(feature = "gzip")]
            b"gzip" | b"GZIP" => DecoderType::Gzip,
            #[cfg(feature = "brotli")]
            b"br" | b"BR" | b"brotli" | b"BROTLI" => DecoderType::Brotli,
            #[cfg(feature = "zstd")]
            b"zstd" | b"ZSTD" => DecoderType::Zstd,
            #[cfg(feature = "deflate")]
            b"deflate" | b"DEFLATE" => DecoderType::Deflate,
            _ => {
                // Fallback to case-insensitive matching for edge cases
                match content_encoding.to_ascii_lowercase().as_str() {
                    #[cfg(feature = "gzip")]
                    "gzip" => DecoderType::Gzip,
                    #[cfg(feature = "brotli")]
                    "br" | "brotli" => DecoderType::Brotli,
                    #[cfg(feature = "zstd")]
                    "zstd" => DecoderType::Zstd,
                    #[cfg(feature = "deflate")]
                    "deflate" => DecoderType::Deflate,
                    _ => DecoderType::Identity,
                }
            }
        }
    }

    /// Check if decoder requires decompression
    #[inline]
    pub(super) const fn needs_decompression(&self) -> bool {
        !matches!(self, DecoderType::Identity)
    }

    /// Get the content-encoding header value for this decoder type
    #[inline]
    pub(super) const fn as_content_encoding(&self) -> &'static str {
        match self {
            #[cfg(feature = "gzip")]
            DecoderType::Gzip => "gzip",
            #[cfg(feature = "brotli")]
            DecoderType::Brotli => "br",
            #[cfg(feature = "zstd")]
            DecoderType::Zstd => "zstd",
            #[cfg(feature = "deflate")]
            DecoderType::Deflate => "deflate",
            DecoderType::Identity => "identity",
        }
    }

    /// Get compression ratio estimate for buffer sizing optimization
    #[inline]
    pub(super) const fn compression_ratio_estimate(&self) -> f32 {
        match self {
            #[cfg(feature = "gzip")]
            DecoderType::Gzip => 3.0,
            #[cfg(feature = "brotli")]
            DecoderType::Brotli => 4.0,
            #[cfg(feature = "zstd")]
            DecoderType::Zstd => 3.5,
            #[cfg(feature = "deflate")]
            DecoderType::Deflate => 2.5,
            DecoderType::Identity => 1.0,
        }
    }

    /// Get optimal chunk size for streaming decompression
    #[inline]
    pub(super) const fn optimal_chunk_size(&self) -> usize {
        match self {
            #[cfg(feature = "gzip")]
            DecoderType::Gzip => 8192,
            #[cfg(feature = "brotli")]
            DecoderType::Brotli => 16384,
            #[cfg(feature = "zstd")]
            DecoderType::Zstd => 32768,
            #[cfg(feature = "deflate")]
            DecoderType::Deflate => 4096,
            DecoderType::Identity => 65536,
        }
    }
}

impl Default for DecoderType {
    #[inline]
    fn default() -> Self {
        DecoderType::Identity
    }
}
