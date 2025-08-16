//! Accept-Encoding header handling and compression format detection
//! Zero-allocation, blazing-fast encoding negotiation with const optimizations

use http::HeaderMap;

/// Accept-Encoding capabilities with feature-gated compression support
#[derive(Clone, Copy, Debug)]
pub struct Accepts {
    #[cfg(feature = "gzip")]
    pub(super) gzip: bool,
    #[cfg(feature = "brotli")]
    pub(super) brotli: bool,
    #[cfg(feature = "zstd")]
    pub(super) zstd: bool,
    #[cfg(feature = "deflate")]
    pub(super) deflate: bool,
}

impl Accepts {
    /// Create accepts with no compression enabled
    #[inline]
    pub const fn none() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }

    /// Create accepts with only gzip enabled
    #[inline]
    pub const fn gzip() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: true,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }

    /// Create accepts with only brotli enabled
    #[inline]
    pub const fn brotli() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: true,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }

    /// Create accepts with only zstd enabled
    #[inline]
    pub const fn zstd() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: true,
            #[cfg(feature = "deflate")]
            deflate: false,
        }
    }

    /// Create accepts with only deflate enabled
    #[inline]
    pub const fn deflate() -> Self {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: false,
            #[cfg(feature = "brotli")]
            brotli: false,
            #[cfg(feature = "zstd")]
            zstd: false,
            #[cfg(feature = "deflate")]
            deflate: true,
        }
    }

    /// Convert to accept-encoding header string with const optimization
    #[inline]
    pub(super) const fn as_str(&self) -> Option<&'static str> {
        match (
            self.is_gzip(),
            self.is_brotli(),
            self.is_zstd(),
            self.is_deflate(),
        ) {
            (true, true, true, true) => Some("gzip, br, zstd, deflate"),
            (true, true, false, true) => Some("gzip, br, deflate"),
            (true, true, true, false) => Some("gzip, br, zstd"),
            (true, true, false, false) => Some("gzip, br"),
            (true, false, true, true) => Some("gzip, zstd, deflate"),
            (true, false, false, true) => Some("gzip, deflate"),
            (false, true, true, true) => Some("br, zstd, deflate"),
            (false, true, false, true) => Some("br, deflate"),
            (true, false, true, false) => Some("gzip, zstd"),
            (true, false, false, false) => Some("gzip"),
            (false, true, true, false) => Some("br, zstd"),
            (false, true, false, false) => Some("br"),
            (false, false, true, true) => Some("zstd, deflate"),
            (false, false, true, false) => Some("zstd"),
            (false, false, false, true) => Some("deflate"),
            (false, false, false, false) => None,
        }
    }

    /// Check if gzip is enabled with const optimization
    #[inline]
    const fn is_gzip(&self) -> bool {
        #[cfg(feature = "gzip")]
        {
            self.gzip
        }

        #[cfg(not(feature = "gzip"))]
        {
            false
        }
    }

    /// Check if brotli is enabled with const optimization
    #[inline]
    const fn is_brotli(&self) -> bool {
        #[cfg(feature = "brotli")]
        {
            self.brotli
        }

        #[cfg(not(feature = "brotli"))]
        {
            false
        }
    }

    /// Check if zstd is enabled with const optimization
    #[inline]
    const fn is_zstd(&self) -> bool {
        #[cfg(feature = "zstd")]
        {
            self.zstd
        }

        #[cfg(not(feature = "zstd"))]
        {
            false
        }
    }

    /// Check if deflate is enabled with const optimization
    #[inline]
    const fn is_deflate(&self) -> bool {
        #[cfg(feature = "deflate")]
        {
            self.deflate
        }

        #[cfg(not(feature = "deflate"))]
        {
            false
        }
    }
}

impl Default for Accepts {
    /// Default enables all available compression formats
    #[inline]
    fn default() -> Accepts {
        Accepts {
            #[cfg(feature = "gzip")]
            gzip: true,
            #[cfg(feature = "brotli")]
            brotli: true,
            #[cfg(feature = "zstd")]
            zstd: true,
            #[cfg(feature = "deflate")]
            deflate: true,
        }
    }
}

/// Parse Accept-Encoding header with zero-allocation string processing
#[inline]
pub fn get_accept_encoding(headers: &HeaderMap) -> Accepts {
    let mut accepts = Accepts::none();

    if let Some(accept_encoding) = headers.get("accept-encoding") {
        if let Ok(accept_str) = accept_encoding.to_str() {
            // Use stack-allocated lowercase conversion for performance
            let accept_lower = accept_str.to_ascii_lowercase();

            #[cfg(feature = "gzip")]
            {
                accepts.gzip = accept_lower.contains("gzip");
            }

            #[cfg(feature = "brotli")]
            {
                accepts.brotli = accept_lower.contains("br");
            }

            #[cfg(feature = "zstd")]
            {
                accepts.zstd = accept_lower.contains("zstd");
            }

            #[cfg(feature = "deflate")]
            {
                accepts.deflate = accept_lower.contains("deflate");
            }
        }
    }

    accepts
}
