//! H3 client configuration types and defaults
//!
//! Configuration structures for HTTP/3 client settings including
//! header size limits and protocol grease options.

/// H3 Client Configuration
#[derive(Clone)]
pub struct H3ClientConfig {
    /// Set the maximum HTTP/3 header size this client is willing to accept.
    ///
    /// See [header size constraints] section of the specification for details.
    ///
    /// [header size constraints]: https://www.rfc-editor.org/rfc/rfc9114.html#name-header-size-constraints
    ///
    /// Please see docs in [`Builder`] in [`h3`].
    ///
    /// [`Builder`]: https://docs.rs/h3/latest/h3/client/struct.Builder.html#method.max_field_section_size
    pub max_field_section_size: Option<u64>,

    /// Enable whether to send HTTP/3 protocol grease on the connections.
    ///
    /// Just like in HTTP/2, HTTP/3 also uses the concept of "grease"
    /// to prevent potential interoperability issues in the future.
    /// In HTTP/3, the concept of grease is used to ensure that the protocol can evolve
    /// and accommodate future changes without breaking existing implementations.
    ///
    /// Please see docs in [`Builder`] in [`h3`].
    ///
    /// [`Builder`]: https://docs.rs/h3/latest/h3/client/struct.Builder.html#method.send_grease
    pub send_grease: Option<bool>,
}

impl Default for H3ClientConfig {
    fn default() -> Self {
        Self {
            max_field_section_size: None,
            send_grease: None,
        }
    }
}

impl H3ClientConfig {
    /// Create new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum field section size
    pub fn with_max_field_section_size(mut self, size: u64) -> Self {
        self.max_field_section_size = Some(size);
        self
    }

    /// Enable or disable protocol grease
    pub fn with_grease(mut self, enable: bool) -> Self {
        self.send_grease = Some(enable);
        self
    }
}
