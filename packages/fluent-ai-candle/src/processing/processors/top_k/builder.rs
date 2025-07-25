//! Builder pattern for Top-K processor construction
//!
//! Provides fluent builder interface for creating TopKProcessor instances.

use crate::processing::traits::ProcessingResult;

use super::core::TopKProcessor;

/// Builder for top-k processor with validation and presets
#[derive(Debug, Clone, Default)]
pub struct TopKBuilder {
    k: Option<usize>,
}

impl TopKBuilder {
    /// Create a new top-k builder
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set k value
    #[inline(always)]
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Use small preset (k=20)
    #[inline(always)]
    pub fn small(mut self) -> Self {
        self.k = Some(20);
        self
    }

    /// Use medium preset (k=50)
    #[inline(always)]
    pub fn medium(mut self) -> Self {
        self.k = Some(50);
        self
    }

    /// Use large preset (k=100)
    #[inline(always)]
    pub fn large(mut self) -> Self {
        self.k = Some(100);
        self
    }

    /// Disable top-k filtering
    #[inline(always)]
    pub fn disabled(mut self) -> Self {
        self.k = Some(0);
        self
    }

    /// Build the top-k processor
    pub fn build(self) -> ProcessingResult<TopKProcessor> {
        let k = self.k.unwrap_or(0); // Default to disabled
        TopKProcessor::new(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_new() {
        let builder = TopKBuilder::new();
        let processor = builder.build().unwrap();
        assert_eq!(processor.k(), 0); // Default to disabled
        assert!(processor.is_identity());
    }

    #[test]
    fn test_builder_k() {
        let processor = TopKBuilder::new().k(75).build().unwrap();
        assert_eq!(processor.k(), 75);
        assert!(!processor.is_identity());
    }

    #[test]
    fn test_builder_presets() {
        let small = TopKBuilder::new().small().build().unwrap();
        assert_eq!(small.k(), 20);
        
        let medium = TopKBuilder::new().medium().build().unwrap();
        assert_eq!(medium.k(), 50);
        
        let large = TopKBuilder::new().large().build().unwrap();
        assert_eq!(large.k(), 100);
    }

    #[test]
    fn test_builder_disabled() {
        let processor = TopKBuilder::new().disabled().build().unwrap();
        assert_eq!(processor.k(), 0);
        assert!(processor.is_identity());
    }

    #[test]
    fn test_builder_chaining() {
        // Later calls should override earlier ones
        let processor = TopKBuilder::new()
            .small()   // k=20
            .medium()  // k=50 (overrides)
            .k(30)     // k=30 (overrides)
            .build()
            .unwrap();
        assert_eq!(processor.k(), 30);
    }

    #[test]
    fn test_builder_validation() {
        let result = TopKBuilder::new()
            .k(super::super::core::MAX_TOP_K + 1)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_default() {
        let builder = TopKBuilder::default();
        let processor = builder.build().unwrap();
        assert_eq!(processor.k(), 0);
    }
}