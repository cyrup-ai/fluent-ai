//! Configuration and trait implementations for Top-K processor
//!
//! Configuration structure and ConfigurableProcessor trait implementation.

use crate::processing::traits::{ConfigurableProcessor, ProcessingResult};

use super::core::TopKProcessor;

/// Configuration for top-k processor
#[derive(Debug, Clone)]
pub struct TopKConfig {
    pub k: usize,
}

impl ConfigurableProcessor for TopKProcessor {
    type Config = TopKConfig;

    fn update_config(&mut self, config: Self::Config) -> ProcessingResult<()> {
        let new_processor = Self::new(config.k)?;

        self.k = new_processor.k;
        self.is_identity = new_processor.is_identity;

        Ok(())
    }

    fn get_config(&self) -> Self::Config {
        TopKConfig { k: self.k }
    }
}