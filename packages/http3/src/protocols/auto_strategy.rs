//! Auto Protocol Strategy with Fallback Support
//!
//! Automatically selects the best protocol and falls back to alternatives on failure.

use std::sync::Arc;

use crate::protocols::strategy_trait::ProtocolStrategy;
use crate::protocols::h2::strategy::H2Strategy;
use crate::protocols::h3::strategy::H3Strategy;
use crate::protocols::strategy::{H2Config, H3Config, ProtocolConfigs};
use crate::protocols::core::HttpVersion;
use crate::http::{HttpRequest, HttpResponse};

/// Auto-selecting Protocol Strategy with Fallback
///
/// Tries protocols in order of preference and automatically
/// falls back to alternatives if the primary protocol fails.
pub struct AutoStrategy {
    /// Primary protocol to try first
    primary: Box<dyn ProtocolStrategy>,
    /// Fallback protocol if primary fails
    fallback: Option<Box<dyn ProtocolStrategy>>,
    /// Protocol preference order
    prefer: Vec<HttpVersion>,
}

impl AutoStrategy {
    /// Create a new auto strategy with preference order
    pub fn new(prefer: Vec<HttpVersion>, configs: ProtocolConfigs) -> Self {
        // Build strategies based on preference
        let (primary, fallback) = Self::build_strategies(&prefer, &configs);
        
        Self {
            primary,
            fallback,
            prefer,
        }
    }
    
    /// Build primary and fallback strategies based on preferences
    fn build_strategies(
        prefer: &[HttpVersion],
        configs: &ProtocolConfigs,
    ) -> (Box<dyn ProtocolStrategy>, Option<Box<dyn ProtocolStrategy>>) {
        let mut strategies = Vec::new();
        
        for version in prefer {
            match version {
                HttpVersion::Http2 => {
                    strategies.push(Box::new(H2Strategy::new(crate::protocols::h2::strategy::H2Config {
                        enable_push: configs.h2.enable_push,
                        max_concurrent_streams: configs.h2.max_concurrent_streams,
                        initial_window_size: configs.h2.initial_window_size,
                        max_frame_size: configs.h2.max_frame_size,
                        max_header_list_size: 16384, // Default value
                    })) as Box<dyn ProtocolStrategy>);
                },
                HttpVersion::Http3 => {
                    strategies.push(Box::new(H3Strategy::new(configs.h3.clone())) as Box<dyn ProtocolStrategy>);
                },
                _ => continue,
            }
        }
        
        // If no preferences or unsupported versions, default to H3 -> H2
        if strategies.is_empty() {
            strategies.push(Box::new(H3Strategy::new(configs.h3.clone())));
            strategies.push(Box::new(H2Strategy::new(crate::protocols::h2::strategy::H2Config {
                enable_push: configs.h2.enable_push,
                max_concurrent_streams: configs.h2.max_concurrent_streams,
                initial_window_size: configs.h2.initial_window_size,
                max_frame_size: configs.h2.max_frame_size,
                max_header_list_size: 16384, // Default value
            })));
        }
        
        let mut iter = strategies.into_iter();
        let primary = iter.next().unwrap();
        let fallback = iter.next();
        
        (primary, fallback)
    }
    
    /// Try executing with primary, fallback on failure
    fn execute_with_fallback(&self, request: HttpRequest) -> HttpResponse {
        // Try primary protocol first
        let response = self.primary.execute(request.clone());
        
        // Check if primary succeeded
        if !response.is_error() {
            return response;
        }
        
        // Try fallback if available
        if let Some(ref fallback) = self.fallback {
            log::warn!(
                "Primary protocol {} failed, trying fallback {}",
                self.primary.protocol_name(),
                fallback.protocol_name()
            );
            
            return fallback.execute(request);
        }
        
        // No fallback available, return the error response
        response
    }
}

impl ProtocolStrategy for AutoStrategy {
    fn execute(&self, request: HttpRequest) -> HttpResponse {
        self.execute_with_fallback(request)
    }
    
    fn protocol_name(&self) -> &'static str {
        "Auto (with fallback)"
    }
    
    fn supports_push(&self) -> bool {
        // Support push if primary protocol supports it
        self.primary.supports_push()
    }
    
    fn max_concurrent_streams(&self) -> usize {
        // Use primary protocol's limit
        self.primary.max_concurrent_streams()
    }
}