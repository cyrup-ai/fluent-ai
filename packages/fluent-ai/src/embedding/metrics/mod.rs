//! Comprehensive metrics and monitoring for embedding operations
//!
//! This module provides enterprise-grade monitoring capabilities including:
//! - Quality assessment with SIMD-optimized similarity validation
//! - Performance monitoring with real-time analytics
//! - Resource utilization tracking with lock-free collection
//! - Anomaly detection and alerting systems

pub mod performance_monitor;
pub mod quality_analyzer;

// Re-export core types
pub use performance_monitor::{
    AlertConfig, AlertSeverity, AnomalyDetector, CacheMetrics, CacheStats, GlobalMetrics,
    LatencyHistogram, LatencyStats, PerformanceAlert, PerformanceMetric, PerformanceMonitor,
    PerformanceMonitorError, ProviderMetrics, ResourceMetrics, ResourceStats, ThroughputWindow,
};
pub use quality_analyzer::{
    OutlierDetector, ProviderPerformanceMetrics, QualityAnalysisError, QualityAnalysisMetrics,
    QualityAnalyzer, QualityDataPoint, RingBuffer, StatisticalSummary,
};
