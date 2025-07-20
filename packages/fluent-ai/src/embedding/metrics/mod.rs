//! Comprehensive metrics and monitoring for embedding operations
//!
//! This module provides enterprise-grade monitoring capabilities including:
//! - Quality assessment with SIMD-optimized similarity validation
//! - Performance monitoring with real-time analytics
//! - Resource utilization tracking with lock-free collection
//! - Anomaly detection and alerting systems

pub mod quality_analyzer;
pub mod performance_monitor;

// Re-export core types
pub use quality_analyzer::{
    QualityAnalyzer, QualityDataPoint, QualityAnalysisMetrics, QualityAnalysisError,
    StatisticalSummary, ProviderPerformanceMetrics, OutlierDetector, RingBuffer
};
pub use performance_monitor::{
    PerformanceMonitor, PerformanceMetric, LatencyHistogram, LatencyStats,
    ThroughputWindow, CacheMetrics, CacheStats, ResourceMetrics, ResourceStats,
    PerformanceAlert, AlertSeverity, AlertConfig, PerformanceMonitorError,
    GlobalMetrics, ProviderMetrics, AnomalyDetector
};