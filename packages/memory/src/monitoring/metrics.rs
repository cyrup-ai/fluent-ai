//! Metrics collection and export

use std::collections::HashMap;

use prometheus::{self, Counter, Gauge, Histogram, HistogramOpts, Registry};

/// Metric types
#[derive(Debug, Clone)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

/// Metric value
#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(f64),
    Gauge(f64),
    Histogram(f64),
}

/// Metrics collector
pub struct MetricsCollector {
    /// Registered metrics
    metrics: HashMap<String, Box<dyn Metric>>,
}

impl MetricsCollector {
    /// Create a new collector
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Register a metric
    pub fn register(&mut self, name: String, metric: Box<dyn Metric>) {
        self.metrics.insert(name, metric);
    }

    /// Record a value
    pub fn record(&self, name: &str, value: f64) {
        if let Some(metric) = self.metrics.get(name) {
            metric.record(value);
        }
    }

    /// Get all metrics
    pub fn collect(&self) -> HashMap<String, MetricValue> {
        self.metrics
            .iter()
            .map(|(name, metric)| (name.clone(), metric.value()))
            .collect()
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Metric trait
pub trait Metric: Send + Sync {
    /// Record a value
    fn record(&self, value: f64);

    /// Get current value
    fn value(&self) -> MetricValue;

    /// Get metric type
    fn metric_type(&self) -> MetricType;
}

/// Counter metric wrapper
pub struct CounterMetric {
    counter: Counter,
    registry: Registry,
}

impl CounterMetric {
    pub fn new(name: &str, help: &str) -> Self {
        let registry = Registry::new();
        let counter = Counter::new(name, help).unwrap_or_else(|e| {
            eprintln!(
                "Warning: Failed to create counter metric '{}': {}. Using fallback counter.",
                name, e
            );
            Counter::new("fallback_counter", "Fallback counter metric").unwrap_or_else(|_| {
                // If even the fallback fails, create a minimal counter
                Counter::new("minimal_counter", "Minimal counter")
                    .expect("Failed to create minimal counter")
            })
        });

        if let Err(e) = registry.register(Box::new(counter.clone())) {
            eprintln!("Failed to register counter metric '{}': {}", name, e);
        }

        Self { counter, registry }
    }
}

impl Metric for CounterMetric {
    fn record(&self, value: f64) {
        self.counter.inc_by(value);
    }

    fn value(&self) -> MetricValue {
        MetricValue::Counter(self.counter.get())
    }

    fn metric_type(&self) -> MetricType {
        MetricType::Counter
    }
}

/// Gauge metric wrapper
pub struct GaugeMetric {
    gauge: Gauge,
    registry: Registry,
}

impl GaugeMetric {
    pub fn new(name: &str, help: &str) -> Self {
        let registry = Registry::new();
        let gauge = Gauge::new(name, help).unwrap_or_else(|e| {
            eprintln!(
                "Warning: Failed to create gauge metric '{}': {}. Using fallback gauge.",
                name, e
            );
            Gauge::new("fallback_gauge", "Fallback gauge metric").unwrap_or_else(|_| {
                // If even the fallback fails, create a minimal gauge
                Gauge::new("minimal_gauge", "Minimal gauge")
                    .expect("Failed to create minimal gauge")
            })
        });

        if let Err(e) = registry.register(Box::new(gauge.clone())) {
            eprintln!("Failed to register gauge metric '{}': {}", name, e);
        }

        Self { gauge, registry }
    }
}

impl Metric for GaugeMetric {
    fn record(&self, value: f64) {
        self.gauge.set(value);
    }

    fn value(&self) -> MetricValue {
        MetricValue::Gauge(self.gauge.get())
    }

    fn metric_type(&self) -> MetricType {
        MetricType::Gauge
    }
}

/// Histogram metric wrapper
pub struct HistogramMetric {
    histogram: Histogram,
    registry: Registry,
}

impl HistogramMetric {
    pub fn new(name: &str, help: &str) -> Self {
        let registry = Registry::new();
        let histogram = Histogram::with_opts(HistogramOpts::new(name, help)).unwrap_or_else(|e| {
            eprintln!(
                "Warning: Failed to create histogram metric '{}': {}. Using fallback histogram.",
                name, e
            );
            Histogram::with_opts(HistogramOpts::new(
                "fallback_histogram",
                "Fallback histogram metric",
            ))
            .unwrap_or_else(|_| {
                // If even the fallback fails, create a minimal histogram
                Histogram::with_opts(HistogramOpts::new("minimal_histogram", "Minimal histogram"))
                    .expect("Failed to create minimal histogram")
            })
        });

        if let Err(e) = registry.register(Box::new(histogram.clone())) {
            eprintln!("Failed to register histogram metric '{}': {}", name, e);
        }

        Self {
            histogram,
            registry,
        }
    }
}

impl Metric for HistogramMetric {
    fn record(&self, value: f64) {
        self.histogram.observe(value);
    }

    fn value(&self) -> MetricValue {
        // Return the sum for simplicity
        MetricValue::Histogram(self.histogram.get_sample_sum())
    }

    fn metric_type(&self) -> MetricType {
        MetricType::Histogram
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_metric_creation() {
        let counter = CounterMetric::new("test_counter", "Test counter metric");
        assert_eq!(counter.metric_type(), MetricType::Counter);

        // Test recording values
        counter.record(1.0);
        counter.record(2.5);

        // Should not panic
        let value = counter.value();
        match value {
            MetricValue::Counter(val) => assert!(val >= 0.0),
            _ => panic!("Expected Counter value"),
        }
    }

    #[test]
    fn test_gauge_metric_creation() {
        let gauge = GaugeMetric::new("test_gauge", "Test gauge metric");
        assert!(matches!(gauge.metric_type(), MetricType::Gauge));

        // Test recording values
        gauge.record(10.0);
        gauge.record(5.0);

        // Should not panic
        let value = gauge.value();
        match value {
            MetricValue::Gauge(_) => {} // Value can be any float
            _ => panic!("Expected Gauge value"),
        }
    }

    #[test]
    fn test_histogram_metric_creation() {
        let histogram = HistogramMetric::new("test_histogram", "Test histogram metric");
        assert!(matches!(histogram.metric_type(), MetricType::Histogram));

        // Test recording values
        histogram.record(1.0);
        histogram.record(2.0);
        histogram.record(3.0);

        // Should not panic
        let value = histogram.value();
        match value {
            MetricValue::Histogram(val) => assert!(val >= 0.0),
            _ => panic!("Expected Histogram value"),
        }
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        // Test adding metrics
        collector.register(
            "counter1".to_string(),
            Box::new(CounterMetric::new("counter1", "Counter 1")),
        );
        collector.register(
            "gauge1".to_string(),
            Box::new(GaugeMetric::new("gauge1", "Gauge 1")),
        );
        collector.register(
            "histogram1".to_string(),
            Box::new(HistogramMetric::new("histogram1", "Histogram 1")),
        );

        // Test recording values
        collector.record("counter1", 5.0);
        collector.record("gauge1", 10.0);
        collector.record("histogram1", 2.5);

        // Test collecting metrics
        let metrics = collector.collect();
        assert_eq!(metrics.len(), 3);
        assert!(metrics.contains_key("counter1"));
        assert!(metrics.contains_key("gauge1"));
        assert!(metrics.contains_key("histogram1"));
    }
}
