use fluent_ai_candle::streaming::formats::utils::*;
use fluent_ai_candle::streaming::formats::*;

#[test]
    fn test_format_parsing() {
        assert_eq!(
            format_utils::parse_format("json").unwrap(),
            OutputFormat::Json
        );
        assert_eq!(
            format_utils::parse_format("SSE").unwrap(),
            OutputFormat::ServerSentEvents
        );
        assert_eq!(
            format_utils::parse_format("WebSocket").unwrap(),
            OutputFormat::WebSocket
        );
        assert_eq!(
            format_utils::parse_format("PLAIN").unwrap(),
            OutputFormat::PlainText
        );

        if let OutputFormat::Custom(name) = format_utils::parse_format("custom:test").unwrap() {
            assert_eq!(name, "test");
        } else {
            panic!("Expected custom format");
        }

        assert!(format_utils::parse_format("invalid").is_err());
    }

    #[test]
    fn test_mime_types() {
        assert_eq!(format_utils::get_mime_type(&OutputFormat::Json), "application/json");
        assert_eq!(
            format_utils::get_mime_type(&OutputFormat::ServerSentEvents),
            "text/event-stream"
        );
        assert_eq!(
            format_utils::get_mime_type(&OutputFormat::PlainText),
            "text/plain; charset=utf-8"
        );
        assert_eq!(
            format_utils::get_mime_type(&OutputFormat::Custom("csv".to_string())),
            "text/csv; charset=utf-8"
        );
    }

    #[test]
    fn test_streaming_support() {
        assert!(format_utils::supports_streaming(&OutputFormat::ServerSentEvents));
        assert!(format_utils::supports_streaming(&OutputFormat::WebSocket));
        assert!(format_utils::supports_streaming(&OutputFormat::Raw));
        assert!(!format_utils::supports_streaming(&OutputFormat::Json)); // Not real-time
    }

    #[test]
    fn test_cache_headers() {
        let sse_headers = format_utils::get_cache_headers(&OutputFormat::ServerSentEvents);
        assert!(sse_headers.contains(&("Cache-Control", "no-cache")));
        assert!(sse_headers.contains(&("Connection", "keep-alive")));

        let ws_headers = format_utils::get_cache_headers(&OutputFormat::WebSocket);
        assert!(ws_headers.contains(&("Connection", "Upgrade")));
        assert!(ws_headers.contains(&("Upgrade", "websocket")));
    }

    #[test]
    fn test_buffer_management() {
        let mut buffer = buffer_utils::FormatBuffer::new(&OutputFormat::Json);
        assert_eq!(buffer.capacity(), 1024);

        let test_data = "test data";
        let result = buffer.write_formatted(test_data).unwrap();
        assert_eq!(result, test_data);

        buffer.clear();
        assert_eq!(buffer.as_str().unwrap(), "");
    }

    #[test]
    fn test_chunk_size_calculation() {
        let json_chunk = buffer_utils::calculate_chunk_size(&OutputFormat::Json, 2000);
        assert!(json_chunk <= 1004); // 1024 - 20 overhead

        let sse_chunk = buffer_utils::calculate_chunk_size(&OutputFormat::ServerSentEvents, 3000);
        assert!(sse_chunk <= 1948); // 2048 - 100 overhead
    }

    #[test]
    fn test_string_capacity_optimization() {
        let json_capacity = buffer_utils::optimize_string_capacity(&OutputFormat::Json, 1000);
        assert_eq!(json_capacity, 1100); // 1000 * 1.1

        let sse_capacity = buffer_utils::optimize_string_capacity(&OutputFormat::ServerSentEvents, 1000);
        assert_eq!(sse_capacity, 1500); // 1000 * 1.5
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = perf_utils::FormatMetrics::new();
        let start = metrics.start_timing();
        
        // Simulate work
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        metrics.end_timing(start);
        metrics.record_sizes(1000, 800);

        assert!(metrics.format_time.as_millis() >= 1);
        assert_eq!(metrics.buffer_size, 1000);
        assert_eq!(metrics.output_size, 800);
        assert_eq!(metrics.compression_ratio, 0.8);
        assert!(metrics.throughput_bps() > 0.0);
    }

    #[test]
    fn test_buffer_capacity_management() {
        let mut buffer = buffer_utils::FormatBuffer::new(&OutputFormat::PlainText);
        let initial_capacity = buffer.capacity();

        buffer.ensure_capacity(initial_capacity * 2);
        assert!(buffer.capacity() >= initial_capacity * 2);
    }
