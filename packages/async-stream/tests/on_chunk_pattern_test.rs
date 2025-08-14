//! Test pattern matching syntax with on_chunk builder
//!
//! Verifies that the immutable builder pattern works with the on_chunk_pattern macro
//! for clean Result<T, E> -> T transformations.

use async_stream::{AsyncStream, on_chunk_pattern};

#[derive(Debug, Default, Clone, PartialEq)]
struct TestData {
    value: i32,
    processed: bool,
}

#[derive(Debug)]
struct TestError {
    message: String,
}

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TestError: {}", self.message)
    }
}

impl std::error::Error for TestError {}

#[test]
fn test_on_chunk_pattern_matching_with_builder() {
    // Test the pattern matching syntax with the builder
    let processor = on_chunk_pattern!(|result| {
        Ok => {
            // Process successful values
            TestData {
                value: result.value * 2,
                processed: true,
            }
        },
        Err(e) => {
            // Handle errors gracefully
            println!("Processing error: {}", e);
            TestData {
                value: -1,
                processed: false,
            }
        }
    });

    // Create stream using the immutable builder pattern
    let stream = AsyncStream::<TestData, _, 32>::on_chunk(processor).with_channel(|sender| {
        // Send some successful values
        let _ = sender.try_send(TestData {
            value: 5,
            processed: false,
        });
        let _ = sender.try_send(TestData {
            value: 10,
            processed: false,
        });

        // Send an error (using send_error method)
        let _ = sender.send_error(TestError {
            message: "test error".to_string(),
        });

        // Send another successful value
        let _ = sender.try_send(TestData {
            value: 15,
            processed: false,
        });
    });

    // Collect results
    let results = stream.collect();

    // Verify pattern matching worked correctly
    assert_eq!(results.len(), 4);

    // First successful value: 5 * 2 = 10
    assert_eq!(
        results[0],
        TestData {
            value: 10,
            processed: true
        }
    );

    // Second successful value: 10 * 2 = 20
    assert_eq!(
        results[1],
        TestData {
            value: 20,
            processed: true
        }
    );

    // Error case: should use default error handling
    assert_eq!(
        results[2],
        TestData {
            value: -1,
            processed: false
        }
    );

    // Third successful value: 15 * 2 = 30
    assert_eq!(
        results[3],
        TestData {
            value: 30,
            processed: true
        }
    );
}

#[test]
fn test_builder_empty_stream_with_processor() {
    let processor = |result: Result<String, impl std::error::Error>| -> String {
        match result {
            Ok(s) => s.to_uppercase(),
            Err(_) => "ERROR".to_string(),
        }
    };

    let stream = AsyncStream::<String, _, 16>::on_chunk(processor).empty();
    let results = stream.collect();

    // Empty stream should have no results
    assert!(results.is_empty());
}

#[test]
fn test_builder_channel_internal_with_processor() {
    let processor = |result: Result<i32, impl std::error::Error>| -> i32 {
        match result {
            Ok(n) => n + 100,
            Err(_) => -999,
        }
    };

    let (sender, stream) = AsyncStream::<i32, _, 8>::on_chunk(processor).channel_internal();

    // Send some values
    let _ = sender.try_send(1);
    let _ = sender.try_send(2);
    let _ = sender.try_send(3);

    let results = stream.collect();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0], 101); // 1 + 100
    assert_eq!(results[1], 102); // 2 + 100  
    assert_eq!(results[2], 103); // 3 + 100
}
