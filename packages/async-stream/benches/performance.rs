//! Performance benchmarks for fluent_ai_async improvements
//!
//! Validates thread pool, dynamic capacity, and notification batching optimizations

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fluent_ai_async::prelude::*;

#[derive(Debug, Clone, Default)]
struct TestChunk {
    data: String,
    error_message: Option<String>,
}

impl MessageChunk for TestChunk {
    fn bad_chunk(error: String) -> Self {
        Self {
            data: "[ERROR]".to_string(),
            error_message: Some(error),
        }
    }

    fn error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }
}

impl TestChunk {
    fn new(data: String) -> Self {
        Self {
            data,
            error_message: None,
        }
    }
}

/// Benchmark thread pool vs raw thread spawning
fn bench_thread_pool_vs_raw_spawning(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_execution");

    for task_count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*task_count as u64));

        // Benchmark work-stealing thread pool
        group.bench_with_input(
            BenchmarkId::new("thread_pool", task_count),
            task_count,
            |b, &task_count| {
                b.iter(|| {
                    let executor = global_executor();
                    let counter = Arc::new(AtomicUsize::new(0));
                    let receivers: Vec<_> = (0..task_count)
                        .map(|_| {
                            let counter_clone = Arc::clone(&counter);
                            executor.execute_with_result(move || {
                                counter_clone.fetch_add(1, Ordering::Relaxed);
                                42
                            })
                        })
                        .collect();

                    // Wait for all tasks to complete
                    for rx in receivers {
                        black_box(rx.recv().expect("Task should complete"));
                    }

                    assert_eq!(counter.load(Ordering::Relaxed), task_count);
                })
            },
        );

        // Benchmark raw thread spawning
        group.bench_with_input(
            BenchmarkId::new("raw_threads", task_count),
            task_count,
            |b, &task_count| {
                b.iter(|| {
                    let counter = Arc::new(AtomicUsize::new(0));
                    let handles: Vec<_> = (0..task_count)
                        .map(|_| {
                            let counter_clone = Arc::clone(&counter);
                            std::thread::spawn(move || {
                                counter_clone.fetch_add(1, Ordering::Relaxed);
                                42
                            })
                        })
                        .collect();

                    // Wait for all threads to complete
                    for handle in handles {
                        black_box(handle.join().expect("Thread should complete"));
                    }

                    assert_eq!(counter.load(Ordering::Relaxed), task_count);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark fixed vs dynamic capacity performance
fn bench_fixed_vs_dynamic_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("capacity_variants");

    for item_count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*item_count as u64));

        // Benchmark fixed capacity (const-generic)
        group.bench_with_input(
            BenchmarkId::new("fixed_capacity", item_count),
            item_count,
            |b, &item_count| {
                b.iter(|| {
                    let (sender, stream) = AsyncStream::<TestChunk, 1024>::channel();

                    std::thread::spawn(move || {
                        for i in 0..item_count {
                            let chunk = TestChunk::new(format!("data_{}", i));
                            if sender.send(chunk).is_err() {
                                break;
                            }
                        }
                    });

                    let results: Vec<TestChunk> = stream.collect();
                    black_box(results);
                })
            },
        );

        // Benchmark dynamic capacity
        group.bench_with_input(
            BenchmarkId::new("dynamic_capacity", item_count),
            item_count,
            |b, &item_count| {
                b.iter(|| {
                    let (sender, stream) = AsyncStream::<TestChunk, 1024>::channel_dynamic();

                    std::thread::spawn(move || {
                        for i in 0..item_count {
                            let chunk = TestChunk::new(format!("data_{}", i));
                            if sender.send(chunk).is_err() {
                                break;
                            }
                        }
                    });

                    let results: Vec<TestChunk> = stream.collect();
                    black_box(results);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark notification batching impact
fn bench_notification_batching(c: &mut Criterion) {
    let mut group = c.benchmark_group("notification_batching");

    for rate in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*rate as u64));

        // High-frequency sending to test batching effectiveness
        group.bench_with_input(
            BenchmarkId::new("high_frequency_send", rate),
            rate,
            |b, &rate| {
                b.iter(|| {
                    let (sender, stream) = AsyncStream::<TestChunk, 2048>::channel();

                    let producer_handle = std::thread::spawn(move || {
                        for i in 0..rate {
                            let chunk = TestChunk::new(format!("chunk_{}", i));
                            if sender.send(chunk).is_err() {
                                break;
                            }

                            // Minimal delay to simulate real-world high-frequency sending
                            if i % 100 == 0 {
                                std::thread::yield_now();
                            }
                        }
                    });

                    let consumer_handle = std::thread::spawn(move || {
                        let results: Vec<TestChunk> = stream.collect();
                        results.len()
                    });

                    producer_handle.join().expect("Producer should complete");
                    let result_count = consumer_handle.join().expect("Consumer should complete");
                    black_box(result_count);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark overall stream throughput
fn bench_stream_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_throughput");

    for payload_size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Bytes(*payload_size as u64 * 1000));

        group.bench_with_input(
            BenchmarkId::new("end_to_end", payload_size),
            payload_size,
            |b, &payload_size| {
                b.iter(|| {
                    let payload = "x".repeat(payload_size);
                    let stream = AsyncStream::<TestChunk, 2048>::with_channel(move |sender| {
                        for i in 0..1000 {
                            let chunk = TestChunk::new(format!("{}_{}", payload, i));
                            if sender.send(chunk).is_err() {
                                break;
                            }
                        }
                    });

                    let results: Vec<TestChunk> = stream.collect();
                    black_box(results);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark error handling performance with collect_or_else
fn bench_error_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_handling");

    for error_rate in [0, 10, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("collect_or_else", error_rate),
            error_rate,
            |b, &error_rate| {
                b.iter(|| {
                    let stream = AsyncStream::<TestChunk, 1024>::with_channel(move |sender| {
                        for i in 0..1000 {
                            if error_rate > 0 && i % (100 / error_rate) == 0 {
                                // Send error chunk
                                let error_chunk = TestChunk::bad_chunk(format!("Error at {}", i));
                                if sender.send(error_chunk).is_err() {
                                    break;
                                }
                            } else {
                                // Send normal chunk
                                let chunk = TestChunk::new(format!("data_{}", i));
                                if sender.send(chunk).is_err() {
                                    break;
                                }
                            }
                        }
                    });

                    let results: Vec<TestChunk> = stream.collect_or_else(|_error_chunk| {
                        vec![TestChunk::new("error_handled".to_string())]
                    });

                    black_box(results);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_thread_pool_vs_raw_spawning,
    bench_fixed_vs_dynamic_capacity,
    bench_notification_batching,
    bench_stream_throughput,
    bench_error_handling
);
criterion_main!(benches);
