use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use fluent_ai_memory::cache::{Cache, CachePolicy, MemoryCache};
use fluent_ai_memory::core::Memory;
use fluent_ai_memory::storage::{graph::GraphDB, vector::VectorStore};
use rand::prelude::*;
use rand::{thread_rng, Rng};
use tokio::runtime::Runtime;

fn setup_benchmark_environment() -> (Arc<GraphDB>, Arc<VectorStore>, Runtime) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let graph_db =
        Arc::new(runtime.block_on(async { GraphDB::new("benchmark_graph.db").await.unwrap() }));
    let vector_store = Arc::new(
        runtime.block_on(async { VectorStore::new("benchmark_vector.db").await.unwrap() }),
    );
    (graph_db, vector_store, runtime)
}

fn create_test_memory(id: &str) -> Memory {
    Memory::new_with_id(id, "Test content", None, None)
}

fn create_memory_vector(size: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.r#gen::<f32>()).collect()
}

fn bench_memory_creation(c: &mut Criterion) {
    c.bench_function("memory_creation", |b| {
        b.iter(|| {
            let memory = create_test_memory("test_id");
            black_box(memory);
        });
    });
}

fn bench_memory_serialization(c: &mut Criterion) {
    let memory = create_test_memory("test_id");
    c.bench_function("memory_serialization", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&memory).unwrap();
            black_box(serialized);
        });
    });
}

fn bench_memory_storage(c: &mut Criterion) {
    let (graph_db, vector_store, runtime) = setup_benchmark_environment();
    let memory = create_test_memory("test_id");
    let vector = create_memory_vector(128);

    c.bench_function("memory_storage", |b| {
        b.iter(|| {
            runtime.block_on(async {
                graph_db.store_entity(&memory).await.unwrap();
                vector_store
                    .store_vector(memory.id(), &vector)
                    .await
                    .unwrap();
            });
        });
    });
}

fn bench_vector_search(c: &mut Criterion) {
    let (_, vector_store, runtime) = setup_benchmark_environment();
    let query_vector = create_memory_vector(128);

    c.bench_function("vector_search", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let results = vector_store
                    .search_vectors(&query_vector, 10)
                    .await
                    .unwrap();
                black_box(results);
            });
        });
    });
}

fn bench_memory_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_cache");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let policies = [CachePolicy::FIFO, CachePolicy::LRU, CachePolicy::LFU];
    let memories: Vec<Memory> = (0..1000)
        .map(|i| create_test_memory(&format!("mem_{}", i)))
        .collect();

    for &policy in &policies {
        let mut cache = MemoryCache::new(100, policy);
        for i in 0..500 {
            cache.put(memories[i].clone());
        }

        // Benchmark cache hits
        group.bench_with_input(
            BenchmarkId::new("cache_hit", format!("{:?}", policy)),
            &policy,
            |b, _| {
                b.iter(|| {
                    let mut rng = thread_rng();
                    let idx = black_box(rng.gen_range(0..500));
                    let id = black_box(&memories[idx].id());
                    cache.get(id)
                });
            },
        );

        // Benchmark cache misses
        group.bench_with_input(
            BenchmarkId::new("cache_miss", format!("{:?}", policy)),
            &policy,
            |b, _| {
                b.iter(|| {
                    let mut rng = thread_rng();
                    let idx = black_box(500 + rng.gen_range(0..500));
                    let id = black_box(&memories[idx].id());
                    cache.get(id)
                });
            },
        );

        // Benchmark cache updates
        group.bench_with_input(
            BenchmarkId::new("cache_update", format!("{:?}", policy)),
            &policy,
            |b, _| {
                b.iter(|| {
                    let mut rng = thread_rng();
                    let idx = black_box(rng.gen_range(0..1000));
                    let memory = black_box(memories[idx].clone());
                    cache.put(memory)
                });
            },
        );
    }

    group.finish();
}

// Benchmark memory operations under concurrent load
fn bench_concurrent_operations(c: &mut Criterion) {
    let (graph_db, vector_store, runtime) = setup_benchmark_environment();

    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);

    // Concurrent memory creation and storage
    let concurrency_levels = [1, 4, 8, 16, 32];

    for &concurrency in &concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_store", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    runtime.block_on(async {
                        let mut handles = Vec::new();
                        for i in 0..concurrency {
                            let graph_db = graph_db.clone();
                            let vector_store = vector_store.clone();
                            let handle = tokio::spawn(async move {
                                let id = format!("concurrent_{}_{}", concurrency, i);
                                let memory = create_test_memory(&id);
                                let vector = create_memory_vector(128);
                                graph_db.store_entity(&memory).await.unwrap();
                                vector_store.store_vector(&id, &vector).await.unwrap();
                            });
                            handles.push(handle);
                        }
                        for handle in handles {
                            handle.await.unwrap();
                        }
                    });
                });
            },
        );

        // Concurrent retrieval
        group.bench_with_input(
            BenchmarkId::new("concurrent_retrieve", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    runtime.block_on(async {
                        let mut handles = Vec::new();
                        for i in 0..concurrency {
                            let graph_db = graph_db.clone();
                            let vector_store = vector_store.clone();
                            let handle = tokio::spawn(async move {
                                let id = format!("concurrent_{}_{}", concurrency, i);
                                let entity = graph_db.get_entity(&id).await.unwrap();
                                let vector = vector_store.get_vector(&id).await.unwrap();
                                (entity, vector)
                            });
                            handles.push(handle);
                        }
                        for handle in handles {
                            handle.await.unwrap();
                        }
                    });
                });
            },
        );

        // Concurrent vector search
        group.bench_with_input(
            BenchmarkId::new("concurrent_search", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    runtime.block_on(async {
                        let mut handles = Vec::new();
                        for _ in 0..concurrency {
                            let vector_store = vector_store.clone();
                            let query_vector = create_memory_vector(128);
                            let handle = tokio::spawn(async move {
                                vector_store
                                    .search_vectors(&query_vector, 10)
                                    .await
                                    .unwrap()
                            });
                            handles.push(handle);
                        }
                        for handle in handles {
                            handle.await.unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_creation,
    bench_memory_serialization,
    bench_memory_storage,
    bench_vector_search,
    bench_memory_cache,
    bench_concurrent_operations,
);
criterion_main!(benches);
