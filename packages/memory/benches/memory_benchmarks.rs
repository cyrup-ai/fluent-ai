use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fluent_ai_memory::{SurrealMemoryManager, MemoryNode, MemoryConfig, memory::MemoryTypeEnum};
use rand::{Rng, rng};
use tokio::runtime::Runtime;

fn setup_benchmark_environment() -> (Arc<SurrealMemoryManager>, Runtime) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let config = MemoryConfig::default();
    let memory_manager = Arc::new(
        runtime.block_on(async { 
            fluent_ai_memory::initialize(&config).await.unwrap()
        })
    );
    (memory_manager, runtime)
}

fn create_test_memory(id: &str) -> MemoryNode {
    MemoryNode::new(
        "Test content".to_string(),
        MemoryTypeEnum::Semantic,
    )
}

fn create_memory_vector(size: usize) -> Vec<f32> {
    let mut rng = rng();
    (0..size).map(|_| rng.random::<f32>()).collect()
}

fn bench_memory_creation(c: &mut Criterion) {
    c.bench_function("memory_creation", |b| {
        b.iter(|| {
            let memory = create_test_memory("test_id");
            std::hint::black_box(memory);
        });
    });
}

fn bench_memory_serialization(c: &mut Criterion) {
    let memory = create_test_memory("test_id");
    c.bench_function("memory_serialization", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&memory).unwrap();
            std::hint::black_box(serialized);
        });
    });
}

fn bench_memory_storage(c: &mut Criterion) {
    let (memory_manager, runtime) = setup_benchmark_environment();
    let memory = create_test_memory("test_id");

    c.bench_function("memory_storage", |b| {
        b.iter(|| {
            runtime.block_on(async {
                memory_manager.store(&memory).await.unwrap();
            });
        });
    });
}

fn bench_memory_retrieval(c: &mut Criterion) {
    let (memory_manager, runtime) = setup_benchmark_environment();
    let memory = create_test_memory("test_id");
    
    // Store the memory first
    runtime.block_on(async {
        memory_manager.store(&memory).await.unwrap();
    });

    c.bench_function("memory_retrieval", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let result = memory_manager.get("test_id").await.unwrap();
                std::hint::black_box(result);
            });
        });
    });
}

fn bench_memory_search(c: &mut Criterion) {
    let (memory_manager, runtime) = setup_benchmark_environment();
    
    // Store some test memories
    runtime.block_on(async {
        for i in 0..100 {
            let memory = create_test_memory(&format!("test_id_{}", i));
            memory_manager.store(&memory).await.unwrap();
        }
    });

    c.bench_function("memory_search", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let results = memory_manager.search("Test", 10).await.unwrap();
                std::hint::black_box(results);
            });
        });
    });
}

fn bench_batch_operations(c: &mut Criterion) {
    let (memory_manager, runtime) = setup_benchmark_environment();
    
    let mut group = c.benchmark_group("batch_operations");
    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("batch_store", size), size, |b, &size| {
            b.iter(|| {
                runtime.block_on(async {
                    let memories: Vec<MemoryNode> = (0..size)
                        .map(|i| create_test_memory(&format!("batch_test_{}", i)))
                        .collect();
                    
                    for memory in memories {
                        memory_manager.store(&memory).await.unwrap();
                    }
                });
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_memory_creation,
    bench_memory_serialization,
    bench_memory_storage,
    bench_memory_retrieval,
    bench_memory_search,
    bench_batch_operations
);
criterion_main!(benches);