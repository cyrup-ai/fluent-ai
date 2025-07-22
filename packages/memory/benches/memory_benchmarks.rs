use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fluent_ai_memory::{MemoryConfig, MemoryNode, SurrealMemoryManager, memory::MemoryTypeEnum};
use rand::{Rng, rng};
use tokio::sync::OnceCell;

// Global memory manager initialized once for all benchmarks
static MEMORY_MANAGER: OnceCell<Arc<SurrealMemoryManager>> = OnceCell::const_new();

async fn get_memory_manager() -> Arc<SurrealMemoryManager> {
    MEMORY_MANAGER
        .get_or_init(|| async {
            let config = MemoryConfig::default();
            Arc::new(
                fluent_ai_memory::initialize(&config)
                    .await
                    .expect("Failed to initialize memory system for benchmarks"),
            )
        })
        .await
        .clone()
}

fn create_test_memory(id: &str) -> MemoryNode {
    MemoryNode::new("Test content".to_string(), MemoryTypeEnum::Semantic)
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
    c.bench_function("memory_storage", |b| {
        b.to_async(tokio::runtime::Runtime::new().expect("Failed to create async runtime"))
            .iter(|| async {
                let memory_manager = get_memory_manager().await;
                let memory = create_test_memory("test_id");
                memory_manager
                    .store(&memory)
                    .await
                    .expect("Failed to store memory in benchmark");
            });
    });
}

fn bench_memory_retrieval(c: &mut Criterion) {
    c.bench_function("memory_retrieval", |b| {
        b.to_async(tokio::runtime::Runtime::new().expect("Failed to create async runtime"))
            .iter(|| async {
                let memory_manager = get_memory_manager().await;
                let memory = create_test_memory("test_id");

                // Store the memory first
                memory_manager
                    .store(&memory)
                    .await
                    .expect("Failed to store memory for retrieval benchmark");

                // Then retrieve it
                let result = memory_manager
                    .get("test_id")
                    .await
                    .expect("Failed to retrieve memory in benchmark");
                std::hint::black_box(result);
            });
    });
}

fn bench_memory_search(c: &mut Criterion) {
    c.bench_function("memory_search", |b| {
        b.to_async(tokio::runtime::Runtime::new().expect("Failed to create async runtime"))
            .iter(|| async {
                let memory_manager = get_memory_manager().await;

                // Store test memories for search
                for i in 0..100 {
                    let memory = create_test_memory(&format!("test_id_{}", i));
                    memory_manager
                        .store(&memory)
                        .await
                        .expect("Failed to store memory for search benchmark");
                }

                // Perform search
                let results = memory_manager
                    .search("Test", 10)
                    .await
                    .expect("Failed to search memories in benchmark");
                std::hint::black_box(results);
            });
    });
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::new("batch_store", size), size, |b, &size| {
            b.to_async(tokio::runtime::Runtime::new().expect("Failed to create async runtime"))
                .iter(|| async {
                    let memory_manager = get_memory_manager().await;
                    let memories: Vec<MemoryNode> = (0..size)
                        .map(|i| create_test_memory(&format!("batch_test_{}", i)))
                        .collect();

                    for memory in memories {
                        memory_manager
                            .store(&memory)
                            .await
                            .expect("Failed to store memory in batch benchmark");
                    }
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
