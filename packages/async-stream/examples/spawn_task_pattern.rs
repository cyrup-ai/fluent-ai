//! # AsyncStream Task Spawning Patterns
//!
//! Demonstrates real-world AsyncStream task spawning patterns used throughout fluent-ai.
//! Shows spawn_task() and spawn_stream() with proper zero-allocation streaming.

use fluent_ai_async::prelude::*;

#[derive(Debug, Clone, Default)]
struct TaskResult {
    task_id: usize,
    result: String,
    status: String,
}

impl MessageChunk for TaskResult {
    fn bad_chunk(error: String) -> Self {
        Self {
            task_id: 0,
            result: format!("ERROR: {}", error),
            status: "failed".to_string(),
        }
    }

    fn is_error(&self) -> bool {
        self.status == "failed"
    }

    fn error(&self) -> Option<&str> {
        if self.is_error() {
            Some(&self.result)
        } else {
            None
        }
    }
}

/// Real-world AsyncStream task spawning patterns
fn main() {
    println!("🚀 Real-World AsyncStream Task Spawning Patterns\n");

    // PATTERN 1: spawn_task() for single background computation
    println!("📋 PATTERN 1: Single Task Spawning");
    let computation_task = spawn_task(|| {
        println!("   ⚙️  Background computation started...");
        std::thread::sleep(std::time::Duration::from_millis(300));
        
        let result = "Computed result: 42";
        println!("   ✅ Computation complete: {}", result);
        result.to_string()
    });

    // Collect the result (this blocks until task completes)
    let result = computation_task.collect();
    println!("   📊 Task result: {}\n", result);

    // PATTERN 2: spawn_stream() for streaming background work
    println!("📋 PATTERN 2: Streaming Task Spawning");
    let task_stream = spawn_stream(move |sender| {
        println!("   ⚙️  Background streaming task started...");
        
        for i in 1..=4 {
            println!("   🔄 Processing task #{}", i);
            std::thread::sleep(std::time::Duration::from_millis(200));
            
            let task_result = TaskResult {
                task_id: i,
                result: format!("Task {} completed successfully", i),
                status: "completed".to_string(),
            };
            
            println!("   ✅ Task #{} done", i);
            emit!(sender, task_result);
        }
        
        println!("   🎯 All streaming tasks complete!");
    });

    // Collect all streaming results
    println!("   📊 Collecting streaming results...");
    let all_results: Vec<TaskResult> = task_stream.collect();
    
    println!("\n📈 Final Results:");
    for result in all_results {
        if result.is_error() {
            println!("   ❌ Error: {}", result.error().unwrap_or("Unknown error"));
        } else {
            println!("   ✅ Task #{}: {}", result.task_id, result.result);
        }
    }
    
    println!("\n💡 Real-World Patterns Demonstrated:");
    println!("   • spawn_task() for single background computations");
    println!("   • spawn_stream() for streaming background work");
    println!("   • .collect() for consuming results");
    println!("   • Zero-allocation streaming with crossbeam primitives");
    println!("   • MessageChunk trait for proper error handling");
    println!("   • This is exactly how it's used throughout fluent-ai codebase");
}