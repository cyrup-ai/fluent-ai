{% if include_examples %}
//! AsyncTask examples - DELETE THIS MODULE IF NOT NEEDED
//!
//! Shows usage of AsyncTask for concrete async types without boxed futures.

use cyrup_sugars::{AsyncTask, ZeroOneOrMany};
use tokio::sync::oneshot;

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ AsyncTask Examples:");
    
    // Single receiver pattern
    let (tx, rx) = oneshot::channel::<String>();
    let task = AsyncTask::new(ZeroOneOrMany::one(rx));
    
    // Send result
    tx.send("Hello from async task!".to_string()).unwrap();
    let result = task.await;
    println!("  Single receiver result: {}", result);
    
    // Multiple receivers (race condition - first to complete wins)  
    // Note: In real use, you'd have actual async operations sending to these channels
    println!("  Multiple receiver pattern available for race conditions");
    
    // From future pattern
    let future_task = AsyncTask::from_future(async { "From future!".to_string() });
    let future_result = future_task.await;
    println!("  From future result: {}", future_result);
    
    // From value pattern
    let value_task = AsyncTask::from_value("Immediate value!".to_string());
    let value_result = value_task.await;
    println!("  From value result: {}", value_result);
    
    println!();
    Ok(())
}
{% endif %}