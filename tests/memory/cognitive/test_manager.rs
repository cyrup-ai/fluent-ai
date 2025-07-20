use fluent_ai_memory::cognitive::manager::{CognitiveMemoryManager, CognitiveSettings, CognitiveMemoryNode};
use fluent_ai_memory::memory::{MemoryNode, MemoryType};

#[tokio::test]
async fn test_cognitive_manager_creation() {
    let settings = CognitiveSettings::default();

    // Would need a test database for full test
    // let manager = CognitiveMemoryManager::new(
    //     "memory://test",
    //     "test_ns",
    //     "test_db",
    //     settings,
    // ).await;

    // assert!(manager.is_ok());
}

#[test]
fn test_cognitive_enhancement() {
    let base_memory = MemoryNode::new("test content".to_string(), MemoryType::Semantic);
    let cognitive_memory = CognitiveMemoryNode::from(base_memory);

    assert!(!cognitive_memory.is_enhanced());
    assert_eq!(cognitive_memory.base_memory.content, "test content");
}