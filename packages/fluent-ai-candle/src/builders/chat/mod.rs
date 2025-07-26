//! Chat-related builders extracted from fluent_ai_domain

pub mod conversation_builder;
pub mod history_manager_builder;
pub mod macro_builder;
pub mod template_builder;

// Re-export for convenience
pub use conversation_builder::CandleConversationBuilder;
pub use history_manager_builder::CandleHistoryManagerBuilder;
pub use macro_builder::CandleMacroBuilder;
pub use template_builder::CandleTemplateBuilder;