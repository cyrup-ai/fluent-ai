use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_skiplist::SkipMap;
use fluent_ai_async::AsyncStream;
use serde::{Deserialize, Serialize};

use crate::chat::message::SearchChatMessage;
use super::SearchError;

// Re-export public types
pub use types::*;
pub use impls::ConversationTagger;

// Submodules
mod types;
mod impls;
