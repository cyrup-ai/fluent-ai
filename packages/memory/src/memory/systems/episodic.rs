// src/memory/episodic.rs
//! Episodic memory implementation for the memory system.
//!
//! Episodic memory stores sequences of events with temporal information,
//! allowing for time-based queries and context-aware retrieval.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use chrono::{DateTime, Utc};
use crossbeam_skiplist::SkipMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::memory::primitives::metadata::MemoryMetadata;
use crate::memory::primitives::node::MemoryNode;
use crate::memory::primitives::types::{BaseMemory, MemoryContent, MemoryType, MemoryTypeEnum};
use crate::memory::repository::MemoryRepository;
use crate::utils::error::Error;
use crate::utils::Result;
use fluent_ai_core::channel::async_stream_channel;
use fluent_ai_core::stream::AsyncStream;

/// Context for an episodic memory event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicContext {
    /// Unique identifier for the context
    pub id: String,

    /// Type of context (e.g., "location", "person", "object")
    pub context_type: String,

    /// Value of the context
    pub value: String,

    /// Additional metadata for the context
    pub metadata: HashMap<String, Value>,
}

impl EpisodicContext {
    /// Create a new episodic context
    pub fn new(id: &str, context_type: &str, value: &str) -> Self {
        Self {
            id: id.to_string(),
            context_type: context_type.to_string(),
            value: value.to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the context
    pub fn with_metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Convert the context to a SurrealDB value
    pub fn to_value(&self) -> Result<Value> {
        let mut obj = serde_json::Map::new();
        obj.insert("id".to_string(), Value::String(self.id.clone()));
        obj.insert(
            "context_type".to_string(),
            Value::String(self.context_type.clone()),
        );
        obj.insert("value".to_string(), Value::String(self.value.clone()));
        obj.insert(
            "metadata".to_string(),
            serde_json::to_value(&self.metadata)?,
        );
        Ok(Value::Object(obj))
    }
}

/// Represents a single event in episodic memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicEvent {
    /// Unique identifier for the event
    pub id: String,

    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,

    /// Content of the event
    pub content: MemoryContent,

    /// Context associated with the event
    pub context: Vec<EpisodicContext>,

    /// Additional metadata for the event
    pub metadata: HashMap<String, Value>,
}

impl EpisodicEvent {
    /// Create a new episodic event
    pub fn new(id: &str, content: MemoryContent) -> Self {
        Self {
            id: id.to_string(),
            timestamp: Utc::now(),
            content,
            context: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a context to the event
    pub fn with_context(mut self, context: EpisodicContext) -> Self {
        self.context.push(context);
        self
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }
}

/// Represents an episodic memory, which is a collection of events
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    /// Base memory properties
    pub base: BaseMemory,

    /// Collection of events, indexed by timestamp for fast temporal queries
    pub events: Arc<ArcSwap<SkipMap<DateTime<Utc>, EpisodicEvent>>>,
}

impl MemoryType for EpisodicMemory {
    fn new(id: &str, name: &str, description: &str) -> Self {
        let mut metadata = MemoryMetadata::with_type(MemoryTypeEnum::Episodic);
        metadata.add_attribute("version".to_string(), json!("1.0"));

        Self {
            base: BaseMemory {
                id: id.to_string(),
                name: name.to_string(),
                description: description.to_string(),
                updated_at: Utc::now(),
                metadata,
                content: MemoryContent::None,
            },
            events: Arc::new(ArcSwap::new(Arc::new(SkipMap::new()))),
        }
    }

    fn from_memory(memory: &BaseMemory) -> Result<Self> {
        let events: SkipMap<DateTime<Utc>, EpisodicEvent> = match &memory.content {
            MemoryContent::Json(val) => serde_json::from_value(val.clone())?,
            MemoryContent::Text(s) => serde_json::from_str(s)?,
            _ => return Err(Error::MemoryError("Invalid content type for episodic memory".to_string())),
        };

        Ok(Self {
            base: memory.clone(),
            events: Arc::new(ArcSwap::new(Arc::new(events))),
        })
    }

    fn to_memory(&self) -> Result<BaseMemory> {
        let mut memory = self.base.clone();
        let events_guard = self.events.load();
        let events_map: HashMap<_, _> = events_guard.iter().map(|entry| (entry.key().clone(), entry.value().clone())).collect();
        memory.content = MemoryContent::Json(serde_json::to_value(events_map)?);
        Ok(memory)
    }

    fn id(&self) -> &str {
        &self.base.id
    }

    fn name(&self) -> &str {
        &self.base.name
    }

    fn description(&self) -> &str {
        &self.base.description
    }

    fn memory_type(&self) -> MemoryTypeEnum {
        MemoryTypeEnum::Episodic
    }
}

impl EpisodicMemory {
    /// Add an event to the episodic memory
    pub fn add_event(&self, event: EpisodicEvent) {
        let new_events = self.events.load().clone();
        new_events.insert(event.timestamp, event);
        self.events.store(new_events);
        self.base.touch();
    }

    /// Retrieve events within a specific time range
    pub fn get_events_in_range(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Vec<EpisodicEvent> {
        self.events
            .load()
            .range(start_time..=end_time)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Find the last N events before a given time
    pub fn get_last_n_events(&self, n: usize, before_time: DateTime<Utc>) -> Vec<EpisodicEvent> {
        self.events
            .load()
            .range(..=before_time)
            .rev()
            .take(n)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Create a new episodic memory and store it in the repository
    pub fn create(
        memory_repo: Arc<MemoryRepository>,
        id: &str,
        name: &str,
        description: &str,
    ) -> AsyncStream<Result<EpisodicMemory>> {
        let (tx, stream) = async_stream_channel();
        let id_string = id.to_string();
        let name_string = name.to_string();
        let description_string = description.to_string();

        tokio::spawn(async move {
            let result = {
                let episodic = EpisodicMemory::new(&id_string, &name_string, &description_string);

                // Convert to MemoryNode for storage
                let mut metadata = MemoryMetadata::new();
                metadata.created_at = episodic.base.metadata.created_at;

                let content = match serde_json::to_string(&episodic.base.content) {
                    Ok(content_str) => content_str,
                    Err(_) => {
                        return Err(crate::utils::error::Error::SerializationError(
                            "Failed to serialize episodic memory content".to_string(),
                        ))
                    }
                };

                let memory_node = MemoryNode {
                    id: episodic.base.id.clone(),
                    content,
                    memory_type: MemoryTypeEnum::Episodic,
                    created_at: episodic.base.metadata.created_at,
                    updated_at: episodic.base.updated_at,
                    embedding: None,
                    metadata,
                };

                // Lock-free create operation
                let created_memory = memory_repo.create(&id_string, &memory_node)?;
                // Convert created MemoryNode to BaseMemory
                let mut metadata = MemoryMetadata::with_type(MemoryTypeEnum::Episodic);
                metadata.created_at = created_memory.created_at;
                // MemoryMetadata doesn't have updated_at field - that's on BaseMemory

                let base_memory = BaseMemory {
                    id: created_memory.id.clone(),
                    name: name_string.clone(),
                    description: description_string.clone(),
                    updated_at: created_memory.updated_at,
                    metadata,
                    content: MemoryContent::text(&created_memory.content),
                };
                let created_episodic = EpisodicMemory::from_memory(&base_memory)?;

                Ok(created_episodic)
            };
            let _ = tx.send(result);
        });
        stream
    }
}
