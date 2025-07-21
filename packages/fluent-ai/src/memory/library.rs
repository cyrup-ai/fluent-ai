//! Library Memory Integration Service
//!
//! This module provides the service logic for integrating domain Library entities
//! with the memory system. It handles library-scoped memory namespace creation,
//! agent integration, and shared memory management.

use std::sync::Arc;

use fluent_ai_domain::{
    Library,
    memory::{Memory, MemoryConfig, MemoryError, MemoryTool},
};
use fluent_ai_memory::utils::config::DatabaseConfig;

/// Library service error types
#[derive(Debug, thiserror::Error)]
pub enum LibraryServiceError {
    /// Memory system initialization error
    #[error("Memory initialization failed for library '{library}': {source}")]
    MemoryInit {
        library: String,
        #[source]
        source: MemoryError,
    },
    /// Library name validation error
    #[error("Invalid library name '{name}': {reason}")]
    InvalidName { name: String, reason: String },
    /// Configuration error
    #[error("Library configuration error: {0}")]
    Config(String),
}

/// Result type for library service operations
pub type LibraryServiceResult<T> = Result<T, LibraryServiceError>;

/// Library memory service for managing library-scoped memory namespaces
///
/// This service provides the integration between domain Library entities
/// and the memory system, implementing the ARCHITECTURE.md pattern:
/// ```rust
/// FluentAi::agent_role("rusty-squire")
///     .memory(Library::named("obsidian_vault"))
///     .into_agent()
/// ```
pub struct LibraryMemoryService;

impl LibraryMemoryService {
    /// Create memory namespace for a library
    ///
    /// This creates an isolated memory namespace using the library name as
    /// the database namespace. Each library gets its own isolated memory space
    /// that can be shared between agents using the same library.
    ///
    /// # Arguments
    /// * `library` - The library to create memory namespace for
    ///
    /// # Returns
    /// Result containing Memory instance scoped to this library's namespace
    ///
    /// # Errors
    /// Returns `LibraryServiceError::MemoryInit` if memory initialization fails
    ///
    /// # Performance
    /// Memory instances are designed for sharing between agents - multiple
    /// agents can safely use the same Memory instance with lock-free operations.
    ///
    /// # Examples
    /// ```rust
    /// let library = Library::named("obsidian_vault");
    /// let memory = LibraryMemoryService::create_memory_namespace(&library).await?;
    /// ```
    pub async fn create_memory_namespace(library: &Library) -> LibraryServiceResult<Memory> {
        // Validate library name
        if let Err(reason) = Library::validate_name(library.name()) {
            return Err(LibraryServiceError::InvalidName {
                name: library.name().to_string(),
                reason,
            });
        }

        // Create library-specific memory configuration
        let config = MemoryConfig {
            database: DatabaseConfig {
                // Use library namespace for isolation
                namespace: library.namespace(),
                database: "fluent_ai".to_string(),
                connection_string: std::env::var("FLUENT_AI_DB_URL")
                    .unwrap_or_else(|_| "mem://".to_string()),
                username: None,
                password: None,
            },
            vector_store: fluent_ai_memory::utils::config::VectorStoreConfig {
                enabled: true,
                dimension: 1536, // OpenAI embedding dimension
                ..Default::default()
            },
            cache: fluent_ai_memory::utils::config::CacheConfig {
                enabled: true,
                max_size: 1000,
                ttl_seconds: 3600,
            },
            provider_model: fluent_ai_provider::completion_provider::ModelConfig {
                model_name: "gpt-4o-mini".to_string(),
                max_tokens: None,
                temperature: None,
                top_p: None,
                frequency_penalty: None,
                presence_penalty: None,
                stop_sequences: Vec::new(),
            },
            logging: fluent_ai_memory::utils::config::LoggingConfig {
                level: "info".to_string(),
                ..Default::default()
            },
        };

        // Create memory instance with library-specific configuration
        Memory::new(config)
            .await
            .map_err(|source| LibraryServiceError::MemoryInit {
                library: library.name().to_string(),
                source,
            })
    }

    /// Create shared memory namespace that can be used by multiple agents
    ///
    /// This is the preferred method for agent integration as it returns an
    /// Arc<Memory> that can be safely shared between multiple agents using
    /// the same library namespace.
    ///
    /// # Arguments
    /// * `library` - The library to create shared memory namespace for
    ///
    /// # Returns
    /// Result containing Arc<Memory> for shared access
    ///
    /// # Examples
    /// ```rust
    /// let library = Library::named("obsidian_vault");
    /// let shared_memory = LibraryMemoryService::create_shared_memory(&library).await?;
    ///
    /// // Both agents share the same memory namespace
    /// let agent1 = Agent::with_memory(shared_memory.clone()).await?;
    /// let agent2 = Agent::with_memory(shared_memory.clone()).await?;
    /// ```
    pub async fn create_shared_memory(library: &Library) -> LibraryServiceResult<Arc<Memory>> {
        let memory = Self::create_memory_namespace(library).await?;
        Ok(Arc::new(memory))
    }

    /// Create memory tool for a library-scoped memory instance
    ///
    /// This creates a MemoryTool that can be used by agents for memorize/recall
    /// operations within the library's memory namespace.
    ///
    /// # Arguments
    /// * `library` - The library to create memory tool for
    ///
    /// # Returns
    /// Result containing MemoryTool scoped to library namespace
    ///
    /// # Examples
    /// ```rust
    /// let library = Library::named("team_knowledge");
    /// let memory_tool = LibraryMemoryService::create_memory_tool(&library).await?;
    /// ```
    pub async fn create_memory_tool(library: &Library) -> LibraryServiceResult<MemoryTool> {
        let memory = Self::create_memory_namespace(library).await?;
        Ok(MemoryTool::new(Arc::new(memory)))
    }

    /// Create shared memory tool for multi-agent collaboration
    ///
    /// This creates a MemoryTool with shared memory that can be used by
    /// multiple agents in the same library namespace.
    ///
    /// # Arguments
    /// * `library` - The library to create shared memory tool for
    ///
    /// # Returns
    /// Result containing MemoryTool with shared memory access
    ///
    /// # Examples
    /// ```rust
    /// let library = Library::named("collaborative_workspace");
    /// let shared_tool = LibraryMemoryService::create_shared_memory_tool(&library).await?;
    /// ```
    pub async fn create_shared_memory_tool(library: &Library) -> LibraryServiceResult<MemoryTool> {
        let shared_memory = Self::create_shared_memory(library).await?;
        Ok(MemoryTool::new(shared_memory))
    }

    /// Extract library name from memory namespace
    ///
    /// This utility method extracts the library name from a memory namespace
    /// that follows the "lib_{name}" format.
    ///
    /// # Arguments
    /// * `namespace` - Memory namespace string
    ///
    /// # Returns
    /// Optional library name if namespace follows library format
    ///
    /// # Examples
    /// ```rust
    /// let name = LibraryMemoryService::extract_library_name("lib_obsidian_vault");
    /// assert_eq!(name, Some("obsidian_vault"));
    /// ```
    pub fn extract_library_name(namespace: &str) -> Option<&str> {
        namespace.strip_prefix("lib_")
    }

    /// Check if a namespace belongs to a library
    ///
    /// # Arguments
    /// * `namespace` - Memory namespace to check
    ///
    /// # Returns
    /// True if namespace follows library format ("lib_*")
    pub fn is_library_namespace(namespace: &str) -> bool {
        namespace.starts_with("lib_")
    }
}

/// Extension trait for Library domain entities
///
/// This trait extends the domain Library struct with service methods
/// for memory integration. This follows the pattern of keeping pure
/// domain entities in the domain package while adding service logic
/// in the application layer.
pub trait LibraryExt {
    /// Create memory namespace for this library
    fn create_memory_namespace(
        &self,
    ) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<Memory>>;

    /// Create shared memory namespace for this library
    fn create_shared_memory(
        &self,
    ) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<Arc<Memory>>>;

    /// Create memory tool for this library
    fn create_memory_tool(
        &self,
    ) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<MemoryTool>>;

    /// Create shared memory tool for this library
    fn create_shared_memory_tool(
        &self,
    ) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<MemoryTool>>;
}

impl LibraryExt for Library {
    fn create_memory_namespace(&self) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<Memory>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let library = self.clone();
        
        tokio::spawn(async move {
            let result = LibraryMemoryService::create_memory_namespace(&library).await;
            let _ = tx.send(result);
        });
        
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }

    fn create_shared_memory(&self) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<Arc<Memory>>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let library = self.clone();
        
        tokio::spawn(async move {
            let result = LibraryMemoryService::create_shared_memory(&library).await;
            let _ = tx.send(result);
        });
        
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }

    fn create_memory_tool(&self) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<MemoryTool>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let library = self.clone();
        
        tokio::spawn(async move {
            let result = LibraryMemoryService::create_memory_tool(&library).await;
            let _ = tx.send(result);
        });
        
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }

    fn create_shared_memory_tool(&self) -> fluent_ai_domain::AsyncStream<LibraryServiceResult<MemoryTool>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let library = self.clone();
        
        tokio::spawn(async move {
            let result = LibraryMemoryService::create_shared_memory_tool(&library).await;
            let _ = tx.send(result);
        });
        
        tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
    }
}
