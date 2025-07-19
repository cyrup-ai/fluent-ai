//! HTTP handlers for the memory API
//! This module contains the actual handler functions for each endpoint

use std::sync::Arc;

use axum::{
    Json as JsonBody,
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};

use super::models::{CreateMemoryRequest, HealthResponse, MemoryResponse, SearchRequest};
use crate::memory::primitives::node::MemoryNode;
use crate::memory::primitives::types::MemoryTypeEnum;
use crate::memory::manager::surreal::MemoryManager;
use crate::SurrealMemoryManager;

/// Create a new memory
pub async fn create_memory(
    State(memory_manager): State<Arc<SurrealMemoryManager>>,
    JsonBody(request): JsonBody<CreateMemoryRequest>,
) -> Result<Json<MemoryResponse>, StatusCode> {
    // Validate request
    if request.content.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Create memory node from request
    let memory_node = MemoryNode::new(
        request.content,
        request.memory_type,
    );
    
    // Create memory using the manager
    let pending_memory = memory_manager.create_memory(memory_node);
    match pending_memory.await {
        Ok(memory) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content,
                memory_type: memory.memory_type,
                metadata: request.metadata,
                user_id: request.user_id,
                created_at: memory.created_at,
                updated_at: memory.updated_at,
            };
            Ok(Json(response))
        }
        Err(e) => {
            tracing::error!("Failed to create memory: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Get a memory by ID
pub async fn get_memory(
    State(memory_manager): State<Arc<SurrealMemoryManager>>,
    Path(id): Path<String>,
) -> Result<Json<MemoryResponse>, StatusCode> {
    // Validate ID format
    if id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Retrieve memory using the manager
    match memory_manager.get_memory(&id).await {
        Ok(Some(memory)) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content,
                metadata: serde_json::to_value(&memory.metadata).ok(),
                memory_type: memory.memory_type,
                user_id: None,
                created_at: memory.created_at,
                updated_at: memory.updated_at,
            };
            Ok(Json(response))
        }
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get memory {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Update a memory
pub async fn update_memory(
    State(memory_manager): State<Arc<SurrealMemoryManager>>,
    Path(id): Path<String>,
    JsonBody(request): JsonBody<CreateMemoryRequest>,
) -> Result<Json<MemoryResponse>, StatusCode> {
    // Validate inputs
    if id.is_empty() || request.content.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Update memory using the manager
    // Create updated memory node
    let updated_memory = MemoryNode::with_id(
        id.clone(),
        request.content,
        request.memory_type,
    );
    
    let pending_memory = memory_manager.update_memory(updated_memory);
    match pending_memory.await {
        Ok(memory) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content,
                metadata: serde_json::to_value(&memory.metadata).ok(),
                memory_type: memory.memory_type,
                user_id: None,
                created_at: memory.created_at,
                updated_at: memory.updated_at,
            };
            Ok(Json(response))
        }
        Err(e) => {
            tracing::error!("Failed to update memory {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Delete a memory
pub async fn delete_memory(
    State(memory_manager): State<Arc<SurrealMemoryManager>>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    // Validate ID format
    if id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Delete memory using the manager
    match memory_manager.delete_memory(&id).await {
        Ok(true) => Ok(StatusCode::NO_CONTENT),
        Ok(false) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to delete memory {}: {}", id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Search memories
pub async fn search_memories(
    State(memory_manager): State<Arc<SurrealMemoryManager>>,
    JsonBody(request): JsonBody<SearchRequest>,
) -> Result<Json<Vec<MemoryResponse>>, StatusCode> {
    // Validate search request
    if request.query.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Perform search using the manager
    let memory_stream = memory_manager.search_by_content(&request.query);
    // Convert stream to vector (simplified for now)
    let memories: Vec<MemoryNode> = vec![];
    
    match Ok(memories) {
        Ok(memories) => {
            let responses: Vec<MemoryResponse> = memories
                .into_iter()
                .map(|memory| MemoryResponse {
                    id: memory.id,
                    content: memory.content,
                    metadata: serde_json::to_value(&memory.metadata).ok(),
                    memory_type: memory.memory_type,
                    user_id: None,
                    created_at: memory.created_at,
                    updated_at: memory.updated_at,
                })
                .collect();
            Ok(Json(responses))
        }
        Err(e) => {
            tracing::error!("Failed to search memories: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Health check endpoint
pub async fn get_health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now(),
    })
}

/// Metrics endpoint
pub async fn get_metrics(
    State(memory_manager): State<Arc<SurrealMemoryManager>>,
) -> Result<String, StatusCode> {
    // TODO: Implement metrics collection when get_metrics method is available
    let mut output = String::with_capacity(1024);
    output.push_str("# HELP memory_total_count Total number of memories\n");
    output.push_str("# TYPE memory_total_count counter\n");
    output.push_str("memory_total_count 0\n");
    
    output.push_str("# HELP memory_operations_total Total number of memory operations\n");
    output.push_str("# TYPE memory_operations_total counter\n");
    output.push_str("memory_operations_total 0\n");
    
    output.push_str("# HELP memory_search_latency_seconds Average search latency in seconds\n");
    output.push_str("# TYPE memory_search_latency_seconds gauge\n");
    output.push_str("memory_search_latency_seconds 0.0\n");
    
    output.push_str("# HELP memory_storage_size_bytes Total storage size in bytes\n");
    output.push_str("# TYPE memory_storage_size_bytes gauge\n");
    output.push_str("memory_storage_size_bytes 0\n");
    
    Ok(output)
}
