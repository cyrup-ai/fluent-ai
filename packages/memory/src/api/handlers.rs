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

    // Create memory using the manager
    match memory_manager.create_memory(
        request.content,
        request.metadata.unwrap_or_default(),
        request.tags.unwrap_or_default(),
    ).await {
        Ok(memory) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content,
                metadata: memory.metadata,
                tags: memory.tags,
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
                metadata: memory.metadata,
                tags: memory.tags,
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
    match memory_manager.update_memory(
        &id,
        request.content,
        request.metadata.unwrap_or_default(),
        request.tags.unwrap_or_default(),
    ).await {
        Ok(Some(memory)) => {
            let response = MemoryResponse {
                id: memory.id,
                content: memory.content,
                metadata: memory.metadata,
                tags: memory.tags,
                created_at: memory.created_at,
                updated_at: memory.updated_at,
            };
            Ok(Json(response))
        }
        Ok(None) => Err(StatusCode::NOT_FOUND),
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
    match memory_manager.search_memories(
        &request.query,
        request.limit.unwrap_or(10),
        request.offset.unwrap_or(0),
        request.tags.as_deref(),
    ).await {
        Ok(memories) => {
            let responses: Vec<MemoryResponse> = memories
                .into_iter()
                .map(|memory| MemoryResponse {
                    id: memory.id,
                    content: memory.content,
                    metadata: memory.metadata,
                    tags: memory.tags,
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
    match memory_manager.get_metrics().await {
        Ok(metrics) => {
            // Format metrics in Prometheus format
            let mut output = String::with_capacity(1024);
            output.push_str("# HELP memory_total_count Total number of memories\n");
            output.push_str("# TYPE memory_total_count counter\n");
            output.push_str(&format!("memory_total_count {}\n", metrics.total_memories));
            
            output.push_str("# HELP memory_operations_total Total number of memory operations\n");
            output.push_str("# TYPE memory_operations_total counter\n");
            output.push_str(&format!("memory_operations_total {}\n", metrics.total_operations));
            
            output.push_str("# HELP memory_search_latency_seconds Average search latency in seconds\n");
            output.push_str("# TYPE memory_search_latency_seconds gauge\n");
            output.push_str(&format!("memory_search_latency_seconds {:.6}\n", metrics.avg_search_latency_ms / 1000.0));
            
            output.push_str("# HELP memory_storage_size_bytes Total storage size in bytes\n");
            output.push_str("# TYPE memory_storage_size_bytes gauge\n");
            output.push_str(&format!("memory_storage_size_bytes {}\n", metrics.storage_size_bytes));
            
            Ok(output)
        }
        Err(e) => {
            tracing::error!("Failed to get metrics: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
