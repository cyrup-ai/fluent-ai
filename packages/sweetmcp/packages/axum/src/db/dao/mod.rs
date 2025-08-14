//! Database Access Object (DAO) module
//!
//! This module provides comprehensive database access functionality with
//! zero allocation patterns and blazing-fast performance.

pub mod core;
pub mod entities;
pub mod implementation;

// Re-export core types and traits for ergonomic usage
pub use core::{
    BaseDao, DaoResult, Entity, EntityMetadata, extract_table_from_id, extract_uuid_from_id,
    utc_now, validate_entity_id,
};

pub use entities::{BaseEntity, PublicUser, PublicUserProfile, User, UserProfile};
pub use implementation::Dao;

/// Create a new DAO for the specified entity type
pub fn dao<T: Entity + 'static>(client: super::client::DatabaseClient) -> Dao<T> {
    Dao::new(client)
}

/// Create a new DAO for User entities
pub fn user_dao(client: super::client::DatabaseClient) -> Dao<User> {
    Dao::new(client)
}
