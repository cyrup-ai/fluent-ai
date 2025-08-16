//! Connection pool manager for H3 client
//!
//! Pool management utilities and connection lifecycle handling.

use std::collections::HashMap;
use std::sync::Arc;

use crate::hyper::async_impl::h3_client::connect::H3Connection;

/// Connection pool manager
pub struct PoolManager {
    connections: HashMap<String, H3Connection>,
    max_connections: usize,
}

impl PoolManager {
    /// Create new pool manager
    pub fn new(max_connections: usize) -> Self {
        Self {
            connections: HashMap::new(),
            max_connections,
        }
    }

    /// Get connection by authority
    pub fn get_connection(&self, authority: &str) -> Option<&H3Connection> {
        self.connections.get(authority)
    }

    /// Add connection to pool
    pub fn add_connection(&mut self, authority: String, connection: H3Connection) {
        if self.connections.len() >= self.max_connections {
            self.evict_oldest();
        }
        self.connections.insert(authority, connection);
    }

    /// Remove connection from pool
    pub fn remove_connection(&mut self, authority: &str) -> Option<H3Connection> {
        self.connections.remove(authority)
    }

    /// Evict oldest connection
    fn evict_oldest(&mut self) {
        if let Some(key) = self.connections.keys().next().cloned() {
            self.connections.remove(&key);
        }
    }

    /// Get pool size
    pub fn size(&self) -> usize {
        self.connections.len()
    }

    /// Check if pool is full
    pub fn is_full(&self) -> bool {
        self.connections.len() >= self.max_connections
    }
}
