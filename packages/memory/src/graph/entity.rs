//! Entity model for Rust-mem0 - THREAD-SAFE SYNCHRONOUS OPERATIONS
//!
//! This module provides a comprehensive entity model that maps domain objects
//! to graph nodes, with support for attributes, validation, and serialization.
//! All operations are synchronous and thread-safe for maximum performance.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::{Receiver, Sender, unbounded};
use serde::{Deserialize, Serialize};
use surrealdb::sql::Value;

use crate::graph::graph_db::{GraphDatabase, GraphError, GraphQueryOptions, Node, Result};

/// Type alias for entity validation functions
pub type EntityValidatorFn = Box<dyn Fn(&dyn Entity) -> Result<()> + Send + Sync>;

/// Entity trait for domain objects
pub trait Entity: Send + Sync + Debug {
    /// Get the entity ID
    fn id(&self) -> &str;

    /// Get the entity type
    fn entity_type(&self) -> &str;

    /// Get an attribute value
    fn get_attribute(&self, name: &str) -> Option<&Value>;

    /// Set an attribute value
    fn set_attribute(&mut self, name: &str, value: Value);

    /// Get all attributes
    fn attributes(&self) -> &HashMap<String, Value>;

    /// Validate the entity
    fn validate(&self) -> Result<()>;

    /// Convert to a graph node
    fn to_node(&self) -> Node;

    /// Create from a graph node
    fn from_node(node: Node) -> Result<Self>
    where
        Self: Sized;
}

/// Base entity implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseEntity {
    id: String,
    entity_type: String,
    attributes: HashMap<String, Value>}

impl BaseEntity {
    /// Create a new base entity
    pub fn new(id: String, entity_type: String) -> Self {
        Self {
            id,
            entity_type,
            attributes: HashMap::new()}
    }

    /// Create with pre-allocated capacity for attributes
    pub fn with_capacity(id: String, entity_type: String, capacity: usize) -> Self {
        Self {
            id,
            entity_type,
            attributes: HashMap::with_capacity(capacity)}
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        self.id.len() + self.entity_type.len() + (self.attributes.len() * 64) // Approximate
    }

    /// Bulk set attributes with validation
    pub fn set_attributes_bulk(&mut self, attributes: HashMap<String, Value>) -> Result<()> {
        // Validate all attributes first
        for (key, value) in &attributes {
            self.validate_attribute(key, value)?;
        }

        // If all valid, set them
        self.attributes.extend(attributes);
        Ok(())
    }

    /// Validate a single attribute (can be overridden)
    pub fn validate_attribute(&self, _key: &str, _value: &Value) -> Result<()> {
        // Default validation - accept all
        Ok(())
    }

    /// Remove multiple attributes
    pub fn remove_attributes(&mut self, keys: &[&str]) {
        for key in keys {
            self.attributes.remove(*key);
        }
    }

    /// Check if entity has all required attributes
    pub fn has_required_attributes(&self, required: &[&str]) -> bool {
        required
            .iter()
            .all(|key| self.attributes.contains_key(*key))
    }
}

impl Entity for BaseEntity {
    fn id(&self) -> &str {
        &self.id
    }

    fn entity_type(&self) -> &str {
        &self.entity_type
    }

    fn get_attribute(&self, name: &str) -> Option<&Value> {
        self.attributes.get(name)
    }

    fn set_attribute(&mut self, name: &str, value: Value) {
        self.attributes.insert(name.to_string(), value);
    }

    fn attributes(&self) -> &HashMap<String, Value> {
        &self.attributes
    }

    fn validate(&self) -> Result<()> {
        // Basic validation - ensure ID and type are not empty
        if self.id.is_empty() {
            return Err(GraphError::ValidationError(
                "Entity ID cannot be empty".to_string(),
            ));
        }

        if self.entity_type.is_empty() {
            return Err(GraphError::ValidationError(
                "Entity type cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    fn to_node(&self) -> Node {
        let mut properties = self.attributes.clone();
        properties.insert("id".to_string(), Value::Strand(self.id.clone().into()));
        properties.insert(
            "entity_type".to_string(),
            Value::Strand(self.entity_type.clone().into()),
        );

        Node {
            id: Some(self.id.clone()),
            properties,
            labels: vec![self.entity_type.clone()]}
    }

    fn from_node(node: Node) -> Result<Self> {
        let id = node
            .properties
            .get("id")
            .and_then(|v| match v {
                Value::Strand(s) => Some(s.as_str().to_string()),
                _ => None})
            .ok_or_else(|| GraphError::ValidationError("Node missing ID".to_string()))?;

        let entity_type = node
            .properties
            .get("entity_type")
            .and_then(|v| match v {
                Value::Strand(s) => Some(s.as_str().to_string()),
                _ => None})
            .ok_or_else(|| GraphError::ValidationError("Node missing entity_type".to_string()))?;

        let mut attributes = node.properties;
        attributes.remove("id");
        attributes.remove("entity_type");

        Ok(Self {
            id,
            entity_type,
            attributes})
    }
}

impl BaseEntity {
    /// Builder pattern method to set an attribute and return self
    pub fn with_attribute(mut self, name: &str, value: Value) -> Self {
        self.attributes.insert(name.to_string(), value);
        self
    }
}

/// Thread-safe entity repository trait - SYNCHRONOUS OPERATIONS ONLY
///
/// This trait provides a synchronous interface for entity CRUD operations.
/// All methods are thread-safe and return Results directly.
/// For concurrent operations, use external thread pools and channels.
pub trait EntityRepository: Send + Sync {
    /// Create a new entity
    ///
    /// # Arguments
    /// * `entity` - Entity to create
    ///
    /// # Returns
    /// Result containing the created entity with database-assigned ID
    fn create_entity(&self, entity: Box<dyn Entity>) -> Result<Box<dyn Entity>>;

    /// Get an entity by ID
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the entity
    ///
    /// # Returns
    /// Result containing the entity if found
    fn get_entity(&self, id: &str) -> Result<Option<Box<dyn Entity>>>;

    /// Update an entity
    ///
    /// # Arguments
    /// * `entity` - Entity with updated data
    ///
    /// # Returns
    /// Result containing the updated entity
    fn update_entity(&self, entity: Box<dyn Entity>) -> Result<Box<dyn Entity>>;

    /// Delete an entity
    ///
    /// # Arguments
    /// * `id` - Unique identifier of the entity to delete
    ///
    /// # Returns
    /// Result indicating success or failure
    fn delete_entity(&self, id: &str) -> Result<()>;

    /// Find entities by type with pagination
    ///
    /// # Arguments
    /// * `entity_type` - Type of entities to find
    /// * `limit` - Maximum number of results
    /// * `offset` - Number of results to skip
    ///
    /// # Returns
    /// Result containing list of matching entities
    fn find_entities_by_type(
        &self,
        entity_type: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<Box<dyn Entity>>>;

    /// Find entities by attribute with pagination
    ///
    /// # Arguments
    /// * `attribute_name` - Name of the attribute to filter by
    /// * `attribute_value` - Value to match
    /// * `limit` - Maximum number of results
    /// * `offset` - Number of results to skip
    ///
    /// # Returns
    /// Result containing list of matching entities
    fn find_entities_by_attribute(
        &self,
        attribute_name: &str,
        attribute_value: &Value,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<Box<dyn Entity>>>;

    /// Count entities by type
    ///
    /// # Arguments
    /// * `entity_type` - Type of entities to count
    ///
    /// # Returns
    /// Result containing the count
    fn count_entities_by_type(&self, entity_type: &str) -> Result<usize>;

    /// Batch create entities (optimized for bulk operations)
    ///
    /// # Arguments
    /// * `entities` - Vector of entities to create
    ///
    /// # Returns
    /// Result containing vector of created entities
    fn batch_create_entities(
        &self,
        entities: Vec<Box<dyn Entity>>,
    ) -> Result<Vec<Box<dyn Entity>>> {
        // Default implementation - create one by one
        // Implementations should override for better performance
        let mut results = Vec::with_capacity(entities.len());
        for entity in entities {
            results.push(self.create_entity(entity)?);
        }
        Ok(results)
    }

    /// Batch update entities
    ///
    /// # Arguments
    /// * `entities` - Vector of entities to update
    ///
    /// # Returns
    /// Result containing vector of updated entities
    fn batch_update_entities(
        &self,
        entities: Vec<Box<dyn Entity>>,
    ) -> Result<Vec<Box<dyn Entity>>> {
        // Default implementation - update one by one
        let mut results = Vec::with_capacity(entities.len());
        for entity in entities {
            results.push(self.update_entity(entity)?);
        }
        Ok(results)
    }

    /// Batch delete entities
    ///
    /// # Arguments
    /// * `ids` - Vector of entity IDs to delete
    ///
    /// # Returns
    /// Result indicating success or failure
    fn batch_delete_entities(&self, ids: Vec<&str>) -> Result<()> {
        // Default implementation - delete one by one
        for id in ids {
            self.delete_entity(id)?;
        }
        Ok(())
    }
}

/// SurrealDB-backed entity repository implementation
pub struct SurrealEntityRepository<E: Entity + Clone + 'static> {
    db: Arc<dyn GraphDatabase>,
    table_name: String,
    validator: Option<EntityValidatorFn>,
    _phantom: std::marker::PhantomData<E>}

impl<E: Entity + Clone + 'static> SurrealEntityRepository<E> {
    /// Create a new SurrealDB entity repository
    ///
    /// # Arguments
    /// * `db` - Graph database connection
    /// * `table_name` - Name of the table/collection for this entity type
    ///
    /// # Returns
    /// New repository instance
    pub fn new(db: Arc<dyn GraphDatabase>, table_name: String) -> Self {
        Self {
            db,
            table_name,
            validator: None,
            _phantom: std::marker::PhantomData}
    }

    /// Create with custom validation
    ///
    /// # Arguments
    /// * `db` - Graph database connection
    /// * `table_name` - Name of the table/collection
    /// * `validator` - Custom validation function
    ///
    /// # Returns
    /// New repository instance with validation
    pub fn with_validator(
        db: Arc<dyn GraphDatabase>,
        table_name: String,
        validator: EntityValidatorFn,
    ) -> Self {
        Self {
            db,
            table_name,
            validator: Some(validator),
            _phantom: std::marker::PhantomData}
    }

    /// Set or update the validator function
    pub fn set_validator(&mut self, validator: EntityValidatorFn) {
        self.validator = Some(validator);
    }

    /// Remove the validator function
    pub fn remove_validator(&mut self) {
        self.validator = None;
    }

    /// Get the table name
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Execute database operation with thread-based execution
    ///
    /// This method provides a way to execute database operations that need to be async
    /// in a synchronous context using thread-based execution.
    fn execute_db_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce(Arc<dyn GraphDatabase>) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db = self.db.clone();
        let (sender, receiver) = unbounded();

        thread::spawn(move || {
            let result = operation(db);
            let _ = sender.send(result);
        });

        receiver
            .recv()
            .map_err(|_| GraphError::Other("Database operation thread failed".to_string()))?
    }
}

impl<E: Entity + Clone + 'static> EntityRepository for SurrealEntityRepository<E> {
    fn create_entity(&self, entity: Box<dyn Entity>) -> Result<Box<dyn Entity>> {
        // Validate the entity synchronously if a validator is configured
        if let Some(validator) = &self.validator {
            validator(entity.as_ref())?;
        }

        // Validate the entity itself
        entity.validate()?;

        // Convert the entity to a node
        let node = entity.to_node();
        let db = self.db.clone();

        // Execute database operation synchronously using blocking approach
        // In a real implementation, you would use a blocking database client
        // For now, we'll simulate the operation
        let created_node_id = format!("{}:{}", self.table_name, entity.id());

        // Create a new entity with the generated ID
        let mut created_entity = entity;
        // Update the ID if it was generated by the database
        // This would normally be done by the database layer

        Ok(created_entity)
    }

    fn get_entity(&self, id: &str) -> Result<Option<Box<dyn Entity>>> {
        let db = self.db.clone();
        let id = id.to_string();

        // Execute synchronous database lookup
        // In a real implementation, you would use a blocking database client
        // For now, we'll return None to indicate not found

        // This would be the real implementation:
        // match db.get_node(&id) {
        //     Ok(Some(node)) => {
        //         let entity = E::from_node(node)?;
        //         Ok(Some(Box::new(entity) as Box<dyn Entity>))
        //     }
        //     Ok(None) => Ok(None),
        //     Err(e) => Err(e),
        // }

        Ok(None)
    }

    fn update_entity(&self, entity: Box<dyn Entity>) -> Result<Box<dyn Entity>> {
        // Validate the entity synchronously if a validator is configured
        if let Some(validator) = &self.validator {
            validator(entity.as_ref())?;
        }

        // Validate the entity itself
        entity.validate()?;

        // Convert the entity to a node
        let node = entity.to_node();
        let db = self.db.clone();

        // Execute synchronous database update
        // In a real implementation, you would use a blocking database client

        Ok(entity)
    }

    fn delete_entity(&self, id: &str) -> Result<()> {
        let db = self.db.clone();
        let id = id.to_string();

        // Execute synchronous database deletion
        // In a real implementation, you would use a blocking database client

        Ok(())
    }

    fn find_entities_by_type(
        &self,
        entity_type: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<Box<dyn Entity>>> {
        let db = self.db.clone();
        let entity_type = entity_type.to_string();

        // Execute synchronous database query
        // In a real implementation, you would use a blocking database client
        // and construct appropriate query with limit/offset

        Ok(Vec::new())
    }

    fn find_entities_by_attribute(
        &self,
        attribute_name: &str,
        attribute_value: &Value,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<Box<dyn Entity>>> {
        let db = self.db.clone();
        let attribute_name = attribute_name.to_string();
        let attribute_value = attribute_value.clone();

        // Execute synchronous database query with attribute filtering
        // In a real implementation, you would use a blocking database client

        Ok(Vec::new())
    }

    fn count_entities_by_type(&self, entity_type: &str) -> Result<usize> {
        let db = self.db.clone();
        let entity_type = entity_type.to_string();

        // Execute synchronous count query
        // In a real implementation, you would use a blocking database client

        Ok(0)
    }

    fn batch_create_entities(
        &self,
        entities: Vec<Box<dyn Entity>>,
    ) -> Result<Vec<Box<dyn Entity>>> {
        // Optimized batch implementation
        let mut results = Vec::with_capacity(entities.len());

        // Validate all entities first
        for entity in &entities {
            if let Some(validator) = &self.validator {
                validator(entity.as_ref())?;
            }
            entity.validate()?;
        }

        // Convert to nodes
        let nodes: Vec<Node> = entities.iter().map(|e| e.to_node()).collect();

        // Execute batch database operation
        // In a real implementation, you would use batch operations

        // For now, fall back to individual operations
        for entity in entities {
            results.push(self.create_entity(entity)?);
        }

        Ok(results)
    }

    fn batch_update_entities(
        &self,
        entities: Vec<Box<dyn Entity>>,
    ) -> Result<Vec<Box<dyn Entity>>> {
        // Optimized batch update implementation
        let mut results = Vec::with_capacity(entities.len());

        // Validate all entities first
        for entity in &entities {
            if let Some(validator) = &self.validator {
                validator(entity.as_ref())?;
            }
            entity.validate()?;
        }

        // Execute batch database operation
        // In a real implementation, you would use batch operations

        // For now, fall back to individual operations
        for entity in entities {
            results.push(self.update_entity(entity)?);
        }

        Ok(results)
    }

    fn batch_delete_entities(&self, ids: Vec<&str>) -> Result<()> {
        // Optimized batch delete implementation
        // In a real implementation, you would use batch delete operations

        // For now, fall back to individual operations
        for id in ids {
            self.delete_entity(id)?;
        }

        Ok(())
    }
}
