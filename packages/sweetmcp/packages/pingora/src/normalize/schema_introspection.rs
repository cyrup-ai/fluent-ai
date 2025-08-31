//! Schema introspection module for upstream GraphQL servers
//!
//! This module provides schema introspection capabilities for SweetMCP Pingora
//! to extract real schema information from upstream GraphQL servers and cache
//! it for fragment resolution.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use super::types::{GraphQLTypeInfo, GraphQLTypeKind};

/// GraphQL introspection query to get full schema information
const INTROSPECTION_QUERY: &str = r#"
query IntrospectionQuery {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      ...FullType
    }
  }
}

fragment FullType on __Type {
  kind
  name
  description
  fields(includeDeprecated: true) {
    name
    description
    args {
      ...InputValue
    }
    type {
      ...TypeRef
    }
    isDeprecated
    deprecationReason
  }
  inputFields {
    ...InputValue
  }
  interfaces {
    ...TypeRef
  }
  enumValues(includeDeprecated: true) {
    name
    description
    isDeprecated
    deprecationReason
  }
  possibleTypes {
    ...TypeRef
  }
}

fragment InputValue on __InputValue {
  name
  description
  type { ...TypeRef }
  defaultValue
}

fragment TypeRef on __Type {
  kind
  name
  ofType {
    kind
    name
    ofType {
      kind
      name
      ofType {
        kind
        name
        ofType {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
              }
            }
          }
        }
      }
    }
  }
}
"#;

/// Introspection response structures
#[derive(Debug, Deserialize)]
struct IntrospectionResponse {
    data: IntrospectionData,
}

#[derive(Debug, Deserialize)]
struct IntrospectionData {
    #[serde(rename = "__schema")]
    schema: SchemaData,
}

#[derive(Debug, Deserialize)]
struct SchemaData {
    #[serde(rename = "queryType")]
    query_type: Option<TypeRef>,
    #[serde(rename = "mutationType")]
    mutation_type: Option<TypeRef>,
    #[serde(rename = "subscriptionType")]
    subscription_type: Option<TypeRef>,
    types: Vec<TypeData>,
}

#[derive(Debug, Deserialize)]
struct TypeData {
    kind: String,
    name: Option<String>,
    description: Option<String>,
    fields: Option<Vec<FieldData>>,
    interfaces: Option<Vec<TypeRef>>,
    #[serde(rename = "possibleTypes")]
    possible_types: Option<Vec<TypeRef>>,
    #[serde(rename = "enumValues")]
    enum_values: Option<Vec<EnumValueData>>,
    #[serde(rename = "inputFields")]
    input_fields: Option<Vec<InputValueData>>,
}

#[derive(Debug, Deserialize)]
struct FieldData {
    name: String,
    description: Option<String>,
    #[serde(rename = "type")]
    field_type: TypeRef,
    args: Vec<InputValueData>,
    #[serde(rename = "isDeprecated")]
    is_deprecated: bool,
    #[serde(rename = "deprecationReason")]
    deprecation_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TypeRef {
    kind: String,
    name: Option<String>,
    #[serde(rename = "ofType")]
    of_type: Option<Box<TypeRef>>,
}

#[derive(Debug, Deserialize)]
struct EnumValueData {
    name: String,
    description: Option<String>,
    #[serde(rename = "isDeprecated")]
    is_deprecated: bool,
    #[serde(rename = "deprecationReason")]
    deprecation_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InputValueData {
    name: String,
    description: Option<String>,
    #[serde(rename = "type")]
    input_type: TypeRef,
    #[serde(rename = "defaultValue")]
    default_value: Option<String>,
}

/// Cached schema information for an upstream server
#[derive(Debug, Clone)]
pub struct CachedSchema {
    pub types: HashMap<String, GraphQLTypeInfo>,
    pub cached_at: Instant,
    pub ttl: Duration,
}

impl CachedSchema {
    /// Check if the cached schema is still valid
    pub fn is_valid(&self) -> bool {
        self.cached_at.elapsed() < self.ttl
    }
}

/// Schema introspection manager for upstream GraphQL servers
pub struct SchemaIntrospector {
    client: Client,
    cache: Arc<RwLock<HashMap<String, CachedSchema>>>,
    default_ttl: Duration,
}

impl SchemaIntrospector {
    /// Create a new schema introspector
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client for schema introspection")?;

        Ok(Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_ttl: Duration::from_secs(3600), // 1 hour default TTL
        })
    }

    /// Create a new schema introspector with custom settings
    pub fn with_settings(timeout: Duration, cache_ttl: Duration) -> Result<Self> {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .context("Failed to create HTTP client for schema introspection")?;

        Ok(Self {
            client,
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_ttl: cache_ttl,
        })
    }

    /// Get schema information for an upstream GraphQL server
    pub async fn get_schema(&self, upstream_url: &str) -> Result<HashMap<String, GraphQLTypeInfo>> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(upstream_url) {
                if cached.is_valid() {
                    debug!("Using cached schema for upstream: {}", upstream_url);
                    return Ok(cached.types.clone());
                }
            }
        }

        // Cache miss or expired, perform introspection
        info!(
            "Performing schema introspection for upstream: {}",
            upstream_url
        );
        let schema_types = self.introspect_schema(upstream_url).await?;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(
                upstream_url.to_string(),
                CachedSchema {
                    types: schema_types.clone(),
                    cached_at: Instant::now(),
                    ttl: self.default_ttl,
                },
            );
        }

        Ok(schema_types)
    }

    /// Perform schema introspection against an upstream GraphQL server
    async fn introspect_schema(
        &self,
        upstream_url: &str,
    ) -> Result<HashMap<String, GraphQLTypeInfo>> {
        let request_body = json!({
            "query": INTROSPECTION_QUERY,
            "variables": {}
        });

        let response = self
            .client
            .post(upstream_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&request_body)
            .send()
            .await
            .context("Failed to send introspection query")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Introspection query failed with status: {}",
                response.status()
            ));
        }

        let introspection_response: IntrospectionResponse = response
            .json()
            .await
            .context("Failed to parse introspection response")?;

        self.parse_introspection_response(introspection_response)
    }

    /// Parse introspection response into GraphQLTypeInfo structures
    fn parse_introspection_response(
        &self,
        response: IntrospectionResponse,
    ) -> Result<HashMap<String, GraphQLTypeInfo>> {
        let mut schema_types = HashMap::new();

        for type_data in response.data.schema.types {
            if let Some(type_name) = &type_data.name {
                // Skip introspection types
                if type_name.starts_with("__") {
                    continue;
                }

                let type_info = self.convert_type_data_to_info(&type_data)?;
                schema_types.insert(type_name.clone(), type_info);
            }
        }

        info!(
            "Parsed {} types from introspection response",
            schema_types.len()
        );
        Ok(schema_types)
    }

    /// Convert TypeData to GraphQLTypeInfo
    fn convert_type_data_to_info(&self, type_data: &TypeData) -> Result<GraphQLTypeInfo> {
        let name = type_data
            .name
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Type data missing name"))?
            .clone();

        let kind = match type_data.kind.as_str() {
            "OBJECT" => GraphQLTypeKind::Object,
            "INTERFACE" => GraphQLTypeKind::Interface,
            "UNION" => GraphQLTypeKind::Union,
            "ENUM" => GraphQLTypeKind::Enum,
            "SCALAR" => GraphQLTypeKind::Scalar,
            "INPUT_OBJECT" => GraphQLTypeKind::InputObject,
            _ => {
                warn!("Unknown GraphQL type kind: {}", type_data.kind);
                GraphQLTypeKind::Object // Default fallback
            }
        };

        let fields = type_data
            .fields
            .as_ref()
            .map(|fields| fields.iter().map(|f| f.name.clone()).collect())
            .unwrap_or_else(Vec::new);

        let interfaces = type_data
            .interfaces
            .as_ref()
            .map(|interfaces| interfaces.iter().filter_map(|i| i.name.clone()).collect())
            .unwrap_or_else(Vec::new);

        let possible_types = type_data
            .possible_types
            .as_ref()
            .map(|possible_types| {
                possible_types
                    .iter()
                    .filter_map(|pt| pt.name.clone())
                    .collect()
            })
            .unwrap_or_else(Vec::new);

        Ok(GraphQLTypeInfo {
            name,
            kind,
            fields,
            interfaces,
            possible_types,
        })
    }

    /// Clear expired entries from the cache
    pub async fn cleanup_cache(&self) {
        let mut cache = self.cache.write().await;
        let initial_size = cache.len();

        cache.retain(|_, cached_schema| cached_schema.is_valid());

        let final_size = cache.len();
        if initial_size > final_size {
            debug!(
                "Cleaned up {} expired schema cache entries",
                initial_size - final_size
            );
        }
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let total = cache.len();
        let valid = cache.values().filter(|c| c.is_valid()).count();
        (total, valid)
    }
}

impl Default for SchemaIntrospector {
    fn default() -> Self {
        // Use a basic configuration that should never fail
        Self {
            client: Client::new(), // Use default client without custom timeout
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_ttl: Duration::from_secs(3600),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_schema_introspector_creation() {
        let introspector = SchemaIntrospector::new().expect("Failed to create introspector");
        let (total, valid) = introspector.cache_stats().await;
        assert_eq!(total, 0);
        assert_eq!(valid, 0);
    }

    #[tokio::test]
    async fn test_cached_schema_validity() {
        let schema = CachedSchema {
            types: HashMap::new(),
            cached_at: Instant::now(),
            ttl: Duration::from_secs(1),
        };

        assert!(schema.is_valid());

        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert!(!schema.is_valid());
    }

    #[test]
    fn test_type_kind_conversion() {
        let introspector = SchemaIntrospector::new().expect("Failed to create introspector");

        let type_data = TypeData {
            kind: "OBJECT".to_string(),
            name: Some("User".to_string()),
            description: None,
            fields: Some(vec![FieldData {
                name: "id".to_string(),
                description: None,
                field_type: TypeRef {
                    kind: "SCALAR".to_string(),
                    name: Some("ID".to_string()),
                    of_type: None,
                },
                args: vec![],
                is_deprecated: false,
                deprecation_reason: None,
            }]),
            interfaces: None,
            possible_types: None,
            enum_values: None,
            input_fields: None,
        };

        let type_info = introspector.convert_type_data_to_info(&type_data).unwrap();
        assert_eq!(type_info.name, "User");
        assert_eq!(type_info.kind, GraphQLTypeKind::Object);
        assert_eq!(type_info.fields, vec!["id"]);
    }
}
