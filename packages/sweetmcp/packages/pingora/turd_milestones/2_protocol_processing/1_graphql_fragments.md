# GraphQL Fragment Handling - Implementation Plan

## Description
Implement proper GraphQL fragment spread resolution in `src/normalize/parsers.rs` lines 92-95. Current implementation skips fragment spreads instead of resolving them.

## Current Problem
```rust
Selection::FragmentSpread(_) => {
    // Fragment spreads would need fragment definition resolution
    // For now, skip fragment spreads
}
```

## Research Summary

### GraphQL Fragment Fundamentals
Based on the [GraphQL specification](./tmp/graphql-spec/spec/Section%202%20--%20Language.md), fragments are the primary unit of composition in GraphQL:

- **Fragment Definitions**: Named, reusable selections of fields with type conditions
- **Fragment Spreads**: References to fragment definitions using the spread operator (`...`)
- **Inline Fragments**: Anonymous fragments with type conditions applied directly in selection sets

### Fragment Types

#### 1. Named Fragment Spreads
```graphql
fragment friendFields on User {
  id
  name
  profilePic(size: 50)
}

query {
  user(id: 4) {
    friends {
      ...friendFields  # Fragment spread
    }
  }
}
```

#### 2. Inline Fragments
```graphql
query {
  search(text: "an") {
    __typename
    ... on Human {
      name
      height
    }
    ... on Droid {
      name
      primaryFunction
    }
  }
}
```

### Validation Requirements (GraphQL Spec Section 5)

1. **Fragment Definition Existence**: Fragment spreads must reference defined fragments
2. **Type Compatibility**: Fragment type conditions must be applicable to the selection context
3. **Circular Reference Detection**: Fragments cannot create circular dependencies
4. **Fragment Usage**: All defined fragments must be used

## Current Codebase Analysis

### Existing Infrastructure
- **Parser**: Uses `async-graphql` crate for GraphQL parsing ([Cargo.toml](./Cargo.toml))
- **Location**: Fragment handling in [`src/normalize/parsers.rs`](./src/normalize/parsers.rs) line 92-95
- **Context**: Part of `extract_fields_from_selection_set` function
- **Types**: Uses `async_graphql::parser::types::*` for AST structures

### Available AST Types (from async-graphql)
```rust
// From async-graphql parser types
Selection::FragmentSpread(fragment_spread) => {
    // fragment_spread.node.fragment_name: Name
    // fragment_spread.node.directives: Vec<Directive>
}

Selection::InlineFragment(inline_fragment) => {
    // inline_fragment.node.type_condition: Option<TypeCondition>
    // inline_fragment.node.selection_set: SelectionSet
    // inline_fragment.node.directives: Vec<Directive>
}
```

## Implementation Plan

### Phase 1: Fragment Storage and Resolution Infrastructure

#### 1.1 Fragment Registry
Create a fragment registry to store and resolve fragment definitions:

```rust
// In src/normalize/types.rs
use std::collections::HashMap;
use async_graphql::parser::types::{FragmentDefinition, TypeCondition};

#[derive(Debug, Clone)]
pub struct FragmentRegistry {
    fragments: HashMap<String, FragmentDefinition>,
    resolution_stack: Vec<String>, // For circular dependency detection
}

impl FragmentRegistry {
    pub fn new() -> Self {
        Self {
            fragments: HashMap::new(),
            resolution_stack: Vec::new(),
        }
    }
    
    pub fn register_fragment(&mut self, name: String, definition: FragmentDefinition) -> Result<(), ConversionError> {
        if self.fragments.contains_key(&name) {
            return Err(ConversionError::ValidationError(
                format!("Fragment '{}' is already defined", name)
            ));
        }
        self.fragments.insert(name, definition);
        Ok(())
    }
    
    pub fn get_fragment(&self, name: &str) -> Option<&FragmentDefinition> {
        self.fragments.get(name)
    }
    
    pub fn validate_no_cycles(&mut self, fragment_name: &str) -> Result<(), ConversionError> {
        if self.resolution_stack.contains(&fragment_name.to_string()) {
            return Err(ConversionError::ValidationError(
                format!("Circular fragment dependency detected: {}", 
                    self.resolution_stack.join(" -> "))
            ));
        }
        Ok(())
    }
    
    pub fn push_resolution(&mut self, name: String) {
        self.resolution_stack.push(name);
    }
    
    pub fn pop_resolution(&mut self) {
        self.resolution_stack.pop();
    }
}
```

#### 1.2 Enhanced Conversion Context
Extend `ProtocolContext` to include fragment registry:

```rust
// In src/normalize/types.rs
#[derive(Debug, Clone)]
pub struct GraphQLContext {
    pub fragment_registry: FragmentRegistry,
    pub type_info: Option<String>, // Current type context for validation
}

impl ProtocolContext {
    pub fn with_graphql_context(mut self, graphql_context: GraphQLContext) -> Self {
        // Store GraphQL-specific context
        self.metadata.options.graphql_context = Some(graphql_context);
        self
    }
}
```

### Phase 2: Fragment Resolution Implementation

#### 2.1 Two-Pass Processing
Modify `graphql_to_json_rpc` function for two-pass processing:

```rust
// In src/normalize/parsers.rs
pub fn graphql_to_json_rpc(
    query: &str,
    variables: Value,
    operation_name: Option<Value>,
    request_id: &str,
) -> Result<Value> {
    debug!("Converting GraphQL query to JSON-RPC with fragment resolution");

    // Parse GraphQL query
    let doc = parse_query(query).map_err(|e| anyhow::anyhow!("GraphQL parse error: {}", e))?;

    // Phase 1: Collect all fragment definitions
    let mut fragment_registry = FragmentRegistry::new();
    for (name, fragment) in &doc.fragments {
        fragment_registry.register_fragment(name.to_string(), fragment.clone())?;
    }

    // Phase 2: Process operations with fragment resolution
    let operation = doc.operations.iter().next();
    
    let (method, params) = match operation {
        Some((name, op)) => {
            let method_name = determine_method_name(operation_name, name);
            
            // Extract fields with fragment resolution
            let mut fields = Vec::new();
            let mut context = GraphQLContext {
                fragment_registry,
                type_info: None, // Would be determined from schema in full implementation
            };
            
            extract_fields_with_fragments(
                &op.node.selection_set.node, 
                &mut fields, 
                &mut context
            )?;

            let params = json!({
                "query": query,
                "variables": variables,
                "operationName": operation_name,
                "fields": fields,
                "operationType": format!("{:?}", op.node.ty),
                "resolvedFragments": true
            });

            (method_name, params)
        }
        None => {
            warn!("No GraphQL operation found, using default");
            // ... existing fallback logic
        }
    };

    Ok(json!({
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
        "params": params,
        "id": request_id
    }))
}
```

#### 2.2 Fragment-Aware Field Extraction
Replace `extract_fields_from_selection_set` with fragment-aware version:

```rust
// In src/normalize/parsers.rs
fn extract_fields_with_fragments(
    selection_set: &SelectionSet,
    fields: &mut Vec<String>,
    context: &mut GraphQLContext,
) -> Result<(), ConversionError> {
    for selection in &selection_set.items {
        match &selection.node {
            Selection::Field(field) => {
                fields.push(field.node.name.node.to_string());

                // Recursively extract nested fields
                if !field.node.selection_set.node.items.is_empty() {
                    extract_fields_with_fragments(
                        &field.node.selection_set.node, 
                        fields, 
                        context
                    )?;
                }
            }
            Selection::InlineFragment(fragment) => {
                // Validate type condition if present
                if let Some(type_condition) = &fragment.node.type_condition {
                    validate_type_condition(type_condition, context)?;
                }
                
                extract_fields_with_fragments(
                    &fragment.node.selection_set.node, 
                    fields, 
                    context
                )?;
            }
            Selection::FragmentSpread(fragment_spread) => {
                let fragment_name = &fragment_spread.node.fragment_name.node;
                
                // Circular dependency check
                context.fragment_registry.validate_no_cycles(fragment_name)?;
                
                // Get fragment definition
                let fragment_def = context.fragment_registry
                    .get_fragment(fragment_name)
                    .ok_or_else(|| ConversionError::ValidationError(
                        format!("Fragment '{}' is not defined", fragment_name)
                    ))?;
                
                // Validate fragment type condition
                validate_type_condition(&fragment_def.node.type_condition, context)?;
                
                // Track resolution for cycle detection
                context.fragment_registry.push_resolution(fragment_name.to_string());
                
                // Recursively resolve fragment
                extract_fields_with_fragments(
                    &fragment_def.node.selection_set.node,
                    fields,
                    context,
                )?;
                
                // Pop resolution stack
                context.fragment_registry.pop_resolution();
            }
        }
    }
    Ok(())
}

fn validate_type_condition(
    type_condition: &TypeCondition,
    context: &GraphQLContext,
) -> Result<(), ConversionError> {
    // In a full implementation, this would validate against a schema
    // For now, we'll do basic validation
    let type_name = &type_condition.node.on.node;
    
    // Basic validation - ensure type name is not empty
    if type_name.is_empty() {
        return Err(ConversionError::ValidationError(
            "Fragment type condition cannot be empty".to_string()
        ));
    }
    
    // TODO: Add schema-based type compatibility validation
    debug!("Validating fragment type condition: {}", type_name);
    Ok(())
}
```

### Phase 3: Performance Optimization

#### 3.1 Fragment Cache
Implement caching for resolved fragments:

```rust
// In src/normalize/types.rs
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct FragmentCache {
    resolved_fields: HashMap<String, Vec<String>>,
    cache_hits: u64,
    cache_misses: u64,
}

impl FragmentCache {
    pub fn new() -> Self {
        Self {
            resolved_fields: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    
    pub fn get_cached_fields(&mut self, fragment_name: &str) -> Option<&Vec<String>> {
        if let Some(fields) = self.resolved_fields.get(fragment_name) {
            self.cache_hits += 1;
            Some(fields)
        } else {
            self.cache_misses += 1;
            None
        }
    }
    
    pub fn cache_fields(&mut self, fragment_name: String, fields: Vec<String>) {
        self.resolved_fields.insert(fragment_name, fields);
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }
}
```

### Phase 4: Error Handling and Validation

#### 4.1 Comprehensive Error Types
Extend `ConversionError` for fragment-specific errors:

```rust
// In src/normalize/types.rs
#[derive(Debug, thiserror::Error)]
pub enum ConversionError {
    // ... existing variants ...
    
    #[error("Fragment error: {0}")]
    FragmentError(String),
    
    #[error("Fragment '{name}' not found")]
    FragmentNotFound { name: String },
    
    #[error("Circular fragment dependency: {cycle}")]
    CircularFragmentDependency { cycle: String },
    
    #[error("Fragment type condition error: {0}")]
    TypeConditionError(String),
    
    #[error("Fragment validation failed: {0}")]
    FragmentValidationError(String),
}
```

## Testing Strategy

### Unit Tests
```rust
// In tests/fragment_resolution_test.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fragment_resolution() {
        let query = r#"
            query {
                user {
                    ...userFields
                }
            }
            
            fragment userFields on User {
                id
                name
                email
            }
        "#;
        
        let result = graphql_to_json_rpc(query, json!({}), None, "test-1");
        assert!(result.is_ok());
        
        let json_rpc = result.unwrap();
        let fields = json_rpc["params"]["fields"].as_array().unwrap();
        assert!(fields.contains(&json!("id")));
        assert!(fields.contains(&json!("name")));
        assert!(fields.contains(&json!("email")));
    }

    #[test]
    fn test_nested_fragment_resolution() {
        let query = r#"
            query {
                user {
                    ...userFields
                }
            }
            
            fragment userFields on User {
                id
                ...profileFields
            }
            
            fragment profileFields on User {
                name
                email
            }
        "#;
        
        let result = graphql_to_json_rpc(query, json!({}), None, "test-2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_circular_dependency_detection() {
        let query = r#"
            query {
                user {
                    ...fragmentA
                }
            }
            
            fragment fragmentA on User {
                id
                ...fragmentB
            }
            
            fragment fragmentB on User {
                name
                ...fragmentA
            }
        "#;
        
        let result = graphql_to_json_rpc(query, json!({}), None, "test-3");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Circular"));
    }

    #[test]
    fn test_missing_fragment_error() {
        let query = r#"
            query {
                user {
                    ...missingFragment
                }
            }
        "#;
        
        let result = graphql_to_json_rpc(query, json!({}), None, "test-4");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not defined"));
    }

    #[test]
    fn test_inline_fragment_resolution() {
        let query = r#"
            query {
                search {
                    ... on User {
                        name
                        email
                    }
                    ... on Post {
                        title
                        content
                    }
                }
            }
        "#;
        
        let result = graphql_to_json_rpc(query, json!({}), None, "test-5");
        assert!(result.is_ok());
    }
}
```

### Integration Tests
```rust
// In tests/integration/fragment_performance_test.rs
#[tokio::test]
async fn test_fragment_cache_performance() {
    // Test that fragment caching improves performance for repeated queries
    let query_with_fragments = r#"
        query {
            users {
                ...userFields
            }
            posts {
                author {
                    ...userFields
                }
            }
        }
        
        fragment userFields on User {
            id
            name
            email
            profile {
                avatar
                bio
            }
        }
    "#;
    
    // Measure performance with and without caching
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = graphql_to_json_rpc(query_with_fragments, json!({}), None, "perf-test");
    }
    let duration = start.elapsed();
    
    // Assert reasonable performance (specific thresholds would depend on requirements)
    assert!(duration.as_millis() < 1000, "Fragment resolution should be fast");
}
```

## Success Criteria
- [x] Research GraphQL fragment specification and best practices
- [ ] Implement fragment definition collection and storage
- [ ] Implement fragment spread resolution with circular dependency detection
- [ ] Add fragment validation and type condition checking
- [ ] Create fragment cache for performance optimization
- [ ] Add comprehensive error handling for fragment-related errors
- [ ] Support both named fragments and inline fragments
- [ ] Validate fragment type conditions against selection context
- [ ] Handle nested fragment spreads correctly
- [ ] Implement performance monitoring and metrics
- [ ] Create comprehensive test suite covering all fragment scenarios

## Technical Resolution Details

### Files to Modify
1. **`src/normalize/parsers.rs`** (lines 92-95 and surrounding functions)
   - Replace fragment spread skip with proper resolution
   - Implement two-pass processing (collect fragments, then resolve)
   - Add fragment-aware field extraction

2. **`src/normalize/types.rs`**
   - Add `FragmentRegistry` and `FragmentCache` structures
   - Extend `ConversionError` with fragment-specific error types
   - Add `GraphQLContext` for fragment resolution context

3. **`src/normalize/mod.rs`**
   - Export new fragment-related types and functions

### Dependencies
- **Existing**: `async-graphql` crate already provides necessary AST types
- **No new dependencies required** - implementation uses existing infrastructure

### Performance Considerations
- **Fragment Cache**: Resolved fragments cached to avoid re-processing
- **Circular Detection**: Efficient stack-based cycle detection
- **Memory Usage**: Fragment registry cleared after each query processing
- **Validation**: Type condition validation optimized for common cases

## References
- [GraphQL Specification - Fragments](./tmp/graphql-spec/spec/Section%202%20--%20Language.md#fragments)
- [GraphQL Validation Rules](./tmp/graphql-spec/spec/Section%205%20--%20Validation.md)
- [async-graphql Documentation](./tmp/async-graphql/README.md)
- [Current Implementation](./src/normalize/parsers.rs#L92-L95)

## Priority
MEDIUM - Protocol completeness

## Dependencies
- Milestone 0 must be completed (foundation safety fixes)
- Milestone 1 must be completed (core security)