//! Query processing and expansion functionality
//!
//! This module handles query parsing, processing, expansion, and normalization
//! for advanced search capabilities.

use std::collections::HashMap;
use std::sync::Arc;

use super::types::{
    ProcessedQuery, QueryMetadata, QueryOperator, SearchOptions,
};

/// Query processor for advanced query handling
pub struct QueryProcessor {
    /// Enable query expansion
    expansion_enabled: bool,
    /// Expansion dictionary for synonyms
    expansion_dict: HashMap<Arc<str>, Vec<Arc<str>>>,
}

impl QueryProcessor {
    /// Create a new query processor
    pub fn new() -> Self {
        Self {
            expansion_enabled: false,
            expansion_dict: HashMap::new(),
        }
    }

    /// Create query processor with expansion enabled
    pub fn with_expansion(expansion_dict: HashMap<Arc<str>, Vec<Arc<str>>>) -> Self {
        Self {
            expansion_enabled: true,
            expansion_dict,
        }
    }

    /// Process a query string
    pub fn process_query(
        &self,
        query: &str,
        options: &SearchOptions,
    ) -> Result<ProcessedQuery, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();

        // Basic query processing
        let terms: Vec<Arc<str>> = query
            .split_whitespace()
            .map(|term| Arc::from(term.to_lowercase()))
            .collect();

        // Apply query expansion if enabled
        let expanded_terms = if options.enable_query_expansion {
            match self.expand_terms_sync(&terms, &options.expansion_dictionary) {
                Ok(terms) => terms,
                Err(_) => Vec::new(), // Fallback to no expansion on error
            }
        } else {
            Vec::new()
        };

        let result = ProcessedQuery {
            original: Arc::from(query.to_string()),
            terms,
            expanded_terms,
            operator: QueryOperator::And, // Default to AND
            metadata: QueryMetadata {
                processed_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                processing_time_us: start_time.elapsed().as_micros() as u64,
                expansion_applied: options.enable_query_expansion,
                normalization_applied: true,
            },
        };

        Ok(result)
    }

    /// Parse query with operators
    pub fn parse_query_with_operators(&self, query: &str) -> Result<ProcessedQuery, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        // Detect query operators
        let (operator, terms) = if query.contains(" AND ") {
            (QueryOperator::And, self.extract_terms_with_operator(query, " AND "))
        } else if query.contains(" OR ") {
            (QueryOperator::Or, self.extract_terms_with_operator(query, " OR "))
        } else if query.contains(" NOT ") {
            (QueryOperator::Not, self.extract_terms_with_operator(query, " NOT "))
        } else if query.starts_with('"') && query.ends_with('"') {
            (QueryOperator::Phrase, vec![Arc::from(query.trim_matches('"').to_lowercase())])
        } else if query.contains(" NEAR ") {
            // Parse proximity query like "term1 NEAR/5 term2"
            let parts: Vec<&str> = query.split(" NEAR").collect();
            if parts.len() == 2 {
                let distance = self.extract_proximity_distance(parts[1])?;
                let terms = parts[0].split_whitespace()
                    .chain(parts[1].split_whitespace().skip(1)) // Skip the distance part
                    .map(|term| Arc::from(term.to_lowercase()))
                    .collect();
                (QueryOperator::Proximity { distance }, terms)
            } else {
                (QueryOperator::And, self.simple_tokenize(query))
            }
        } else {
            (QueryOperator::And, self.simple_tokenize(query))
        };

        let result = ProcessedQuery {
            original: Arc::from(query.to_string()),
            terms,
            expanded_terms: Vec::new(),
            operator,
            metadata: QueryMetadata {
                processed_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                processing_time_us: start_time.elapsed().as_micros() as u64,
                expansion_applied: false,
                normalization_applied: true,
            },
        };

        Ok(result)
    }

    /// Expand query terms using synonyms (synchronous for streams-only architecture)  
    fn expand_terms_sync(
        &self,
        terms: &[Arc<str>],
        dictionary: &HashMap<Arc<str>, Vec<Arc<str>>>,
    ) -> Result<Vec<Arc<str>>, Box<dyn std::error::Error + Send + Sync>> {
        let mut expanded = Vec::new();

        for term in terms {
            if let Some(synonyms) = dictionary.get(term) {
                expanded.extend(synonyms.clone());
            }
        }

        Ok(expanded)
    }

    /// Extract terms with specific operator
    fn extract_terms_with_operator(&self, query: &str, operator: &str) -> Vec<Arc<str>> {
        query.split(operator)
            .flat_map(|part| part.split_whitespace())
            .map(|term| Arc::from(term.to_lowercase()))
            .collect()
    }

    /// Simple tokenization
    fn simple_tokenize(&self, query: &str) -> Vec<Arc<str>> {
        query.split_whitespace()
            .map(|term| Arc::from(term.to_lowercase()))
            .collect()
    }

    /// Extract proximity distance from query part
    fn extract_proximity_distance(&self, part: &str) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        // Look for pattern like "/5" in "NEAR/5"
        if part.starts_with('/') {
            let distance_str = part.chars().skip(1).take_while(|c| c.is_ascii_digit()).collect::<String>();
            if !distance_str.is_empty() {
                return Ok(distance_str.parse()?);
            }
        }
        
        // Default proximity distance
        Ok(5)
    }

    /// Normalize query terms
    pub fn normalize_terms(&self, terms: &[Arc<str>]) -> Vec<Arc<str>> {
        terms.iter()
            .map(|term| {
                // Remove punctuation and normalize whitespace
                let normalized: String = term
                    .chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_lowercase();
                Arc::from(normalized)
            })
            .filter(|term: &Arc<str>| !term.is_empty())
            .collect()
    }

    /// Add synonym to expansion dictionary
    pub fn add_synonym(&mut self, term: Arc<str>, synonyms: Vec<Arc<str>>) {
        self.expansion_dict.insert(term, synonyms);
        self.expansion_enabled = true;
    }

    /// Remove synonym from expansion dictionary
    pub fn remove_synonym(&mut self, term: &Arc<str>) {
        self.expansion_dict.remove(term);
    }

    /// Get all synonyms for a term
    pub fn get_synonyms(&self, term: &Arc<str>) -> Option<&Vec<Arc<str>>> {
        self.expansion_dict.get(term)
    }

    /// Check if expansion is enabled
    pub fn is_expansion_enabled(&self) -> bool {
        self.expansion_enabled
    }

    /// Enable/disable expansion
    pub fn set_expansion_enabled(&mut self, enabled: bool) {
        self.expansion_enabled = enabled;
    }

    /// Get expansion dictionary size
    pub fn expansion_dict_size(&self) -> usize {
        self.expansion_dict.len()
    }

    /// Clear expansion dictionary
    pub fn clear_expansion_dict(&mut self) {
        self.expansion_dict.clear();
        self.expansion_enabled = false;
    }

    /// Validate query syntax
    pub fn validate_query(&self, query: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if query.trim().is_empty() {
            return Err("Query cannot be empty".into());
        }

        // Check for unbalanced quotes
        let quote_count = query.chars().filter(|&c| c == '"').count();
        if quote_count % 2 != 0 {
            return Err("Unbalanced quotes in query".into());
        }

        // Check for valid proximity syntax
        if query.contains(" NEAR") {
            let parts: Vec<&str> = query.split(" NEAR").collect();
            if parts.len() != 2 {
                return Err("Invalid NEAR operator syntax".into());
            }
        }

        Ok(())
    }

    /// Get query complexity score
    pub fn get_query_complexity(&self, query: &str) -> u32 {
        let mut complexity = 0;

        // Base complexity for terms
        complexity += query.split_whitespace().count() as u32;

        // Add complexity for operators
        if query.contains(" AND ") { complexity += 1; }
        if query.contains(" OR ") { complexity += 2; }
        if query.contains(" NOT ") { complexity += 1; }
        if query.contains(" NEAR") { complexity += 3; }
        if query.contains('"') { complexity += 2; }

        complexity
    }
}

impl Default for QueryProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for QueryProcessor {
    fn clone(&self) -> Self {
        Self {
            expansion_enabled: self.expansion_enabled,
            expansion_dict: self.expansion_dict.clone(),
        }
    }
}

impl std::fmt::Debug for QueryProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryProcessor")
            .field("expansion_enabled", &self.expansion_enabled)
            .field("expansion_dict_size", &self.expansion_dict.len())
            .finish()
    }
}