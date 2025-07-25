//! Query processing and parsing system
//!
//! This module provides comprehensive query processing with expansion, normalization,
//! and optimization using zero-allocation patterns and streaming-first architecture.

use std::sync::Arc;
use std::time::Instant;
use fluent_ai_async::{AsyncStream, emit, handle_error};

use super::types::{QueryOperator, SearchOptions};

/// Processed query structure
#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    /// Original query
    pub original: Arc<str>,
    /// Processed terms
    pub terms: Vec<Arc<str>>,
    /// Expanded terms (synonyms, etc.)
    pub expanded_terms: Vec<Arc<str>>,
    /// Query operator
    pub operator: QueryOperator,
    /// Processing metadata
    pub metadata: QueryMetadata}

/// Query metadata
#[derive(Debug, Clone)]
pub struct QueryMetadata {
    /// Processing timestamp
    pub processed_at: u64,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Expansion applied
    pub expansion_applied: bool,
    /// Normalization applied
    pub normalization_applied: bool}

/// Query processor for handling search queries
#[derive(Debug, Clone)]
pub struct QueryProcessor {
    /// Enable query expansion
    expansion_enabled: bool,
    /// Expansion dictionary for synonyms
    expansion_dict: HashMap<Arc<str>, Vec<Arc<str>>>}

impl QueryProcessor {
    /// Create a new query processor
    pub fn new() -> Self {
        Self {
            expansion_enabled: false,
            expansion_dict: HashMap::new()}
    }

    /// Create query processor with options
    pub fn with_options(options: &SearchOptions) -> Self {
        Self {
            expansion_enabled: options.enable_query_expansion,
            expansion_dict: options.expansion_dictionary.clone()}
    }

    /// Process a query string with fluent-ai-async streaming architecture
    pub fn process_query(
        &self,
        query: &str,
        options: &SearchOptions,
    ) -> AsyncStream<ProcessedQuery> {
        let query = query.to_string();
        let options = options.clone();
        
        AsyncStream::with_channel(move |sender| {
            let start_time = Instant::now();

            // Basic query processing
            let terms: Vec<Arc<str>> = query
                .split_whitespace()
                .map(|term| Arc::from(term.to_lowercase()))
                .collect();

            // Apply query expansion if enabled
            let expanded_terms = if options.enable_query_expansion {
                let mut expanded = Vec::new();
                for term in &terms {
                    if let Some(synonyms) = options.expansion_dictionary.get(term) {
                        expanded.extend(synonyms.clone());
                    }
                }
                expanded
            } else {
                Vec::new()
            };

            emit!(sender, ProcessedQuery {
                original: Arc::from(query.as_str()),
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
                    normalization_applied: true}});
        })
    }

    /// Expand query terms using synonyms with fluent-ai-async streaming architecture
    pub fn expand_terms(
        &self,
        terms: &[Arc<str>],
        dictionary: &HashMap<Arc<str>, Vec<Arc<str>>>,
    ) -> AsyncStream<Vec<Arc<str>>> {
        let terms = terms.to_vec();
        let dictionary = dictionary.clone();
        
        AsyncStream::with_channel(move |sender| {
            let mut expanded = Vec::new();

            for term in &terms {
                if let Some(synonyms) = dictionary.get(term) {
                    expanded.extend(synonyms.clone());
                }
            }

            emit!(sender, expanded);
        })
    }

    /// Normalize query terms with fluent-ai-async streaming architecture
    pub fn normalize_terms(&self, terms: &[Arc<str>]) -> AsyncStream<Vec<Arc<str>>> {
        let terms = terms.to_vec();
        
        AsyncStream::with_channel(move |sender| {
            let normalized: Vec<Arc<str>> = terms
                .iter()
                .map(|term| {
                    // Apply normalization rules
                    let normalized = term
                        .chars()
                        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                        .collect::<String>()
                        .to_lowercase()
                        .trim()
                        .to_string();
                    Arc::from(normalized.as_str())
                })
                .filter(|term| !term.is_empty())
                .collect();

            emit!(sender, normalized);
        })
    }

    /// Parse query operators with fluent-ai-async streaming architecture
    pub fn parse_operators(&self, query: &str) -> AsyncStream<QueryOperator> {
        let query = query.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let operator = if query.contains(" AND ") {
                QueryOperator::And
            } else if query.contains(" OR ") {
                QueryOperator::Or
            } else if query.contains(" NOT ") {
                QueryOperator::Not
            } else if query.starts_with('"') && query.ends_with('"') {
                QueryOperator::Phrase
            } else if query.contains(" NEAR ") {
                // Extract distance if specified
                QueryOperator::Proximity { distance: 5 } // Default distance
            } else {
                QueryOperator::And // Default operator
            };

            emit!(sender, operator);
        })
    }

    /// Validate query syntax with fluent-ai-async streaming architecture
    pub fn validate_query(&self, query: &str) -> AsyncStream<Result<(), QueryValidationError>> {
        let query = query.to_string();
        
        AsyncStream::with_channel(move |sender| {
            // Basic validation rules
            if query.trim().is_empty() {
                emit!(sender, Err(QueryValidationError::EmptyQuery));
                return;
            }

            if query.len() > 1000 {
                emit!(sender, Err(QueryValidationError::QueryTooLong));
                return;
            }

            // Check for balanced quotes
            let quote_count = query.chars().filter(|&c| c == '"').count();
            if quote_count % 2 != 0 {
                emit!(sender, Err(QueryValidationError::UnbalancedQuotes));
                return;
            }

            // Check for valid operators
            let invalid_operators = ["AND AND", "OR OR", "NOT NOT"];
            for invalid in &invalid_operators {
                if query.contains(invalid) {
                    emit!(sender, Err(QueryValidationError::InvalidOperator));
                    return;
                }
            }

            emit!(sender, Ok(()));
        })
    }

    /// Get query suggestions with fluent-ai-async streaming architecture
    pub fn get_suggestions(&self, partial_query: &str) -> AsyncStream<Vec<Arc<str>>> {
        let partial_query = partial_query.to_string();
        
        AsyncStream::with_channel(move |sender| {
            let mut suggestions = Vec::new();

            // Simple suggestion logic - would be more sophisticated in production
            if partial_query.len() >= 2 {
                // Add common search patterns
                suggestions.push(Arc::from(format!("{} AND", partial_query).as_str()));
                suggestions.push(Arc::from(format!("{} OR", partial_query).as_str()));
                suggestions.push(Arc::from(format!("\"{}\"", partial_query).as_str()));
            }

            emit!(sender, suggestions);
        })
    }

    /// Enable query expansion
    pub fn enable_expansion(&mut self, dictionary: HashMap<Arc<str>, Vec<Arc<str>>>) {
        self.expansion_enabled = true;
        self.expansion_dict = dictionary;
    }

    /// Disable query expansion
    pub fn disable_expansion(&mut self) {
        self.expansion_enabled = false;
        self.expansion_dict.clear();
    }
}

/// Query validation error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum QueryValidationError {
    #[error("Query cannot be empty")]
    EmptyQuery,
    
    #[error("Query is too long (max 1000 characters)")]
    QueryTooLong,
    
    #[error("Unbalanced quotes in query")]
    UnbalancedQuotes,
    
    #[error("Invalid operator usage")]
    InvalidOperator,
    
    #[error("Unsupported query syntax")]
    UnsupportedSyntax}

impl Default for QueryProcessor {
    fn default() -> Self {
        Self::new()
    }
}
