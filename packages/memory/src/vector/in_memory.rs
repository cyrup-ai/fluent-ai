//! In-memory vector store implementation

use std::cmp::Ordering;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use surrealdb::sql::Value;

use super::vector_store::{VectorSearchResult, VectorStore};
use crate::constants::ERROR_VECTOR_NOT_FOUND;
use crate::utils::error::{Error, Result};

/// In-memory vector store implementation
pub struct InMemoryVectorStore {
    vectors: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, HashMap<String, Value>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

impl VectorStore for InMemoryVectorStore {
    fn add_vector(
        &mut self,
        id: &str,
        vector: Vec<f32>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        self.vectors.insert(id.to_string(), vector);
        Box::pin(async { Ok(()) })
    }

    fn get_vector(
        &self,
        id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<(Vec<f32>, Option<HashMap<String, Value>>)>> + Send>>
    {
        let result = if let Some(vector) = self.vectors.get(id) {
            let metadata = self.metadata.get(id).cloned();
            Ok((vector.clone(), metadata))
        } else {
            Err(Error::NotFound(ERROR_VECTOR_NOT_FOUND.to_string()))
        };
        Box::pin(async move { result })
    }

    fn update_vector(
        &mut self,
        id: &str,
        vector: Vec<f32>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        if self.vectors.contains_key(id) {
            self.vectors.insert(id.to_string(), vector);
            Box::pin(async { Ok(()) })
        } else {
            Box::pin(async move { Err(Error::NotFound(ERROR_VECTOR_NOT_FOUND.to_string())) })
        }
    }

    fn delete_vector(&mut self, id: &str) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        self.vectors.remove(id);
        self.metadata.remove(id);
        Box::pin(async { Ok(()) })
    }

    fn search_similar(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send>> {
        let mut results: Vec<(String, f32)> = Vec::new();

        // Simple cosine similarity search
        for (id, vector) in &self.vectors {
            let similarity = cosine_similarity(query_vector, vector);
            results.push((id.clone(), similarity));
        }

        // Sort by similarity (descending) - custom comparison to handle NaN values
        results.sort_by(|a, b| {
            match (a.1.is_nan(), b.1.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater, // NaN goes to end
                (false, true) => Ordering::Less,    // NaN goes to end
                (false, false) => {
                    // Safe to compare non-NaN values (descending order)
                    if b.1 > a.1 {
                        Ordering::Greater
                    } else if b.1 < a.1 {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    }
                }
            }
        });

        // Take top k results
        let top_k: Vec<String> = results.into_iter().take(limit).map(|(id, _)| id).collect();

        Box::pin(async move { Ok(top_k) })
    }

    fn search(
        &self,
        query_vector: &[f32],
        limit: Option<usize>,
        filters: Option<HashMap<String, Value>>,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<Vec<(String, Vec<f32>, f32, Option<HashMap<String, Value>>)>>,
                > + Send,
        >,
    > {
        let mut results: Vec<VectorSearchResult> = Vec::new();

        // Simple cosine similarity search
        for (id, vector) in &self.vectors {
            // Apply filters if any
            if let Some(ref filters) = filters {
                if let Some(metadata) = self.metadata.get(id) {
                    let mut matches = true;
                    for (key, value) in filters {
                        if metadata.get(key) != Some(value) {
                            matches = false;
                            break;
                        }
                    }
                    if !matches {
                        continue;
                    }
                } else {
                    continue; // No metadata, skip if filters are present
                }
            }

            let similarity = cosine_similarity(query_vector, vector);
            let metadata = self.metadata.get(id).cloned();
            results.push((id.clone(), vector.clone(), similarity, metadata));
        }

        // Sort by similarity (descending) - custom comparison to handle NaN values
        results.sort_by(|a, b| {
            match (a.2.is_nan(), b.2.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater, // NaN goes to end
                (false, true) => Ordering::Less,    // NaN goes to end
                (false, false) => {
                    // Safe to compare non-NaN values (descending order)
                    if b.2 > a.2 {
                        Ordering::Greater
                    } else if b.2 < a.2 {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    }
                }
            }
        });

        // Take top k results
        if let Some(limit) = limit {
            results.truncate(limit);
        }

        Box::pin(async move { Ok(results) })
    }

    fn count(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send>> {
        let count = self.vectors.len();
        Box::pin(async move { Ok(count) })
    }

    fn clear(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
        // Note: This requires mutable access, but the trait signature doesn't allow it
        // For now, we return an error
        Box::pin(async {
            Err(Error::Other(
                "Clear operation requires mutable access".to_string(),
            ))
        })
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
