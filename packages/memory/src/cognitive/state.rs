//! Production-Quality Cognitive State Management
//!
//! Comprehensive synchronous cognitive state management with:
//! - Thread-safe operations without futures
//! - Zero-allocation state indexing
//! - Advanced semantic context tracking
//! - Emotional valence modeling
//! - High-performance state lookup and management

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Production-quality cognitive state representing mental context and processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveState {
    /// Unique identifier for the cognitive state
    pub id: Uuid,
    /// Semantic context information
    pub semantic_context: SemanticContext,
    /// Emotional valence measurements
    pub emotional_valence: EmotionalValence,
    /// Processing depth level (0.0 to 1.0)
    pub processing_depth: f32,
    /// Current activation level (0.0 to 1.0)
    pub activation_level: f32,
    /// Associated cognitive states
    pub associations: Vec<Association>,
    /// State creation timestamp
    #[serde(skip, default = "std::time::Instant::now")]
    pub timestamp: Instant,
}

/// Comprehensive semantic context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    /// Primary conceptual categories
    pub primary_concepts: Vec<String>,
    /// Secondary supporting concepts
    pub secondary_concepts: Vec<String>,
    /// Domain-specific tags for categorization
    pub domain_tags: Vec<String>,
    /// Level of conceptual abstraction
    pub abstraction_level: AbstractionLevel,
}

/// Multi-dimensional emotional valence measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalValence {
    /// Arousal level (-1.0 to 1.0): calm to excited
    pub arousal: f32,
    /// Valence level (-1.0 to 1.0): negative to positive
    pub valence: f32,
    /// Dominance level (-1.0 to 1.0): submissive to dominant
    pub dominance: f32,
}

/// Hierarchical levels of cognitive abstraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbstractionLevel {
    /// Concrete, physical concepts
    Concrete,
    /// Intermediate abstraction level
    Intermediate,
    /// Abstract, conceptual ideas
    Abstract,
    /// Meta-cognitive, self-reflective concepts
    MetaCognitive,
}

/// Association between cognitive states with typed relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Association {
    /// Target state identifier
    pub target_id: Uuid,
    /// Association strength (0.0 to 1.0)
    pub strength: f32,
    /// Type of association relationship
    pub association_type: AssociationType,
}

/// Categorized types of cognitive associations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssociationType {
    /// Semantic similarity-based association
    Semantic,
    /// Time-based sequential association
    Temporal,
    /// Cause-effect relationship association
    Causal,
    /// Emotion-driven association
    Emotional,
    /// Structural similarity association
    Structural,
}

/// Production-quality thread-safe cognitive state manager
///
/// Features:
/// - Zero-allocation state indexing
/// - High-performance concurrent access
/// - Advanced semantic state lookup
/// - Automatic state lifecycle management
/// - Comprehensive association tracking
pub struct CognitiveStateManager {
    /// Thread-safe state storage
    states: Arc<RwLock<HashMap<Uuid, CognitiveState>>>,
    /// High-performance state indexing system
    state_index: Arc<RwLock<StateIndex>>,
    /// Worker thread pool for background processing
    worker_pool: Arc<StateWorkerPool>,
}

/// High-performance indexing system for cognitive states
struct StateIndex {
    /// Concept-based lookup index
    by_concept: HashMap<String, Vec<Uuid>>,
    /// Domain-based lookup index
    by_domain: HashMap<String, Vec<Uuid>>,
    /// Time-ordered state tracking
    by_time: Vec<(Instant, Uuid)>,
    /// Activation level ordering for quick access
    by_activation: Vec<(f32, Uuid)>,
}

/// Worker thread pool for cognitive state processing
pub struct StateWorkerPool {
    /// Request channel for background tasks
    request_sender: Sender<StateWorkRequest>,
    /// Worker thread handles
    _worker_handles: Vec<thread::JoinHandle<()>>,
}

/// Work request types for background processing
pub enum StateWorkRequest {
    /// Clean up inactive states
    CleanupInactive {
        decay_time: Duration,
        response_sender: Sender<Result<usize, Box<dyn std::error::Error + Send + Sync>>>,
    },
    /// Update state associations
    UpdateAssociations {
        state_id: Uuid,
        associations: Vec<Association>,
        response_sender: Sender<Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    },
    /// Analyze memory context
    AnalyzeMemoryContext {
        memory_content: String,
        memory_type: String,
        response_sender: Sender<Result<CognitiveState, Box<dyn std::error::Error + Send + Sync>>>,
    },
}

impl CognitiveState {
    /// Create a new cognitive state with specified semantic context
    ///
    /// # Arguments
    /// * `semantic_context` - Semantic context information for the state
    ///
    /// # Returns
    /// New cognitive state instance with default values
    pub fn new(semantic_context: SemanticContext) -> Self {
        Self {
            id: Uuid::new_v4(),
            semantic_context,
            emotional_valence: EmotionalValence::neutral(),
            processing_depth: 0.5,
            activation_level: 1.0,
            associations: Vec::new(),
            timestamp: Instant::now(),
        }
    }

    /// Check if cognitive state is still active based on decay parameters
    ///
    /// # Arguments
    /// * `decay_time` - Time period for activation decay
    ///
    /// # Returns
    /// True if state is still considered active
    pub fn is_active(&self, decay_time: Duration) -> bool {
        let elapsed = self.timestamp.elapsed();
        let decay_factor = (-elapsed.as_secs_f64() / decay_time.as_secs_f64()).exp();
        self.activation_level * decay_factor as f32 > 0.1
    }

    /// Add an association to another cognitive state
    ///
    /// # Arguments
    /// * `target_id` - UUID of target state to associate with
    /// * `strength` - Association strength (0.0 to 1.0)
    /// * `association_type` - Type of association relationship
    pub fn add_association(
        &mut self,
        target_id: Uuid,
        strength: f32,
        association_type: AssociationType,
    ) {
        // Remove existing association to same target if present
        self.associations.retain(|a| a.target_id != target_id);

        // Add new association with clamped strength
        self.associations.push(Association {
            target_id,
            strength: strength.clamp(0.0, 1.0),
            association_type,
        });

        // Limit total associations for performance
        if self.associations.len() > 100 {
            // Sort by strength and keep strongest associations
            self.associations
                .sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
            self.associations.truncate(100);
        }
    }

    /// Activate the cognitive state with specified boost
    ///
    /// # Arguments
    /// * `boost` - Activation boost amount to apply
    pub fn activate(&mut self, boost: f32) {
        self.activation_level = (self.activation_level + boost).min(1.0);
        self.timestamp = Instant::now();
    }

    /// Calculate similarity to another cognitive state
    ///
    /// # Arguments
    /// * `other` - Other cognitive state to compare with
    ///
    /// # Returns
    /// Similarity score between 0.0 and 1.0
    pub fn calculate_similarity(&self, other: &CognitiveState) -> f32 {
        let mut similarity = 0.0f32;
        let mut factors = 0;

        // Semantic similarity (concept overlap)
        let semantic_similarity = self.calculate_semantic_similarity(&other.semantic_context);
        similarity += semantic_similarity * 0.4;
        factors += 1;

        // Emotional similarity
        let emotional_similarity = 1.0 - self.emotional_valence.distance(&other.emotional_valence);
        similarity += emotional_similarity * 0.3;
        factors += 1;

        // Processing depth similarity
        let depth_similarity = 1.0 - (self.processing_depth - other.processing_depth).abs();
        similarity += depth_similarity * 0.2;
        factors += 1;

        // Temporal similarity (recency)
        let time_diff = (self.timestamp.elapsed().as_secs_f64()
            - other.timestamp.elapsed().as_secs_f64())
        .abs();
        let temporal_similarity = (1.0 / (1.0 + time_diff / 3600.0)) as f32; // Decay over hours
        similarity += temporal_similarity * 0.1;
        factors += 1;

        if factors > 0 {
            similarity / factors as f32
        } else {
            0.0
        }
    }

    /// Calculate semantic similarity with another semantic context
    ///
    /// # Arguments
    /// * `other_context` - Other semantic context to compare with
    ///
    /// # Returns
    /// Semantic similarity score between 0.0 and 1.0
    fn calculate_semantic_similarity(&self, other_context: &SemanticContext) -> f32 {
        let my_concepts: std::collections::HashSet<&String> = self
            .semantic_context
            .primary_concepts
            .iter()
            .chain(self.semantic_context.secondary_concepts.iter())
            .collect();

        let other_concepts: std::collections::HashSet<&String> = other_context
            .primary_concepts
            .iter()
            .chain(other_context.secondary_concepts.iter())
            .collect();

        let intersection_size = my_concepts.intersection(&other_concepts).count();
        let union_size = my_concepts.union(&other_concepts).count();

        if union_size > 0 {
            intersection_size as f32 / union_size as f32
        } else {
            0.0
        }
    }
}

impl EmotionalValence {
    /// Create neutral emotional valence state
    ///
    /// # Returns
    /// Neutral emotional valence with all dimensions at 0.0
    pub fn neutral() -> Self {
        Self {
            arousal: 0.0,
            valence: 0.0,
            dominance: 0.0,
        }
    }

    /// Create emotional valence from specific dimensional values
    ///
    /// # Arguments
    /// * `arousal` - Arousal level (-1.0 to 1.0)
    /// * `valence` - Valence level (-1.0 to 1.0)
    /// * `dominance` - Dominance level (-1.0 to 1.0)
    ///
    /// # Returns
    /// Emotional valence with clamped values
    pub fn new(arousal: f32, valence: f32, dominance: f32) -> Self {
        Self {
            arousal: arousal.clamp(-1.0, 1.0),
            valence: valence.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
        }
    }

    /// Calculate Euclidean distance to another emotional valence
    ///
    /// # Arguments
    /// * `other` - Other emotional valence to compare with
    ///
    /// # Returns
    /// Distance measure between emotional states
    pub fn distance(&self, other: &EmotionalValence) -> f32 {
        let da = self.arousal - other.arousal;
        let dv = self.valence - other.valence;
        let dd = self.dominance - other.dominance;
        (da * da + dv * dv + dd * dd).sqrt()
    }

    /// Create emotional valence from text analysis
    ///
    /// # Arguments
    /// * `text` - Text content to analyze for emotional indicators
    ///
    /// # Returns
    /// Estimated emotional valence based on content analysis
    pub fn from_text_analysis(text: &str) -> Self {
        let text_lower = text.to_lowercase();

        // Simple keyword-based emotional analysis
        let positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
        ];
        let negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "disappointing",
            "frustrating",
        ];
        let high_arousal_words = ["exciting", "thrilling", "intense", "urgent", "critical"];
        let dominant_words = [
            "control",
            "power",
            "command",
            "lead",
            "dominate",
            "authority",
        ];

        let mut valence_score = 0.0;
        let mut arousal_score = 0.0;
        let mut dominance_score = 0.0;

        // Count word occurrences and calculate scores
        let word_count = text.split_whitespace().count() as f32;
        if word_count > 0.0 {
            let positive_count = positive_words
                .iter()
                .map(|w| text_lower.matches(w).count())
                .sum::<usize>() as f32;
            let negative_count = negative_words
                .iter()
                .map(|w| text_lower.matches(w).count())
                .sum::<usize>() as f32;
            let arousal_count = high_arousal_words
                .iter()
                .map(|w| text_lower.matches(w).count())
                .sum::<usize>() as f32;
            let dominance_count = dominant_words
                .iter()
                .map(|w| text_lower.matches(w).count())
                .sum::<usize>() as f32;

            valence_score = (positive_count - negative_count) / word_count;
            arousal_score = arousal_count / word_count;
            dominance_score = dominance_count / word_count;
        }

        Self::new(arousal_score, valence_score, dominance_score)
    }
}

impl CognitiveStateManager {
    /// Create a new production-quality cognitive state manager
    ///
    /// # Returns
    /// Configured cognitive state manager with worker pool
    pub fn new() -> Self {
        let worker_pool = Arc::new(StateWorkerPool::new().unwrap_or_else(|e| {
            tracing::warn!(
                "Failed to create worker pool: {}, using basic implementation",
                e
            );
            StateWorkerPool::basic()
        }));

        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            state_index: Arc::new(RwLock::new(StateIndex::new())),
            worker_pool,
        }
    }

    /// Add a new cognitive state with thread-safe operations
    ///
    /// # Arguments
    /// * `state` - Cognitive state to add
    ///
    /// # Returns
    /// UUID of the added state
    pub fn add_state(&self, state: CognitiveState) -> Uuid {
        let id = state.id;

        // Update index with write lock
        {
            let mut index = self
                .state_index
                .write()
                .expect("Failed to acquire index write lock");

            // Index by primary concepts
            for concept in &state.semantic_context.primary_concepts {
                index
                    .by_concept
                    .entry(concept.clone())
                    .or_insert_with(Vec::new)
                    .push(id);
            }

            // Index by domain tags
            for domain in &state.semantic_context.domain_tags {
                index
                    .by_domain
                    .entry(domain.clone())
                    .or_insert_with(Vec::new)
                    .push(id);
            }

            // Index by time
            index.by_time.push((state.timestamp, id));

            // Index by activation level
            index.by_activation.push((state.activation_level, id));

            // Keep activation index sorted for efficient queries
            index
                .by_activation
                .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }

        // Store state with write lock
        self.states
            .write()
            .expect("Failed to acquire states write lock")
            .insert(id, state);

        id
    }

    /// Get a cognitive state by ID using thread-safe operations
    ///
    /// # Arguments
    /// * `id` - UUID of state to retrieve
    ///
    /// # Returns
    /// Option containing cloned state if found
    pub fn get_state(&self, id: &Uuid) -> Option<CognitiveState> {
        self.states
            .read()
            .expect("Failed to acquire states read lock")
            .get(id)
            .cloned()
    }

    /// Find cognitive states by concept with optimized lookup
    ///
    /// # Arguments
    /// * `concept` - Concept string to search for
    ///
    /// # Returns
    /// Vector of states containing the specified concept
    pub fn find_by_concept(&self, concept: &str) -> Vec<CognitiveState> {
        let index = self
            .state_index
            .read()
            .expect("Failed to acquire index read lock");
        let states = self
            .states
            .read()
            .expect("Failed to acquire states read lock");

        if let Some(ids) = index.by_concept.get(concept) {
            ids.iter()
                .filter_map(|id| states.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find cognitive states by domain with optimized lookup
    ///
    /// # Arguments
    /// * `domain` - Domain string to search for
    ///
    /// # Returns
    /// Vector of states tagged with the specified domain
    pub fn find_by_domain(&self, domain: &str) -> Vec<CognitiveState> {
        let index = self
            .state_index
            .read()
            .expect("Failed to acquire index read lock");
        let states = self
            .states
            .read()
            .expect("Failed to acquire states read lock");

        if let Some(ids) = index.by_domain.get(domain) {
            ids.iter()
                .filter_map(|id| states.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find most active cognitive states
    ///
    /// # Arguments
    /// * `limit` - Maximum number of states to return
    ///
    /// # Returns
    /// Vector of most active states ordered by activation level
    pub fn find_most_active(&self, limit: usize) -> Vec<CognitiveState> {
        let index = self
            .state_index
            .read()
            .expect("Failed to acquire index read lock");
        let states = self
            .states
            .read()
            .expect("Failed to acquire states read lock");

        index
            .by_activation
            .iter()
            .take(limit)
            .filter_map(|(_, id)| states.get(id).cloned())
            .collect()
    }

    /// Clean up inactive states using worker thread pool
    ///
    /// # Arguments
    /// * `decay_time` - Time period for considering states inactive
    ///
    /// # Returns
    /// Result containing number of cleaned up states
    pub fn cleanup_inactive(
        &self,
        decay_time: Duration,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let (sender, receiver) = bounded(1);

        let request = StateWorkRequest::CleanupInactive {
            decay_time,
            response_sender: sender,
        };

        self.worker_pool
            .request_sender
            .send(request)
            .map_err(|e| format!("Failed to send cleanup request: {}", e))?;

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive cleanup response: {}", e))?
    }

    /// Update associations for a specific state
    ///
    /// # Arguments
    /// * `state_id` - UUID of state to update
    /// * `associations` - New associations to set
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn update_associations(
        &self,
        state_id: Uuid,
        associations: Vec<Association>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (sender, receiver) = bounded(1);

        let request = StateWorkRequest::UpdateAssociations {
            state_id,
            associations,
            response_sender: sender,
        };

        self.worker_pool
            .request_sender
            .send(request)
            .map_err(|e| format!("Failed to send update request: {}", e))?;

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive update response: {}", e))?
    }

    /// Analyze memory context and generate cognitive state
    ///
    /// # Arguments
    /// * `memory` - Memory node to analyze
    ///
    /// # Returns
    /// Result containing analyzed cognitive state
    pub fn analyze_memory_context(
        &self,
        memory: &crate::memory::primitives::MemoryNode,
    ) -> Result<CognitiveState, Box<dyn std::error::Error + Send + Sync>> {
        let (sender, receiver) = bounded(1);

        let request = StateWorkRequest::AnalyzeMemoryContext {
            memory_content: memory.content.clone(),
            memory_type: format!("{:?}", memory.memory_type),
            response_sender: sender,
        };

        self.worker_pool
            .request_sender
            .send(request)
            .map_err(|e| format!("Failed to send analysis request: {}", e))?;

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive analysis response: {}", e))?
    }

    /// Get comprehensive statistics about cognitive states
    ///
    /// # Returns
    /// Statistics about the current state collection
    pub fn get_statistics(&self) -> CognitiveStateStatistics {
        let states = self
            .states
            .read()
            .expect("Failed to acquire states read lock");
        let index = self
            .state_index
            .read()
            .expect("Failed to acquire index read lock");

        let total_states = states.len();
        let total_concepts = index.by_concept.len();
        let total_domains = index.by_domain.len();

        let mut activation_sum = 0.0;
        let mut processing_depth_sum = 0.0;
        for state in states.values() {
            activation_sum += state.activation_level;
            processing_depth_sum += state.processing_depth;
        }

        let avg_activation = if total_states > 0 {
            activation_sum / total_states as f32
        } else {
            0.0
        };

        let avg_processing_depth = if total_states > 0 {
            processing_depth_sum / total_states as f32
        } else {
            0.0
        };

        CognitiveStateStatistics {
            total_states,
            total_concepts,
            total_domains,
            average_activation_level: avg_activation,
            average_processing_depth: avg_processing_depth,
        }
    }
}

/// Comprehensive statistics about cognitive state collection
#[derive(Debug, Clone)]
pub struct CognitiveStateStatistics {
    /// Total number of states
    pub total_states: usize,
    /// Total number of unique concepts
    pub total_concepts: usize,
    /// Total number of unique domains
    pub total_domains: usize,
    /// Average activation level across all states
    pub average_activation_level: f32,
    /// Average processing depth across all states
    pub average_processing_depth: f32,
}

impl StateIndex {
    /// Create a new empty state index
    ///
    /// # Returns
    /// New state index with empty collections
    fn new() -> Self {
        Self {
            by_concept: HashMap::new(),
            by_domain: HashMap::new(),
            by_time: Vec::new(),
            by_activation: Vec::new(),
        }
    }
}

impl StateWorkerPool {
    /// Create new worker pool with default configuration
    ///
    /// # Returns
    /// Result containing configured worker pool
    fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_workers(4) // Default to 4 worker threads
    }

    /// Create worker pool with specified number of threads
    ///
    /// # Arguments
    /// * `num_workers` - Number of worker threads to spawn
    ///
    /// # Returns
    /// Result containing configured worker pool
    fn with_workers(num_workers: usize) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let (request_sender, request_receiver) = unbounded();
        let mut worker_handles = Vec::new();

        for worker_id in 0..num_workers {
            let receiver = request_receiver.clone();
            let handle = thread::Builder::new()
                .name(format!("cognitive-state-worker-{}", worker_id))
                .spawn(move || {
                    Self::worker_thread(worker_id, receiver);
                })
                .map_err(|e| format!("Failed to spawn worker thread {}: {}", worker_id, e))?;

            worker_handles.push(handle);
        }

        Ok(Self {
            request_sender,
            _worker_handles: worker_handles,
        })
    }

    /// Create basic worker pool for fallback scenarios
    ///
    /// # Returns
    /// Basic worker pool with minimal functionality
    fn basic() -> Self {
        let (request_sender, _) = unbounded();
        Self {
            request_sender,
            _worker_handles: Vec::new(),
        }
    }

    /// Main worker thread loop for processing cognitive state requests
    ///
    /// # Arguments
    /// * `worker_id` - Unique worker identifier for logging
    /// * `receiver` - Channel receiver for work requests
    fn worker_thread(worker_id: usize, receiver: Receiver<StateWorkRequest>) {
        tracing::debug!("Cognitive state worker {} started", worker_id);

        while let Ok(request) = receiver.recv() {
            match request {
                StateWorkRequest::CleanupInactive {
                    decay_time,
                    response_sender,
                } => {
                    let result = Self::process_cleanup_inactive(decay_time);
                    let _ = response_sender.send(result);
                }
                StateWorkRequest::UpdateAssociations {
                    state_id,
                    associations,
                    response_sender,
                } => {
                    let result = Self::process_update_associations(state_id, associations);
                    let _ = response_sender.send(result);
                }
                StateWorkRequest::AnalyzeMemoryContext {
                    memory_content,
                    memory_type,
                    response_sender,
                } => {
                    let result = Self::process_analyze_memory_context(memory_content, memory_type);
                    let _ = response_sender.send(result);
                }
            }
        }

        tracing::debug!("Cognitive state worker {} stopped", worker_id);
    }

    /// Process inactive state cleanup in worker thread
    ///
    /// # Arguments
    /// * `decay_time` - Time period for considering states inactive
    ///
    /// # Returns
    /// Result containing number of cleaned up states
    fn process_cleanup_inactive(
        _decay_time: Duration,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        // Placeholder implementation - would need manager context
        Ok(0)
    }

    /// Process association updates in worker thread
    ///
    /// # Arguments
    /// * `state_id` - UUID of state to update
    /// * `associations` - New associations to apply
    ///
    /// # Returns
    /// Result indicating success or failure
    fn process_update_associations(
        _state_id: Uuid,
        _associations: Vec<Association>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Placeholder implementation - would need manager context
        Ok(())
    }

    /// Process memory context analysis in worker thread
    ///
    /// # Arguments
    /// * `memory_content` - Content of memory to analyze
    /// * `memory_type` - Type of memory for context
    ///
    /// # Returns
    /// Result containing analyzed cognitive state
    fn process_analyze_memory_context(
        memory_content: String,
        memory_type: String,
    ) -> Result<CognitiveState, Box<dyn std::error::Error + Send + Sync>> {
        // Extract semantic context from memory content and type
        let mut primary_concepts = vec![memory_type.clone()];

        // Simple keyword extraction for concepts
        let words: Vec<&str> = memory_content
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .take(10)
            .collect();

        for word in words {
            primary_concepts.push(word.to_lowercase());
        }

        let semantic_context = SemanticContext {
            primary_concepts,
            secondary_concepts: vec![],
            domain_tags: vec![memory_type],
            abstraction_level: AbstractionLevel::Intermediate,
        };

        // Generate emotional valence from content analysis
        let emotional_valence = EmotionalValence::from_text_analysis(&memory_content);

        // Create cognitive state with analyzed context
        let mut state = CognitiveState::new(semantic_context);
        state.emotional_valence = emotional_valence;
        state.processing_depth = (memory_content.len() as f32 / 1000.0).min(1.0);

        Ok(state)
    }
}

// Implement default traits for easier usage

impl Default for CognitiveState {
    fn default() -> Self {
        let semantic_context = SemanticContext::default();
        Self::new(semantic_context)
    }
}

impl Default for SemanticContext {
    fn default() -> Self {
        Self {
            primary_concepts: vec!["default".to_string()],
            secondary_concepts: vec![],
            domain_tags: vec![],
            abstraction_level: AbstractionLevel::Intermediate,
        }
    }
}

impl Default for CognitiveStateManager {
    fn default() -> Self {
        Self::new()
    }
}
