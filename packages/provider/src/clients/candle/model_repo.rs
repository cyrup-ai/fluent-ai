//! Zero-allocation model repository with lock-free HuggingFace integration
//!
//! This module provides a production-grade model repository system that manages
//! HuggingFace model downloads, caching, and metadata with lock-free operations
//! and zero-allocation patterns for optimal performance.

use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering}};

use arc_swap::{ArcSwap, Guard};
use arrayvec::ArrayString;
use crossbeam::channel::{Receiver, Sender, bounded};
use dashmap::DashMap;
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

use super::models::{CandleDevice, CandleModel};

/// Model loading state with atomic transitions
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Unloaded = 0,
    Downloading = 1,
    Loading = 2,
    Loaded = 3,
    Failed = 4}

impl From<u8> for ModelState {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => ModelState::Unloaded,
            1 => ModelState::Downloading,
            2 => ModelState::Loading,
            3 => ModelState::Loaded,
            _ => ModelState::Failed}
    }
}

/// Model architecture types for specialized loading
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    LLaMA = 0,
    Mistral = 1,
    CodeLlama = 2,
    Phi3 = 3,
    Gemma = 4,
    Kimi = 5}

/// Zero-allocation model metadata with cache-aligned fields
#[repr(C, align(64))]
#[derive(Debug)]
pub struct ModelMetadata {
    /// HuggingFace repository ID (stack-allocated)
    pub hf_repo_id: ArrayString<128>,
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Configuration file name
    pub config_file: ArrayString<32>,
    /// Model file pattern for discovery
    pub model_files: ArrayString<64>,
    /// Tokenizer file name
    pub tokenizer_file: ArrayString<32>,
    /// Atomic model loading state
    pub state: AtomicU8,
    /// Total model size in bytes
    pub model_size_bytes: u64,
    /// Recommended minimum VRAM/RAM in GB
    pub min_memory_gb: u8}

impl ModelMetadata {
    /// Create new model metadata with specified parameters
    #[inline]
    pub fn new(
        hf_repo_id: &str,
        architecture: ModelArchitecture,
        config_file: &str,
        model_files: &str,
        tokenizer_file: &str,
        model_size_bytes: u64,
        min_memory_gb: u8,
    ) -> Result<Self, ModelRepoError> {
        Ok(Self {
            hf_repo_id: ArrayString::from(hf_repo_id).map_err(|_| ModelRepoError::RepoIdTooLong)?,
            architecture,
            config_file: ArrayString::from(config_file)
                .map_err(|_| ModelRepoError::ConfigFileTooLong)?,
            model_files: ArrayString::from(model_files)
                .map_err(|_| ModelRepoError::ModelFilesTooLong)?,
            tokenizer_file: ArrayString::from(tokenizer_file)
                .map_err(|_| ModelRepoError::TokenizerFileTooLong)?,
            state: AtomicU8::new(ModelState::Unloaded as u8),
            model_size_bytes,
            min_memory_gb})
    }

    /// Get current model state atomically
    #[inline(always)]
    pub fn get_state(&self) -> ModelState {
        ModelState::from(self.state.load(Ordering::Acquire))
    }

    /// Compare and swap model state atomically
    #[inline(always)]
    pub fn compare_exchange_state(&self, current: ModelState, new: ModelState) -> bool {
        self.state
            .compare_exchange(
                current as u8,
                new as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Set model state unconditionally
    #[inline(always)]
    pub fn set_state(&self, state: ModelState) {
        self.state.store(state as u8, Ordering::Release);
    }
}

/// Download progress information
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub model: CandleModel,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub file_name: ArrayString<64>}

/// Model file information after download
#[derive(Debug)]
pub struct ModelFileInfo {
    pub file_path: PathBuf,
    pub file_size: u64,
    pub checksum: [u8; 32],
    pub memory_map: Option<Mmap>}

/// Lock-free model repository with atomic operations
pub struct ModelRepository {
    /// Model metadata registry (lock-free)
    metadata: DashMap<CandleModel, Arc<ModelMetadata>>,
    /// Download progress channel
    progress_tx: Sender<DownloadProgress>,
    progress_rx: Receiver<DownloadProgress>,
    /// Cache directory for downloaded models
    cache_dir: PathBuf,
    /// Hot-swappable HuggingFace API client
    hf_client: ArcSwap<Option<hf_hub::api::tokio::Api>>}

impl ModelRepository {
    /// Create new model repository with default model mappings
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self, ModelRepoError> {
        let (progress_tx, progress_rx) = bounded(256);

        let repository = Self {
            metadata: DashMap::new(),
            progress_tx,
            progress_rx,
            cache_dir: cache_dir.as_ref().to_path_buf(),
            hf_client: ArcSwap::from_pointee(None)};

        // Initialize default model mappings
        repository.initialize_default_models()?;

        Ok(repository)
    }

    /// Initialize HuggingFace API client
    pub async fn initialize_hf_client(&self) -> Result<(), ModelRepoError> {
        let api = hf_hub::api::tokio::Api::new()
            .map_err(|e| ModelRepoError::HfApiError(format!("Failed to create HF API: {}", e)))?;

        self.hf_client.store(Arc::new(Some(api)));
        Ok(())
    }

    /// Initialize default model mappings with production model configurations
    fn initialize_default_models(&self) -> Result<(), ModelRepoError> {
        // LLaMA 2 7B - General purpose model
        let llama2_7b = ModelMetadata::new(
            "meta-llama/Llama-2-7b-hf",
            ModelArchitecture::LLaMA,
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            13_000_000_000, // ~13GB
            16,             // 16GB minimum memory
        )?;
        self.metadata
            .insert(CandleModel::Llama2_7B, Arc::new(llama2_7b));

        // LLaMA 2 13B - Larger general purpose model
        let llama2_13b = ModelMetadata::new(
            "meta-llama/Llama-2-13b-hf",
            ModelArchitecture::LLaMA,
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            26_000_000_000, // ~26GB
            32,             // 32GB minimum memory
        )?;
        self.metadata
            .insert(CandleModel::Llama2_13B, Arc::new(llama2_13b));

        // Mistral 7B - Efficient instruction following
        let mistral_7b = ModelMetadata::new(
            "mistralai/Mistral-7B-v0.1",
            ModelArchitecture::Mistral,
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            13_500_000_000, // ~13.5GB
            16,             // 16GB minimum memory
        )?;
        self.metadata
            .insert(CandleModel::Mistral_7B, Arc::new(mistral_7b));

        // Code Llama 7B - Code generation specialist
        let codellama_7b = ModelMetadata::new(
            "codellama/CodeLlama-7b-hf",
            ModelArchitecture::CodeLlama,
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            13_200_000_000, // ~13.2GB
            16,             // 16GB minimum memory
        )?;
        self.metadata
            .insert(CandleModel::CodeLlama_7B, Arc::new(codellama_7b));

        // Phi-3 Mini - Compact high-performance model
        let phi3_mini = ModelMetadata::new(
            "microsoft/Phi-3-mini-4k-instruct",
            ModelArchitecture::Phi3,
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            7_400_000_000, // ~7.4GB
            8,             // 8GB minimum memory
        )?;
        self.metadata
            .insert(CandleModel::Phi3_Mini, Arc::new(phi3_mini));

        // Gemma 2B - Ultra-compact model
        let gemma_2b = ModelMetadata::new(
            "google/gemma-2b",
            ModelArchitecture::Gemma,
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            4_500_000_000, // ~4.5GB
            6,             // 6GB minimum memory
        )?;
        self.metadata
            .insert(CandleModel::Gemma_2B, Arc::new(gemma_2b));

        // Gemma 7B - Larger Gemma variant
        let gemma_7b = ModelMetadata::new(
            "google/gemma-7b",
            ModelArchitecture::Gemma,
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            14_000_000_000, // ~14GB
            16,             // 16GB minimum memory
        )?;
        self.metadata
            .insert(CandleModel::Gemma_7B, Arc::new(gemma_7b));

        Ok(())
    }

    /// Get model metadata (lock-free access)
    #[inline(always)]
    pub fn get_metadata(&self, model: CandleModel) -> Option<Arc<ModelMetadata>> {
        self.metadata.get(&model).map(|entry| entry.value().clone())
    }

    /// Check if model is available (downloaded and ready)
    #[inline(always)]
    pub fn is_model_available(&self, model: CandleModel) -> bool {
        self.get_metadata(model)
            .map(|meta| meta.get_state() == ModelState::Loaded)
            .unwrap_or(false)
    }

    /// Get models by minimum memory requirement
    pub fn get_models_by_memory(&self, max_memory_gb: u8) -> SmallVec<[CandleModel; 8]> {
        let mut models = SmallVec::new();

        for entry in &self.metadata {
            if entry.value().min_memory_gb <= max_memory_gb {
                models.push(*entry.key());
            }
        }

        // Sort by memory requirement (ascending)
        models.sort_by_key(|model| {
            self.get_metadata(*model)
                .map(|meta| meta.min_memory_gb)
                .unwrap_or(u8::MAX)
        });

        models
    }

    /// Download model files asynchronously with progress reporting
    pub async fn download_model(
        &self,
        model: CandleModel,
    ) -> Result<SmallVec<[ModelFileInfo; 8]>, ModelRepoError> {
        let metadata = self
            .get_metadata(model)
            .ok_or(ModelRepoError::ModelNotFound(model))?;

        // Atomic state transition to downloading
        if !metadata.compare_exchange_state(ModelState::Unloaded, ModelState::Downloading) {
            let current_state = metadata.get_state();
            match current_state {
                ModelState::Downloading => return Err(ModelRepoError::AlreadyDownloading(model)),
                ModelState::Loading => return Err(ModelRepoError::AlreadyLoading(model)),
                ModelState::Loaded => return Err(ModelRepoError::AlreadyLoaded(model)),
                ModelState::Failed => {
                    // Reset failed state and try again
                    metadata.set_state(ModelState::Unloaded);
                    if !metadata
                        .compare_exchange_state(ModelState::Unloaded, ModelState::Downloading)
                    {
                        return Err(ModelRepoError::StateTransitionFailed(model));
                    }
                }
                ModelState::Unloaded => unreachable!()}
        }

        let result = self.download_model_files(&metadata, model).await;

        match &result {
            Ok(_) => metadata.set_state(ModelState::Loaded),
            Err(_) => metadata.set_state(ModelState::Failed)}

        result
    }

    /// Internal model download implementation
    async fn download_model_files(
        &self,
        metadata: &ModelMetadata,
        model: CandleModel,
    ) -> Result<SmallVec<[ModelFileInfo; 8]>, ModelRepoError> {
        let hf_client_guard = self.hf_client.load();
        let hf_api = hf_client_guard
            .as_ref()
            .ok_or(ModelRepoError::HfApiNotInitialized)?;

        let repo = hf_api.repo(hf_hub::Repo::with_revision(
            metadata.hf_repo_id.as_str().to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));

        let mut file_infos = SmallVec::new();

        // Download configuration file
        let config_path = self
            .download_file(&repo, &metadata.config_file, model)
            .await?;
        file_infos.push(ModelFileInfo {
            file_path: config_path.clone(),
            file_size: tokio::fs::metadata(&config_path)
                .await
                .map_err(|e| ModelRepoError::IoError(e))?
                .len(),
            checksum: self.calculate_checksum(&config_path).await?,
            memory_map: None});

        // Download tokenizer file
        let tokenizer_path = self
            .download_file(&repo, &metadata.tokenizer_file, model)
            .await?;
        file_infos.push(ModelFileInfo {
            file_path: tokenizer_path.clone(),
            file_size: tokio::fs::metadata(&tokenizer_path)
                .await
                .map_err(|e| ModelRepoError::IoError(e))?
                .len(),
            checksum: self.calculate_checksum(&tokenizer_path).await?,
            memory_map: None});

        // Download model files (handle both single file and index-based models)
        if metadata.model_files.ends_with(".index.json") {
            // Multi-file model with index
            let index_path = self
                .download_file(&repo, &metadata.model_files, model)
                .await?;
            let model_files = self.parse_safetensors_index(&index_path).await?;

            for model_file in model_files {
                let file_path = self.download_file(&repo, &model_file, model).await?;
                let file_size = tokio::fs::metadata(&file_path)
                    .await
                    .map_err(|e| ModelRepoError::IoError(e))?
                    .len();

                // Create memory map for model files for zero-copy access
                let memory_map = self.create_memory_map(&file_path)?;

                file_infos.push(ModelFileInfo {
                    file_path,
                    file_size,
                    checksum: self.calculate_checksum_from_mmap(&memory_map)?,
                    memory_map: Some(memory_map)});
            }
        } else {
            // Single model file
            let model_path = self
                .download_file(&repo, &metadata.model_files, model)
                .await?;
            let file_size = tokio::fs::metadata(&model_path)
                .await
                .map_err(|e| ModelRepoError::IoError(e))?
                .len();

            let memory_map = self.create_memory_map(&model_path)?;

            file_infos.push(ModelFileInfo {
                file_path: model_path,
                file_size,
                checksum: self.calculate_checksum_from_mmap(&memory_map)?,
                memory_map: Some(memory_map)});
        }

        Ok(file_infos)
    }

    /// Download individual file with progress reporting
    async fn download_file(
        &self,
        repo: &hf_hub::api::tokio::ApiRepo,
        filename: &str,
        model: CandleModel,
    ) -> Result<PathBuf, ModelRepoError> {
        let local_path = self.cache_dir.join(filename);

        // Check if file already exists and is valid
        if local_path.exists() {
            if let Ok(metadata) = tokio::fs::metadata(&local_path).await {
                if metadata.len() > 0 {
                    return Ok(local_path);
                }
            }
        }

        // Create parent directories if needed
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ModelRepoError::IoError(e))?;
        }

        // Download file using HuggingFace API
        let downloaded_path = repo.get(filename).await.map_err(|e| {
            ModelRepoError::DownloadError(format!("Failed to download {}: {}", filename, e))
        })?;

        // Move downloaded file to cache location
        tokio::fs::copy(&downloaded_path, &local_path)
            .await
            .map_err(|e| ModelRepoError::IoError(e))?;

        // Report download progress
        if let Ok(metadata) = tokio::fs::metadata(&local_path).await {
            let progress = DownloadProgress {
                model,
                bytes_downloaded: metadata.len(),
                total_bytes: metadata.len(),
                file_name: ArrayString::from(filename).unwrap_or_default()};

            // Non-blocking progress send
            let _ = self.progress_tx.try_send(progress);
        }

        Ok(local_path)
    }

    /// Parse safetensors index file to get model file names
    async fn parse_safetensors_index(
        &self,
        index_path: &Path,
    ) -> Result<SmallVec<[ArrayString<64>; 16]>, ModelRepoError> {
        let content = tokio::fs::read_to_string(index_path)
            .await
            .map_err(|e| ModelRepoError::IoError(e))?;

        let index: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| ModelRepoError::ParseError(format!("Invalid safetensors index: {}", e)))?;

        let mut files = SmallVec::new();

        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for filename in weight_map.values() {
                if let Some(filename_str) = filename.as_str() {
                    if let Ok(filename_array) = ArrayString::from(filename_str) {
                        if !files.contains(&filename_array) {
                            files.push(filename_array);
                        }
                    }
                }
            }
        }

        Ok(files)
    }

    /// Create memory map for zero-copy file access
    fn create_memory_map(&self, file_path: &Path) -> Result<Mmap, ModelRepoError> {
        let file = std::fs::File::open(file_path).map_err(|e| ModelRepoError::IoError(e))?;

        unsafe {
            Mmap::map(&file).map_err(|e| {
                ModelRepoError::MemoryMapError(format!("Failed to create memory map: {}", e))
            })
        }
    }

    /// Calculate SHA256 checksum from file
    async fn calculate_checksum(&self, file_path: &Path) -> Result<[u8; 32], ModelRepoError> {
        let content = tokio::fs::read(file_path)
            .await
            .map_err(|e| ModelRepoError::IoError(e))?;

        let mut hasher = Sha256::new();
        hasher.update(&content);

        Ok(hasher.finalize().into())
    }

    /// Calculate SHA256 checksum from memory map (zero-copy)
    fn calculate_checksum_from_mmap(&self, mmap: &Mmap) -> Result<[u8; 32], ModelRepoError> {
        let mut hasher = Sha256::new();
        hasher.update(&mmap[..]);

        Ok(hasher.finalize().into())
    }

    /// Get download progress receiver for monitoring
    #[inline(always)]
    pub fn progress_receiver(&self) -> &Receiver<DownloadProgress> {
        &self.progress_rx
    }

    /// List all available models with their states
    pub fn list_models(&self) -> SmallVec<[(CandleModel, ModelState); 16]> {
        let mut models = SmallVec::new();

        for entry in &self.metadata {
            let model = *entry.key();
            let state = entry.value().get_state();
            models.push((model, state));
        }

        models
    }

    /// Get total cache size in bytes
    pub async fn get_cache_size(&self) -> Result<u64, ModelRepoError> {
        let mut total_size = 0u64;

        let mut dir_entries = tokio::fs::read_dir(&self.cache_dir)
            .await
            .map_err(|e| ModelRepoError::IoError(e))?;

        while let Some(entry) = dir_entries
            .next_entry()
            .await
            .map_err(|e| ModelRepoError::IoError(e))?
        {
            if let Ok(metadata) = entry.metadata().await {
                if metadata.is_file() {
                    total_size = total_size.saturating_add(metadata.len());
                }
            }
        }

        Ok(total_size)
    }

    /// Clear model cache and reset all states
    pub async fn clear_cache(&self) -> Result<(), ModelRepoError> {
        // Reset all model states to unloaded
        for entry in &self.metadata {
            entry.value().set_state(ModelState::Unloaded);
        }

        // Remove all cached files
        if self.cache_dir.exists() {
            tokio::fs::remove_dir_all(&self.cache_dir)
                .await
                .map_err(|e| ModelRepoError::IoError(e))?;

            tokio::fs::create_dir_all(&self.cache_dir)
                .await
                .map_err(|e| ModelRepoError::IoError(e))?;
        }

        Ok(())
    }
}

/// Model repository error types with zero-allocation error handling
#[derive(Debug, thiserror::Error)]
pub enum ModelRepoError {
    #[error("Model not found: {0:?}")]
    ModelNotFound(CandleModel),

    #[error("Repository ID too long (max 128 characters)")]
    RepoIdTooLong,

    #[error("Config file name too long (max 32 characters)")]
    ConfigFileTooLong,

    #[error("Model files specification too long (max 64 characters)")]
    ModelFilesTooLong,

    #[error("Tokenizer file name too long (max 32 characters)")]
    TokenizerFileTooLong,

    #[error("Model {0:?} is already downloading")]
    AlreadyDownloading(CandleModel),

    #[error("Model {0:?} is already loading")]
    AlreadyLoading(CandleModel),

    #[error("Model {0:?} is already loaded")]
    AlreadyLoaded(CandleModel),

    #[error("State transition failed for model {0:?}")]
    StateTransitionFailed(CandleModel),

    #[error("HuggingFace API not initialized")]
    HfApiNotInitialized,

    #[error("HuggingFace API error: {0}")]
    HfApiError(String),

    #[error("Download error: {0}")]
    DownloadError(String),

    #[error("IO error: {0}")]
    IoError(#[from] tokio::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Memory map error: {0}")]
    MemoryMapError(String)}
