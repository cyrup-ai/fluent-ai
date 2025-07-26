//! Kimi-K2 model integration layer
//!
//! Zero-allocation configuration types and public re-exports for the
//! `moonshotai/Kimi-K2-Instruct` checkpoints.
//! Heavy logic (loader, tokenizer) resides in sub-modules.

#![allow(clippy::module_inception)]

use arrayvec::ArrayVec;

pub mod adapter;
pub mod integration;
pub mod loader;
pub mod model;
pub mod tokenizer;

/// Quantisation formats supported by Kimi-K2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// Full-precision FP16 weights (≈ 40 GB VRAM).
    Fp16,
    /// NVIDIA FP8 quantised weights (≈ 13 GB VRAM).
    Fp8}

/// Static configuration describing a concrete Kimi-K2 variant.
#[derive(Debug, Clone)]
pub struct KimiK2Config {
    /// Hugging Face repository id.
    pub repo: ArrayVec<u8, 128>,
    /// Safetensors shard index filename.
    pub index_file: ArrayVec<u8, 128>,
    /// Selected quantisation format.
    pub quant: QuantFormat,
    /// Maximum sequence length supported.
    pub max_seq_len: u32}

impl KimiK2Config {
    /// Construct a new configuration.
    #[inline]
    pub fn new(repo: &str, quant: QuantFormat) -> Self {
        let mut repo_buf: ArrayVec<u8, 128> = ArrayVec::new();
        let _ = repo_buf.try_extend_from_slice(repo.as_bytes());
        let mut index_buf: ArrayVec<u8, 128> = ArrayVec::new();
        let _ = index_buf.try_extend_from_slice(b"model.safetensors.index.json");
        Self {
            repo: repo_buf,
            index_file: index_buf,
            quant,
            max_seq_len: 32_768}
    }
}

/// Static byte arrays for configuration data
static KIMI_K2_REPO_BYTES: [u8; 128] = {
    let mut arr = [0u8; 128];
    let repo = b"moonshotai/Kimi-K2-Instruct";
    let mut i = 0;
    while i < repo.len() {
        arr[i] = repo[i];
        i += 1;
    }
    arr
};

static KIMI_K2_INDEX_BYTES: [u8; 128] = {
    let mut arr = [0u8; 128];
    let index = b"model.safetensors.index.json";
    let mut i = 0;
    while i < index.len() {
        arr[i] = index[i];
        i += 1;
    }
    arr
};

/// Get pre-defined FP16 configuration.
pub fn kimi_k2_fp16() -> KimiK2Config {
    KimiK2Config {
        repo: ArrayVec::from(KIMI_K2_REPO_BYTES),
        index_file: ArrayVec::from(KIMI_K2_INDEX_BYTES),
        quant: QuantFormat::Fp16,
        max_seq_len: 32_768}
}

/// Get pre-defined FP8 configuration.
pub fn kimi_k2_fp8() -> KimiK2Config {
    KimiK2Config {
        repo: ArrayVec::from(KIMI_K2_REPO_BYTES),
        index_file: ArrayVec::from(KIMI_K2_INDEX_BYTES),
        quant: QuantFormat::Fp8,
        max_seq_len: 32_768}
}
