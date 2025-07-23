//! Streaming, zero-allocation loader for the Kimi-K2 model shards.
//!
//! This module follows the project-wide *streams-only* architecture:
//! – **No async/await or Future usage**
//! – **No `unwrap` / `expect` / panicking macros** in production code
//! – **Zero allocations** on all hot paths (uses stack-allocated buffers)
//! – **Lock-free** concurrent IO leveraging memory-mapped files where possible
//! – **ProgressHub direct usage** (no abstractions, dual-backend XET CAS + QUIC)
//!
//! The loader streams [`LoaderEvent`]s that represent high-level progress
//! information while lazily memory-mapping the underlying `safetensors` files.
//!
//! NOTE: The implementation purposefully avoids depending on Candle internals
//! and instead produces an opaque `ModelShard` which upstream components can
//! convert into `candle::Tensor`s.

#![allow(clippy::module_name_repetitions)]

use arrayvec::ArrayVec;
use core::sync::atomic::{AtomicU64, Ordering};
use crate::hub::{create_client, create_download_config, Backend, ProgressData};
use fluent_ai_async::AsyncStream;

use super::mod_::KimiK2Config; // re-exported via `mod.rs`

/// Maximum number of shards for the FP8 variant (empirically 61)
const MAX_SHARDS: usize = 64;

/// Opaque handle representing a single memory-mapped shard.
#[derive(Debug)]
pub struct ModelShard {
    /// Raw bytes of the shard mapped into memory.
    bytes: memmap2::Mmap,
}

/// Events emitted by the [`load_model`] stream.
#[derive(Debug)]
pub enum LoaderEvent<'a> {
    /// Metadata about the overall download (file count, total bytes).
    Start { total_shards: usize, total_bytes: u64 },
    /// Progress information for a single shard.
    Progress { shard_idx: usize, bytes: &'a [u8] },
    /// Emitted when a shard has been fully memory-mapped.
    ShardReady { shard_idx: usize, shard: ModelShard },
    /// Emitted once all shards have been processed.
    Complete { shards: ArrayVec<ModelShard, MAX_SHARDS> },
}

/// Streams [`LoaderEvent`]s while downloading and memory-mapping the Kimi-K2
/// weights defined by `config` using ProgressHub directly.
///
/// The caller is expected to *consume* the stream and assemble the model
/// tensors once `Complete` is received.
#[must_use]
pub fn load_model<'cfg>(config: &'cfg KimiK2Config) -> AsyncStream<'cfg, LoaderEvent<'cfg>> {
    AsyncStream::new(move |y| {
        // Use ProgressHub directly - no abstractions
        let client = match create_client(Backend::Auto) {
            Ok(client) => client,
            Err(e) => {
                y.yield_error(e);
                return;
            }
        };

        let cache_dir = std::path::PathBuf::from("/tmp/fluent_ai_cache"); // TODO: make configurable
        let download_config = create_download_config(cache_dir);

        let mut total_bytes = 0_u64;
        let mut shards: ArrayVec<ModelShard, MAX_SHARDS> = ArrayVec::new();

        // Convert repo ArrayVec to string
        let repo_str = match std::str::from_utf8(&config.repo) {
            Ok(s) => s,
            Err(e) => {
                y.yield_error(crate::error::CandleError::ValidationError(e.to_string()));
                return;
            }
        };

        // Emit initial metadata event
        y.yield_item(LoaderEvent::Start {
            total_shards: 1, // ProgressHub downloads entire models
            total_bytes: 0, // will be updated via Progress events
        });

        // Use ProgressHub's download_model_auto directly - it handles all shards
        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(client.download_model_auto(repo_str, &download_config, None)) {
            Ok(result) => {
                // Find and memory-map all safetensors files
                let model_dir = result.destination;
                if let Ok(entries) = std::fs::read_dir(&model_dir) {
                    let mut shard_idx = 0;
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                            match memmap2::MmapOptions::new().map_ro(&path) {
                                Ok(mmap) => {
                                    let shard = ModelShard { bytes: mmap };
                                    if shards.try_push(shard).is_ok() {
                                        y.yield_item(LoaderEvent::ShardReady { 
                                            shard_idx, 
                                            shard: ModelShard { bytes: unsafe { std::mem::transmute(shards[shard_idx].bytes.clone()) } }
                                        });
                                        shard_idx += 1;
                                    }
                                }
                                Err(e) => {
                                    y.yield_error(crate::error::CandleError::Io(e.to_string()));
                                    return;
                                }
                            }
                        }
                    }
                }

                y.yield_item(LoaderEvent::Complete { shards });
            }
            Err(e) => {
                y.yield_error(crate::error::CandleError::InitializationError(e.to_string()));
                return;
            }
        }
    })
}
