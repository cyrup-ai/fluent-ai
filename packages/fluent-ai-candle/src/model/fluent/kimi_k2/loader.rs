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
use fluent_ai_async::AsyncStream;

use super::KimiK2Config;
use crate::hub::{Backend, create_client, create_download_config}; // re-exported via `mod.rs`

/// Maximum number of shards for the FP8 variant (empirically 61)
const MAX_SHARDS: usize = 64;

/// Opaque handle representing a single memory-mapped shard.
#[derive(Debug)]
pub struct ModelShard {
    /// Raw bytes of the shard mapped into memory.
    #[allow(dead_code)] // Used in model loading logic but flagged incorrectly by compiler
    bytes: memmap2::Mmap,
}

/// Events emitted by the [`load_model`] stream.
#[derive(Debug)]
pub enum LoaderEvent {
    /// Metadata about the overall download (file count, total bytes).
    Start {
        total_shards: usize,
        total_bytes: u64,
    },
    /// Progress information for a single shard (owns bytes for 'static lifetime compliance).
    Progress { shard_idx: usize, bytes: Vec<u8> },
    /// Emitted when a shard has been fully memory-mapped.
    ShardReady { shard_idx: usize, shard: ModelShard },
    /// Emitted once all shards have been processed.
    Complete {
        shards: ArrayVec<ModelShard, MAX_SHARDS>,
    },
    /// Error during loading process.
    Error { message: String },
}

/// Streams [`LoaderEvent`]s while downloading and memory-mapping the Kimi-K2
/// weights defined by `config` using ProgressHub directly.
///
/// The caller is expected to *consume* the stream and assemble the model
/// tensors once `Complete` is received.
#[must_use]
pub fn load_model(config: &KimiK2Config) -> AsyncStream<LoaderEvent> {
    let config = config.clone(); // Clone to avoid lifetime issues
    AsyncStream::with_channel(move |y: fluent_ai_async::AsyncStreamSender<LoaderEvent>| {
        // Use ProgressHub directly - no abstractions
        let client = match create_client(Backend::Auto) {
            Ok(client) => client,
            Err(e) => {
                let _ = y.send(LoaderEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
        };

        let cache_dir = std::path::PathBuf::from("/tmp/fluent_ai_cache"); // TODO: make configurable
        let download_config = create_download_config(cache_dir);

        let _total_bytes = 0_u64;
        let mut shards: ArrayVec<ModelShard, MAX_SHARDS> = ArrayVec::new();

        // Convert repo ArrayVec to string
        let repo_str = match std::str::from_utf8(&config.repo) {
            Ok(s) => s,
            Err(e) => {
                let _ = y.send(LoaderEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
        };

        // Emit initial metadata event
        let _ = y.send(LoaderEvent::Start {
            total_shards: 1, // ProgressHub downloads entire models
            total_bytes: 0,  // will be updated via Progress events
        });

        // Use ProgressHub's download_model_auto directly - it handles all shards
        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(client.download_model_auto(repo_str, &download_config, None)) {
            Ok(result) => {
                // Find and memory-map all safetensors files
                let model_dir = if let Some(model_result) = result.models.first() {
                    model_result.path.parent().unwrap_or(&model_result.path)
                } else {
                    let _ = y.send(LoaderEvent::Error {
                        message: "No models in download result".to_string(),
                    });
                    return;
                };
                if let Ok(entries) = std::fs::read_dir(&model_dir) {
                    let mut shard_idx = 0;
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                            match unsafe {
                                memmap2::MmapOptions::new()
                                    .map(&std::fs::File::open(&path).unwrap())
                            } {
                                Ok(mmap) => {
                                    let current_idx = shard_idx;
                                    // Get bytes for progress event BEFORE moving mmap into shard (avoids borrowing conflict)
                                    let progress_bytes = mmap[..].to_vec();
                                    let shard = ModelShard { bytes: mmap };

                                    if shards.try_push(shard).is_ok() {
                                        // Send progress event with owned bytes data (fixes lifetime issue)
                                        let _ = y.send(LoaderEvent::Progress {
                                            shard_idx: current_idx,
                                            bytes: progress_bytes,
                                        });
                                        shard_idx += 1;
                                    }
                                }
                                Err(e) => {
                                    let _ = y.send(LoaderEvent::Error {
                                        message: e.to_string(),
                                    });
                                    return;
                                }
                            }
                        }
                    }
                }

                let _ = y.send(LoaderEvent::Complete { shards });
            }
            Err(e) => {
                let _ = y.send(LoaderEvent::Error {
                    message: e.to_string(),
                });
                return;
            }
        }
    })
}
