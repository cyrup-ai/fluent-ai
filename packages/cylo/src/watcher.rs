use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;
use tracing::{error, info};

use crate::error::StorageError;
use crate::state::PipelineEvent;

// A simplified file watcher implementation without using watchexec
pub fn watch_directory(path: PathBuf, tx: mpsc::Sender<PipelineEvent>) -> Result<(), StorageError> {
    info!("Starting file watcher for directory: {}", path.display());

    // Clone the path and sender for the handler
    let path_to_watch = path.clone();
    let tx_clone = tx.clone();

    // Start the watcher in a background thread
    thread::spawn(move || {
        info!(
            "File watcher thread started for {}",
            path_to_watch.display()
        );

        // Create tokio runtime for future compatibility with watchexec
        match Runtime::new() {
            Ok(_rt) => {
                // Simple polling implementation that periodically checks for changes
                // This is a temporary solution until we properly integrate watchexec

                info!("Using simplified polling watcher (watchexec integration pending)");

                // Send initial notification
                if let Err(e) = tx_clone.send(PipelineEvent::FileChanged(path_to_watch.clone())) {
                    error!("Failed to send initial file event: {}", e);
                }

                // Poll for changes every 5 seconds
                // In a real implementation, we would use watchexec to get actual file events
                let mut previous_time = std::fs::metadata(&path_to_watch)
                    .ok()
                    .and_then(|m| m.modified().ok());

                loop {
                    // Sleep for a while before checking again
                    thread::sleep(Duration::from_secs(5));

                    // Check if the directory has been modified
                    if let Ok(metadata) = std::fs::metadata(&path_to_watch) {
                        if let Ok(modified_time) = metadata.modified() {
                            if let Some(prev) = previous_time {
                                if modified_time > prev {
                                    info!("Directory changed: {}", path_to_watch.display());

                                    // Send notification about the change
                                    if let Err(e) = tx_clone
                                        .send(PipelineEvent::FileChanged(path_to_watch.clone()))
                                    {
                                        error!("Failed to send file change event: {}", e);
                                        break;
                                    }

                                    // Update previous time
                                    previous_time = Some(modified_time);
                                }
                            } else {
                                previous_time = Some(modified_time);
                            }
                        }
                    }
                }
            }
            Err(e) => error!("Failed to create runtime: {}", e)}

        info!("File watcher thread exited");
    });

    info!("File watcher initialized successfully");
    Ok(())
}
