use std::fs;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use cylo::{
    cli::Cli,
    error::ExecError,
    jail::{self, JailConfig},
    repo::{init_repository, RepoConfig},
    state::{ExecutionFlow, PipelineEvent, State},
    watcher::watch_directory,
};
use tracing::{error, info, warn, Level};

fn main() -> Result<(), ExecError> {
    let cli = Cli::parse();

    // Setup logging
    let level = if cli.is_debug() {
        Level::DEBUG
    } else {
        Level::INFO
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    info!("Starting secure code execution environment");

    // Initialize repository with git filters
    let repo_config = RepoConfig {
        path: PathBuf::from("."),
        init_git: true,
        setup_filters: true,
    };

    if let Err(e) = init_repository(&repo_config) {
        warn!("Failed to initialize repository: {:?}", e);
        info!("Continuing without repository initialization");
    } else {
        info!("Repository initialized successfully");
    }

    info!(
        "Running in Linux-only mode with native ramdisk isolation (will prompt for sudo if needed)"
    );

    // Initialize the file system jail
    let watch_path = PathBuf::from("./watched_dir");
    let jail_config = JailConfig {
        allowed_dir: watch_path.clone(),
        enable_landlock: cli.is_landlock_enabled(),
        check_apparmor: true,
    };

    if let Err(e) = jail::init_jail(&jail_config) {
        error!("Failed to initialize file system jail: {:?}", e);
        info!("Continuing with ramdisk-only isolation");
    }

    let (tx, rx) = mpsc::channel::<PipelineEvent>();
    let flow = Arc::new(Mutex::new(ExecutionFlow::default()));

    // Spawn a file watcher thread
    {
        let tx_clone = tx.clone();

        // Create the watched directory if it doesn't exist
        if !watch_path.exists() {
            fs::create_dir_all(&watch_path).map_err(|e| {
                ExecError::RuntimeError(format!("Failed to create watched directory: {}", e))
            })?;
        }

        thread::spawn(move || {
            if let Err(e) = watch_directory(watch_path, tx_clone) {
                error!("File watcher encountered an error: {:?}", e);
            }
        });
    }

    // Send the code execution request
    if let Some(exec_args) = cli.get_exec_args() {
        tx.send(PipelineEvent::ExecuteCode {
            language: exec_args.lang().to_string(),
            code: exec_args.code().to_string(),
        })
        .map_err(|e| ExecError::RuntimeError(format!("Failed to send execution request: {}", e)))?;
    } else {
        return Err(ExecError::InvalidCode("No code execution requested".into()));
    }

    // Main event loop: handle events from the file watcher or other triggers
    loop {
        let mut flow_locked = flow.lock().unwrap();
        let current_state = flow_locked.state();

        // Check if execution is already completed
        if let State::Failed | State::Done = current_state {
            info!("Execution completed with state: {:?}", current_state);
            break;
        }

        // In Processing state, we need to regularly check for task outcomes
        // even if no explicit events arrive
        if current_state == State::Processing {
            // Send a dummy event to trigger outcome checking
            flow_locked.handle(&PipelineEvent::StepSuccess);
            drop(flow_locked);

            // Don't wait too long before checking again
            thread::sleep(Duration::from_millis(100));
            continue;
        }
        drop(flow_locked);

        // Process external events from watchers or step completions
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(evt) => {
                let mut flow_locked = flow.lock().unwrap();
                flow_locked.handle(&evt);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // On timeout, just loop and check state again
            }
            Err(e) => {
                error!("Channel closed or error: {:?}", e);
                break;
            }
        }
    }

    // Return error if execution failed
    let final_state = flow.lock().unwrap().state();
    match final_state {
        State::Failed => Err(ExecError::RuntimeError("Execution failed".into())),
        _ => Ok(()),
    }
}
