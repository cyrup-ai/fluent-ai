//! Firecracker-based secure execution environment

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::{error, info, warn};

use crate::config::RamdiskConfig;
use crate::error::StorageError;

/// Configuration for Firecracker VM
#[derive(Debug, Clone)]
pub struct FirecrackerConfig {
    /// Path to the Firecracker binary
    pub binary_path: PathBuf,
    /// Path to the kernel image
    pub kernel_path: PathBuf,
    /// Path to the root filesystem image
    pub rootfs_path: PathBuf,
    /// Memory size in MB
    pub mem_size_mib: u32,
    /// Number of vCPUs
    pub vcpu_count: u32,
    /// Network configuration (optional)
    pub network_config: Option<NetworkConfig>,
}

/// Network configuration for Firecracker VM
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Host interface name
    pub host_interface: String,
    /// Guest MAC address
    pub guest_mac: String,
    /// IP configuration
    pub ip_config: String,
}

impl Default for FirecrackerConfig {
    fn default() -> Self {
        Self {
            binary_path: PathBuf::from("/usr/bin/firecracker"),
            kernel_path: PathBuf::from("/var/lib/firecracker/vmlinux"),
            rootfs_path: PathBuf::from("/var/lib/firecracker/rootfs.ext4"),
            mem_size_mib: 512,
            vcpu_count: 1,
            network_config: None,
        }
    }
}

/// Firecracker VM manager
pub struct FirecrackerVM {
    config: FirecrackerConfig,
    socket_path: PathBuf,
    api_socket: Option<PathBuf>,
    vm_id: String,
}

impl FirecrackerVM {
    /// Create a new Firecracker VM manager
    pub fn new(config: FirecrackerConfig, vm_id: impl Into<String>) -> Self {
        let vm_id = vm_id.into();
        let socket_path = PathBuf::from(format!("/tmp/firecracker-{}.sock", vm_id));

        Self {
            config,
            socket_path,
            api_socket: None,
            vm_id,
        }
    }

    /// Start the Firecracker VM
    pub fn start(&mut self) -> Result<()> {
        info!("Starting Firecracker VM with ID: {}", self.vm_id);

        // Check if Firecracker binary exists
        if !self.config.binary_path.exists() {
            return Err(anyhow::anyhow!(
                "Firecracker binary not found at {:?}",
                self.config.binary_path
            ));
        }

        // Remove socket file if it exists
        if self.socket_path.exists() {
            fs::remove_file(&self.socket_path)?;
        }

        // Start Firecracker process
        let _cmd = Command::new(&self.config.binary_path)
            .arg("--api-sock")
            .arg(&self.socket_path)
            .arg("--id")
            .arg(&self.vm_id)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to start Firecracker process")?;

        // Wait for socket to be created
        let mut attempts = 0;
        while !self.socket_path.exists() && attempts < 10 {
            thread::sleep(Duration::from_millis(100));
            attempts += 1;
        }

        if !self.socket_path.exists() {
            return Err(anyhow::anyhow!("Firecracker API socket not created"));
        }

        self.api_socket = Some(self.socket_path.clone());

        // Configure the VM
        self.configure_vm()?;

        info!("Firecracker VM started successfully");
        Ok(())
    }

    /// Configure the VM using the API
    fn configure_vm(&self) -> Result<()> {
        // This would normally use the Firecracker API client
        // For now, we'll use curl commands for simplicity

        // Configure boot source
        let boot_source = format!(
            r#"{{
                "kernel_image_path": "{kernel}",
                "boot_args": "console=ttyS0 reboot=k panic=1 pci=off"
            }}"#,
            kernel = self.config.kernel_path.display()
        );

        self.api_put("boot-source", &boot_source)?;

        // Configure machine config
        let machine_config = format!(
            r#"{{
                "vcpu_count": {vcpu},
                "mem_size_mib": {mem},
                "ht_enabled": false
            }}"#,
            vcpu = self.config.vcpu_count,
            mem = self.config.mem_size_mib
        );

        self.api_put("machine-config", &machine_config)?;

        // Configure rootfs
        let rootfs_config = format!(
            r#"{{
                "drive_id": "rootfs",
                "path_on_host": "{rootfs}",
                "is_root_device": true,
                "is_read_only": false
            }}"#,
            rootfs = self.config.rootfs_path.display()
        );

        self.api_put("drives/rootfs", &rootfs_config)?;

        // Configure network if provided
        if let Some(net_config) = &self.config.network_config {
            let network_config = format!(
                r#"{{
                    "iface_id": "eth0",
                    "host_dev_name": "{host_if}",
                    "guest_mac": "{guest_mac}",
                    "allow_mmds_requests": true
                }}"#,
                host_if = net_config.host_interface,
                guest_mac = net_config.guest_mac
            );

            self.api_put("network-interfaces/eth0", &network_config)?;
        }

        // Start the VM
        self.api_put("actions", r#"{"action_type": "InstanceStart"}"#)?;

        Ok(())
    }

    /// Make a PUT request to the Firecracker API
    fn api_put(&self, path: &str, body: &str) -> Result<()> {
        if let Some(socket) = &self.api_socket {
            let url = format!("http://localhost/{}", path);

            // Write the request to a temporary file
            let tmp_file = format!("/tmp/fc-request-{}.json", path.replace("/", "-"));
            let mut file = File::create(&tmp_file)?;
            file.write_all(body.as_bytes())?;

            // Use curl to make the request
            let output = Command::new("curl")
                .arg("--unix-socket")
                .arg(socket)
                .arg("-X")
                .arg("PUT")
                .arg("-H")
                .arg("Content-Type: application/json")
                .arg("-d")
                .arg(format!("@{}", tmp_file))
                .arg(url)
                .output()?;

            // Clean up
            let _ = fs::remove_file(tmp_file);

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow::anyhow!("API request failed: {}", stderr));
            }
        } else {
            return Err(anyhow::anyhow!("API socket not available"));
        }

        Ok(())
    }

    /// Stop the Firecracker VM
    pub fn stop(&self) -> Result<(), StorageError> {
        info!("Stopping Firecracker VM with ID: {}", self.vm_id);

        if let Some(socket) = &self.api_socket {
            // Send shutdown request
            if let Err(e) = self.api_put("actions", r#"{"action_type": "SendCtrlAltDel"}"#) {
                warn!("Failed to send shutdown request: {}", e);
                // Continue with cleanup even if shutdown request fails
            }

            // Wait for VM to shut down
            thread::sleep(Duration::from_secs(2));

            // Clean up socket file
            if socket.exists() {
                if let Err(e) = fs::remove_file(socket) {
                    warn!("Failed to remove socket file: {}", e);
                    // Continue anyway
                }
            }
        }

        info!("Firecracker VM stopped successfully");
        Ok(())
    }

    /// Execute code in the Firecracker VM
    pub fn execute_code(&self, language: &str, code: &str) -> Result<String> {
        info!("Executing {} code in Firecracker VM", language);

        // Create a temporary file with the code
        let tmp_dir = "/tmp";
        let file_name = format!(
            "code-{}-{}.{}",
            self.vm_id,
            language,
            Self::get_file_extension(language)
        );
        let file_path = format!("{}/{}", tmp_dir, file_name);

        let mut file = File::create(&file_path)?;
        file.write_all(code.as_bytes())?;

        // Copy the file to the VM
        self.copy_to_vm(&file_path, &format!("/tmp/{}", file_name))?;

        // Execute the code in the VM
        let cmd = Self::get_execution_command(language, &format!("/tmp/{}", file_name));
        let output = self.execute_command(&cmd)?;

        // Clean up
        fs::remove_file(file_path)?;

        Ok(output)
    }

    /// Copy a file to the VM
    fn copy_to_vm(&self, host_path: &str, guest_path: &str) -> Result<()> {
        // In a real implementation, this would use SSH or another method to copy files
        // For now, we'll simulate it
        info!("Copying {} to VM at {}", host_path, guest_path);
        Ok(())
    }

    /// Execute a command in the VM
    fn execute_command(&self, command: &str) -> Result<String> {
        // In a real implementation, this would use SSH or another method to execute commands
        // For now, we'll simulate it
        info!("Executing command in VM: {}", command);
        Ok(format!("Output from command: {}", command))
    }

    /// Get the file extension for a language
    fn get_file_extension(language: &str) -> &'static str {
        match language {
            "go" => "go",
            "rust" => "rs",
            "python" => "py",
            "js" => "js",
            _ => "txt",
        }
    }

    /// Get the execution command for a language
    fn get_execution_command(language: &str, file_path: &str) -> String {
        match language {
            "go" => format!("go run {}", file_path),
            "rust" => format!("rust-script {}", file_path),
            "python" => format!("python3 {}", file_path),
            "js" => format!("node {}", file_path),
            _ => format!("cat {}", file_path),
        }
    }
}

/// Create a Firecracker-based execution environment
pub fn create_firecracker_environment(
    config: &RamdiskConfig,
) -> Result<FirecrackerVM, StorageError> {
    // Convert RamdiskConfig to FirecrackerConfig
    let fc_config = FirecrackerConfig {
        binary_path: PathBuf::from("/usr/bin/firecracker"),
        kernel_path: PathBuf::from("/var/lib/firecracker/vmlinux"),
        rootfs_path: PathBuf::from("/var/lib/firecracker/rootfs.ext4"),
        mem_size_mib: (config.size_gb * 1024) as u32,
        vcpu_count: 1,
        network_config: None,
    };

    let vm_id = format!("cylo-{}", uuid::Uuid::new_v4());
    let mut vm = FirecrackerVM::new(fc_config, vm_id);

    match vm.start() {
        Ok(_) => {
            info!("Firecracker VM started successfully");
            Ok(vm)
        }
        Err(e) => {
            error!("Failed to start Firecracker VM: {}", e);
            Err(StorageError::Other(e))
        }
    }
}

/// Check if Firecracker is available on the system
pub fn is_firecracker_available() -> bool {
    Command::new("which")
        .arg("firecracker")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}
