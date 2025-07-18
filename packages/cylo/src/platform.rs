// ============================================================================
// File: packages/cylo/src/platform.rs
// ----------------------------------------------------------------------------
// Platform detection utilities for Cylo execution environments.
//
// Provides comprehensive platform and capability detection for:
// - Operating system and architecture detection
// - Backend availability and feature support
// - Runtime capability verification
// - Performance optimization hints
// ============================================================================

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

/// Global platform information cache
static PLATFORM_INFO: OnceLock<PlatformInfo> = OnceLock::new();

/// Comprehensive platform information
///
/// Contains detected platform capabilities, available backends,
/// and performance characteristics for optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Operating system name
    pub os: OperatingSystem,

    /// CPU architecture
    pub arch: Architecture,

    /// Available execution backends
    pub available_backends: Vec<BackendAvailability>,

    /// Platform capabilities
    pub capabilities: PlatformCapabilities,

    /// Performance characteristics
    pub performance: PerformanceHints,

    /// Detection timestamp
    pub detected_at: SystemTime,
}

/// Operating system enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatingSystem {
    /// Linux distribution
    Linux {
        /// Distribution name (e.g., "Ubuntu", "Alpine")
        distribution: Option<String>,
        /// Kernel version
        kernel_version: Option<String>,
    },
    /// macOS
    MacOS {
        /// macOS version (e.g., "14.0")
        version: Option<String>,
    },
    /// Windows
    Windows {
        /// Windows version
        version: Option<String>,
    },
    /// Unknown/other OS
    Unknown {
        /// OS name if detectable
        name: String,
    },
}

/// CPU architecture enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    /// ARM64/AArch64 (Apple Silicon, etc.)
    Arm64,
    /// x86_64/AMD64
    X86_64,
    /// ARM32
    Arm,
    /// x86 32-bit
    X86,
    /// Unknown architecture
    Unknown(String),
}

/// Backend availability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendAvailability {
    /// Backend name
    pub name: &'static str,

    /// Whether backend is available
    pub available: bool,

    /// Availability reason (why available/unavailable)
    pub reason: String,

    /// Backend-specific capabilities
    pub capabilities: HashMap<String, String>,

    /// Performance rating (0-100)
    pub performance_rating: u8,
}

/// Platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    /// Virtualization support
    pub virtualization: VirtualizationSupport,

    /// Container runtime support
    pub containers: ContainerSupport,

    /// Security features
    pub security: SecurityFeatures,

    /// Network capabilities
    pub network: NetworkCapabilities,

    /// File system features
    pub filesystem: FilesystemFeatures,
}

/// Virtualization support details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualizationSupport {
    /// Hardware virtualization available
    pub hardware_virtualization: bool,

    /// KVM available (Linux)
    pub kvm_available: bool,

    /// Hyper-V available (Windows)
    pub hyperv_available: bool,

    /// Hypervisor.framework available (macOS)
    pub hypervisor_framework: bool,

    /// Nested virtualization support
    pub nested_virtualization: bool,
}

/// Container runtime support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSupport {
    /// Docker available
    pub docker_available: bool,

    /// Podman available
    pub podman_available: bool,

    /// Apple containerization available
    pub apple_containers: bool,

    /// Native container runtimes
    pub native_runtimes: Vec<String>,
}

/// Security features available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFeatures {
    /// LandLock support (Linux)
    pub landlock: bool,

    /// SELinux support
    pub selinux: bool,

    /// AppArmor support
    pub apparmor: bool,

    /// Gatekeeper (macOS)
    pub gatekeeper: bool,

    /// Windows Defender
    pub windows_defender: bool,

    /// Secure boot
    pub secure_boot: bool,
}

/// Network capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    /// Network namespaces support
    pub network_namespaces: bool,

    /// Bridge networking
    pub bridge_networking: bool,

    /// Host networking
    pub host_networking: bool,

    /// Custom DNS support
    pub custom_dns: bool,
}

/// Filesystem features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemFeatures {
    /// Overlay filesystem support
    pub overlay_fs: bool,

    /// Bind mounts support
    pub bind_mounts: bool,

    /// Temporary filesystems
    pub tmpfs: bool,

    /// Extended attributes
    pub extended_attributes: bool,

    /// File capabilities
    pub file_capabilities: bool,
}

/// Performance optimization hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHints {
    /// Recommended backend for this platform
    pub recommended_backend: Option<&'static str>,

    /// CPU core count
    pub cpu_cores: u32,

    /// Available memory in bytes
    pub available_memory: u64,

    /// Temporary directory performance
    pub tmpdir_performance: TmpDirPerformance,

    /// I/O characteristics
    pub io_characteristics: IOCharacteristics,
}

/// Temporary directory performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmpDirPerformance {
    /// Temporary directory path
    pub path: String,

    /// Whether it's in-memory (tmpfs, ramdisk)
    pub in_memory: bool,

    /// Estimated throughput in MB/s
    pub estimated_throughput: u32,
}

/// I/O performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOCharacteristics {
    /// Disk type (SSD, HDD, NVMe)
    pub disk_type: String,

    /// Sequential read speed estimate (MB/s)
    pub sequential_read_mbps: u32,

    /// Sequential write speed estimate (MB/s)
    pub sequential_write_mbps: u32,

    /// Random IOPS estimate
    pub random_iops: u32,
}

impl PlatformInfo {
    /// Get or detect platform information
    ///
    /// Uses cached detection results for performance.
    ///
    /// # Returns
    /// Platform information
    pub fn get() -> &'static PlatformInfo {
        PLATFORM_INFO.get_or_init(|| Self::detect())
    }

    /// Force re-detection of platform information
    ///
    /// # Returns
    /// Newly detected platform information
    pub fn detect() -> PlatformInfo {
        let start_time = SystemTime::now();

        let os = Self::detect_operating_system();
        let arch = Self::detect_architecture();
        let capabilities = Self::detect_capabilities(&os);
        let available_backends = Self::detect_available_backends(&os, &arch, &capabilities);
        let performance = Self::detect_performance_hints(&os, &arch, &available_backends);

        PlatformInfo {
            os,
            arch,
            available_backends,
            capabilities,
            performance,
            detected_at: start_time,
        }
    }

    /// Detect operating system
    fn detect_operating_system() -> OperatingSystem {
        #[cfg(target_os = "linux")]
        {
            let distribution = Self::detect_linux_distribution();
            let kernel_version = Self::detect_kernel_version();
            OperatingSystem::Linux {
                distribution,
                kernel_version,
            }
        }

        #[cfg(target_os = "macos")]
        {
            let version = Self::detect_macos_version();
            OperatingSystem::MacOS { version }
        }

        #[cfg(target_os = "windows")]
        {
            let version = Self::detect_windows_version();
            OperatingSystem::Windows { version }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            OperatingSystem::Unknown {
                name: std::env::consts::OS.to_string(),
            }
        }
    }

    /// Detect CPU architecture
    fn detect_architecture() -> Architecture {
        match std::env::consts::ARCH {
            "aarch64" => Architecture::Arm64,
            "x86_64" => Architecture::X86_64,
            "arm" => Architecture::Arm,
            "x86" => Architecture::X86,
            other => Architecture::Unknown(other.to_string()),
        }
    }

    /// Detect platform capabilities
    fn detect_capabilities(os: &OperatingSystem) -> PlatformCapabilities {
        PlatformCapabilities {
            virtualization: Self::detect_virtualization_support(os),
            containers: Self::detect_container_support(os),
            security: Self::detect_security_features(os),
            network: Self::detect_network_capabilities(os),
            filesystem: Self::detect_filesystem_features(os),
        }
    }

    /// Detect available backends
    fn detect_available_backends(
        os: &OperatingSystem,
        arch: &Architecture,
        capabilities: &PlatformCapabilities,
    ) -> Vec<BackendAvailability> {
        let mut backends = Vec::new();

        // Apple backend availability
        backends.push(Self::detect_apple_backend_availability(os, arch));

        // LandLock backend availability
        backends.push(Self::detect_landlock_backend_availability(os, capabilities));

        // FireCracker backend availability
        backends.push(Self::detect_firecracker_backend_availability(
            os,
            capabilities,
        ));

        backends
    }

    /// Detect performance hints
    fn detect_performance_hints(
        _os: &OperatingSystem,
        _arch: &Architecture,
        backends: &[BackendAvailability],
    ) -> PerformanceHints {
        let recommended_backend = backends
            .iter()
            .filter(|b| b.available)
            .max_by_key(|b| b.performance_rating)
            .map(|b| b.name);

        PerformanceHints {
            recommended_backend,
            cpu_cores: Self::detect_cpu_cores(),
            available_memory: Self::detect_available_memory(),
            tmpdir_performance: Self::detect_tmpdir_performance(),
            io_characteristics: Self::detect_io_characteristics(),
        }
    }

    // Platform-specific detection methods

    #[cfg(target_os = "linux")]
    fn detect_linux_distribution() -> Option<String> {
        use std::fs;

        // Try to read distribution from os-release
        if let Ok(content) = fs::read_to_string("/etc/os-release") {
            for line in content.lines() {
                if line.starts_with("ID=") {
                    return Some(line[3..].trim_matches('"').to_string());
                }
            }
        }

        // Fallback methods
        if std::path::Path::new("/etc/alpine-release").exists() {
            Some("alpine".to_string())
        } else if std::path::Path::new("/etc/debian_version").exists() {
            Some("debian".to_string())
        } else {
            None
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn detect_linux_distribution() -> Option<String> {
        None
    }

    #[cfg(target_os = "linux")]
    fn detect_kernel_version() -> Option<String> {
        use std::fs;

        fs::read_to_string("/proc/version")
            .ok()
            .and_then(|content| content.split_whitespace().nth(2).map(|v| v.to_string()))
    }

    #[cfg(not(target_os = "linux"))]
    fn detect_kernel_version() -> Option<String> {
        None
    }

    #[cfg(target_os = "macos")]
    fn detect_macos_version() -> Option<String> {
        use std::process::Command;

        Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    None
                }
            })
    }

    #[cfg(not(target_os = "macos"))]
    fn detect_macos_version() -> Option<String> {
        None
    }

    #[cfg(target_os = "windows")]
    fn detect_windows_version() -> Option<String> {
        // Windows version detection would go here
        None
    }

    #[cfg(not(target_os = "windows"))]
    fn detect_windows_version() -> Option<String> {
        None
    }

    fn detect_virtualization_support(_os: &OperatingSystem) -> VirtualizationSupport {
        VirtualizationSupport {
            hardware_virtualization: Self::has_hardware_virtualization(),
            kvm_available: Self::has_kvm_support(),
            hyperv_available: Self::has_hyperv_support(),
            hypervisor_framework: Self::has_hypervisor_framework(),
            nested_virtualization: false, // Complex to detect
        }
    }

    fn detect_container_support(os: &OperatingSystem) -> ContainerSupport {
        ContainerSupport {
            docker_available: Self::is_command_available("docker"),
            podman_available: Self::is_command_available("podman"),
            apple_containers: Self::is_command_available("container")
                && matches!(os, OperatingSystem::MacOS { .. }),
            native_runtimes: Self::detect_native_runtimes(),
        }
    }

    fn detect_security_features(os: &OperatingSystem) -> SecurityFeatures {
        SecurityFeatures {
            landlock: Self::has_landlock_support(),
            selinux: Self::has_selinux_support(),
            apparmor: Self::has_apparmor_support(),
            gatekeeper: matches!(os, OperatingSystem::MacOS { .. }),
            windows_defender: matches!(os, OperatingSystem::Windows { .. }),
            secure_boot: Self::has_secure_boot(),
        }
    }

    fn detect_network_capabilities(_os: &OperatingSystem) -> NetworkCapabilities {
        NetworkCapabilities {
            network_namespaces: std::path::Path::new("/proc/self/ns/net").exists(),
            bridge_networking: Self::is_command_available("ip"),
            host_networking: true,
            custom_dns: true,
        }
    }

    fn detect_filesystem_features(_os: &OperatingSystem) -> FilesystemFeatures {
        FilesystemFeatures {
            overlay_fs: std::path::Path::new("/sys/module/overlay").exists(),
            bind_mounts: true, // Generally available on Unix systems
            tmpfs: std::path::Path::new("/proc/filesystems").exists(),
            extended_attributes: true,
            file_capabilities: std::path::Path::new("/proc/sys/kernel/cap_last_cap").exists(),
        }
    }

    fn detect_apple_backend_availability(
        os: &OperatingSystem,
        arch: &Architecture,
    ) -> BackendAvailability {
        let available = matches!(os, OperatingSystem::MacOS { .. })
            && matches!(arch, Architecture::Arm64)
            && Self::is_command_available("container");

        let (reason, performance_rating) = if available {
            (
                "Apple Silicon macOS with containerization support".to_string(),
                95,
            )
        } else if matches!(os, OperatingSystem::MacOS { .. }) {
            (
                "macOS detected but not Apple Silicon or containerization not available"
                    .to_string(),
                0,
            )
        } else {
            (
                "Apple containerization only available on macOS".to_string(),
                0,
            )
        };

        BackendAvailability {
            name: "Apple",
            available,
            reason,
            capabilities: HashMap::new(),
            performance_rating,
        }
    }

    fn detect_landlock_backend_availability(
        os: &OperatingSystem,
        capabilities: &PlatformCapabilities,
    ) -> BackendAvailability {
        let available = matches!(os, OperatingSystem::Linux { .. })
            && capabilities.security.landlock
            && Self::is_command_available("bwrap");

        let (reason, performance_rating) = if available {
            ("Linux with LandLock and bubblewrap support".to_string(), 85)
        } else if matches!(os, OperatingSystem::Linux { .. }) {
            (
                "Linux detected but LandLock or bubblewrap not available".to_string(),
                0,
            )
        } else {
            ("LandLock only available on Linux".to_string(), 0)
        };

        BackendAvailability {
            name: "LandLock",
            available,
            reason,
            capabilities: HashMap::new(),
            performance_rating,
        }
    }

    fn detect_firecracker_backend_availability(
        os: &OperatingSystem,
        capabilities: &PlatformCapabilities,
    ) -> BackendAvailability {
        let available = matches!(os, OperatingSystem::Linux { .. })
            && capabilities.virtualization.kvm_available
            && Self::is_command_available("firecracker");

        let (reason, performance_rating) = if available {
            ("Linux with KVM and FireCracker support".to_string(), 90)
        } else if matches!(os, OperatingSystem::Linux { .. }) {
            (
                "Linux detected but KVM or FireCracker not available".to_string(),
                0,
            )
        } else {
            ("FireCracker only available on Linux".to_string(), 0)
        };

        BackendAvailability {
            name: "FireCracker",
            available,
            reason,
            capabilities: HashMap::new(),
            performance_rating,
        }
    }

    // Utility detection methods

    fn has_hardware_virtualization() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/cpuinfo")
                .map(|content| content.contains("vmx") || content.contains("svm"))
                .unwrap_or(false)
        }

        #[cfg(target_os = "macos")]
        {
            // Apple Silicon has virtualization support
            std::env::consts::ARCH == "aarch64"
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        false
    }

    fn has_kvm_support() -> bool {
        std::path::Path::new("/dev/kvm").exists()
    }

    fn has_hyperv_support() -> bool {
        #[cfg(target_os = "windows")]
        {
            // Windows Hyper-V detection would go here
            false
        }

        #[cfg(not(target_os = "windows"))]
        false
    }

    fn has_hypervisor_framework() -> bool {
        #[cfg(target_os = "macos")]
        {
            std::path::Path::new("/System/Library/Frameworks/Hypervisor.framework").exists()
        }

        #[cfg(not(target_os = "macos"))]
        false
    }

    fn has_landlock_support() -> bool {
        std::path::Path::new("/sys/kernel/security/landlock").exists()
    }

    fn has_selinux_support() -> bool {
        std::path::Path::new("/sys/fs/selinux").exists()
    }

    fn has_apparmor_support() -> bool {
        std::path::Path::new("/sys/kernel/security/apparmor").exists()
    }

    fn has_secure_boot() -> bool {
        std::path::Path::new(
            "/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c",
        )
        .exists()
    }

    fn is_command_available(command: &str) -> bool {
        use std::process::Command;

        Command::new("which")
            .arg(command)
            .output()
            .map(|output| output.status.success())
            .unwrap_or_else(|_| {
                // Fallback: try to run the command with --version or --help
                Command::new(command)
                    .arg("--version")
                    .output()
                    .map(|output| output.status.success())
                    .unwrap_or(false)
            })
    }

    fn detect_native_runtimes() -> Vec<String> {
        let mut runtimes = Vec::new();

        let commands = ["runc", "crun", "containerd", "buildkit"];
        for cmd in &commands {
            if Self::is_command_available(cmd) {
                runtimes.push(cmd.to_string());
            }
        }

        runtimes
    }

    fn detect_cpu_cores() -> u32 {
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1)
    }

    fn detect_available_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;

            if let Ok(content) = fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024; // Convert to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback: return a reasonable default
        4 * 1024 * 1024 * 1024 // 4GB
    }

    fn detect_tmpdir_performance() -> TmpDirPerformance {
        let tmp_path = std::env::temp_dir();
        let path = tmp_path.display().to_string();

        // Check if it's likely in-memory
        let in_memory = path.contains("/tmp");

        let estimated_throughput = if in_memory {
            5000 // 5GB/s for RAM
        } else {
            500 // 500MB/s for SSD
        };

        TmpDirPerformance {
            path,
            in_memory,
            estimated_throughput,
        }
    }

    fn detect_io_characteristics() -> IOCharacteristics {
        // This is a simplified implementation
        // Real implementation would benchmark I/O performance
        IOCharacteristics {
            disk_type: "SSD".to_string(),
            sequential_read_mbps: 500,
            sequential_write_mbps: 400,
            random_iops: 50000,
        }
    }
}

/// Public API functions

/// Get current platform information
pub fn detect_platform() -> &'static PlatformInfo {
    PlatformInfo::get()
}

/// Check if running on Apple Silicon
pub fn is_apple_silicon() -> bool {
    let info = detect_platform();
    matches!(info.os, OperatingSystem::MacOS { .. }) && matches!(info.arch, Architecture::Arm64)
}

/// Check if running on Linux
pub fn is_linux() -> bool {
    matches!(detect_platform().os, OperatingSystem::Linux { .. })
}

/// Check if LandLock is available
pub fn has_landlock() -> bool {
    detect_platform().capabilities.security.landlock
}

/// Check if KVM is available
pub fn has_kvm() -> bool {
    detect_platform().capabilities.virtualization.kvm_available
}

/// Get recommended backend for current platform
pub fn get_recommended_backend() -> Option<&'static str> {
    detect_platform().performance.recommended_backend
}

/// Get available backends for current platform
pub fn get_available_backends() -> Vec<&'static str> {
    detect_platform()
        .available_backends
        .iter()
        .filter(|b| b.available)
        .map(|b| b.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_detection() {
        let info = detect_platform();

        // Basic sanity checks
        assert!(info.performance.cpu_cores > 0);
        assert!(info.performance.available_memory > 0);
        assert!(!info.performance.tmpdir_performance.path.is_empty());

        // Should have at least some architecture detection
        assert!(!matches!(info.arch, Architecture::Unknown(_)));

        // Should detect current OS correctly
        #[cfg(target_os = "linux")]
        assert!(matches!(info.os, OperatingSystem::Linux { .. }));

        #[cfg(target_os = "macos")]
        assert!(matches!(info.os, OperatingSystem::MacOS { .. }));
    }

    #[test]
    fn backend_availability() {
        let backends = get_available_backends();

        // Should have at least one backend available or give reasonable reasons
        if backends.is_empty() {
            let info = detect_platform();
            for backend in &info.available_backends {
                assert!(!backend.reason.is_empty());
            }
        }
    }

    #[test]
    fn utility_functions() {
        // These should not panic
        let _ = is_apple_silicon();
        let _ = is_linux();
        let _ = has_landlock();
        let _ = has_kvm();
        let _ = get_recommended_backend();
    }

    #[test]
    fn platform_specific_detection() {
        let info = detect_platform();

        #[cfg(target_os = "macos")]
        {
            if is_apple_silicon() {
                assert!(info.available_backends.iter().any(|b| b.name == "Apple"));
            }
        }

        #[cfg(target_os = "linux")]
        {
            assert!(info.available_backends.iter().any(|b| b.name == "LandLock"));
            assert!(info
                .available_backends
                .iter()
                .any(|b| b.name == "FireCracker"));
        }
    }
}
