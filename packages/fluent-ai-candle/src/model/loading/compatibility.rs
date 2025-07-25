//! Model compatibility checking and validation
//!
//! This module handles checking model compatibility with the current system
//! including device support, version requirements, and feature compatibility.

use crate::error::{CandleError, CandleResult};
use super::metadata::ModelMetadata;

/// Check if the current system meets the requirements for the model
pub fn check_system_requirements(
    metadata: &ModelMetadata,
    device: &candle_core::Device,
) -> CandleResult<()> {
    // Check device compatibility
    if !is_device_compatible(metadata, device) {
        return Err(CandleError::IncompatibleDevice {
            msg: format!("Model '{}' is not compatible with device {:?}", 
                         metadata.architecture, device),
        });
    }

    // Check for required features
    if let Some(features) = metadata.config.get("required_features") {
        if !has_required_features(features) {
            return Err(CandleError::IncompatibleModel {
                msg: format!("Missing required features: {}", features),
            });
        }
    }

    // Check version compatibility
    if let Some(min_version) = metadata.config.get("min_candle_version") {
        if !check_version(min_version) {
            return Err(CandleError::IncompatibleVersion {
                msg: format!("Candle version too old, minimum required: {}", min_version),
            });
        }
    }

    Ok(())
}

/// Check if the device is compatible with the model
fn is_device_compatible(metadata: &ModelMetadata, device: &candle_core::Device) -> bool {
    // Check for device-specific requirements
    if let Some(device_type) = metadata.config.get("device_type") {
        match device_type.as_str() {
            "cuda" => {
                if !device.is_cuda() {
                    return false;
                }
            }
            "metal" => {
                if !device.is_metal() {
                    return false;
                }
            }
            "cpu" => {
                if !device.is_cpu() {
                    return false;
                }
            }
            _ => {}
        }
    }

    // Check for CUDA compute capability requirements
    if device.is_cuda() {
        if let Some(min_cc) = metadata.config.get("min_cuda_compute_capability") {
            if let Ok(min_cc) = min_cc.parse::<f32>() {
                let cc = device.cuda_compute_capability().unwrap_or(0.0);
                if cc < min_cc {
                    return false;
                }
            }
        }
    }

    true
}

/// Check if all required features are available
fn has_required_features(required: &str) -> bool {
    for feature in required.split(',') {
        let feature = feature.trim();
        if !is_feature_available(feature) {
            return false;
        }
    }
    true
}

/// Check if a specific feature is available
fn is_feature_available(feature: &str) -> bool {
    // Check for CPU features
    if feature.starts_with("cpu_") {
        return has_cpu_feature(feature);
    }
    
    // Check for GPU features
    if feature.starts_with("gpu_") {
        return has_gpu_feature(feature);
    }
    
    // Default to true for unknown features
    true
}

/// Check for CPU features
#[cfg(target_arch = "x86_64")]
fn has_cpu_feature(feature: &str) -> bool {
    use std::arch::x86_64::*;
    
    match feature {
        "cpu_avx" => is_x86_feature_detected!("avx"),
        "cpu_avx2" => is_x86_feature_detected!("avx2"),
        "cpu_avx512" => is_x86_feature_detected!("avx512f"),
        "cpu_fma" => is_x86_feature_detected!("fma"),
        _ => true,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn has_cpu_feature(_feature: &str) -> bool {
    // On non-x86_64, assume all CPU features are available
    true
}

/// Check for GPU features
fn has_gpu_feature(_feature: &str) -> bool {
    // This would check for GPU-specific features
    // For now, assume all GPU features are available
    true
}

/// Check if the current version meets the minimum required version
fn check_version(required: &str) -> bool {
    use semver::{Version, VersionReq};
    
    let current = env!("CARGO_PKG_VERSION");
    let current_version = Version::parse(current).unwrap_or_else(|_| Version::new(0, 0, 0));
    let required = format!(">={}", required);
    
    if let Ok(req) = VersionReq::parse(&required) {
        req.matches(&current_version)
    } else {
        // If we can't parse the version requirement, assume it's compatible
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_check_system_requirements() {
        let mut metadata = ModelMetadata::default();
        let device = Device::Cpu;
        
        // Test with no requirements
        assert!(check_system_requirements(&metadata, &device).is_ok());
        
        // Test with device requirement
        metadata.config.insert("device_type".to_string(), "cpu".to_string());
        assert!(check_system_requirements(&metadata, &device).is_ok());
        
        // Test with incompatible device
        metadata.config.insert("device_type".to_string(), "cuda".to_string());
        assert!(check_system_requirements(&metadata, &device).is_err());
    }
}