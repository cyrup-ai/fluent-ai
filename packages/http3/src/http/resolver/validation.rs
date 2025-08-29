//! DNS hostname validation
//!
//! This module provides security-conscious hostname validation for DNS operations.

/// Validate hostname format for security
pub fn validate_hostname(hostname: &str) -> Result<(), String> {
    if hostname.is_empty() {
        return Err("Empty hostname".to_string());
    }
    if hostname.len() > 253 {
        return Err("Hostname too long (max 253 characters)".to_string());
    }

    // Check for invalid characters
    if !hostname
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
    {
        return Err("Invalid characters in hostname".to_string());
    }

    // Check label constraints
    for label in hostname.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err("Invalid label length".to_string());
        }
        if label.starts_with('-') || label.ends_with('-') {
            return Err("Invalid label format".to_string());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_hostnames() {
        assert!(validate_hostname("example.com").is_ok());
        assert!(validate_hostname("sub.example.com").is_ok());
        assert!(validate_hostname("test-host.example.org").is_ok());
        assert!(validate_hostname("a.b.c.d").is_ok());
    }

    #[test]
    fn test_invalid_hostnames() {
        assert!(validate_hostname("").is_err());
        assert!(validate_hostname("-invalid.com").is_err());
        assert!(validate_hostname("invalid-.com").is_err());
        assert!(validate_hostname("invalid..com").is_err());
        assert!(validate_hostname("invalid_hostname.com").is_err());
    }

    #[test]
    fn test_hostname_length_limits() {
        let long_hostname = "a".repeat(254);
        assert!(validate_hostname(&long_hostname).is_err());

        let long_label = format!("{}.com", "a".repeat(64));
        assert!(validate_hostname(&long_label).is_err());
    }
}
