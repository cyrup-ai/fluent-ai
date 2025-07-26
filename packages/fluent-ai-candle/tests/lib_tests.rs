use fluent_ai_candle::{
    constants, memory, device, VERSION, BUILD_INFO
};

#[test]
fn test_constants() {
    assert!(constants::MAX_MODEL_FILE_SIZE > 0);
    assert!(constants::DEFAULT_TOKEN_BUFFER_SIZE > 0);
    assert!(constants::MAX_VOCAB_SIZE > 0);
}

#[test]
fn test_memory_tracking() {
    memory::reset_tracking();
    assert_eq!(memory::current_usage(), 0);

    memory::track_allocation(1024);
    assert_eq!(memory::current_usage(), 1024);

    memory::track_deallocation(512);
    assert_eq!(memory::current_usage(), 512);

    memory::reset_tracking();
    assert_eq!(memory::current_usage(), 0);
}

#[test]
fn test_device_auto_selection() -> Result<(), Box<dyn std::error::Error>> {
    let device = device::auto_device();
    assert!(device.is_ok());

    let device = device.map_err(|_| "Failed to create test device")?;
    let info = device::device_info(&device);
    assert!(!info.is_empty());
    Ok(())
}

#[test]
fn test_version_info() {
    assert!(!VERSION.is_empty());
    assert!(!BUILD_INFO.is_empty());
    assert!(BUILD_INFO.contains("fluent_ai_candle"));
}