use domain::capabilities::{ModelCapabilities, Capability};

#[test]
fn test_capabilities() {
    let mut caps = ModelCapabilities::new()
        .with_capability(Capability::Vision)
        .with_capability(Capability::FunctionCalling);

    assert!(caps.has_capability(Capability::Vision));
    assert!(caps.has_capability(Capability::FunctionCalling));
    assert!(!caps.has_capability(Capability::Streaming));

    caps.set_capability(Capability::Streaming, true);
    assert!(caps.has_capability(Capability::Streaming));

    assert!(caps.has_all_capabilities(&[Capability::Vision, Capability::FunctionCalling]));
    assert!(caps.has_any_capability(&[Capability::Vision, Capability::BatchProcessing]));
    assert!(!caps.has_all_capabilities(&[Capability::Vision, Capability::BatchProcessing]));
}
