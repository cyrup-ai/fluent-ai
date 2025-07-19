use domain::usage::Usage;

#[test]
fn test_usage_creation() {
    let usage = Usage::new(10, 20);
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
}

#[test]
fn test_usage_addition() {
    let a = Usage::new(5, 10);
    let b = Usage::new(15, 20);
    let c = a + b;
    
    assert_eq!(c.prompt_tokens, 20);
    assert_eq!(c.completion_tokens, 30);
    assert_eq!(c.total_tokens, 50);
}

#[test]
fn test_usage_zero() {
    let zero = Usage::zero();
    assert!(zero.is_zero());
    assert_eq!(zero.prompt_tokens, 0);
    assert_eq!(zero.completion_tokens, 0);
    assert_eq!(zero.total_tokens, 0);
}
