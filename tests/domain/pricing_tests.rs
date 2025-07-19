use domain::pricing::PricingTier;
use float_cmp::assert_approx_eq;

#[test]
fn test_classify_pricing_tier() {
    // Test UltraLow tier
    assert_eq!(PricingTier::classify(0.3, 1.2), PricingTier::UltraLow);
    
    // Test Low tier
    assert_eq!(PricingTier::classify(0.6, 2.5), PricingTier::Low);
    
    // Test Medium tier
    assert_eq!(PricingTier::classify(2.0, 10.0), PricingTier::Medium);
    
    // Test High tier
    assert_eq!(PricingTier::classify(10.0, 30.0), PricingTier::High);
    
    // Test Premium tier (high input cost)
    assert_eq!(PricingTier::classify(25.0, 50.0), PricingTier::Premium);
    
    // Test Premium tier (high output cost)
    assert_eq!(PricingTier::classify(15.0, 70.0), PricingTier::Premium);
}

#[test]
fn test_cost_range() {
    let (min_in, max_in, min_out, max_out) = PricingTier::Medium.cost_range();
    assert_approx_eq!(f64, min_in, 1.0);
    assert_approx_eq!(f64, max_in, 5.0);
    assert_approx_eq!(f64, min_out, 3.0);
    assert_approx_eq!(f64, max_out, 15.0);
}

#[test]
fn test_is_recommended() {
    // High quality requirements should recommend High or Premium
    assert!(!PricingTier::Low.is_recommended(true, false));
    assert!(PricingTier::High.is_recommended(true, false));
    
    // Budget constrained should recommend UltraLow or Low
    assert!(PricingTier::UltraLow.is_recommended(false, true));
    assert!(!PricingTier::Medium.is_recommended(false, true));
    
    // Balanced requirements should recommend Low or Medium
    assert!(PricingTier::Low.is_recommended(false, false));
    assert!(PricingTier::Medium.is_recommended(false, false));
    assert!(!PricingTier::UltraLow.is_recommended(false, false));
}

#[test]
fn test_display() {
    assert_eq!(PricingTier::UltraLow.to_string(), "Ultra Low");
    assert_eq!(PricingTier::Low.to_string(), "Low");
    assert_eq!(PricingTier::Medium.to_string(), "Medium");
    assert_eq!(PricingTier::High.to_string(), "High");
    assert_eq!(PricingTier::Premium.to_string(), "Premium");
}
