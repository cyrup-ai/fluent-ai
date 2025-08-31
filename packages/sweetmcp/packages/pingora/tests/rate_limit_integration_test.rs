//! Integration tests for rate limiting functionality
//!
//! Tests the HybridAlgorithm integration and ensures proper rate limiting behavior.

use std::time::Duration;

use sweetmcp_pingora::rate_limit::{
    AdvancedRateLimitManager, RateLimitAlgorithmType, RateLimitConfig, SlidingWindowConfig,
    TokenBucketConfig,
};

#[tokio::test]
async fn test_hybrid_algorithm_integration() {
    // Create restrictive configurations for both algorithms
    let token_config = TokenBucketConfig {
        capacity: 5,
        refill_rate: 1.0, // 1 token per second
        initial_tokens: 5,
    };

    let window_config = SlidingWindowConfig {
        window_size: 10, // 10 second window
        max_requests: 3, // Only 3 requests per 10 seconds
        sub_windows: 5,
    };

    let rate_limit_config = RateLimitConfig {
        token_bucket: token_config,
        sliding_window: window_config,
        algorithm: RateLimitAlgorithmType::Hybrid,
        enabled: true,
    };

    let mut manager = AdvancedRateLimitManager::new(1.0, 5, 10);
    manager.update_config(rate_limit_config);

    let endpoint = "test_endpoint";
    let peer_id = Some("test_peer");

    // Test that requests are allowed when both algorithms permit them
    // First 3 requests should be allowed (within sliding window limit)
    for i in 1..=3 {
        let allowed = manager.check_request(endpoint, peer_id, 1);
        assert!(
            allowed,
            "Request {} should be allowed by hybrid algorithm",
            i
        );
    }

    // 4th request should be denied by sliding window (exceeds 3 requests per 10 seconds)
    let allowed = manager.check_request(endpoint, peer_id, 1);
    assert!(
        !allowed,
        "Request 4 should be denied by sliding window limit in hybrid algorithm"
    );

    // Test token bucket exhaustion
    // Reset by creating new manager with fresh state
    let mut manager2 = AdvancedRateLimitManager::new(1.0, 5, 10);
    manager2.update_config(RateLimitConfig {
        token_bucket: TokenBucketConfig {
            capacity: 2,
            refill_rate: 0.1, // Very slow refill
            initial_tokens: 2,
        },
        sliding_window: SlidingWindowConfig {
            window_size: 60,
            max_requests: 10, // High sliding window limit
            sub_windows: 6,
        },
        algorithm: RateLimitAlgorithmType::Hybrid,
        enabled: true,
    });

    // First 2 requests should be allowed
    for i in 1..=2 {
        let allowed = manager2.check_request(endpoint, peer_id, 1);
        assert!(allowed, "Request {} should be allowed", i);
    }

    // 3rd request should be denied by token bucket (no tokens left)
    let allowed = manager2.check_request(endpoint, peer_id, 1);
    assert!(
        !allowed,
        "Request 3 should be denied by token bucket exhaustion in hybrid algorithm"
    );
}

#[tokio::test]
async fn test_hybrid_vs_single_algorithm_behavior() {
    // Test that hybrid algorithm is more restrictive than individual algorithms
    let token_config = TokenBucketConfig {
        capacity: 10,
        refill_rate: 5.0,
        initial_tokens: 10,
    };

    let window_config = SlidingWindowConfig {
        window_size: 5,
        max_requests: 3,
        sub_windows: 5,
    };

    // Test TokenBucket alone
    let mut token_manager = AdvancedRateLimitManager::new(5.0, 10, 5);
    token_manager.update_config(RateLimitConfig {
        token_bucket: token_config.clone(),
        sliding_window: window_config.clone(),
        algorithm: RateLimitAlgorithmType::TokenBucket,
        enabled: true,
    });

    // Test SlidingWindow alone
    let mut window_manager = AdvancedRateLimitManager::new(5.0, 10, 5);
    window_manager.update_config(RateLimitConfig {
        token_bucket: token_config.clone(),
        sliding_window: window_config.clone(),
        algorithm: RateLimitAlgorithmType::SlidingWindow,
        enabled: true,
    });

    // Test Hybrid
    let mut hybrid_manager = AdvancedRateLimitManager::new(5.0, 10, 5);
    hybrid_manager.update_config(RateLimitConfig {
        token_bucket: token_config,
        sliding_window: window_config,
        algorithm: RateLimitAlgorithmType::Hybrid,
        enabled: true,
    });

    let endpoint = "comparison_test";
    let peer_id = Some("comparison_peer");

    // Make 5 requests rapidly
    let mut token_allowed = 0;
    let mut window_allowed = 0;
    let mut hybrid_allowed = 0;

    for _ in 0..5 {
        if token_manager.check_request(endpoint, peer_id, 1) {
            token_allowed += 1;
        }
        if window_manager.check_request(endpoint, peer_id, 1) {
            window_allowed += 1;
        }
        if hybrid_manager.check_request(endpoint, peer_id, 1) {
            hybrid_allowed += 1;
        }
    }

    // Hybrid should be most restrictive (limited by sliding window to 3)
    assert_eq!(
        window_allowed, 3,
        "Sliding window should allow exactly 3 requests"
    );
    assert!(
        token_allowed >= 5,
        "Token bucket should allow all 5 requests"
    );
    assert_eq!(
        hybrid_allowed, 3,
        "Hybrid should be limited by the more restrictive sliding window"
    );
}

#[tokio::test]
async fn test_hybrid_algorithm_endpoint_and_peer_consistency() {
    // Verify that both endpoint and peer limiters use HybridAlgorithm when configured
    let config = RateLimitConfig {
        token_bucket: TokenBucketConfig {
            capacity: 2,
            refill_rate: 0.5,
            initial_tokens: 2,
        },
        sliding_window: SlidingWindowConfig {
            window_size: 10,
            max_requests: 1, // Very restrictive
            sub_windows: 5,
        },
        algorithm: RateLimitAlgorithmType::Hybrid,
        enabled: true,
    };

    let mut manager = AdvancedRateLimitManager::new(0.5, 2, 10);
    manager.update_config(config);

    // Test endpoint limiting
    let endpoint1 = "endpoint1";
    let endpoint2 = "endpoint2";
    let peer_id = Some("test_peer");

    // First request to endpoint1 should be allowed
    assert!(manager.check_request(endpoint1, peer_id, 1));

    // Second request to endpoint1 should be denied by sliding window
    assert!(!manager.check_request(endpoint1, peer_id, 1));

    // First request to endpoint2 should be allowed (separate limiter)
    assert!(manager.check_request(endpoint2, peer_id, 1));

    // Test peer limiting
    let endpoint = "test_endpoint";
    let peer1 = Some("peer1");
    let peer2 = Some("peer2");

    // Reset manager for peer testing
    let mut manager2 = AdvancedRateLimitManager::new(0.5, 2, 10);
    manager2.update_config(RateLimitConfig {
        token_bucket: TokenBucketConfig {
            capacity: 2,
            refill_rate: 0.5,
            initial_tokens: 2,
        },
        sliding_window: SlidingWindowConfig {
            window_size: 10,
            max_requests: 1,
            sub_windows: 5,
        },
        algorithm: RateLimitAlgorithmType::Hybrid,
        enabled: true,
    });

    // First request from peer1 should be allowed
    assert!(manager2.check_request(endpoint, peer1, 1));

    // Second request from peer1 should be denied by sliding window
    assert!(!manager2.check_request(endpoint, peer1, 1));

    // First request from peer2 should be allowed (separate limiter)
    assert!(manager2.check_request(endpoint, peer2, 1));
}
