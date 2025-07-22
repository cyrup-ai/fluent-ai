//! Nucleus (top-p) sampling implementation

use rand::Rng;

use crate::logits::LogitsResult;

/// Prepare logits for nucleus sampling
pub fn prepare_nucleus_sampling_simd(logits: &mut [f32], top_p: f64) -> LogitsResult<()> {
    if top_p <= 0.0 || top_p > 1.0 {
        return Err(crate::logits::LogitsError::SamplingError(
            "top_p must be in (0, 1]".to_string(),
        ));
    }

    println!(
        "[DEBUG] prepare_nucleus_sampling_simd - Input logits: {:?}",
        logits
    );

    // Create a vector of (index, value) pairs
    let mut sorted: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();

    println!("[DEBUG] Before sort: {:?}", sorted);

    // Sort in descending order by probability
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("[DEBUG] After sort: {:?}", sorted);

    // Calculate cumulative probabilities
    let mut cumsum = 0.0f64;
    let mut cutoff = logits.len(); // Default to keeping all if top_p is very large

    println!("[DEBUG] Top-p threshold: {}", top_p);

    // First, find the cutoff point based on cumulative probability
    for (i, &(_, prob)) in sorted.iter().enumerate() {
        cumsum += prob as f64;
        println!("[DEBUG] i={}, prob={}, cumsum={}", i, prob, cumsum);

        if cumsum >= top_p {
            cutoff = i + 1; // Include the current element in the nucleus
            println!("[DEBUG] Reached threshold at i={}, cutoff={}", i, cutoff);
            break;
        }
    }

    println!("[DEBUG] Final cutoff: {}", cutoff);

    // Create a set of indices to keep for O(1) lookup
    let keep_indices: std::collections::HashSet<_> =
        sorted[..cutoff].iter().map(|&(idx, _)| idx).collect();

    // Apply mask to logits not in the nucleus
    for (i, logit) in logits.iter_mut().enumerate() {
        if !keep_indices.contains(&i) {
            *logit = f32::NEG_INFINITY;
            println!("[DEBUG] Setting logits[{}] to -inf", i);
        }
    }

    println!("[DEBUG] Final logits: {:?}", logits);

    Ok(())
}

/// Sample from nucleus
pub fn sample_from_nucleus<R: Rng>(probs: &[f32], rng: &mut R) -> LogitsResult<usize> {
    let sum: f32 = probs.iter().sum();
    let threshold = rng.gen_range(0.0..sum);

    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= threshold {
            return Ok(i);
        }
    }

    Ok(probs.len() - 1)
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_nucleus_sampling() {
        let mut logits = vec![0.1, 0.2, 0.3, 0.4];
        println!("Original logits: {:?}", logits);

        prepare_nucleus_sampling_simd(&mut logits, 0.6).unwrap();
        println!("After nucleus sampling: {:?}", logits);

        // Check which elements are kept
        let kept: Vec<_> = logits
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v > f32::NEG_INFINITY)
            .map(|(i, _)| i)
            .collect();
        println!("Kept indices: {:?}", kept);

        // Only the top elements should remain
        assert!(
            logits[0] > f32::NEG_INFINITY,
            "Expected index 0 to be kept, but it was set to -inf"
        );
        assert!(
            logits[1] > f32::NEG_INFINITY,
            "Expected index 1 to be kept, but it was set to -inf"
        );
        assert_eq!(
            logits[2],
            f32::NEG_INFINITY,
            "Expected index 2 to be set to -inf, but it was {}",
            logits[2]
        );
        assert_eq!(
            logits[3],
            f32::NEG_INFINITY,
            "Expected index 3 to be set to -inf, but it was {}",
            logits[3]
        );
    }

    #[test]
    fn test_sample_from_nucleus() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng = thread_rng();
        let idx = sample_from_nucleus(&probs, &mut rng).unwrap();
        assert!(idx < probs.len());
    }
}
