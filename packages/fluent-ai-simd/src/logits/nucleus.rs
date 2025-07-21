//! Nucleus (top-p) sampling implementation

use crate::logits::LogitsResult;
use rand::Rng;

/// Prepare logits for nucleus sampling
pub fn prepare_nucleus_sampling_simd(logits: &mut [f32], top_p: f64) -> LogitsResult<()> {
    if top_p <= 0.0 || top_p > 1.0 {
        return Err(crate::logits::LogitsError::SamplingError(
            "top_p must be in (0, 1]".to_string(),
        ));
    }

    let mut sorted: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    // Sort in descending order
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate cumulative probabilities
    let mut cumsum = 0.0f64;
    let mut cutoff = 0;
    
    for (i, &(_, prob)) in sorted.iter().enumerate() {
        cumsum += prob as f64;
        if cumsum >= top_p && i > 0 {
            cutoff = i;
            break;
        }
    }

    // Apply mask
    for i in cutoff..logits.len() {
        logits[sorted[i].0] = f32::NEG_INFINITY;
    }

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
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_nucleus_sampling() {
        let mut logits = vec![0.1, 0.2, 0.3, 0.4];
        prepare_nucleus_sampling_simd(&mut logits, 0.6).unwrap();
        
        // Only the top elements should remain
        assert!(logits[0] > f32::NEG_INFINITY);
        assert!(logits[1] > f32::NEG_INFINITY);
        assert!(logits[2] == f32::NEG_INFINITY);
        assert!(logits[3] == f32::NEG_INFINITY);
    }
    
    #[test]
    fn test_sample_from_nucleus() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng = thread_rng();
        let idx = sample_from_nucleus(&probs, &mut rng).unwrap();
        assert!(idx < probs.len());
    }
}
