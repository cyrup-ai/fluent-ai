//! SIMD-optimized token processing operations for blazing-fast performance
//!
//! These functions provide vectorized operations using CPU SIMD instructions
//! (AVX2/FMA3 on x86_64, NEON on ARM64) for maximum throughput

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized temperature scaling for logits array
/// Scales f32 logits by temperature using vectorized operations
#[inline(always)]
pub fn scale_logits_by_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return; // No scaling needed
    }

    let inv_temp = 1.0 / temperature;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                scale_logits_avx2_fma(logits, inv_temp);
                return;
            }
        }
    }

    // Fallback optimized scalar with manual unrolling for ILP
    let (chunks_4, remainder) = logits.split_at_mut(logits.len() - (logits.len() % 4));

    for chunk in chunks_4.chunks_exact_mut(4) {
        chunk[0] *= inv_temp;
        chunk[1] *= inv_temp;
        chunk[2] *= inv_temp;
        chunk[3] *= inv_temp;
    }

    for value in remainder {
        *value *= inv_temp;
    }
}

/// SIMD-optimized cumulative sum for probability calculations
/// Computes prefix sum using vectorized operations for top-p sampling
#[inline(always)]
pub fn cumulative_sum_f32(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    if input.is_empty() {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && input.len() >= 8 {
            unsafe {
                cumulative_sum_avx2(input, output);
                return;
            }
        }
    }

    // Fallback optimized scalar implementation
    output[0] = input[0];
    for i in 1..input.len() {
        output[i] = output[i - 1] + input[i];
    }
}

/// SIMD-optimized multinomial sampling index finder
/// Finds first index where cumulative probability >= random value
#[inline(always)]
pub fn find_sample_index(cumulative_probs: &[f32], random_val: f32) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && cumulative_probs.len() >= 8 {
            unsafe {
                return find_sample_index_avx2(cumulative_probs, random_val);
            }
        }
    }

    // Fallback binary search for O(log n) performance
    match cumulative_probs.binary_search_by(|&x| {
        if x < random_val {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }) {
        Ok(idx) => idx,
        Err(idx) => idx.min(cumulative_probs.len().saturating_sub(1))}
}

// AVX2/FMA3 implementations for x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_logits_avx2_fma(logits: &mut [f32], inv_temp: f32) {
    let inv_temp_vec = _mm256_set1_ps(inv_temp);
    let chunks = logits.chunks_exact_mut(8);
    let remainder = chunks.into_remainder();

    for chunk in chunks {
        let logits_vec = _mm256_loadu_ps(chunk.as_ptr());
        let scaled = _mm256_mul_ps(logits_vec, inv_temp_vec);
        _mm256_storeu_ps(chunk.as_mut_ptr(), scaled);
    }

    // Handle remainder with scalar
    for value in remainder {
        *value *= inv_temp;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cumulative_sum_avx2(input: &[f32], output: &mut [f32]) {
    // Handle small arrays with scalar
    if input.len() < 16 {
        output[0] = input[0];
        for i in 1..input.len() {
            output[i] = output[i - 1] + input[i];
        }
        return;
    }

    // Vectorized prefix sum using segmented approach
    let mut running_sum = 0.0f32;
    let chunks = input.chunks_exact(8);
    let remainder = chunks.into_remainder();

    for (chunk_idx, chunk) in chunks.enumerate() {
        let chunk_vec = _mm256_loadu_ps(chunk.as_ptr());

        // Compute prefix sum within chunk using horizontal adds
        let mut accumulator = _mm256_setzero_ps();
        let mut current = chunk_vec;

        // Step 1: [a, b, c, d, e, f, g, h] -> [a, a+b, c, c+d, e, e+f, g, g+h]
        let shifted = _mm256_permute_ps(current, 0b10010011);
        let mask1 = _mm256_set_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let masked = _mm256_mul_ps(shifted, mask1);
        current = _mm256_add_ps(current, masked);

        // Step 2: Add running sum to all elements
        let running_sum_vec = _mm256_set1_ps(running_sum);
        let result = _mm256_add_ps(current, running_sum_vec);

        // Store result
        _mm256_storeu_ps(output[chunk_idx * 8..].as_mut_ptr(), result);

        // Extract last element as new running sum
        let mut temp: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), result);
        running_sum = temp[7];
    }

    // Handle remainder
    let offset = chunks.len() * 8;
    for (i, &val) in remainder.iter().enumerate() {
        running_sum += val;
        output[offset + i] = running_sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_sample_index_avx2(cumulative_probs: &[f32], random_val: f32) -> usize {
    let target = _mm256_set1_ps(random_val);
    let chunks = cumulative_probs.chunks_exact(8);

    for (chunk_idx, chunk) in chunks.enumerate() {
        let probs = _mm256_loadu_ps(chunk.as_ptr());
        let cmp = _mm256_cmp_ps(probs, target, _CMP_GE_OQ);
        let mask = _mm256_movemask_ps(cmp);

        if mask != 0 {
            // Found match - get first set bit position
            let first_bit = mask.trailing_zeros() as usize;
            return chunk_idx * 8 + first_bit;
        }
    }

    // Check remainder with scalar
    let offset = chunks.len() * 8;
    for (i, &prob) in cumulative_probs[offset..].iter().enumerate() {
        if prob >= random_val {
            return offset + i;
        }
    }

    cumulative_probs.len().saturating_sub(1)
}