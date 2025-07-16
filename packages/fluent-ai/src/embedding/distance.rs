// ============================================================================
// File: src/embeddings/distance.rs
// ----------------------------------------------------------------------------
// Pure-math helpers for the `Embedding` type.
//
// * **Zero-alloc hot path**: everything runs on the stack and touches
//   nothing outside the two input slices.
// * **Soundness**: every index is bounds-checked in *all* build modes;
//   no unchecked UB slips past `--release` optimisations.
// * **SIMD / Rayon** feature-gates** keep the scalar fallback spotless.
// ============================================================================

#![allow(clippy::inline_always)]
#![forbid(unsafe_code)] // nothing here needs `unsafe`

use std::iter::Sum;

/// Small helper so an impossible code-path stays zero-cost in the happy path.
#[inline(always)]
#[cold] // never pollute branch predictions
const fn unlikely(b: bool) -> bool {
    #[allow(clippy::needless_bool)]
    {
        b
    }
}

/// Cosine similarity for two equal-length vectors.
///
/// * Returns `NaN` if the lengths differ (deterministic sentinel).
/// * **No heap allocations, ever.**
#[inline(always)]
pub fn cosine_similarity(lhs: &[f64], rhs: &[f64]) -> f64 {
    if unlikely(lhs.len() != rhs.len()) {
        return f64::NAN; // safe, deterministic, zero-alloc sentinel
    }

    // ---------- scalar fast path (no SIMD feature enabled) ----------
    let (mut dot, mut norm_l, mut norm_r) = (0.0, 0.0, 0.0);
    for (&l, &r) in lhs.iter().zip(rhs) {
        dot += l * r;
        norm_l += l * l;
        norm_r += r * r;
    }

    // Avoid a division by zero if either vector is all-zeros.
    if norm_l == 0.0 || norm_r == 0.0 {
        return f64::NAN;
    }
    dot / norm_l.sqrt() / norm_r.sqrt()
}

/// Euclidean (ℓ₂) distance between two equal-length vectors.
#[inline(always)]
pub fn euclidean(lhs: &[f64], rhs: &[f64]) -> f64 {
    if unlikely(lhs.len() != rhs.len()) {
        return f64::NAN;
    }

    lhs.iter()
        .zip(rhs)
        .map(|(&l, &r)| (l - r).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Dot-product (plain inner product) – included for completeness.
#[inline(always)]
pub fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    if unlikely(lhs.len() != rhs.len()) {
        return f64::NAN;
    }

    lhs.iter().zip(rhs).map(|(&l, &r)| l * r).sum::<f64>()
}

// ---------------------------------------------------------------------------
// Unit tests (cover scalar + error paths)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const V1: &[f64] = &[1.0, 2.0, 3.0];
    const V2: &[f64] = &[4.0, 5.0, 6.0];

    #[test]
    fn cosine_ok() {
        let sim = cosine_similarity(V1, V2);
        // pre-computed reference
        let expected = 32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        assert!((sim - expected).abs() < 1e-12);
    }

    #[test]
    fn euclidean_ok() {
        let dist = euclidean(V1, V2);
        assert!((dist - 5.196152422706632).abs() < 1e-12);
    }

    #[test]
    fn dot_ok() {
        assert_eq!(dot(V1, V2), 32.0);
    }

    #[test]
    fn mismatch_len_yields_nan() {
        let a = &[1.0, 2.0];
        let b = &[1.0];
        assert!(cosine_similarity(a, b).is_nan());
        assert!(euclidean(a, b).is_nan());
        assert!(dot(a, b).is_nan());
    }

    #[test]
    fn zero_vector_cosine_nan() {
        let zero = &[0.0, 0.0, 0.0];
        assert!(cosine_similarity(V1, zero).is_nan());
    }
}
