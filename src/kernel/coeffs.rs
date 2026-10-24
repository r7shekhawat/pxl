//! Precomputed filter weights for one axis of a separable resample.
//!
//! Resampling is separable, so each axis is described independently by a
//! [`Plan`]: for every output index we store the first source index of a
//! fixed-width window (`taps` consecutive source samples) and one fixed-point
//! weight per tap. All backends (portable, AVX2, NEON) consume the same plan,
//! which keeps their outputs bit-identical to each other.
//!
//! # Filter
//!
//! The filter is the triangle ("bilinear") kernel with pixel-centre alignment:
//! output sample `i` is centred at source coordinate `(i + 0.5) * ratio - 0.5`.
//! When downscaling, the tent widens to `ratio` source pixels on each side so
//! every source pixel contributes and the result is anti-aliased; when
//! upscaling it is the familiar two-tap linear interpolation. This matches
//! `image::imageops::FilterType::Triangle` and Pillow's `BILINEAR`.
//!
//! At the image border, taps that fall outside the source are dropped and the
//! remaining weights are renormalised (no edge replication).
//!
//! # Fixed point
//!
//! Weights are stored as `i16` with [`PRECISION`] fractional bits. Triangle
//! weights are non-negative and at most 1.0, so they fit comfortably; the
//! per-window sum is forced to exactly [`ONE`] so constant regions resample to
//! exactly the same value. An 8-bit sample times a weight is < 2^22 and a
//! whole window sums to at most `255 * ONE`, so `i32` accumulation never
//! overflows.

/// Number of fractional bits in a fixed-point weight.
pub const PRECISION: u32 = 14;
/// The fixed-point representation of 1.0.
pub const ONE: i32 = 1 << PRECISION;
/// Rounding bias added before shifting an accumulator back to 8 bits.
pub const ROUND: i32 = 1 << (PRECISION - 1);

/// Resampling weights for one axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Window width: every output index reads exactly this many consecutive
    /// source samples. Always `>= 1` and `<= src_len`.
    pub taps: usize,
    /// First source index of the window for each output index
    /// (`starts.len() == dst_len`). Invariant: `start + taps <= src_len`.
    pub starts: Vec<usize>,
    /// `taps` weights per output index, row-major (`coefs.len() == dst_len * taps`).
    /// Unused window slots (padding) hold weight 0.
    pub coefs: Vec<i16>,
}

impl Plan {
    /// Build the plan for resampling `src_len` samples to `dst_len`.
    ///
    /// `align` rounds `taps` up to a multiple of the given value (padding with
    /// zero weights) so that SIMD backends can consume whole chunks. If the
    /// source is too small to fit a padded window, `taps` is clamped to
    /// `src_len` instead and may not be aligned; callers must check
    /// [`Plan::taps`] before assuming alignment.
    ///
    /// # Panics
    ///
    /// Panics if `src_len`, `dst_len` or `align` is zero.
    pub fn new(src_len: usize, dst_len: usize, align: usize) -> Plan {
        assert!(
            src_len > 0 && dst_len > 0,
            "plan dimensions must be non-zero"
        );
        assert!(align > 0, "plan alignment must be non-zero");

        let ratio = src_len as f64 / dst_len as f64;
        // Tent half-width in source samples: widen when downscaling.
        let support = ratio.max(1.0);

        // Pass 1: the (in-range, non-zero-weight) source span of every output
        // sample, and the widest span seen.
        let mut spans = Vec::with_capacity(dst_len);
        let mut max_taps = 0usize;
        for i in 0..dst_len {
            let center = (i as f64 + 0.5) * ratio - 0.5;
            // Integers strictly inside (center - support, center + support).
            let lo = ((center - support).floor() as i64 + 1).max(0) as usize;
            let hi = ((center + support).ceil() as i64 - 1).min(src_len as i64 - 1) as usize;
            debug_assert!(lo <= hi, "empty window for output {i}");
            max_taps = max_taps.max(hi - lo + 1);
            spans.push((center, lo, hi));
        }

        let taps = max_taps.div_ceil(align) * align;
        let taps = taps.min(src_len);
        debug_assert!(taps >= max_taps);

        // Pass 2: place each span inside an in-bounds window of `taps` samples
        // and quantise the normalised weights.
        let mut starts = Vec::with_capacity(dst_len);
        let mut coefs = vec![0i16; dst_len * taps];
        let mut weights = Vec::with_capacity(taps);
        for (i, &(center, lo, hi)) in spans.iter().enumerate() {
            let start = lo.min(src_len - taps);
            debug_assert!(start + taps > hi);
            starts.push(start);

            weights.clear();
            weights.extend((lo..=hi).map(|k| 1.0 - (k as f64 - center).abs() / support));
            let sum: f64 = weights.iter().sum();
            debug_assert!(sum > 0.0);

            let out = &mut coefs[i * taps..(i + 1) * taps];
            let mut qsum = 0i32;
            let mut largest = 0usize;
            for (j, w) in weights.iter().enumerate() {
                let q = (w / sum * ONE as f64).round() as i32;
                let slot = lo - start + j;
                out[slot] = q as i16;
                qsum += q;
                if q > i32::from(out[largest]) {
                    largest = slot;
                }
            }
            // Fold the quantisation residual into the largest weight so each
            // window sums to exactly ONE.
            let residual = ONE - qsum;
            out[largest] = (i32::from(out[largest]) + residual) as i16;
            debug_assert!(out[largest] > 0);
        }

        Plan {
            taps,
            starts,
            coefs,
        }
    }

    /// Number of output samples this plan produces.
    pub fn dst_len(&self) -> usize {
        self.starts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_invariants(plan: &Plan, src_len: usize, dst_len: usize) {
        assert_eq!(plan.starts.len(), dst_len);
        assert_eq!(plan.coefs.len(), dst_len * plan.taps);
        assert!(plan.taps >= 1 && plan.taps <= src_len);
        for (i, &start) in plan.starts.iter().enumerate() {
            assert!(start + plan.taps <= src_len, "window out of bounds at {i}");
            let window = &plan.coefs[i * plan.taps..(i + 1) * plan.taps];
            assert!(window.iter().all(|&c| c >= 0), "negative weight at {i}");
            let sum: i32 = window.iter().map(|&c| i32::from(c)).sum();
            assert_eq!(sum, ONE, "weights at {i} do not sum to ONE");
        }
    }

    #[test]
    fn identity_plan_is_a_single_unit_tap() {
        let plan = Plan::new(7, 7, 1);
        check_invariants(&plan, 7, 7);
        assert_eq!(plan.taps, 1);
        assert_eq!(plan.starts, (0..7).collect::<Vec<_>>());
        assert!(plan.coefs.iter().all(|&c| i32::from(c) == ONE));
    }

    #[test]
    fn upscale_uses_two_taps_and_linear_weights() {
        // 2x upscale: centres at -0.25, 0.25, 0.75, 1.25, ... -> weights 3/4, 1/4.
        let plan = Plan::new(4, 8, 1);
        check_invariants(&plan, 4, 8);
        assert_eq!(plan.taps, 2);
        // Output 1 is centred at 0.25: 75% of source 0, 25% of source 1.
        assert_eq!(plan.starts[1], 0);
        assert_eq!(plan.coefs[2..4], [(ONE * 3 / 4) as i16, (ONE / 4) as i16]);
        // Output 2 is centred at 0.75: 25% of source 0, 75% of source 1.
        assert_eq!(plan.starts[2], 0);
        assert_eq!(plan.coefs[4..6], [(ONE / 4) as i16, (ONE * 3 / 4) as i16]);
    }

    #[test]
    fn border_windows_are_renormalised_not_replicated() {
        // First output of a 2x upscale is centred at -0.25: only source 0 is in
        // range, so it gets the full weight.
        let plan = Plan::new(4, 8, 1);
        assert_eq!(plan.starts[0], 0);
        assert_eq!(plan.coefs[0..2], [ONE as i16, 0]);
    }

    #[test]
    fn downscale_widens_the_window() {
        let plan = Plan::new(100, 25, 1);
        check_invariants(&plan, 100, 25);
        // Ratio 4: centres sit at 4i + 1.5, so the tent (half-width 4)
        // covers the 8 integers in (4i - 2.5, 4i + 5.5).
        assert_eq!(plan.taps, 8);
        assert_eq!(plan.starts[1], 2);
    }

    #[test]
    fn alignment_pads_with_zero_weights() {
        let plan = Plan::new(100, 50, 8);
        check_invariants(&plan, 100, 50);
        assert_eq!(plan.taps, 8);
        let aligned = Plan::new(100, 50, 1);
        // Same non-zero weights, just different placement.
        for i in 0..50 {
            let a: Vec<i16> = plan.coefs[i * 8..(i + 1) * 8]
                .iter()
                .copied()
                .filter(|&c| c != 0)
                .collect();
            let b: Vec<i16> = aligned.coefs[i * aligned.taps..(i + 1) * aligned.taps]
                .iter()
                .copied()
                .filter(|&c| c != 0)
                .collect();
            assert_eq!(a, b, "output {i}");
        }
    }

    #[test]
    fn alignment_is_clamped_for_tiny_sources() {
        let plan = Plan::new(3, 9, 16);
        check_invariants(&plan, 3, 9);
        assert_eq!(plan.taps, 3);
        assert!(plan.starts.iter().all(|&s| s == 0));
    }

    #[test]
    fn invariants_hold_across_many_ratios() {
        for &src in &[1usize, 2, 3, 5, 16, 17, 100, 1023, 4000] {
            for &dst in &[1usize, 2, 3, 7, 16, 99, 800, 4001] {
                for &align in &[1usize, 2, 4, 8, 16] {
                    let plan = Plan::new(src, dst, align);
                    check_invariants(&plan, src, dst);
                    if plan.taps < src {
                        assert_eq!(plan.taps % align, 0, "src={src} dst={dst} align={align}");
                    }
                }
            }
        }
    }
}
