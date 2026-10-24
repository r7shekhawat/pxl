//! AVX2 implementation of the two resampling passes (x86 / x86_64).
//!
//! Both passes are built around `vpmaddwd` (`_mm256_madd_epi16`), which
//! multiplies 16 pairs of `i16` and sums each adjacent pair into an `i32`. Data
//! is shuffled so that each adjacent pair holds *the same channel from two
//! neighbouring taps*, turning the multiply-add into a two-tap filter step.
//!
//! # Horizontal pass
//!
//! Every output pixel gathers `taps` source pixels. A 16-byte load covers
//! several taps (4 pixels for RGB/RGBA, 8 for LA, 16 for L); `pshufb`
//! rearranges the bytes into "channel-major pairs" and `vpmovzxbw` widens them
//! to `i16`. The matching weights are broadcast into the same layout with one
//! more shuffle. The `i32` partial sums are accumulated in a `__m256i` and
//! reduced only once per output pixel.
//!
//! # Vertical pass
//!
//! Rows are processed two taps at a time: 16 bytes from row `k` and 16 bytes
//! from row `k + 1` are widened and interleaved, so `vpmaddwd` with the
//! broadcast pair `(w_k, w_k+1)` accumulates both taps at once.
//!
//! # Preconditions
//!
//! The plans must have been built with the alignment reported by
//! [`h_align`] / [`V_ALIGN`]; the dispatcher checks `taps` and falls back to the
//! portable code otherwise.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::coeffs::{Plan, ROUND};
#[cfg(test)]
use super::scalar;

/// Required `taps` alignment of the horizontal plan for `C` channels.
pub const fn h_align(channels: usize) -> usize {
    match channels {
        1 => 16,
        2 => 8,
        _ => 4,
    }
}

/// Required `taps` alignment of the vertical plan.
pub const V_ALIGN: usize = 2;

/// Whether the horizontal kernel can run on this plan.
pub fn supports_hpass(channels: usize, plan: &Plan) -> bool {
    plan.taps.is_multiple_of(h_align(channels))
}

/// Whether the vertical kernel can run on this plan.
pub fn supports_vpass(plan: &Plan) -> bool {
    plan.taps.is_multiple_of(V_ALIGN)
}

/// Rearrange 16 loaded bytes into channel-major pairs of taps (see module docs).
/// `0x80` entries produce zero bytes. Layout of the result for each channel
/// count, as `i16` lanes after `vpmovzxbw`:
/// * 4: `[r0 r1 g0 g1 b0 b1 a0 a1 | r2 r3 g2 g3 b2 b3 a2 a3]`
/// * 3: `[r0 r1 g0 g1 b0 b1  0  0 | r2 r3 g2 g3 b2 b3  0  0]`
/// * 2: `[l0 l1 a0 a1 l2 l3 a2 a3 | l4 l5 a4 a5 l6 l7 a6 a7]`
const fn pixel_shuffle<const C: usize>() -> [i8; 16] {
    const Z: i8 = -128;
    match C {
        4 => [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15],
        3 => [0, 3, 1, 4, 2, 5, Z, Z, 6, 9, 7, 10, 8, 11, Z, Z],
        2 => [0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15],
        _ => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    }
}

/// Byte shuffle that expands a chunk of weights (held identically in both
/// 128-bit lanes) into the layout matching [`pixel_shuffle`].
/// * 4 / 3 channels, weights `[w0 w1 w2 w3 ..]`: `[w0 w1 x4 | w2 w3 x4]`
/// * 2 channels, weights `[w0 .. w7]`: `[w0 w1 w0 w1 w2 w3 w2 w3 | w4 w5 w4 w5 w6 w7 w6 w7]`
const fn coef_shuffle<const C: usize>() -> [i8; 32] {
    match C {
        2 => [
            0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, //
            8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15,
        ],
        _ => [
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, //
            4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
        ],
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn set_epi8(bytes: [i8; 16]) -> __m128i {
    _mm_loadu_si128(bytes.as_ptr() as *const __m128i)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn set256_epi8(bytes: [i8; 32]) -> __m256i {
    _mm256_loadu_si256(bytes.as_ptr() as *const __m256i)
}

/// Load the 12 bytes of 4 RGB pixels without reading past `ptr + 12`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load12(ptr: *const u8) -> __m128i {
    let lo = _mm_loadl_epi64(ptr as *const __m128i);
    let hi = _mm_cvtsi32_si128(std::ptr::read_unaligned(ptr.add(8) as *const i32));
    _mm_unpacklo_epi64(lo, hi)
}

/// Horizontal pass; see `scalar::hpass` for the contract.
///
/// # Safety
///
/// Requires AVX2. `plan.taps` must be a multiple of `h_align(C)`.
#[target_feature(enable = "avx2")]
pub unsafe fn hpass<const C: usize>(
    src: &[u8],
    src_w: usize,
    rows: usize,
    dst: &mut [u8],
    plan: &Plan,
) {
    let dst_w = plan.dst_len();
    let taps = plan.taps;
    assert_eq!(src.len(), src_w * rows * C);
    assert_eq!(dst.len(), dst_w * rows * C);
    assert!(C >= 1 && C <= 4);
    assert!(supports_hpass(C, plan));

    let chunk = h_align(C);
    let pixel_mask = set_epi8(pixel_shuffle::<C>());
    let coef_mask = set256_epi8(coef_shuffle::<C>());
    let round = _mm_set1_epi32(ROUND);

    for y in 0..rows {
        let drow = dst.as_mut_ptr().add(y * dst_w * C);
        for x in 0..dst_w {
            let start = *plan.starts.get_unchecked(x);
            let coefs = plan.coefs.as_ptr().add(x * taps);
            debug_assert!(start + taps <= src_w);

            let mut acc = _mm256_setzero_si256();
            let mut k = 0;
            while k < taps {
                let off = (y * src_w + start + k) * C;
                let p = src.as_ptr().add(off);
                let bytes = if C == 3 {
                    // 4 RGB pixels are 12 bytes; a 16-byte load is fine unless
                    // it would run past the end of the buffer.
                    if off + 16 <= src.len() {
                        _mm_loadu_si128(p as *const __m128i)
                    } else {
                        load12(p)
                    }
                } else {
                    _mm_loadu_si128(p as *const __m128i)
                };
                let px = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(bytes, pixel_mask));

                let w = match C {
                    1 => _mm256_loadu_si256(coefs.add(k) as *const __m256i),
                    2 => {
                        let w = _mm_loadu_si128(coefs.add(k) as *const __m128i);
                        _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(w), coef_mask)
                    }
                    _ => {
                        let w = std::ptr::read_unaligned(coefs.add(k) as *const i64);
                        _mm256_shuffle_epi8(_mm256_set1_epi64x(w), coef_mask)
                    }
                };

                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(px, w));
                k += chunk;
            }

            // Fold the two 128-bit lanes, then finish the per-channel reduction.
            let mut sum = _mm_add_epi32(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256::<1>(acc),
            );
            if C <= 2 {
                // [l a l a] -> [l+l a+a ..]
                sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b01_00_11_10>(sum));
            }
            if C == 1 {
                sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b10_11_00_01>(sum));
            }
            let v = _mm_srai_epi32::<14>(_mm_add_epi32(sum, round));
            let v16 = _mm_packs_epi32(v, v);
            let v8 = _mm_packus_epi16(v16, v16);
            let out = _mm_cvtsi128_si32(v8) as u32;

            let o = drow.add(x * C);
            match C {
                4 => std::ptr::write_unaligned(o as *mut u32, out),
                3 => std::ptr::copy_nonoverlapping(out.to_le_bytes().as_ptr(), o, 3),
                2 => std::ptr::write_unaligned(o as *mut u16, out as u16),
                _ => *o = out as u8,
            }
        }
    }
}

/// Vertical pass; see `scalar::vpass` for the contract.
///
/// # Safety
///
/// Requires AVX2. `plan.taps` must be a multiple of [`V_ALIGN`].
#[target_feature(enable = "avx2")]
pub unsafe fn vpass(src: &[u8], row_len: usize, dst: &mut [u8], plan: &Plan) {
    let taps = plan.taps;
    assert_eq!(src.len() % row_len, 0);
    assert_eq!(dst.len(), plan.dst_len() * row_len);
    assert!(supports_vpass(plan));

    let src_h = src.len() / row_len;
    let round = _mm256_set1_epi32(ROUND);
    let vec_len = row_len & !15;

    for y in 0..plan.dst_len() {
        let start = *plan.starts.get_unchecked(y);
        let coefs = &plan.coefs[y * taps..(y + 1) * taps];
        debug_assert!(start + taps <= src_h);
        let drow = dst.as_mut_ptr().add(y * row_len);

        let mut x = 0;
        while x < vec_len {
            let mut acc_lo = _mm256_setzero_si256();
            let mut acc_hi = _mm256_setzero_si256();
            for (k, pair) in coefs.as_chunks::<2>().0.iter().enumerate() {
                let w = _mm256_set1_epi32(
                    i32::from(pair[0] as u16) | (i32::from(pair[1] as u16) << 16),
                );
                let row_a = src.as_ptr().add((start + 2 * k) * row_len + x);
                let row_b = row_a.add(row_len);
                let a = _mm256_cvtepu8_epi16(_mm_loadu_si128(row_a as *const __m128i));
                let b = _mm256_cvtepu8_epi16(_mm_loadu_si128(row_b as *const __m128i));
                acc_lo =
                    _mm256_add_epi32(acc_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), w));
                acc_hi =
                    _mm256_add_epi32(acc_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), w));
            }
            // acc_lo = bytes [0..4 | 8..12], acc_hi = [4..8 | 12..16]; packing
            // within lanes restores the natural order.
            let lo = _mm256_srai_epi32::<14>(_mm256_add_epi32(acc_lo, round));
            let hi = _mm256_srai_epi32::<14>(_mm256_add_epi32(acc_hi, round));
            let v16 = _mm256_packs_epi32(lo, hi);
            let v8 = _mm256_packus_epi16(v16, v16);
            let out = _mm_unpacklo_epi64(
                _mm256_castsi256_si128(v8),
                _mm256_extracti128_si256::<1>(v8),
            );
            _mm_storeu_si128(drow.add(x) as *mut __m128i, out);
            x += 16;
        }

        // Scalar tail for the last < 16 bytes of the row.
        for x in vec_len..row_len {
            let mut acc = ROUND;
            for (k, &w) in coefs.iter().enumerate() {
                acc += i32::from(w) * i32::from(*src.get_unchecked((start + k) * row_len + x));
            }
            *drow.add(x) = (acc >> 14).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_hpass<const C: usize>(src_w: usize, rows: usize, dst_w: usize) {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let src: Vec<u8> = (0..src_w * rows * C)
            .map(|i| (i * 7 + i / 3) as u8)
            .collect();
        let plan = Plan::new(src_w, dst_w, h_align(C));
        if !supports_hpass(C, &plan) {
            return;
        }
        let mut want = vec![0u8; dst_w * rows * C];
        scalar::hpass::<C>(&src, src_w, rows, &mut want, &plan);
        let mut got = vec![0u8; dst_w * rows * C];
        unsafe { hpass::<C>(&src, src_w, rows, &mut got, &plan) };
        assert_eq!(got, want, "C={C} {src_w}x{rows} -> {dst_w}");
    }

    #[test]
    fn hpass_matches_scalar() {
        for &(sw, dw) in &[
            (64, 16),
            (17, 5),
            (5, 17),
            (100, 33),
            (33, 100),
            (16, 16),
            (9, 64),
        ] {
            check_hpass::<1>(sw, 3, dw);
            check_hpass::<2>(sw, 3, dw);
            check_hpass::<3>(sw, 3, dw);
            check_hpass::<4>(sw, 3, dw);
        }
    }

    #[test]
    fn vpass_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &(sh, dh) in &[(64, 16), (17, 5), (5, 17), (100, 33), (33, 100), (16, 16)] {
            for &row_len in &[1usize, 15, 16, 17, 48, 100] {
                let src: Vec<u8> = (0..sh * row_len).map(|i| (i * 13 + i / 5) as u8).collect();
                let plan = Plan::new(sh, dh, V_ALIGN);
                if !supports_vpass(&plan) {
                    continue;
                }
                let mut want = vec![0u8; dh * row_len];
                scalar::vpass(&src, row_len, &mut want, &plan);
                let mut got = vec![0u8; dh * row_len];
                unsafe { vpass(&src, row_len, &mut got, &plan) };
                assert_eq!(got, want, "{sh}->{dh} row_len={row_len}");
            }
        }
    }
}
