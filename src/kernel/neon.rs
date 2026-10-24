//! NEON implementation of the two resampling passes (AArch64).
//!
//! NEON's widening multiply-accumulate (`vmlal_s16` and friends) multiplies
//! four `i16` lanes by either a vector or a single broadcast lane and adds the
//! `i32` products into a four-lane accumulator, so no pair interleaving is
//! needed: a 4-byte RGBA pixel maps directly onto one accumulator.
//!
//! # Horizontal pass
//!
//! A 16-byte load covers several taps (4 pixels for RGB/RGBA, 8 for LA, 16
//! for L). RGB is expanded to 4-byte pixels with a table lookup (`tbl`) so it
//! shares the RGBA code path; LA is multiplied by zipped weight pairs and
//! reduced at the end; L uses a plain dot product.
//!
//! # Vertical pass
//!
//! 16 bytes of each source row are widened to `i16` and multiplied by the
//! broadcast row weight into four `i32` accumulators.
//!
//! # Preconditions
//!
//! The plans must have been built with the alignment reported by
//! [`h_align`] / [`V_ALIGN`]; the dispatcher checks `taps` and falls back to
//! the portable code otherwise.

use std::arch::aarch64::*;

use super::coeffs::{Plan, PRECISION, ROUND};
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
pub const V_ALIGN: usize = 1;

/// Whether the horizontal kernel can run on this plan.
pub fn supports_hpass(channels: usize, plan: &Plan) -> bool {
    plan.taps.is_multiple_of(h_align(channels))
}

/// Whether the vertical kernel can run on this plan.
pub fn supports_vpass(plan: &Plan) -> bool {
    plan.taps.is_multiple_of(V_ALIGN)
}

/// Table that spreads 4 packed RGB pixels into 4 RGB0 pixels.
const RGB_TO_RGBX: [u8; 16] = [0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255];

/// Load the 12 bytes of 4 RGB pixels without reading past `ptr + 12`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load12(ptr: *const u8) -> uint8x16_t {
    let lo = vld1_u8(ptr);
    let hi = vdup_n_u32(std::ptr::read_unaligned(ptr.add(8) as *const u32));
    vcombine_u8(lo, vreinterpret_u8_u32(hi))
}

/// Round an `i32x4` accumulator to 8 bits: `(acc + ROUND) >> PRECISION`,
/// saturated to `0..=255`, returned in the low 4 bytes.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn narrow4(acc: int32x4_t) -> uint8x8_t {
    let v16 = vrshrn_n_s32::<{ PRECISION as i32 }>(acc);
    vqmovun_s16(vcombine_s16(v16, v16))
}

/// Horizontal pass; see `scalar::hpass` for the contract.
///
/// # Safety
///
/// Requires NEON. `plan.taps` must be a multiple of `h_align(C)`.
#[target_feature(enable = "neon")]
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
    let rgb_table = vld1q_u8(RGB_TO_RGBX.as_ptr());

    for y in 0..rows {
        let drow = dst.as_mut_ptr().add(y * dst_w * C);
        for x in 0..dst_w {
            let start = *plan.starts.get_unchecked(x);
            let coefs = plan.coefs.as_ptr().add(x * taps);
            debug_assert!(start + taps <= src_w);

            let mut acc = vdupq_n_s32(0);
            let mut k = 0;
            while k < taps {
                let off = (y * src_w + start + k) * C;
                let p = src.as_ptr().add(off);
                let bytes = if C == 3 {
                    // 4 RGB pixels are 12 bytes; a 16-byte load is fine unless
                    // it would run past the end of the buffer.
                    let raw = if off + 16 <= src.len() {
                        vld1q_u8(p)
                    } else {
                        load12(p)
                    };
                    vqtbl1q_u8(raw, rgb_table)
                } else {
                    vld1q_u8(p)
                };
                let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes)));
                let hi = vreinterpretq_s16_u16(vmovl_high_u8(bytes));

                match C {
                    1 => {
                        // 16 taps: plain dot product into 4 partial sums.
                        let w0 = vld1q_s16(coefs.add(k));
                        let w1 = vld1q_s16(coefs.add(k + 8));
                        acc = vmlal_s16(acc, vget_low_s16(lo), vget_low_s16(w0));
                        acc = vmlal_high_s16(acc, lo, w0);
                        acc = vmlal_s16(acc, vget_low_s16(hi), vget_low_s16(w1));
                        acc = vmlal_high_s16(acc, hi, w1);
                    }
                    2 => {
                        // 8 taps of [l a]: weights zipped to [w0 w0 w1 w1 ..] so
                        // lanes hold [l_even a_even l_odd a_odd] partial sums.
                        let w = vld1q_s16(coefs.add(k));
                        let w01 = vzip1q_s16(w, w);
                        let w23 = vzip2q_s16(w, w);
                        acc = vmlal_s16(acc, vget_low_s16(lo), vget_low_s16(w01));
                        acc = vmlal_high_s16(acc, lo, w01);
                        acc = vmlal_s16(acc, vget_low_s16(hi), vget_low_s16(w23));
                        acc = vmlal_high_s16(acc, hi, w23);
                    }
                    _ => {
                        // 4 taps of 4-byte pixels: one lane-broadcast MLA each.
                        let w = vld1_s16(coefs.add(k));
                        acc = vmlal_lane_s16::<0>(acc, vget_low_s16(lo), w);
                        acc = vmlal_lane_s16::<1>(acc, vget_high_s16(lo), w);
                        acc = vmlal_lane_s16::<2>(acc, vget_low_s16(hi), w);
                        acc = vmlal_lane_s16::<3>(acc, vget_high_s16(hi), w);
                    }
                }
                k += chunk;
            }

            let o = drow.add(x * C);
            match C {
                4 => {
                    let v = vget_lane_u32::<0>(vreinterpret_u32_u8(narrow4(acc)));
                    std::ptr::write_unaligned(o as *mut u32, v);
                }
                3 => {
                    let v = vget_lane_u32::<0>(vreinterpret_u32_u8(narrow4(acc)));
                    std::ptr::copy_nonoverlapping(v.to_le_bytes().as_ptr(), o, 3);
                }
                2 => {
                    // [l_even a_even l_odd a_odd] -> [l a]
                    let s = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
                    let v = vget_lane_u16::<0>(vreinterpret_u16_u8(narrow4(vcombine_s32(s, s))));
                    std::ptr::write_unaligned(o as *mut u16, v);
                }
                _ => {
                    let total = vaddvq_s32(acc);
                    *o = ((total + ROUND) >> PRECISION).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Vertical pass; see `scalar::vpass` for the contract.
///
/// # Safety
///
/// Requires NEON.
#[target_feature(enable = "neon")]
pub unsafe fn vpass(src: &[u8], row_len: usize, dst: &mut [u8], plan: &Plan) {
    let taps = plan.taps;
    assert_eq!(src.len() % row_len, 0);
    assert_eq!(dst.len(), plan.dst_len() * row_len);
    assert!(supports_vpass(plan));

    let src_h = src.len() / row_len;
    let vec_len = row_len & !15;

    for y in 0..plan.dst_len() {
        let start = *plan.starts.get_unchecked(y);
        let coefs = &plan.coefs[y * taps..(y + 1) * taps];
        debug_assert!(start + taps <= src_h);
        let drow = dst.as_mut_ptr().add(y * row_len);

        let mut x = 0;
        while x < vec_len {
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut acc2 = vdupq_n_s32(0);
            let mut acc3 = vdupq_n_s32(0);
            for (k, &w) in coefs.iter().enumerate() {
                let bytes = vld1q_u8(src.as_ptr().add((start + k) * row_len + x));
                let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes)));
                let hi = vreinterpretq_s16_u16(vmovl_high_u8(bytes));
                acc0 = vmlal_n_s16(acc0, vget_low_s16(lo), w);
                acc1 = vmlal_n_s16(acc1, vget_high_s16(lo), w);
                acc2 = vmlal_n_s16(acc2, vget_low_s16(hi), w);
                acc3 = vmlal_n_s16(acc3, vget_high_s16(hi), w);
            }
            let lo = vcombine_s16(
                vrshrn_n_s32::<{ PRECISION as i32 }>(acc0),
                vrshrn_n_s32::<{ PRECISION as i32 }>(acc1),
            );
            let hi = vcombine_s16(
                vrshrn_n_s32::<{ PRECISION as i32 }>(acc2),
                vrshrn_n_s32::<{ PRECISION as i32 }>(acc3),
            );
            vst1q_u8(drow.add(x), vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi)));
            x += 16;
        }

        // Scalar tail for the last < 16 bytes of the row.
        for x in vec_len..row_len {
            let mut acc = ROUND;
            for (k, &w) in coefs.iter().enumerate() {
                acc += i32::from(w) * i32::from(*src.get_unchecked((start + k) * row_len + x));
            }
            *drow.add(x) = (acc >> PRECISION).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_hpass<const C: usize>(src_w: usize, rows: usize, dst_w: usize) {
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
        for &(sh, dh) in &[(64, 16), (17, 5), (5, 17), (100, 33), (33, 100), (16, 16)] {
            for &row_len in &[1usize, 15, 16, 17, 48, 100] {
                let src: Vec<u8> = (0..sh * row_len).map(|i| (i * 13 + i / 5) as u8).collect();
                let plan = Plan::new(sh, dh, V_ALIGN);
                let mut want = vec![0u8; dh * row_len];
                scalar::vpass(&src, row_len, &mut want, &plan);
                let mut got = vec![0u8; dh * row_len];
                unsafe { vpass(&src, row_len, &mut got, &plan) };
                assert_eq!(got, want, "{sh}->{dh} row_len={row_len}");
            }
        }
    }
}
