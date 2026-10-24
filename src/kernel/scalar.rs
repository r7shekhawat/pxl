//! Portable reference implementation of the two resampling passes.
//!
//! This is plain Rust (LLVM may auto-vectorise parts of it) and is the
//! fallback on every platform. The SIMD backends must produce output that is
//! bit-identical to this module, which is what `tests/` checks.

use super::coeffs::{Plan, PRECISION, ROUND};

/// Horizontal pass: resample every row of `src` (`rows` rows of `src_w`
/// pixels, `C` bytes each) into `dst` (`rows` rows of `plan.dst_len()` pixels).
pub fn hpass<const C: usize>(src: &[u8], src_w: usize, rows: usize, dst: &mut [u8], plan: &Plan) {
    let dst_w = plan.dst_len();
    let taps = plan.taps;
    assert_eq!(src.len(), src_w * rows * C);
    assert_eq!(dst.len(), dst_w * rows * C);

    for (srow, drow) in src
        .chunks_exact(src_w * C)
        .zip(dst.chunks_exact_mut(dst_w * C))
    {
        let outputs = drow.as_chunks_mut::<C>().0.iter_mut();
        for ((out, &start), coefs) in outputs.zip(&plan.starts).zip(plan.coefs.chunks_exact(taps)) {
            let window = srow[start * C..(start + taps) * C].as_chunks::<C>().0;
            let mut acc = [ROUND; C];
            for (px, &w) in window.iter().zip(coefs) {
                let w = i32::from(w);
                for (a, &p) in acc.iter_mut().zip(px) {
                    *a += w * i32::from(p);
                }
            }
            for (o, a) in out.iter_mut().zip(acc) {
                *o = (a >> PRECISION).clamp(0, 255) as u8;
            }
        }
    }
}

/// Vertical pass: resample the columns of `src` (rows of `row_len` bytes) into
/// `dst` (`plan.dst_len()` rows of `row_len` bytes). Channel-agnostic.
pub fn vpass(src: &[u8], row_len: usize, dst: &mut [u8], plan: &Plan) {
    let taps = plan.taps;
    assert_eq!(src.len() % row_len, 0);
    assert_eq!(dst.len(), plan.dst_len() * row_len);

    let mut acc = vec![0i32; row_len];
    for ((drow, &start), coefs) in dst
        .chunks_exact_mut(row_len)
        .zip(&plan.starts)
        .zip(plan.coefs.chunks_exact(taps))
    {
        acc.fill(ROUND);
        let window = &src[start * row_len..(start + taps) * row_len];
        for (srow, &w) in window.chunks_exact(row_len).zip(coefs) {
            let w = i32::from(w);
            for (a, &p) in acc.iter_mut().zip(srow) {
                *a += w * i32::from(p);
            }
        }
        for (o, &a) in drow.iter_mut().zip(&acc) {
            *o = (a >> PRECISION).clamp(0, 255) as u8;
        }
    }
}
