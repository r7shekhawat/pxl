//! SIMD-accelerated image resampling kernels.
//!
//! The entry point is [`resize_bilinear`], which resizes an interleaved 8-bit
//! image (1 to 4 channels) with a separable triangle filter. See
//! [`coeffs`] for the exact filter definition and fixed-point scheme.
//!
//! The same precomputed weights drive three interchangeable backends, selected
//! at runtime by [`Backend::detect`]:
//!
//! * [`Backend::Scalar`] — portable Rust, always available.
//! * `Backend::Avx2` — x86/x86_64 with AVX2.
//! * `Backend::Neon` — AArch64.
//!
//! All backends produce bit-identical output.
//!
//! # Example
//!
//! ```
//! use pxl::kernel::resize_bilinear;
//!
//! // A 4x2 RGB image, shrunk to 2x1.
//! let src = [
//!     10, 20, 30,  10, 20, 30,  200, 100, 0,  200, 100, 0, //
//!     10, 20, 30,  10, 20, 30,  200, 100, 0,  200, 100, 0,
//! ];
//! let out = resize_bilinear(&src, 4, 2, 3, 2, 1).unwrap();
//! assert_eq!(out.len(), 2 * 1 * 3);
//! ```

pub mod coeffs;
pub mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;

use std::fmt;

use coeffs::Plan;

/// Largest number of interleaved channels the kernels accept.
pub const MAX_CHANNELS: usize = 4;

/// Which implementation of the resampling passes to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Portable Rust. Always available.
    Scalar,
    /// AVX2 intrinsics (x86 / x86_64). Requires runtime CPU support.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    /// NEON intrinsics (AArch64). Always available on AArch64.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Backend {
    /// The fastest backend supported by the current CPU.
    pub fn detect() -> Backend {
        Backend::simd().unwrap_or(Backend::Scalar)
    }

    /// Every backend the current CPU can run, slowest first.
    pub fn available() -> Vec<Backend> {
        std::iter::once(Backend::Scalar)
            .chain(Backend::simd())
            .collect()
    }

    /// Whether the current CPU can run this backend.
    pub fn is_supported(self) -> bool {
        self == Backend::Scalar || Backend::simd() == Some(self)
    }

    /// The SIMD backend for this CPU, if any.
    fn simd() -> Option<Backend> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            is_x86_feature_detected!("avx2").then_some(Backend::Avx2)
        }
        #[cfg(target_arch = "aarch64")]
        {
            Some(Backend::Neon)
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            None
        }
    }

    /// `taps` alignment this backend wants for the horizontal plan.
    #[cfg_attr(
        not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
        allow(unused_variables)
    )]
    fn h_align(self, channels: usize) -> usize {
        match self {
            Backend::Scalar => 1,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 => avx2::h_align(channels),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => neon::h_align(channels),
        }
    }

    /// `taps` alignment this backend wants for the vertical plan.
    fn v_align(self) -> usize {
        match self {
            Backend::Scalar => 1,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 => avx2::V_ALIGN,
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => neon::V_ALIGN,
        }
    }

    fn hpass<const C: usize>(
        self,
        src: &[u8],
        src_w: usize,
        rows: usize,
        dst: &mut [u8],
        plan: &Plan,
    ) {
        match self {
            Backend::Scalar => scalar::hpass::<C>(src, src_w, rows, dst, plan),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 if avx2::supports_hpass(C, plan) => {
                // SAFETY: the caller verified AVX2 support via `is_supported`,
                // and the plan alignment was just checked.
                unsafe { avx2::hpass::<C>(src, src_w, rows, dst, plan) }
            }
            #[cfg(target_arch = "aarch64")]
            Backend::Neon if neon::supports_hpass(C, plan) => {
                // SAFETY: NEON is mandatory on AArch64; plan alignment checked.
                unsafe { neon::hpass::<C>(src, src_w, rows, dst, plan) }
            }
            #[allow(unreachable_patterns)]
            _ => scalar::hpass::<C>(src, src_w, rows, dst, plan),
        }
    }

    fn vpass(self, src: &[u8], row_len: usize, dst: &mut [u8], plan: &Plan) {
        match self {
            Backend::Scalar => scalar::vpass(src, row_len, dst, plan),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 if avx2::supports_vpass(plan) => {
                // SAFETY: as in `hpass`.
                unsafe { avx2::vpass(src, row_len, dst, plan) }
            }
            #[cfg(target_arch = "aarch64")]
            Backend::Neon if neon::supports_vpass(plan) => {
                // SAFETY: as in `hpass`.
                unsafe { neon::vpass(src, row_len, dst, plan) }
            }
            #[allow(unreachable_patterns)]
            _ => scalar::vpass(src, row_len, dst, plan),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Backend::Scalar => "scalar",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 => "avx2",
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => "neon",
        })
    }
}

/// Errors reported by [`resize_bilinear`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResizeError {
    /// A source or destination dimension is zero.
    ZeroDimension,
    /// The channel count is not in `1..=MAX_CHANNELS`.
    UnsupportedChannels(usize),
    /// `src.len()` does not equal `src_w * src_h * channels`.
    BufferSize { expected: usize, actual: usize },
    /// The source, destination or intermediate buffer size overflows `usize`.
    TooLarge,
    /// The requested backend cannot run on this CPU.
    UnsupportedBackend(Backend),
}

impl fmt::Display for ResizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResizeError::ZeroDimension => write!(f, "image dimensions must be non-zero"),
            ResizeError::UnsupportedChannels(c) => {
                write!(
                    f,
                    "unsupported channel count {c} (expected 1..={MAX_CHANNELS})"
                )
            }
            ResizeError::BufferSize { expected, actual } => {
                write!(f, "source buffer has {actual} bytes, expected {expected}")
            }
            ResizeError::TooLarge => write!(f, "image is too large to address"),
            ResizeError::UnsupportedBackend(b) => {
                write!(f, "backend {b} is not supported on this CPU")
            }
        }
    }
}

impl std::error::Error for ResizeError {}

/// Resize an interleaved 8-bit image with a bilinear (triangle) filter using
/// the fastest backend available on this CPU.
///
/// `src` holds `src_h` rows of `src_w` pixels with `channels` bytes each
/// (1 = L, 2 = LA, 3 = RGB, 4 = RGBA), tightly packed. The result has the same
/// layout with `dst_w * dst_h` pixels. Alpha is treated as an ordinary channel
/// (no premultiplication).
///
/// Downscaling is anti-aliased: the filter footprint grows with the shrink
/// ratio so every source pixel contributes. See [`coeffs`] for details.
pub fn resize_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    channels: usize,
    dst_w: usize,
    dst_h: usize,
) -> Result<Vec<u8>, ResizeError> {
    resize_bilinear_with(Backend::detect(), src, src_w, src_h, channels, dst_w, dst_h)
}

/// [`resize_bilinear`] with an explicit [`Backend`].
///
/// Fails with [`ResizeError::UnsupportedBackend`] if the CPU lacks the
/// required instructions.
pub fn resize_bilinear_with(
    backend: Backend,
    src: &[u8],
    src_w: usize,
    src_h: usize,
    channels: usize,
    dst_w: usize,
    dst_h: usize,
) -> Result<Vec<u8>, ResizeError> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err(ResizeError::ZeroDimension);
    }
    if !(1..=MAX_CHANNELS).contains(&channels) {
        return Err(ResizeError::UnsupportedChannels(channels));
    }
    // Every buffer (source, destination and either intermediate) is bounded
    // by this product, so one check covers all of them.
    src_w
        .max(dst_w)
        .checked_mul(src_h.max(dst_h))
        .and_then(|n| n.checked_mul(channels))
        .ok_or(ResizeError::TooLarge)?;
    let expected = src_w * src_h * channels;
    if src.len() != expected {
        return Err(ResizeError::BufferSize {
            expected,
            actual: src.len(),
        });
    }
    if !backend.is_supported() {
        return Err(ResizeError::UnsupportedBackend(backend));
    }

    // A pass whose axis is unchanged is an identity (single unit weight) and
    // is skipped. Otherwise the two passes run in whichever order is cheaper:
    // the horizontal pass gathers and shuffles, so per sample-tap it costs
    // roughly 1.5x the streaming vertical pass. For a downscale that means
    // shrinking the row count first; for an upscale, widening rows first.
    let xplan = (dst_w != src_w).then(|| Plan::new(src_w, dst_w, backend.h_align(channels)));
    let yplan = (dst_h != src_h).then(|| Plan::new(src_h, dst_h, backend.v_align()));
    let hpass = |src: &[u8], src_w: usize, rows: usize, dst: &mut [u8], plan: &Plan| match channels
    {
        1 => backend.hpass::<1>(src, src_w, rows, dst, plan),
        2 => backend.hpass::<2>(src, src_w, rows, dst, plan),
        3 => backend.hpass::<3>(src, src_w, rows, dst, plan),
        _ => backend.hpass::<4>(src, src_w, rows, dst, plan),
    };

    let mut out = vec![0u8; dst_w * dst_h * channels];
    match (xplan, yplan) {
        (None, None) => out.copy_from_slice(src),
        (Some(xplan), None) => hpass(src, src_w, src_h, &mut out, &xplan),
        (None, Some(yplan)) => backend.vpass(src, src_w * channels, &mut out, &yplan),
        (Some(xplan), Some(yplan)) => {
            // Estimated from the geometry only (not `Plan::taps`, which
            // carries backend-specific padding) so every backend picks the
            // same order and stays bit-identical.
            const H_COST: usize = 3;
            const V_COST: usize = 2;
            let taps_x = window_taps(src_w, dst_w);
            let taps_y = window_taps(src_h, dst_h);
            let h_first = H_COST * dst_w * src_h * taps_x + V_COST * dst_w * dst_h * taps_y;
            let v_first = V_COST * src_w * dst_h * taps_y + H_COST * dst_w * dst_h * taps_x;
            if h_first <= v_first {
                let mut mid = vec![0u8; dst_w * src_h * channels];
                hpass(src, src_w, src_h, &mut mid, &xplan);
                backend.vpass(&mid, dst_w * channels, &mut out, &yplan);
            } else {
                let mut mid = vec![0u8; src_w * dst_h * channels];
                backend.vpass(src, src_w * channels, &mut mid, &yplan);
                hpass(&mid, src_w, dst_h, &mut out, &xplan);
            }
        }
    }
    Ok(out)
}

/// Approximate filter window width (source samples per output sample) for a
/// `src_len -> dst_len` resample: two taps when upscaling, `2 * ratio` when
/// downscaling.
fn window_taps(src_len: usize, dst_len: usize) -> usize {
    (2 * src_len).div_ceil(dst_len).max(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            resize_bilinear(&[0; 12], 0, 1, 3, 1, 1),
            Err(ResizeError::ZeroDimension)
        );
        assert_eq!(
            resize_bilinear(&[0; 12], 2, 2, 3, 0, 1),
            Err(ResizeError::ZeroDimension)
        );
        assert_eq!(
            resize_bilinear(&[0; 12], 2, 2, 5, 1, 1),
            Err(ResizeError::UnsupportedChannels(5))
        );
        assert_eq!(resize_bilinear(&[0; 12], 2, 2, 3, 1, 1), Ok(vec![0, 0, 0]));
        assert_eq!(
            resize_bilinear(&[0; 11], 2, 2, 3, 1, 1),
            Err(ResizeError::BufferSize {
                expected: 12,
                actual: 11
            })
        );
        assert_eq!(
            resize_bilinear(&[0; 12], 2, 2, 3, usize::MAX / 2, 3),
            Err(ResizeError::TooLarge)
        );
    }

    #[test]
    fn same_size_is_identity() {
        let src: Vec<u8> = (0..5 * 4 * 3).map(|i| (i * 37) as u8).collect();
        for backend in Backend::available() {
            assert_eq!(
                resize_bilinear_with(backend, &src, 5, 4, 3, 5, 4).unwrap(),
                src,
                "{backend}"
            );
        }
    }

    #[test]
    fn detect_is_available() {
        assert!(Backend::detect().is_supported());
        assert_eq!(Backend::available().last(), Some(&Backend::detect()));
    }
}
