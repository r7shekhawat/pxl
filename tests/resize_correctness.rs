//! Correctness tests for the bilinear resize kernel and the file-level glue.
//!
//! Strategy:
//! * the portable backend is checked against an independent oracle
//!   (`image`'s f32 triangle filter) and against analytic properties;
//! * every SIMD backend the host CPU supports is checked to be bit-identical
//!   to the portable backend;
//! * the CLI path (`resize_file`) is exercised end to end on temporary files.
//!
//! Randomised cases use a fixed-seed PRNG so failures are reproducible.

use std::path::PathBuf;

use image::{DynamicImage, ImageBuffer, Luma, LumaA, Pixel, Rgb, Rgba};
use pxl::kernel::{resize_bilinear, resize_bilinear_with, Backend};
use pxl::ops::resize::{fit_height, resize_file, resize_image};

/// xorshift64* — tiny deterministic PRNG so the tests need no `rand` dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `lo..=hi`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() % (hi - lo + 1) as u64) as usize
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// A random resize case biased towards small, awkward sizes (odd widths,
/// rows that are not a multiple of the SIMD width, sources narrower than a
/// filter window, 1-pixel edges) as well as realistic thumbnail ratios.
fn random_case(rng: &mut Rng) -> (usize, usize, usize, usize, usize) {
    let channels = rng.range(1, 4);
    let (src_w, src_h, dst_w, dst_h) = match rng.range(0, 3) {
        0 => (
            rng.range(1, 8),
            rng.range(1, 8),
            rng.range(1, 40),
            rng.range(1, 40),
        ),
        1 => (
            rng.range(1, 200),
            rng.range(1, 200),
            rng.range(1, 200),
            rng.range(1, 200),
        ),
        2 => (
            rng.range(200, 900),
            rng.range(1, 100),
            rng.range(1, 120),
            rng.range(1, 60),
        ),
        _ => (
            rng.range(1, 100),
            rng.range(1, 100),
            rng.range(1, 300),
            rng.range(1, 100),
        ),
    };
    (src_w, src_h, channels, dst_w, dst_h)
}

/// Resize raw samples with `image`'s own triangle filter, as an oracle.
fn image_crate_resize<P>(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8>
where
    P: Pixel<Subpixel = u8> + 'static,
{
    let img: ImageBuffer<P, Vec<u8>> =
        ImageBuffer::from_raw(src_w as u32, src_h as u32, src.to_vec()).unwrap();
    image::imageops::resize(
        &img,
        dst_w as u32,
        dst_h as u32,
        image::imageops::FilterType::Triangle,
    )
    .into_raw()
}

fn oracle(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    channels: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    match channels {
        1 => image_crate_resize::<Luma<u8>>(src, src_w, src_h, dst_w, dst_h),
        2 => image_crate_resize::<LumaA<u8>>(src, src_w, src_h, dst_w, dst_h),
        3 => image_crate_resize::<Rgb<u8>>(src, src_w, src_h, dst_w, dst_h),
        4 => image_crate_resize::<Rgba<u8>>(src, src_w, src_h, dst_w, dst_h),
        _ => unreachable!(),
    }
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Kernel: analytic properties
// ---------------------------------------------------------------------------

#[test]
fn upscale_is_linear_interpolation_between_pixel_centres() {
    // 2x upscale of [0, 200]: output centres land at source coordinates
    // -0.25, 0.25, 0.75, 1.25, so the middle samples are 1/4 and 3/4 blends
    // and the edge samples clamp to the nearest source pixel.
    let out = resize_bilinear(&[0, 200], 2, 1, 1, 4, 1).unwrap();
    assert_eq!(out, [0, 50, 150, 200]);

    // Same thing vertically.
    let out = resize_bilinear(&[0, 200], 1, 2, 1, 1, 4).unwrap();
    assert_eq!(out, [0, 50, 150, 200]);
}

#[test]
fn downscale_is_a_weighted_average_of_every_source_pixel() {
    // 4 -> 1: the tent spans all four pixels with weights 5:7:7:5, so
    // (10*5 + 20*7 + 30*7 + 40*5) / 24 = 25 exactly.
    let out = resize_bilinear(&[10, 20, 30, 40], 4, 1, 1, 1, 1).unwrap();
    assert_eq!(out, [25]);

    // Plain 2-tap sampling would have dropped pixels entirely; with the widened
    // tent, a single bright pixel must still influence the shrunken result.
    let mut src = vec![0u8; 64];
    src[20] = 255;
    let out = resize_bilinear(&src, 64, 1, 1, 8, 1).unwrap();
    assert!(
        out.iter().any(|&p| p > 0),
        "bright pixel lost on 8x shrink: {out:?}"
    );
}

#[test]
fn channels_are_independent_and_interleaving_is_preserved() {
    // Every channel carries its own ramp; after a resize each channel must
    // still only depend on itself (no cross-talk from the SIMD shuffles).
    let mut rng = Rng::new(7);
    for channels in 1..=4 {
        let (src_w, src_h, dst_w, dst_h) = (37, 11, 19, 23);
        let src = rng.bytes(src_w * src_h * channels);
        let combined = resize_bilinear(&src, src_w, src_h, channels, dst_w, dst_h).unwrap();
        for c in 0..channels {
            let plane: Vec<u8> = src.iter().skip(c).step_by(channels).copied().collect();
            let alone = resize_bilinear(&plane, src_w, src_h, 1, dst_w, dst_h).unwrap();
            let extracted: Vec<u8> = combined.iter().skip(c).step_by(channels).copied().collect();
            assert_eq!(extracted, alone, "channel {c} of {channels}");
        }
    }
}

#[test]
fn constant_images_stay_exactly_constant() {
    let mut rng = Rng::new(11);
    for _ in 0..100 {
        let (src_w, src_h, channels, dst_w, dst_h) = random_case(&mut rng);
        let value = rng.next() as u8;
        let src = vec![value; src_w * src_h * channels];
        for backend in Backend::available() {
            let out =
                resize_bilinear_with(backend, &src, src_w, src_h, channels, dst_w, dst_h).unwrap();
            assert_eq!(out.len(), dst_w * dst_h * channels);
            assert!(
                out.iter().all(|&p| p == value),
                "{backend}: {src_w}x{src_h}x{channels} -> {dst_w}x{dst_h} value {value}"
            );
        }
    }
}

#[test]
fn output_never_leaves_the_source_range() {
    // The filter is a convex combination, so no ringing or overshoot: every
    // output sample must lie within [min, max] of its channel in the source.
    let mut rng = Rng::new(13);
    for _ in 0..100 {
        let (src_w, src_h, channels, dst_w, dst_h) = random_case(&mut rng);
        let lo = rng.next() as u8;
        let hi = lo.max(rng.next() as u8);
        let src: Vec<u8> = rng
            .bytes(src_w * src_h * channels)
            .into_iter()
            .map(|b| lo + (u16::from(b) * u16::from(hi - lo) / 255) as u8)
            .collect();
        let out = resize_bilinear(&src, src_w, src_h, channels, dst_w, dst_h).unwrap();
        assert!(
            out.iter().all(|&p| (lo..=hi).contains(&p)),
            "{src_w}x{src_h}x{channels} -> {dst_w}x{dst_h}: sample outside [{lo}, {hi}]"
        );
    }
}

// ---------------------------------------------------------------------------
// Kernel: oracle comparison and backend equivalence
// ---------------------------------------------------------------------------

#[test]
fn matches_image_crate_triangle_filter_within_one_lsb() {
    // The kernel uses 14-bit fixed-point weights and an 8-bit intermediate
    // between the two passes, while `image` computes in f32 and rounds once.
    // The two therefore differ by at most one quantisation step.
    let mut rng = Rng::new(17);
    let mut total_diff = 0u64;
    let mut total_samples = 0u64;
    for _ in 0..300 {
        let (src_w, src_h, channels, dst_w, dst_h) = random_case(&mut rng);
        let src = rng.bytes(src_w * src_h * channels);
        let want = oracle(&src, src_w, src_h, channels, dst_w, dst_h);
        let got = resize_bilinear_with(Backend::Scalar, &src, src_w, src_h, channels, dst_w, dst_h)
            .unwrap();
        assert_eq!(got.len(), want.len());
        let diff = max_abs_diff(&got, &want);
        assert!(
            diff <= 1,
            "{src_w}x{src_h}x{channels} -> {dst_w}x{dst_h}: max diff {diff} vs image crate"
        );
        total_diff += got
            .iter()
            .zip(&want)
            .map(|(a, b)| u64::from(a.abs_diff(*b)))
            .sum::<u64>();
        total_samples += got.len() as u64;
    }
    // Differences should be rare rounding flips, not a systematic bias.
    let mean = total_diff as f64 / total_samples as f64;
    assert!(mean < 0.3, "mean abs diff vs image crate too high: {mean}");
}

#[test]
fn matches_image_crate_on_a_photo_sized_shrink() {
    // A realistic (if modest) thumbnail job on smooth content, so rounding is
    // checked on gradients and not only on noise.
    let (src_w, src_h, dst_w) = (1200, 900, 300);
    let dst_h = fit_height(src_w as u32, src_h as u32, dst_w as u32) as usize;
    let mut rng = Rng::new(19);
    for channels in [1, 3, 4] {
        // Smooth-ish content: a noisy gradient, so neighbouring values differ.
        let src: Vec<u8> = (0..src_w * src_h * channels)
            .map(|i| {
                let x = (i / channels) % src_w;
                let y = (i / channels) / src_w;
                ((x + y) / 16) as u8 ^ (rng.next() as u8 & 0x1f)
            })
            .collect();
        let want = oracle(&src, src_w, src_h, channels, dst_w, dst_h);
        for backend in Backend::available() {
            let got =
                resize_bilinear_with(backend, &src, src_w, src_h, channels, dst_w, dst_h).unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(diff <= 1, "{backend} channels={channels}: max diff {diff}");
        }
    }
}

#[test]
fn simd_backends_are_bit_identical_to_scalar() {
    let backends = Backend::available();
    eprintln!("backends under test: {backends:?}");
    let mut rng = Rng::new(23);
    for _ in 0..400 {
        let (src_w, src_h, channels, dst_w, dst_h) = random_case(&mut rng);
        let src = rng.bytes(src_w * src_h * channels);
        let want =
            resize_bilinear_with(Backend::Scalar, &src, src_w, src_h, channels, dst_w, dst_h)
                .unwrap();
        for &backend in &backends {
            let got =
                resize_bilinear_with(backend, &src, src_w, src_h, channels, dst_w, dst_h).unwrap();
            assert!(
                got == want,
                "{backend} differs from scalar for {src_w}x{src_h}x{channels} -> {dst_w}x{dst_h} \
                 (max diff {})",
                max_abs_diff(&got, &want)
            );
        }
    }
}

#[test]
fn detected_backend_is_used_by_default() {
    let mut rng = Rng::new(29);
    let (src_w, src_h, channels, dst_w, dst_h) = (123, 77, 3, 50, 31);
    let src = rng.bytes(src_w * src_h * channels);
    let auto = resize_bilinear(&src, src_w, src_h, channels, dst_w, dst_h).unwrap();
    let explicit = resize_bilinear_with(
        Backend::detect(),
        &src,
        src_w,
        src_h,
        channels,
        dst_w,
        dst_h,
    )
    .unwrap();
    assert_eq!(auto, explicit);
}

// ---------------------------------------------------------------------------
// File-level glue
// ---------------------------------------------------------------------------

#[test]
fn fit_height_preserves_aspect_ratio_and_never_yields_zero() {
    let mut rng = Rng::new(31);
    for _ in 0..1000 {
        let src_w = rng.range(1, 10_000) as u32;
        let src_h = rng.range(1, 10_000) as u32;
        let width = rng.range(1, 10_000) as u32;
        let height = fit_height(src_w, src_h, width);
        assert!(height >= 1);
        let aspect_in = f64::from(src_h) / f64::from(src_w);
        let aspect_out = f64::from(height) / f64::from(width);
        // Integer rounding of the height can move the aspect ratio by at most
        // one pixel row over the output width (plus the clamp to 1 row).
        let epsilon = 1.0 / f64::from(width) + 1e-9;
        assert!(
            (aspect_out - aspect_in).abs() <= epsilon || height == 1,
            "{src_w}x{src_h} -> {width}x{height}: aspect {aspect_in} vs {aspect_out}"
        );
    }
    assert_eq!(fit_height(4000, 3000, 800), 600);
    assert_eq!(fit_height(4000, 1, 800), 1);
}

#[test]
fn resize_image_keeps_the_native_colour_type_for_8bit_images() {
    let mut rng = Rng::new(37);
    let (w, h) = (40u32, 30u32);
    let cases: Vec<DynamicImage> = vec![
        ImageBuffer::<Luma<u8>, _>::from_raw(w, h, rng.bytes(40 * 30))
            .unwrap()
            .into(),
        ImageBuffer::<LumaA<u8>, _>::from_raw(w, h, rng.bytes(40 * 30 * 2))
            .unwrap()
            .into(),
        ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, rng.bytes(40 * 30 * 3))
            .unwrap()
            .into(),
        ImageBuffer::<Rgba<u8>, _>::from_raw(w, h, rng.bytes(40 * 30 * 4))
            .unwrap()
            .into(),
    ];
    for img in cases {
        let out = resize_image(&img, 16, 12).unwrap();
        assert_eq!((out.width(), out.height()), (16, 12));
        assert_eq!(out.color(), img.color(), "colour type changed by resize");
        let want = oracle(
            img.as_bytes(),
            40,
            30,
            img.color().channel_count().into(),
            16,
            12,
        );
        assert!(max_abs_diff(out.as_bytes(), &want) <= 1);
    }
}

#[test]
fn resize_image_falls_back_for_non_8bit_samples() {
    let img: DynamicImage = ImageBuffer::<Luma<u16>, _>::from_raw(20, 10, vec![1000u16; 200])
        .unwrap()
        .into();
    let out = resize_image(&img, 10, 5).unwrap();
    assert_eq!((out.width(), out.height()), (10, 5));
}

/// Fresh scratch directory under the system temp dir; removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("pxl-test-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn resize_file_writes_an_aspect_preserving_image_with_the_same_name() {
    let tmp = TempDir::new("resize-file");
    let out_dir = tmp.0.join("thumbs");
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut rng = Rng::new(41);
    let input = tmp.0.join("photo.png");
    let src: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(300, 200, rng.bytes(300 * 200 * 3)).unwrap();
    src.save(&input).unwrap();

    resize_file(&input, &out_dir, 120).unwrap();

    let out = image::open(out_dir.join("photo.png")).unwrap();
    assert_eq!((out.width(), out.height()), (120, 80));
    assert_eq!(out.color(), image::ColorType::Rgb8);
    let want = oracle(src.as_raw(), 300, 200, 3, 120, 80);
    assert!(max_abs_diff(out.as_bytes(), &want) <= 1);

    // Upscaling works too, and a missing input is a clean error.
    resize_file(&input, &out_dir, 450).unwrap();
    let out = image::open(out_dir.join("photo.png")).unwrap();
    assert_eq!((out.width(), out.height()), (450, 300));
    assert!(resize_file(&tmp.0.join("missing.png"), &out_dir, 10).is_err());
}
