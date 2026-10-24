//! Resize throughput: `image`'s triangle filter vs. the pxl kernel backends.
//!
//! Uses `benches/fixtures/sample.png` if present, otherwise a synthetic
//! 4000x3000 RGB image, so the benchmark runs out of the box.
//!
//! ```sh
//! cargo bench --bench resize
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgb};
use pxl::kernel::{resize_bilinear_with, Backend};
use pxl::ops::resize::fit_height;

fn load_sample() -> DynamicImage {
    if let Ok(img) = image::open("benches/fixtures/sample.png") {
        return img;
    }
    let (w, h) = (4000u32, 3000u32);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let img = ImageBuffer::from_fn(w, h, |x, y| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state & 0x0f) as u8;
        Rgb([
            (x / 16) as u8 ^ noise,
            (y / 12) as u8 ^ noise,
            ((x + y) / 32) as u8,
        ])
    });
    DynamicImage::ImageRgb8(img)
}

fn bench_resize(c: &mut Criterion) {
    let img = load_sample();
    let (w, h) = (img.width(), img.height());

    for &(name, dst_w) in &[("thumb_800w", 800u32), ("upscale_1.5x", w * 3 / 2)] {
        let dst_h = fit_height(w, h, dst_w);
        let mut group = c.benchmark_group(name);
        group.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
        group.sample_size(20);

        for (label, img) in [
            ("rgb", DynamicImage::ImageRgb8(img.to_rgb8())),
            ("rgba", DynamicImage::ImageRgba8(img.to_rgba8())),
        ] {
            group.bench_with_input(BenchmarkId::new("image_triangle", label), &img, |b, img| {
                b.iter(|| {
                    black_box(image::imageops::resize(
                        black_box(img),
                        dst_w,
                        dst_h,
                        FilterType::Triangle,
                    ))
                })
            });

            let channels = usize::from(img.color().channel_count());
            let bytes = img.as_bytes();
            for backend in Backend::available() {
                group.bench_with_input(
                    BenchmarkId::new(format!("pxl_{backend}"), label),
                    &bytes,
                    |b, bytes| {
                        b.iter(|| {
                            black_box(
                                resize_bilinear_with(
                                    backend,
                                    black_box(bytes),
                                    w as usize,
                                    h as usize,
                                    channels,
                                    dst_w as usize,
                                    dst_h as usize,
                                )
                                .unwrap(),
                            )
                        })
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
