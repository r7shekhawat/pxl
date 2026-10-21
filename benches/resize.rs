use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::{imageops::FilterType, GenericImageView};

fn bench_naive_resize(c: &mut Criterion) {
    // Placeholder benchmark — replace with your SIMD kernel in week 2.
    // Compare against the `image` crate baseline and, externally, ImageMagick.
    let img = image::open("benches/fixtures/sample.png").expect("fixture");
    c.bench_function("resize_triangle_800w", |b| {
        b.iter(|| {
            let (w, h) = img.dimensions();
            let h2 = (h as u64 * 800 / w as u64) as u32;
            black_box(image::imageops::resize(black_box(&img), 800, h2, FilterType::Triangle));
        })
    });
}

criterion_group!(benches, bench_naive_resize);
criterion_main!(benches);
