# pxl

A fast batch image processor for the command line. Resize, convert, and
compress images with SIMD-accelerated kernels and parallel execution —
built because ImageMagick was too slow for my photo workflow.

## Status
Early development. See [ROADMAP.md](ROADMAP.md) for planned work.

## Install
```sh
cargo install --git https://github.com/r7shekhawat/pxl
```

## Usage
```sh
pxl resize ./photos --out ./thumbs --width 800
pxl convert ./photos --to webp
```

## Resize kernel
`pxl resize` uses its own separable bilinear (triangle-filter) resampler,
`pxl::kernel::resize_bilinear`, rather than the `image` crate's generic path:

* 14-bit fixed-point weights, `i32` accumulation, 8-bit L / LA / RGB / RGBA
  in their native layout (no RGBA round trip).
* Anti-aliased downscaling: the filter footprint widens with the shrink
  ratio, matching `image::FilterType::Triangle` / Pillow `BILINEAR` to within
  1 LSB.
* Runtime-dispatched backends: AVX2 (x86/x86_64), NEON (AArch64) and a
  portable fallback, all bit-identical to each other.

```sh
cargo test                 # correctness: oracle, properties, backend equivalence
cargo bench --bench resize # image crate vs. pxl backends, 4000x3000 -> 800 wide
```

## License
MIT
