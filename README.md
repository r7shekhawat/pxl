# pxl

A fast batch image processor for the command line. Resize, convert, and
compress images with SIMD-accelerated kernels and parallel execution —
built because ImageMagick was too slow for my photo workflow.

## Status
Early development. See [ROADMAP.md](ROADMAP.md) for planned work.

## Install
```sh
cargo install --git https://github.com/YOUR_USERNAME/pxl
```

## Usage
```sh
pxl resize ./photos --out ./thumbs --width 800
pxl convert ./photos --to webp
```

## License
MIT
