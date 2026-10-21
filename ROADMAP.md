# Roadmap

## v0.1 (in progress)
- [x] `resize` and `convert` subcommands with parallel batch processing
- [ ] SIMD-accelerated bilinear resize kernel (target: 2× ImageMagick)
- [ ] Benchmark harness + published results
- [ ] `compress --max-size` with quality binary search
- [ ] `watch` mode for folder auto-processing
- [ ] Preset system (`pxl.toml`)
- [ ] Shell completions, man page, CI
- [ ] GitHub Release with prebuilt binaries

## v0.2 (planned)
- [ ] AVIF support
- [ ] Tiled processing for very large images
- [ ] SSIM-targeted compression
