use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Pixel};
use std::path::Path;

use crate::kernel::resize_bilinear;

/// Resize a single image file to `width`, preserving aspect ratio, and write
/// it to `out_dir` under the same file name.
pub fn resize_file(input: &Path, out_dir: &Path, width: u32) -> Result<()> {
    let img = image::open(input).with_context(|| format!("opening {}", input.display()))?;
    let height = fit_height(img.width(), img.height(), width);
    let resized = resize_image(&img, width, height)
        .with_context(|| format!("resizing {}", input.display()))?;
    let out = out_dir.join(input.file_name().context("input path has no file name")?);
    resized
        .save(&out)
        .with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

/// Height that preserves the aspect ratio of a `src_w` x `src_h` image when
/// it is scaled to `width` (rounded down, but never below 1).
pub fn fit_height(src_w: u32, src_h: u32, width: u32) -> u32 {
    let h = u64::from(src_h) * u64::from(width) / u64::from(src_w.max(1));
    h.clamp(1, u64::from(u32::MAX)) as u32
}

/// Resize a decoded image to exactly `width` x `height` with the bilinear
/// kernel.
///
/// 8-bit images (L, LA, RGB, RGBA) go through the SIMD kernel in their native
/// layout. Other sample formats (16-bit, float) are rare for this tool's
/// purpose and fall back to the `image` crate's triangle filter, which
/// converts to RGBA8.
pub fn resize_image(img: &DynamicImage, width: u32, height: u32) -> Result<DynamicImage> {
    Ok(match img {
        DynamicImage::ImageLuma8(b) => resize_buffer(b, width, height)?.into(),
        DynamicImage::ImageLumaA8(b) => resize_buffer(b, width, height)?.into(),
        DynamicImage::ImageRgb8(b) => resize_buffer(b, width, height)?.into(),
        DynamicImage::ImageRgba8(b) => resize_buffer(b, width, height)?.into(),
        other => {
            image::imageops::resize(other, width, height, image::imageops::FilterType::Triangle)
                .into()
        }
    })
}

/// Resize one 8-bit `ImageBuffer` with the kernel, keeping its pixel type.
pub fn resize_buffer<P>(
    img: &ImageBuffer<P, Vec<u8>>,
    width: u32,
    height: u32,
) -> Result<ImageBuffer<P, Vec<u8>>>
where
    P: Pixel<Subpixel = u8> + 'static,
{
    let channels = usize::from(P::CHANNEL_COUNT);
    let data = resize_bilinear(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        channels,
        width as usize,
        height as usize,
    )?;
    ImageBuffer::from_raw(width, height, data).context("resized buffer has unexpected size")
}
