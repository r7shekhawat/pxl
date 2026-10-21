use anyhow::{Context, Result};
use std::path::Path;

/// Resize a single image to `width`, preserving aspect ratio.
/// TODO(week 2): replace this naive image::resize with a custom SIMD kernel.
pub fn resize_file(input: &Path, out_dir: &Path, width: u32) -> Result<()> {
    let img = image::open(input).with_context(|| format!("opening {}", input.display()))?;
    let height = ((img.height() as u64 * width as u64) / img.width().max(1) as u64) as u32;
    let resized = image::imageops::resize(&img, width, height, image::imageops::FilterType::Triangle);
    let out = out_dir.join(input.file_name().unwrap());
    resized.save(&out)?;
    Ok(())
}
