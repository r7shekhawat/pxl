use anyhow::{bail, Context, Result};
use std::path::Path;

pub fn convert_file(input: &Path, to: &str) -> Result<()> {
    let img = image::open(input).with_context(|| format!("opening {}", input.display()))?;
    let stem = input.file_stem().unwrap().to_string_lossy();
    let out = Path::new("converted").join(format!("{stem}.{to}"));
    match to {
        "png" => img.save_with_format(&out, image::ImageFormat::Png)?,
        "jpg" | "jpeg" => img.save_with_format(&out, image::ImageFormat::Jpeg)?,
        "webp" => img.save_with_format(&out, image::ImageFormat::WebP)?,
        other => bail!("unsupported format: {other}"),
    }
    Ok(())
}
