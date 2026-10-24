use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

pub fn collect_images(input: &str) -> Result<Vec<PathBuf>> {
    let p = Path::new(input);
    if p.is_file() {
        return Ok(vec![p.to_path_buf()]);
    }
    let mut v = Vec::new();
    for e in fs::read_dir(p).with_context(|| format!("reading {}", input))? {
        let e = e?;
        if is_image(e.path()) {
            v.push(e.path());
        }
    }
    Ok(v)
}

fn is_image(p: PathBuf) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
            .as_str(),
        "png" | "jpg" | "jpeg" | "webp"
    )
}

pub fn run_resize(input: &str, out: &str, width: u32) -> Result<()> {
    let files = collect_images(input)?;
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::with_template("{bar:40} {pos}/{len} {msg}")?);
    fs::create_dir_all(out)?;
    files.par_iter().for_each(|f| {
        if let Err(e) = crate::ops::resize::resize_file(f, Path::new(out), width) {
            pb.suspend(|| eprintln!("{}: {}", f.display(), e));
        }
        pb.inc(1);
    });
    pb.finish_with_message("done");
    Ok(())
}

pub fn run_convert(input: &str, to: &str) -> Result<()> {
    let files = collect_images(input)?;
    let pb = ProgressBar::new(files.len() as u64);
    fs::create_dir_all("converted")?;
    files.par_iter().for_each(|f| {
        if let Err(e) = crate::ops::convert::convert_file(f, to) {
            pb.suspend(|| eprintln!("{}: {}", f.display(), e));
        }
        pb.inc(1);
    });
    pb.finish_with_message("done");
    Ok(())
}
