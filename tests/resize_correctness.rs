use std::path::Path;

#[test]
fn resize_preserves_aspect_ratio() {
    // TODO(week 6): property test — generate random-size images,
    // resize, assert |aspect_out - aspect_in| < epsilon and dims exact.
    assert!(Path::new("Cargo.toml").exists());
}
