//! Release builds embed the OCR models: set `HYPER_PIXEL_BUNDLE_OCR` to a directory holding
//! `text-detection.onnx` and `text-recognition.onnx` (see `packaging/build-release.sh`).
//! Without it the models are looked up on disk at runtime.

use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-env-changed=HYPER_PIXEL_BUNDLE_OCR");
    println!("cargo::rustc-check-cfg=cfg(hp_bundled_ocr)");
    let Some(dir) = std::env::var_os("HYPER_PIXEL_BUNDLE_OCR").filter(|d| !d.is_empty()) else {
        return;
    };
    let dir = std::fs::canonicalize(PathBuf::from(&dir))
        .unwrap_or_else(|e| panic!("HYPER_PIXEL_BUNDLE_OCR={}: {e}", dir.to_string_lossy()));
    for (var, file) in [
        ("HP_OCR_DETECTION", "text-detection.onnx"),
        ("HP_OCR_RECOGNITION", "text-recognition.onnx"),
    ] {
        let path = dir.join(file);
        assert!(path.is_file(), "missing {}", path.display());
        println!("cargo::rerun-if-changed={}", path.display());
        println!("cargo::rustc-env={var}={}", path.display());
    }
    println!("cargo::rustc-cfg=hp_bundled_ocr");
}
