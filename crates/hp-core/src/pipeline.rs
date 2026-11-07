//! Shared "scan → files" logic used by both the GUI and the CLI.

use crate::color::ColorPreset;
use crate::crop;
use crate::detect::ScanGeometry;
use crate::export::{self, ExportError, ExportOptions, ImageFormat};
use crate::geom::RotatedRect;
use image::DynamicImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One photo to extract from a scan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhotoRegion {
    /// Bed millimetres.
    pub rect: RotatedRect,
    /// Extra clockwise quarter turns applied after straightening.
    pub quarter_turns: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotoOutput {
    pub color: ColorPreset,
    /// Trim this much from each edge (mm) so no background shows.
    pub inset_mm: f64,
}

impl Default for PhotoOutput {
    fn default() -> Self {
        Self {
            color: ColorPreset::None,
            inset_mm: 0.4,
        }
    }
}

/// Extract, straighten and colour-correct each region (in parallel).
pub fn extract_photos(
    img: &DynamicImage,
    geo: ScanGeometry,
    regions: &[PhotoRegion],
    out: &PhotoOutput,
) -> Vec<DynamicImage> {
    regions
        .par_iter()
        .map(|r| {
            let mut p = crop::extract(img, geo, &r.rect, out.inset_mm);
            if let Some(cp) = out.color.params() {
                p = crate::color::apply(&p, &cp);
            }
            crop::rotate_quarter(&p, r.quarter_turns)
        })
        .collect()
}

/// File naming. Tokens: `{date}` (YYYY-MM-DD), `{time}` (HHMMSS), `{n}` (4-digit counter),
/// `{name}` (a free label, e.g. album name).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Naming {
    pub pattern: String,
    pub label: String,
}

impl Default for Naming {
    fn default() -> Self {
        Self {
            pattern: "{date}_{n}".into(),
            label: "scan".into(),
        }
    }
}

/// Local date/time strings for naming, supplied by the caller (keeps hp-core clock-free).
#[derive(Debug, Clone)]
pub struct Stamp {
    pub date: String,
    pub time: String,
}

fn render_name(n: &Naming, stamp: &Stamp, counter: u32) -> String {
    let s = n
        .pattern
        .replace("{date}", &stamp.date)
        .replace("{time}", &stamp.time)
        .replace("{name}", &n.label)
        .replace("{n}", &format!("{counter:04}"));
    s.chars()
        .map(|c| if c == '/' || c == '\0' { '_' } else { c })
        .collect()
}

/// Find the next path that doesn't exist yet, starting the counter at `start`.
/// Returns the path and the counter used.
pub fn next_free_path(
    dir: &Path,
    n: &Naming,
    stamp: &Stamp,
    ext: &str,
    start: u32,
) -> (PathBuf, u32) {
    let mut c = start.max(1);
    loop {
        let p = dir.join(format!("{}.{ext}", render_name(n, stamp, c)));
        // If the pattern has no counter, append one on collision.
        let p = if !n.pattern.contains("{n}") && c > 1 {
            dir.join(format!("{}_{c}.{ext}", render_name(n, stamp, c)))
        } else {
            p
        };
        if !p.exists() {
            return (p, c);
        }
        c += 1;
    }
}

/// Save images with sequential names. Returns the written paths.
pub fn save_all(
    images: &[DynamicImage],
    dir: &Path,
    naming: &Naming,
    stamp: &Stamp,
    opts: &ExportOptions,
) -> Result<Vec<PathBuf>, ExportError> {
    let mut out = Vec::new();
    let mut counter = 1;
    for img in images {
        let (path, c) = next_free_path(dir, naming, stamp, opts.format.extension(), counter);
        counter = c + 1;
        // Reserve the name immediately so parallel encoders cannot collide.
        export::write_atomic(&path, &[])?;
        out.push((path, img));
    }
    out.par_iter()
        .map(|(p, img)| export::save(img, p, opts))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(out.into_iter().map(|(p, _)| p).collect())
}

/// Recommended defaults for the user's hardware (CanoScan LiDE 400 class flatbed).
pub mod presets {
    use super::*;
    /// Photo prints: 600 DPI is beyond what consumer prints resolve; JPEG q95 keeps files
    /// small enough for thousands of photos while being visually lossless.
    pub fn photos() -> (f64, ExportOptions) {
        (
            600.0,
            ExportOptions {
                format: ImageFormat::Jpeg,
                jpeg_quality: 95,
                dpi: 600.0,
            },
        )
    }
    /// Archival: lossless 16-bit, larger files.
    pub fn archival() -> (f64, ExportOptions) {
        (
            1200.0,
            ExportOptions {
                format: ImageFormat::Tiff,
                jpeg_quality: 95,
                dpi: 1200.0,
            },
        )
    }
    /// Documents: 300 DPI searchable PDF.
    pub fn documents() -> (f64, ExportOptions) {
        (
            300.0,
            ExportOptions {
                format: ImageFormat::Pdf,
                jpeg_quality: 85,
                dpi: 300.0,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naming_and_collisions() {
        let dir = std::env::temp_dir().join(format!("hp-naming-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = Naming::default();
        let st = Stamp {
            date: "2026-11-07".into(),
            time: "101500".into(),
        };
        let (p, c) = next_free_path(&dir, &n, &st, "jpg", 1);
        assert_eq!(p.file_name().unwrap(), "2026-11-07_0001.jpg");
        std::fs::write(&p, b"x").unwrap();
        let (p2, c2) = next_free_path(&dir, &n, &st, "jpg", c);
        assert_eq!(p2.file_name().unwrap(), "2026-11-07_0002.jpg");
        assert_eq!(c2, 2);
        let imgs = vec![DynamicImage::new_rgb8(4, 4), DynamicImage::new_rgb8(4, 4)];
        let saved = save_all(
            &imgs,
            &dir,
            &n,
            &st,
            &ExportOptions {
                format: ImageFormat::Png,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(saved.len(), 2);
        assert!(
            saved[0]
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with("0001.png")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
