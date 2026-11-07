//! `hyper-pixel-cli` — headless tools around the Hyper Pixel engine.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use hp_core::color::ColorPreset;
use hp_core::eval::Evaluator;
use hp_core::export::{ExportOptions, ImageFormat};
use hp_core::pipeline::{Naming, PhotoOutput, PhotoRegion, Stamp, extract_photos, save_all};
use hp_core::{BackgroundHint, DetectParams, RotatedRect, ScanGeometry, detect_photos};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "hyper-pixel-cli",
    version,
    about = "Hyper Pixel command-line tools"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Bg {
    Auto,
    Dark,
    Light,
}

#[derive(Clone, Copy, ValueEnum)]
enum Color {
    None,
    Auto,
    Restore,
    Vivid,
    Mono,
}

#[derive(Subcommand)]
enum Cmd {
    /// Split existing flatbed scans containing several photos into one file per photo.
    Split {
        inputs: Vec<PathBuf>,
        #[arg(short, long, default_value = "split")]
        out: PathBuf,
        /// Scan resolution if the file does not record it.
        #[arg(long, default_value_t = 300.0)]
        dpi: f64,
        #[arg(long, value_enum, default_value_t = Bg::Auto)]
        background: Bg,
        #[arg(long, value_enum, default_value_t = Color::None)]
        color: Color,
        /// Output extension: jpg, png, tif, jxl, webp, bmp.
        #[arg(long, default_value = "jpg")]
        format: String,
        #[arg(long, default_value_t = 95)]
        quality: u8,
        /// Also write <input>.overlay.png showing detections.
        #[arg(long)]
        overlay: bool,
        /// Write <input>.json with detections (a starting point for ground truth).
        #[arg(long)]
        write_truth: bool,
    },
    /// Score the detector on synthetic scenes.
    EvalSynth {
        #[arg(long, default_value_t = 200)]
        n: u64,
        #[arg(long, default_value_t = 75.0)]
        dpi: f64,
        /// Minimum gap between photos (mm).
        #[arg(long, default_value_t = 5.0)]
        gap: f64,
    },
    /// Score the detector on a corpus directory of `<name>.<img>` + `<name>.json` pairs.
    Eval { dir: PathBuf },
    /// Run OCR on images; writes a searchable PDF and prints the text.
    Ocr {
        inputs: Vec<PathBuf>,
        #[arg(short, long)]
        pdf: Option<PathBuf>,
        #[arg(long, default_value_t = 300.0)]
        dpi: f64,
    },
    /// Download OCR models for offline use.
    FetchModels,
    /// List scanners.
    Devices,
}

/// Ground truth / detection record stored next to corpus images.
#[derive(Serialize, Deserialize)]
struct Truth {
    dpi: f64,
    photos: Vec<RotatedRect>,
}

fn stamp() -> Stamp {
    let now = chrono::Local::now();
    Stamp {
        date: now.format("%Y-%m-%d").to_string(),
        time: now.format("%H%M%S").to_string(),
    }
}

/// Read DPI from PNG pHYs / JPEG JFIF density, if present.
fn file_dpi(path: &Path) -> Option<f64> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.starts_with(b"\x89PNG") {
        let pos = bytes.windows(4).position(|w| w == b"pHYs")?;
        let d = &bytes[pos + 4..pos + 13];
        let ppu = u32::from_be_bytes([d[0], d[1], d[2], d[3]]) as f64;
        return (d[8] == 1 && ppu > 0.0).then_some(ppu * 0.0254);
    }
    if bytes.starts_with(&[0xFF, 0xD8]) {
        let pos = bytes.windows(5).position(|w| w == b"JFIF\0")?;
        let d = &bytes[pos + 5..pos + 12];
        let unit = d[2];
        let x = u16::from_be_bytes([d[3], d[4]]) as f64;
        return match unit {
            1 if x > 1.0 => Some(x),
            2 if x > 1.0 => Some(x * 2.54),
            _ => None,
        };
    }
    None
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    match Cli::parse().cmd {
        Cmd::Split {
            inputs,
            out,
            dpi,
            background,
            color,
            format,
            quality,
            overlay,
            write_truth,
        } => {
            let fmt = ImageFormat::from_extension(&format).context("unknown format")?;
            let params = DetectParams {
                background: match background {
                    Bg::Auto => BackgroundHint::Auto,
                    Bg::Dark => BackgroundHint::Dark,
                    Bg::Light => BackgroundHint::Light,
                },
                ..Default::default()
            };
            let color = match color {
                Color::None => ColorPreset::None,
                Color::Auto => ColorPreset::Auto,
                Color::Restore => ColorPreset::RestoreFaded,
                Color::Vivid => ColorPreset::Vivid,
                Color::Mono => ColorPreset::Monochrome,
            };
            for input in inputs {
                let img =
                    image::open(&input).with_context(|| format!("reading {}", input.display()))?;
                let dpi = file_dpi(&input).unwrap_or(dpi);
                let geo = ScanGeometry::new(dpi);
                let t = std::time::Instant::now();
                let rep = detect_photos(&img, geo, &params);
                log::info!(
                    "{}: {} photo(s) at {dpi} dpi in {:.0} ms",
                    input.display(),
                    rep.photos.len(),
                    t.elapsed().as_secs_f64() * 1e3
                );
                if overlay {
                    let mut o = img.to_rgb8();
                    let rects: Vec<_> = rep.photos.iter().map(|d| geo.mm_to_px(&d.rect)).collect();
                    hp_core::raster::draw_rects(
                        &mut o,
                        &rects,
                        [255, 40, 40],
                        (dpi / 100.0).max(1.0) as u32 * 2 + 1,
                    );
                    o.save(input.with_extension("overlay.png"))?;
                }
                if write_truth {
                    let t = Truth {
                        dpi,
                        photos: rep.photos.iter().map(|d| d.rect).collect(),
                    };
                    std::fs::write(
                        input.with_extension("json"),
                        serde_json::to_string_pretty(&t)?,
                    )?;
                }
                let regions: Vec<PhotoRegion> = rep
                    .photos
                    .iter()
                    .map(|d| PhotoRegion {
                        rect: d.rect,
                        quarter_turns: 0,
                    })
                    .collect();
                let photos = extract_photos(
                    &img,
                    geo,
                    &regions,
                    &PhotoOutput {
                        color,
                        ..Default::default()
                    },
                );
                let stem = input
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("scan")
                    .to_string();
                let naming = Naming {
                    pattern: "{name}_{n}".into(),
                    label: stem,
                };
                let saved = save_all(
                    &photos,
                    &out,
                    &naming,
                    &stamp(),
                    &ExportOptions {
                        format: fmt,
                        jpeg_quality: quality,
                        dpi,
                    },
                )?;
                for p in saved {
                    println!("{}", p.display());
                }
            }
        }
        Cmd::EvalSynth { n, dpi, gap } => {
            let mut ev = Evaluator::new();
            for seed in 0..n {
                let scene = hp_core::synth::render_scene(&hp_core::synth::SceneParams {
                    seed: seed + 1000,
                    dpi,
                    dark_background: seed % 2 == 0,
                    min_gap_mm: gap,
                    ..Default::default()
                });
                let rep = detect_photos(
                    &scene.image,
                    ScanGeometry::new(dpi),
                    &DetectParams::default(),
                );
                ev.add(
                    &scene.truth,
                    &rep.photos.iter().map(|d| d.rect).collect::<Vec<_>>(),
                );
            }
            println!("{}", serde_json::to_string_pretty(ev.stats())?);
        }
        Cmd::Eval { dir } => {
            let mut ev = Evaluator::new();
            let mut entries: Vec<_> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .collect();
            entries.sort();
            for json in entries
                .iter()
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
            {
                let truth: Truth = serde_json::from_str(&std::fs::read_to_string(json)?)?;
                let Some(img_path) = entries.iter().find(|p| {
                    p.file_stem() == json.file_stem()
                        && *p != json
                        && !p.to_string_lossy().contains(".overlay")
                }) else {
                    continue;
                };
                let img = image::open(img_path)?;
                let rep =
                    detect_photos(&img, ScanGeometry::new(truth.dpi), &DetectParams::default());
                let det: Vec<_> = rep.photos.iter().map(|d| d.rect).collect();
                let before = ev.stats().perfect_scenes;
                ev.add(&truth.photos, &det);
                let ok = ev.stats().perfect_scenes > before;
                println!(
                    "{} {}",
                    if ok { "ok  " } else { "FAIL" },
                    img_path.display()
                );
            }
            println!("{}", serde_json::to_string_pretty(ev.stats())?);
        }
        Cmd::Ocr { inputs, pdf, dpi } => {
            let ocr = hp_ocr::Ocr::load_default()
                .context("OCR models missing — run `hyper-pixel-cli fetch-models`")?;
            let mut pages = Vec::new();
            for input in &inputs {
                let img = image::open(input)?;
                let t = std::time::Instant::now();
                let page = ocr.recognize(&img)?;
                log::info!(
                    "{}: {} words in {:.1} s",
                    input.display(),
                    page.words.len(),
                    t.elapsed().as_secs_f64()
                );
                print!("{}", page.text);
                pages.push((img, page.words));
            }
            if let Some(pdf) = pdf {
                let bytes = hp_core::export::pdf_document(
                    &pages,
                    &ExportOptions {
                        format: ImageFormat::Pdf,
                        jpeg_quality: 85,
                        dpi,
                    },
                    "OCR",
                )?;
                std::fs::write(&pdf, bytes)?;
                log::info!("wrote {}", pdf.display());
            }
        }
        Cmd::FetchModels => {
            let dir = hp_ocr::download_models(&mut |d, t| {
                eprint!("\r{:.1}/{:.1} MB", d as f64 / 1e6, t as f64 / 1e6)
            })?;
            eprintln!("\nmodels in {}", dir.display());
        }
        Cmd::Devices => {
            let (mut b, err) = hp_scan::default_backend();
            if let Some(e) = err {
                eprintln!("warning: {e}");
            }
            let devs = b.list_devices()?;
            if devs.is_empty() {
                bail!("no scanners found");
            }
            for d in devs {
                println!("{}\t{}", d.name, d.label());
            }
        }
    }
    Ok(())
}
