//! `hyper-pixel-cli` — headless tools around the Hyper Pixel engine.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use hp_core::color::ColorPreset;
use hp_core::eval::Evaluator;
use hp_core::export::{ExportOptions, ImageFormat};
use hp_core::pipeline::{
    Naming, PhotoOutput, PhotoRegion, Stamp, enhance_photo, extract_photos_with, save_all,
};
use hp_core::restore::DustLevel;
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

#[derive(Clone, Copy, ValueEnum)]
enum Dust {
    Off,
    Light,
    Normal,
    Strong,
}

impl From<Dust> for DustLevel {
    fn from(d: Dust) -> Self {
        match d {
            Dust::Off => DustLevel::Off,
            Dust::Light => DustLevel::Light,
            Dust::Normal => DustLevel::Normal,
            Dust::Strong => DustLevel::Strong,
        }
    }
}

impl From<Color> for ColorPreset {
    fn from(c: Color) -> Self {
        match c {
            Color::None => ColorPreset::None,
            Color::Auto => ColorPreset::Auto,
            Color::Restore => ColorPreset::RestoreFaded,
            Color::Vivid => ColorPreset::Vivid,
            Color::Mono => ColorPreset::Monochrome,
        }
    }
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Models {
    /// OCR models (~12 MB).
    Ocr,
    /// LaMa inpainting and DDColor colourisation (~410 MB).
    Ai,
    All,
}

/// Restoration options shared by `split` and `restore`.
#[derive(clap::Args, Clone)]
struct RestoreArgs {
    /// Dust and scratch removal.
    #[arg(long, value_enum, default_value_t = Dust::Off)]
    dust: Dust,
    /// Fill large defects with the LaMa model (needs `fetch-models ai`).
    #[arg(long)]
    ai_fill: bool,
    /// Colourise black-and-white photos with DDColor (needs `fetch-models ai`).
    #[arg(long)]
    colorize: bool,
}

impl RestoreArgs {
    fn enhancer(&self) -> Result<hp_ml::MlEnhancer> {
        let e = hp_ml::MlEnhancer::load(self.ai_fill, self.colorize);
        if (self.ai_fill && e.inpainter.is_none()) || (self.colorize && e.colorizer.is_none()) {
            bail!("AI models missing — run `hyper-pixel-cli fetch-models ai`");
        }
        Ok(e)
    }
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
        #[command(flatten)]
        restore: RestoreArgs,
        /// Output extension: jpg, png, tif, jxl, webp, bmp.
        #[arg(long, default_value = "jpg")]
        format: String,
        #[arg(long, default_value_t = 95)]
        quality: u8,
        /// Also write <input>.overlay.png showing detections.
        #[arg(long)]
        overlay: bool,
        /// Write <input>.json with detections (a starting point for ground truth); an existing
        /// file is left alone.
        #[arg(long)]
        write_truth: bool,
    },
    /// Restore single photos (already cropped scans or old files): dust and scratch
    /// removal, colour correction, colourisation. Writes `<out>/<name>.<format>`.
    Restore {
        inputs: Vec<PathBuf>,
        #[arg(short, long, default_value = "restored")]
        out: PathBuf,
        /// Resolution if the file does not record it (dust sizes are in millimetres).
        #[arg(long, default_value_t = 600.0)]
        dpi: f64,
        #[arg(long, value_enum, default_value_t = Color::None)]
        color: Color,
        #[command(flatten)]
        restore: RestoreArgs,
        /// Output extension: jpg, png, tif, jxl, webp, bmp.
        #[arg(long, default_value = "png")]
        format: String,
        #[arg(long, default_value_t = 95)]
        quality: u8,
        /// Also write `<name>.defects.png` with the detected defects in red.
        #[arg(long)]
        show_defects: bool,
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
    Eval {
        dir: PathBuf,
        /// Downscale each scan to this resolution first (simulates a low-dpi preview).
        #[arg(long)]
        at_dpi: Option<f64>,
    },
    /// Run OCR on images; writes a searchable PDF and prints the text.
    Ocr {
        inputs: Vec<PathBuf>,
        #[arg(short, long)]
        pdf: Option<PathBuf>,
        #[arg(long, default_value_t = 300.0)]
        dpi: f64,
    },
    /// Download models for offline use.
    FetchModels {
        #[arg(value_enum, default_value_t = Models::Ocr)]
        which: Models,
    },
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
            restore,
            format,
            quality,
            overlay,
            write_truth,
        } => {
            let enhancer = restore.enhancer()?;
            let fmt = ImageFormat::from_extension(&format).context("unknown format")?;
            let params = DetectParams {
                background: match background {
                    Bg::Auto => BackgroundHint::Auto,
                    Bg::Dark => BackgroundHint::Dark,
                    Bg::Light => BackgroundHint::Light,
                },
                ..Default::default()
            };
            let output = PhotoOutput {
                color: color.into(),
                dust: restore.dust.into(),
                colorize: restore.colorize,
                ..Default::default()
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
                    // Never clobber a (possibly hand-corrected) ground-truth file.
                    let json = input.with_extension("json");
                    if json.exists() {
                        log::warn!("{} exists, not overwriting it", json.display());
                    } else {
                        std::fs::write(&json, serde_json::to_string_pretty(&t)?)?;
                    }
                }
                let regions: Vec<PhotoRegion> = rep
                    .photos
                    .iter()
                    .map(|d| PhotoRegion {
                        rect: d.rect,
                        quarter_turns: 0,
                    })
                    .collect();
                let photos = extract_photos_with(&img, geo, &regions, &output, Some(&enhancer));
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
        Cmd::Restore {
            inputs,
            out,
            dpi,
            color,
            restore,
            format,
            quality,
            show_defects,
        } => {
            let fmt = ImageFormat::from_extension(&format).context("unknown format")?;
            let enhancer = restore.enhancer()?;
            let output = PhotoOutput {
                color: color.into(),
                dust: restore.dust.into(),
                colorize: restore.colorize,
                ..Default::default()
            };
            std::fs::create_dir_all(&out)?;
            for input in inputs {
                let img =
                    image::open(&input).with_context(|| format!("reading {}", input.display()))?;
                let dpi = file_dpi(&input).unwrap_or(dpi);
                let stem = input
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("photo");
                let t = std::time::Instant::now();
                if show_defects && let Some(p) = output.dust.params() {
                    let d = hp_core::restore::find_defects(&img, dpi, &p);
                    log::info!("{}: {} defect(s)", input.display(), d.items.len());
                    hp_core::restore::overlay(&img, &d.mask)
                        .save(out.join(format!("{stem}.defects.png")))?;
                }
                let res = enhance_photo(img, dpi, &output, Some(&enhancer));
                let dest = out.join(format!("{stem}.{}", fmt.extension()));
                hp_core::export::save(
                    &res,
                    &dest,
                    &ExportOptions {
                        format: fmt,
                        jpeg_quality: quality,
                        dpi,
                    },
                )?;
                log::info!(
                    "{} -> {} in {:.1} s",
                    input.display(),
                    dest.display(),
                    t.elapsed().as_secs_f64()
                );
                println!("{}", dest.display());
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
        Cmd::Eval { dir, at_dpi } => {
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
                let mut img = image::open(img_path)?;
                let mut dpi = truth.dpi;
                if let Some(target) = at_dpi.filter(|&d| d < truth.dpi) {
                    let s = target / truth.dpi;
                    let orig_w = img.width() as f64;
                    img = img.resize_exact(
                        (img.width() as f64 * s).round() as u32,
                        (img.height() as f64 * s).round() as u32,
                        image::imageops::FilterType::Triangle,
                    );
                    dpi = truth.dpi * img.width() as f64 / orig_w;
                }
                let t0 = std::time::Instant::now();
                let rep = detect_photos(&img, ScanGeometry::new(dpi), &DetectParams::default());
                log::info!("detected in {} ms", t0.elapsed().as_millis());
                let det: Vec<_> = rep.photos.iter().map(|d| d.rect).collect();
                let before = ev.stats().perfect_scenes;
                ev.add(&truth.photos, &det);
                let ok = ev.stats().perfect_scenes > before;
                println!(
                    "{} {} ({} truth, {} detected)",
                    if ok { "ok  " } else { "FAIL" },
                    img_path.display(),
                    truth.photos.len(),
                    det.len()
                );
                for (i, t) in truth.photos.iter().enumerate() {
                    let best = det
                        .iter()
                        .map(|d| (t.iou(d), hp_core::eval::corner_error(t, d)))
                        .max_by(|a, b| a.0.total_cmp(&b.0));
                    match best {
                        Some((iou, ce)) => {
                            println!("    photo {i}: IoU {iou:.4}, worst corner {ce:.2} mm")
                        }
                        None => println!("    photo {i}: not detected"),
                    }
                }
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
        Cmd::FetchModels { which } => {
            let progress =
                &mut |d: u64, t: u64| eprint!("\r{:.1}/{:.1} MB", d as f64 / 1e6, t as f64 / 1e6);
            if which != Models::Ai {
                let dir = hp_ocr::download_models(progress)?;
                eprintln!("\nOCR models in {}", dir.display());
            }
            if which != Models::Ocr {
                for spec in [hp_ml::LAMA, hp_ml::DDCOLOR] {
                    eprintln!("{} ({})", spec.label, spec.licence);
                    let dir = hp_ml::models::download(&spec, progress)?;
                    eprintln!("\n{} in {}", spec.label, dir.display());
                }
            }
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
