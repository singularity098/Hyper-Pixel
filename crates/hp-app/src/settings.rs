//! Persisted user settings (~/.config/hyper-pixel/settings.json).

use hp_core::BackgroundHint;
use hp_core::color::ColorPreset;
use hp_core::document::DocColorMode;
use hp_core::export::ImageFormat;
use hp_core::restore::DustLevel;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Mode {
    #[default]
    Photos,
    Documents,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PhotoColor {
    #[default]
    Color,
    Gray,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub device: String,
    pub mode: Mode,
    pub photo_dpi: f64,
    pub doc_dpi: f64,
    pub preview_dpi: f64,
    pub photo_color: PhotoColor,
    pub source: String,
    pub background: BackgroundHint,
    pub auto_detect: bool,
    pub color_preset: ColorPreset,
    /// Dust and scratch removal on saved photos.
    pub dust: DustLevel,
    /// LaMa for large defects (needs the AI models).
    pub ai_fill: bool,
    /// Colourise black-and-white photos (needs the AI models).
    pub colorize: bool,
    pub deskew: bool,
    pub whiten: bool,
    pub doc_mode: DocColorMode,
    pub ocr: bool,
    pub output_dir: PathBuf,
    pub photo_format: ImageFormat,
    pub doc_format: ImageFormat,
    pub jpeg_quality: u8,
    pub high_bit_depth: bool,
    pub name_pattern: String,
    pub inset_mm: f64,
    /// Show a magnifier loupe while placing crop-box corners.
    pub magnifier: bool,
    /// Loupe magnification relative to the on-screen preview.
    pub loupe_zoom: f64,
}

impl Default for Settings {
    fn default() -> Self {
        let pictures = dirs::picture_dir()
            .or_else(dirs::home_dir)
            .unwrap_or_else(|| PathBuf::from("."));
        Self {
            device: String::new(),
            mode: Mode::Photos,
            // CanoScan LiDE 400: 600 dpi captures everything a photo print holds; 300 dpi
            // is the sweet spot for text/OCR. Previews at 150 dpi are still quick and show
            // the thin gaps between touching prints, which 75 dpi does not.
            photo_dpi: 600.0,
            doc_dpi: 300.0,
            preview_dpi: PREVIEW_DPI,
            photo_color: PhotoColor::Color,
            source: String::new(),
            background: BackgroundHint::Auto,
            auto_detect: true,
            color_preset: ColorPreset::None,
            dust: DustLevel::Off,
            ai_fill: false,
            colorize: false,
            deskew: true,
            whiten: true,
            doc_mode: DocColorMode::Color,
            ocr: true,
            output_dir: pictures.join("Hyper Pixel"),
            photo_format: ImageFormat::Jpeg,
            doc_format: ImageFormat::Pdf,
            jpeg_quality: 95,
            high_bit_depth: false,
            name_pattern: "{date}_{n}".into(),
            inset_mm: 0.4,
            magnifier: true,
            loupe_zoom: 4.0,
        }
    }
}

fn path() -> Option<PathBuf> {
    Some(
        dirs::config_dir()?
            .join("hyper-pixel")
            .join("settings.json"),
    )
}

/// Default preview resolution.
const PREVIEW_DPI: f64 = 150.0;
/// Preview resolution saved by older versions as their default (not user-adjustable then).
const OLD_PREVIEW_DPI: f64 = 75.0;

impl Settings {
    pub fn load() -> Self {
        let mut s: Settings = path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if s.preview_dpi == OLD_PREVIEW_DPI {
            s.preview_dpi = PREVIEW_DPI;
        }
        s
    }

    pub fn save(&self) {
        let Some(p) = path() else { return };
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(p, s);
        }
    }
}
