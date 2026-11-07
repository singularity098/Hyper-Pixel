//! Optical character recognition using the pure-Rust `ocrs` engine.
//!
//! Models (~12 MB) are looked up in, in order:
//! 1. `$HYPER_PIXEL_MODELS`
//! 2. `$APPDIR/usr/share/hyper-pixel/models` (bundled inside the AppImage)
//! 3. `<exe dir>/../share/hyper-pixel/models`
//! 4. `$XDG_DATA_HOME/hyper-pixel/models` (downloaded on demand)

use hp_core::pdf::TextWord;
use image::DynamicImage;
use ocrs::{ImageSource, OcrEngine, OcrEngineParams, TextItem};
use std::path::{Path, PathBuf};

pub const DETECTION_MODEL: &str = "text-detection.onnx";
pub const RECOGNITION_MODEL: &str = "text-recognition.onnx";
const MODEL_BASE_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com";

#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error("OCR models not found; download them first")]
    ModelsMissing,
    #[error("model download failed: {0}")]
    Download(String),
    #[error("OCR failed: {0}")]
    Engine(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, OcrError>;

fn has_models(dir: &Path) -> bool {
    dir.join(DETECTION_MODEL).is_file() && dir.join(RECOGNITION_MODEL).is_file()
}

/// Where downloaded models are stored.
pub fn user_model_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hyper-pixel")
        .join("models")
}

/// Find a directory containing both models.
pub fn find_models() -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("HYPER_PIXEL_MODELS") {
        cands.push(p.into());
    }
    if let Some(p) = std::env::var_os("APPDIR") {
        cands.push(PathBuf::from(p).join("usr/share/hyper-pixel/models"));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(bin) = exe.parent()
    {
        cands.push(bin.join("../share/hyper-pixel/models"));
    }
    cands.push(user_model_dir());
    cands.into_iter().find(|d| has_models(d))
}

/// Download the models into [`user_model_dir`]. `progress` receives (bytes done, total).
pub fn download_models(progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    let dir = user_model_dir();
    std::fs::create_dir_all(&dir)?;
    for name in [DETECTION_MODEL, RECOGNITION_MODEL] {
        let dest = dir.join(name);
        if dest.is_file() {
            continue;
        }
        let url = format!("{MODEL_BASE_URL}/{name}");
        log::info!("downloading {url}");
        let mut resp = ureq::get(&url)
            .call()
            .map_err(|e| OcrError::Download(e.to_string()))?;
        let total = resp.body().content_length().unwrap_or(0);
        let mut reader = resp.body_mut().as_reader();
        let tmp = dest.with_extension("part");
        let mut file = std::fs::File::create(&tmp)?;
        let mut buf = vec![0u8; 64 * 1024];
        let mut done = 0u64;
        loop {
            let n = std::io::Read::read(&mut reader, &mut buf)?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut file, &buf[..n])?;
            done += n as u64;
            progress(done, total);
        }
        drop(file);
        std::fs::rename(&tmp, &dest)?;
    }
    Ok(dir)
}

/// Text found on one page.
#[derive(Debug, Clone, Default)]
pub struct OcrPage {
    /// Words with pixel boxes, for a searchable PDF layer.
    pub words: Vec<TextWord>,
    /// Plain text, one line per recognised line.
    pub text: String,
}

pub struct Ocr {
    engine: OcrEngine,
}

impl Ocr {
    pub fn load(dir: &Path) -> Result<Self> {
        if !has_models(dir) {
            return Err(OcrError::ModelsMissing);
        }
        let det = rten::Model::load_file(dir.join(DETECTION_MODEL))
            .map_err(|e| OcrError::Engine(e.to_string()))?;
        let rec = rten::Model::load_file(dir.join(RECOGNITION_MODEL))
            .map_err(|e| OcrError::Engine(e.to_string()))?;
        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(det),
            recognition_model: Some(rec),
            ..Default::default()
        })
        .map_err(|e| OcrError::Engine(e.to_string()))?;
        Ok(Self { engine })
    }

    /// Load from the first location that has models.
    pub fn load_default() -> Result<Self> {
        Self::load(&find_models().ok_or(OcrError::ModelsMissing)?)
    }

    pub fn recognize(&self, img: &DynamicImage) -> Result<OcrPage> {
        let rgb = img.to_rgb8();
        let e = |e: anyhow::Error| OcrError::Engine(e.to_string());
        let src = ImageSource::from_bytes(rgb.as_raw(), rgb.dimensions())
            .map_err(|e| OcrError::Engine(e.to_string()))?;
        let input = self.engine.prepare_input(src).map_err(e)?;
        let word_rects = self.engine.detect_words(&input).map_err(e)?;
        let lines = self.engine.find_text_lines(&input, &word_rects);
        let texts = self.engine.recognize_text(&input, &lines).map_err(e)?;
        let mut page = OcrPage::default();
        for line in texts.iter().flatten() {
            let s = line.to_string();
            if s.trim().is_empty() {
                continue;
            }
            page.text.push_str(s.trim());
            page.text.push('\n');
            for word in line.words() {
                let r = word.bounding_rect();
                let text: String = word.chars().iter().map(|c| c.char).collect();
                page.words.push(TextWord {
                    text,
                    x: r.left() as f32,
                    y: r.top() as f32,
                    w: r.width() as f32,
                    h: r.height() as f32,
                });
            }
        }
        Ok(page)
    }
}
