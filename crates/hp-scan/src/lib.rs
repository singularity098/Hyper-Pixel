//! Scanner access for Hyper Pixel.
//!
//! [`Backend`] enumerates and opens devices; [`Device`] exposes options and performs scans.
//! Two backends are provided:
//!
//! * [`sane::SaneBackend`] – the system's SANE library, loaded at runtime with `libloading`
//!   (never linked or bundled, so the host's drivers and configuration are used).
//! * [`mock::MockBackend`] – synthetic scanners for development, demos and CI.
//!
//! SANE is not thread-safe; create and use backends and devices on a single thread.

pub mod mock;
pub mod sane;

use image::DynamicImage;
use std::sync::atomic::AtomicBool;

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error(
        "SANE library not found ({0}). Install your distribution's 'sane' / 'libsane1' package."
    )]
    LibraryMissing(String),
    #[error("{0}")]
    Sane(String),
    #[error("scan cancelled")]
    Cancelled,
    #[error("no more documents in the feeder")]
    NoDocuments,
    #[error("device not found: {0}")]
    NoDevice(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ScanError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Backend-specific identifier used to open the device.
    pub name: String,
    pub vendor: String,
    pub model: String,
    pub kind: String,
}

impl DeviceInfo {
    pub fn label(&self) -> String {
        let vm = format!("{} {}", self.vendor, self.model).trim().to_string();
        if vm.is_empty() { self.name.clone() } else { vm }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptValue {
    Bool(bool),
    Int(i32),
    Fixed(f64),
    Str(String),
    Button,
}

impl OptValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            OptValue::Int(v) => Some(*v as f64),
            OptValue::Fixed(v) => Some(*v),
            OptValue::Bool(b) => Some(*b as i32 as f64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptKind {
    Bool,
    Int,
    Fixed,
    Str,
    Button,
    Group,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    None,
    Pixel,
    Bit,
    Mm,
    Dpi,
    Percent,
    Microsecond,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    None,
    Range { min: f64, max: f64, quant: f64 },
    List(Vec<f64>),
    Strings(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct OptionDesc {
    pub index: usize,
    pub name: String,
    pub title: String,
    pub desc: String,
    pub kind: OptKind,
    pub unit: Unit,
    pub constraint: Constraint,
    pub active: bool,
    pub settable: bool,
    pub advanced: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    Color,
    Gray,
    Lineart,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScanSettings {
    pub resolution: f64,
    pub mode: ColorMode,
    /// Bits per sample (8 or 16). `None` = device default.
    pub depth: Option<u8>,
    /// Scan area in bed millimetres: (x, y, width, height). `None` = full bed.
    pub area_mm: Option<(f64, f64, f64, f64)>,
    /// Document source (e.g. "Flatbed", "ADF"). `None` = device default.
    pub source: Option<String>,
}

impl Default for ScanSettings {
    fn default() -> Self {
        Self {
            resolution: 300.0,
            mode: ColorMode::Color,
            depth: None,
            area_mm: None,
            source: None,
        }
    }
}

/// What a device can do, distilled from its options.
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    pub resolutions: Vec<f64>,
    pub modes: Vec<ColorMode>,
    pub depths: Vec<u8>,
    pub sources: Vec<String>,
    /// Bed size in mm, if the device reports its scan area.
    pub bed_mm: Option<(f64, f64)>,
}

impl Capabilities {
    /// Closest supported resolution to `want`.
    pub fn nearest_resolution(&self, want: f64) -> f64 {
        self.resolutions
            .iter()
            .cloned()
            .min_by(|a, b| (a - want).abs().total_cmp(&(b - want).abs()))
            .unwrap_or(want)
    }
    pub fn has_feeder(&self) -> bool {
        self.sources.iter().any(|s| is_feeder_source(s))
    }
}

pub fn is_feeder_source(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.contains("adf") || l.contains("feeder") || l.contains("duplex")
}

/// A finished scan.
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub image: DynamicImage,
    pub dpi: f64,
    /// Bed position of the image's top-left corner, mm.
    pub origin_mm: (f64, f64),
}

/// Raw lines received so far, for progressive display.
pub struct FrameView<'a> {
    pub width: u32,
    pub lines: u32,
    pub total_lines: Option<u32>,
    pub channels: u8,
    /// 8 or 16 (16-bit samples are native-endian). 1-bit data is expanded to 8 before display.
    pub depth: u8,
    pub data: &'a [u8],
}

pub struct Progress<'a> {
    /// 0..1, or negative if unknown.
    pub fraction: f32,
    pub frame: Option<FrameView<'a>>,
}

pub type ProgressFn<'a> = dyn FnMut(Progress<'_>) + 'a;

pub trait Device {
    fn info(&self) -> &DeviceInfo;
    fn options(&mut self) -> Result<Vec<OptionDesc>>;
    fn get_option(&mut self, name: &str) -> Result<OptValue>;
    fn set_option(&mut self, name: &str, value: OptValue) -> Result<()>;
    fn capabilities(&mut self) -> Result<Capabilities>;
    /// Perform one scan (one page).
    fn scan(
        &mut self,
        settings: &ScanSettings,
        progress: &mut ProgressFn<'_>,
        cancel: &AtomicBool,
    ) -> Result<ScanResult>;
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn list_devices(&mut self) -> Result<Vec<DeviceInfo>>;
    fn open(&mut self, name: &str) -> Result<Box<dyn Device>>;
}

/// Scan pages until the feeder runs out (or once, for a flatbed).
pub fn scan_pages(
    dev: &mut dyn Device,
    settings: &ScanSettings,
    progress: &mut ProgressFn<'_>,
    cancel: &AtomicBool,
    on_page: &mut dyn FnMut(ScanResult),
) -> Result<usize> {
    let feeder = settings.source.as_deref().is_some_and(is_feeder_source);
    let mut n = 0;
    loop {
        match dev.scan(settings, progress, cancel) {
            Ok(r) => {
                on_page(r);
                n += 1;
                if !feeder {
                    return Ok(n);
                }
            }
            Err(ScanError::NoDocuments) if n > 0 => return Ok(n),
            Err(e) => return Err(e),
        }
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(n);
        }
    }
}

/// Convert raw scanner bytes into an image.
pub fn raw_to_image(
    width: u32,
    height: u32,
    channels: u8,
    depth: u8,
    data: &[u8],
) -> Result<DynamicImage> {
    let px = width as usize * height as usize;
    let err = || {
        ScanError::Other(format!(
            "bad frame {width}x{height}x{channels}@{depth} with {} bytes",
            data.len()
        ))
    };
    Ok(match (channels, depth) {
        (1, 1) => {
            let stride = (width as usize).div_ceil(8);
            if data.len() < stride * height as usize {
                return Err(err());
            }
            let mut out = Vec::with_capacity(px);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    // SANE lineart: 1 = black.
                    let bit = data[y * stride + x / 8] & (0x80 >> (x % 8)) != 0;
                    out.push(if bit { 0 } else { 255 });
                }
            }
            DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(width, height, out).ok_or_else(err)?,
            )
        }
        (1, 8) => DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(width, height, data.get(..px).ok_or_else(err)?.to_vec())
                .ok_or_else(err)?,
        ),
        (3, 8) => DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(width, height, data.get(..px * 3).ok_or_else(err)?.to_vec())
                .ok_or_else(err)?,
        ),
        (1, 16) => {
            let v: Vec<u16> = data
                .get(..px * 2)
                .ok_or_else(err)?
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                .collect();
            DynamicImage::ImageLuma16(
                image::ImageBuffer::from_raw(width, height, v).ok_or_else(err)?,
            )
        }
        (3, 16) => {
            let v: Vec<u16> = data
                .get(..px * 6)
                .ok_or_else(err)?
                .chunks_exact(2)
                .map(|c| u16::from_ne_bytes([c[0], c[1]]))
                .collect();
            DynamicImage::ImageRgb16(
                image::ImageBuffer::from_raw(width, height, v).ok_or_else(err)?,
            )
        }
        _ => return Err(err()),
    })
}

/// Open the best available backend: SANE if the library loads, else the mock backend.
/// Set `HYPER_PIXEL_MOCK=1` to force the mock backend.
pub fn default_backend() -> (Box<dyn Backend>, Option<ScanError>) {
    if std::env::var_os("HYPER_PIXEL_MOCK").is_some() {
        return (Box::new(mock::MockBackend::new()), None);
    }
    match sane::SaneBackend::new() {
        Ok(b) => (Box::new(b), None),
        Err(e) => (Box::new(mock::MockBackend::new()), Some(e)),
    }
}
