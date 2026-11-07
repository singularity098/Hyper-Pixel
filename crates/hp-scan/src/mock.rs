//! Synthetic scanners for development, demos and tests. They behave like an A4 flatbed
//! (CanoScan LiDE 400 sized) and an ADF document scanner.

use crate::*;
use hp_core::synth::{Rng, SceneParams, render_scene};
use image::{DynamicImage, Rgb, RgbImage};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

const BED: (f64, f64) = (216.0, 297.0);

pub struct MockBackend {
    seed: u64,
}

impl MockBackend {
    pub fn new() -> Self {
        let seed = std::env::var("HYPER_PIXEL_MOCK_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);
        Self { seed }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn devices() -> Vec<DeviceInfo> {
    vec![
        DeviceInfo {
            name: "mock:photos".into(),
            vendor: "Hyper Pixel".into(),
            model: "Demo flatbed (photos)".into(),
            kind: "flatbed scanner".into(),
        },
        DeviceInfo {
            name: "mock:documents".into(),
            vendor: "Hyper Pixel".into(),
            model: "Demo document scanner".into(),
            kind: "sheetfed scanner".into(),
        },
    ]
}

impl Backend for MockBackend {
    fn name(&self) -> &'static str {
        "Demo"
    }
    fn list_devices(&mut self) -> Result<Vec<DeviceInfo>> {
        Ok(devices())
    }
    fn open(&mut self, name: &str) -> Result<Box<dyn Device>> {
        let info = devices()
            .into_iter()
            .find(|d| d.name == name)
            .ok_or_else(|| ScanError::NoDevice(name.into()))?;
        let documents = name == "mock:documents";
        Ok(Box::new(MockDevice {
            info,
            documents,
            seed: self.seed,
            pages_left: 3,
            values: HashMap::new(),
            delay: true,
        }))
    }
}

pub struct MockDevice {
    info: DeviceInfo,
    documents: bool,
    seed: u64,
    pages_left: u32,
    values: HashMap<String, OptValue>,
    /// Simulate scanner speed.
    pub delay: bool,
}

impl MockDevice {
    pub fn new_photos(seed: u64) -> Self {
        Self {
            info: devices()[0].clone(),
            documents: false,
            seed,
            pages_left: 0,
            values: HashMap::new(),
            delay: false,
        }
    }

    fn render_bed(&self, dpi: f64, page: u32) -> DynamicImage {
        if self.documents {
            render_document(dpi, self.seed + page as u64)
        } else {
            render_scene(&SceneParams {
                seed: self.seed,
                dpi,
                dark_background: true,
                count: Some(4),
                ..Default::default()
            })
            .image
        }
    }
}

/// A fake text page: paper with slightly skewed lines of word-like blocks.
pub fn render_document(dpi: f64, seed: u64) -> DynamicImage {
    let mut rng = Rng::new(seed);
    let ppm = dpi / 25.4;
    let (w, h) = ((BED.0 * ppm) as u32, (BED.1 * ppm) as u32);
    let mut img = RgbImage::from_pixel(w, h, Rgb([244, 241, 232]));
    let skew = rng.range(-1.5, 1.5).to_radians();
    let line_h = 4.2 * ppm;
    let mut y = 25.0 * ppm;
    while y < (BED.1 - 25.0) * ppm {
        let mut x = 20.0 * ppm;
        let end = (BED.0 - 20.0) * ppm * rng.range(0.7, 1.0);
        while x < end {
            let ww = rng.range(4.0, 18.0) * ppm;
            for yy in 0..(line_h * 0.55) as u32 {
                for xx in 0..ww as u32 {
                    let px = x + xx as f64;
                    let py = y + yy as f64 + (px - w as f64 / 2.0) * skew.tan();
                    if px >= 0.0 && py >= 0.0 && (px as u32) < w && (py as u32) < h {
                        img.put_pixel(px as u32, py as u32, Rgb([30, 30, 35]));
                    }
                }
            }
            x += ww + 2.0 * ppm;
        }
        y += line_h * if rng.f64() < 0.1 { 3.0 } else { 1.6 };
    }
    DynamicImage::ImageRgb8(img)
}

impl Device for MockDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn options(&mut self) -> Result<Vec<OptionDesc>> {
        let mk = |index, name: &str, title: &str, kind, unit, constraint| OptionDesc {
            index,
            name: name.into(),
            title: title.into(),
            desc: String::new(),
            kind,
            unit,
            constraint,
            active: true,
            settable: true,
            advanced: false,
        };
        let mut v = vec![
            mk(
                1,
                "resolution",
                "Resolution",
                OptKind::Int,
                Unit::Dpi,
                Constraint::List(vec![75.0, 150.0, 300.0, 600.0, 1200.0]),
            ),
            mk(
                2,
                "mode",
                "Mode",
                OptKind::Str,
                Unit::None,
                Constraint::Strings(vec!["Color".into(), "Gray".into(), "Lineart".into()]),
            ),
        ];
        if self.documents {
            v.push(mk(
                3,
                "source",
                "Source",
                OptKind::Str,
                Unit::None,
                Constraint::Strings(vec!["Flatbed".into(), "ADF".into()]),
            ));
        }
        Ok(v)
    }

    fn get_option(&mut self, name: &str) -> Result<OptValue> {
        self.values
            .get(name)
            .cloned()
            .ok_or_else(|| ScanError::Other(format!("no option {name}")))
    }

    fn set_option(&mut self, name: &str, value: OptValue) -> Result<()> {
        self.values.insert(name.into(), value);
        Ok(())
    }

    fn capabilities(&mut self) -> Result<Capabilities> {
        Ok(Capabilities {
            resolutions: vec![75.0, 150.0, 300.0, 600.0, 1200.0],
            modes: vec![ColorMode::Color, ColorMode::Gray, ColorMode::Lineart],
            depths: vec![8, 16],
            sources: if self.documents {
                vec!["Flatbed".into(), "ADF".into()]
            } else {
                vec!["Flatbed".into()]
            },
            bed_mm: Some(BED),
        })
    }

    fn scan(
        &mut self,
        s: &ScanSettings,
        progress: &mut ProgressFn<'_>,
        cancel: &AtomicBool,
    ) -> Result<ScanResult> {
        let feeder = s.source.as_deref().is_some_and(is_feeder_source);
        if feeder {
            if self.pages_left == 0 {
                self.pages_left = 3;
                return Err(ScanError::NoDocuments);
            }
            self.pages_left -= 1;
        }
        let dpi = self.capabilities()?.nearest_resolution(s.resolution);
        let page = 3 - self.pages_left;
        let bed = self.render_bed(dpi, page);
        let (ax, ay, aw, ah) = s.area_mm.unwrap_or((0.0, 0.0, BED.0, BED.1));
        let ppm = dpi / 25.4;
        let x0 = ((ax * ppm).round() as u32).min(bed.width() - 1);
        let y0 = ((ay * ppm).round() as u32).min(bed.height() - 1);
        let w = ((aw * ppm).round() as u32).clamp(1, bed.width() - x0);
        let h = ((ah * ppm).round() as u32).clamp(1, bed.height() - y0);
        let mut img = bed.crop_imm(x0, y0, w, h);
        img = match s.mode {
            ColorMode::Color => img,
            ColorMode::Gray => DynamicImage::ImageLuma8(img.to_luma8()),
            ColorMode::Lineart => {
                let mut g = img.to_luma8();
                g.pixels_mut()
                    .for_each(|p| p[0] = if p[0] > 128 { 255 } else { 0 });
                DynamicImage::ImageLuma8(g)
            }
        };
        if s.depth == Some(16) && s.mode != ColorMode::Lineart {
            img = match img {
                DynamicImage::ImageLuma8(_) => DynamicImage::ImageLuma16(img.to_luma16()),
                _ => DynamicImage::ImageRgb16(img.to_rgb16()),
            };
        }
        // Stream rows back as a real scanner would.
        let (channels, depth): (u8, u8) = match &img {
            DynamicImage::ImageLuma8(_) => (1, 8),
            DynamicImage::ImageLuma16(_) => (1, 16),
            DynamicImage::ImageRgb16(_) => (3, 16),
            _ => (3, 8),
        };
        let bytes: Vec<u8> = match &img {
            DynamicImage::ImageLuma16(b) => {
                b.as_raw().iter().flat_map(|v| v.to_ne_bytes()).collect()
            }
            DynamicImage::ImageRgb16(b) => {
                b.as_raw().iter().flat_map(|v| v.to_ne_bytes()).collect()
            }
            DynamicImage::ImageLuma8(b) => b.as_raw().clone(),
            _ => img.to_rgb8().into_raw(),
        };
        let bpl = w as usize * channels as usize * (depth as usize / 8);
        // Roughly LiDE 400 speed: ~8 s for a 300 dpi A4 page, scaled with pixel count.
        let total_time = if self.delay {
            (w as f64 * h as f64 / (2480.0 * 3508.0) * 8.0).clamp(0.4, 25.0)
        } else {
            0.0
        };
        let chunks = 40u32;
        for c in 1..=chunks {
            if cancel.load(Ordering::Relaxed) {
                return Err(ScanError::Cancelled);
            }
            let lines = (h as u64 * c as u64 / chunks as u64) as u32;
            if total_time > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(total_time / chunks as f64));
            }
            progress(Progress {
                fraction: lines as f32 / h as f32,
                frame: Some(FrameView {
                    width: w,
                    lines,
                    total_lines: Some(h),
                    channels,
                    depth,
                    data: &bytes[..lines as usize * bpl],
                }),
            });
        }
        Ok(ScanResult {
            image: img,
            dpi,
            origin_mm: (x0 as f64 / ppm, y0 as f64 / ppm),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_scan_area_and_detection_roundtrip() {
        let mut dev = MockDevice::new_photos(3);
        let cancel = AtomicBool::new(false);
        let preview = dev
            .scan(
                &ScanSettings {
                    resolution: 75.0,
                    ..Default::default()
                },
                &mut |_| {},
                &cancel,
            )
            .unwrap();
        let geo = hp_core::ScanGeometry {
            dpi: preview.dpi,
            origin_mm: preview.origin_mm,
        };
        let rep = hp_core::detect_photos(&preview.image, geo, &Default::default());
        assert_eq!(rep.photos.len(), 4);
        // Scan only part of the bed at a higher resolution; detections must map consistently.
        let area = (10.0, 20.0, 150.0, 200.0);
        let part = dev
            .scan(
                &ScanSettings {
                    resolution: 150.0,
                    area_mm: Some(area),
                    ..Default::default()
                },
                &mut |_| {},
                &cancel,
            )
            .unwrap();
        assert!((part.origin_mm.0 - 10.0).abs() < 0.2 && (part.origin_mm.1 - 20.0).abs() < 0.2);
        assert_eq!(part.image.width(), (150.0 / 25.4 * 150.0f64).round() as u32);
    }

    #[test]
    fn raw_conversion() {
        let img = raw_to_image(9, 1, 1, 1, &[0b1000_0000, 0b1000_0000])
            .unwrap()
            .to_luma8();
        assert_eq!(img.get_pixel(0, 0)[0], 0);
        assert_eq!(img.get_pixel(1, 0)[0], 255);
        assert_eq!(img.get_pixel(8, 0)[0], 0);
    }
}
