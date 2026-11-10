//! Hyper Pixel desktop application.

mod editor;
mod settings;
mod worker;

use editor::{CropBox, Editor, rotate_handle};
use hp_core::color::ColorPreset;
use hp_core::document::{DocColorMode, DocParams};
use hp_core::export::{ExportOptions, ImageFormat};
use hp_core::pdf::TextWord;
use hp_core::pipeline::{
    Enhancer, Naming, PhotoOutput, PhotoRegion, Stamp, extract_photos_with, save_all,
};
use hp_core::restore::DustLevel;
use hp_core::{BackgroundHint, DetectParams, Point, RotatedRect, ScanGeometry, detect_photos};
use hp_scan::{Capabilities, ColorMode, DeviceInfo, ScanResult, ScanSettings, is_feeder_source};
use image::DynamicImage;
use settings::{Mode, PhotoColor, Settings};
use slint::{
    ComponentHandle, Image, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, VecModel,
};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use worker::{Command, Event, Purpose, Worker};

slint::include_modules!();

const BACKGROUNDS: [(&str, BackgroundHint); 3] = [
    ("Auto-detect", BackgroundHint::Auto),
    ("Black sheet (best)", BackgroundHint::Dark),
    ("White lid", BackgroundHint::Light),
];
const PHOTO_COLORS: [(&str, PhotoColor); 2] = [
    ("Colour", PhotoColor::Color),
    ("Greyscale", PhotoColor::Gray),
];
/// Margin added around the photos when scanning only part of the bed (mm).
const AREA_MARGIN_MM: f64 = 4.0;

struct Preview {
    image: DynamicImage,
    geo: ScanGeometry,
    /// Displayed preview pixels per preview-scan pixel.
    display_scale: f64,
}

struct ResultEntry {
    path: PathBuf,
    thumb: SharedPixelBuffer<Rgb8Pixel>,
}

struct App {
    ui: slint::Weak<MainWindow>,
    settings: Settings,
    worker: Worker,
    devices: Vec<DeviceInfo>,
    caps: Capabilities,
    preview: Option<Preview>,
    editor: Editor,
    results: Vec<ResultEntry>,
    pages: Vec<(DynamicImage, Vec<TextWord>, f64)>,
    ocr: Option<Arc<hp_ocr::Ocr>>,
    ml: Option<Arc<hp_ml::MlEnhancer>>,
    busy: bool,
    /// Boxes captured when a final photo scan was started.
    scan_boxes: Vec<CropBox>,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
}

fn with_app<R>(f: impl FnOnce(&mut App) -> R) -> Option<R> {
    APP.with(|a| a.borrow_mut().as_mut().map(f))
}

/// Run `f` on the UI thread with the app state.
fn post(f: impl FnOnce(&mut App) + Send + 'static) {
    let _ = slint::invoke_from_event_loop(move || {
        with_app(f);
    });
}

fn stamp() -> Stamp {
    let now = chrono::Local::now();
    Stamp {
        date: now.format("%Y-%m-%d").to_string(),
        time: now.format("%H%M%S").to_string(),
    }
}

fn thumb(img: &DynamicImage) -> SharedPixelBuffer<Rgb8Pixel> {
    let t = img.thumbnail(256, 192).to_rgb8();
    SharedPixelBuffer::clone_from_slice(t.as_raw(), t.width(), t.height())
}

fn to_buffer(img: &DynamicImage, max: u32) -> (SharedPixelBuffer<Rgb8Pixel>, f64) {
    let scale = (max as f64 / img.width().max(img.height()) as f64).min(1.0);
    let small = if scale < 1.0 {
        img.resize(
            (img.width() as f64 * scale) as u32,
            (img.height() as f64 * scale) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img.clone()
    };
    let rgb = small.to_rgb8();
    let s = rgb.width() as f64 / img.width() as f64;
    (
        SharedPixelBuffer::clone_from_slice(rgb.as_raw(), rgb.width(), rgb.height()),
        s,
    )
}

fn open_path(p: &Path) {
    if let Err(e) = std::process::Command::new("xdg-open").arg(p).spawn() {
        log::warn!("xdg-open failed: {e}");
    }
}

fn strings(v: impl IntoIterator<Item = String>) -> ModelRc<SharedString> {
    ModelRc::new(VecModel::from(
        v.into_iter().map(SharedString::from).collect::<Vec<_>>(),
    ))
}

fn nearest_index(list: &[f64], want: f64) -> i32 {
    list.iter()
        .enumerate()
        .min_by(|a, b| (a.1 - want).abs().total_cmp(&(b.1 - want).abs()))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

impl App {
    fn ui(&self) -> MainWindow {
        self.ui.upgrade().expect("window alive")
    }

    fn set_status(&self, s: impl Into<SharedString>) {
        self.ui().set_status(s.into());
    }

    fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
        let ui = self.ui();
        ui.set_busy(busy);
        if !busy {
            ui.set_progress(0.0);
        }
    }

    fn mode(&self) -> Mode {
        self.settings.mode
    }

    // ---------------------------------------------------------------- settings <-> UI --
    fn push_settings(&self) {
        let ui = self.ui();
        let s = &self.settings;
        ui.set_mode(if s.mode == Mode::Photos { 0 } else { 1 });
        ui.set_color_modes(strings(PHOTO_COLORS.iter().map(|c| c.0.to_string())));
        ui.set_color_mode_index(
            PHOTO_COLORS
                .iter()
                .position(|c| c.1 == s.photo_color)
                .unwrap_or(0) as i32,
        );
        ui.set_backgrounds(strings(BACKGROUNDS.iter().map(|b| b.0.to_string())));
        ui.set_background_index(
            BACKGROUNDS
                .iter()
                .position(|b| b.1 == s.background)
                .unwrap_or(0) as i32,
        );
        ui.set_auto_detect(s.auto_detect);
        ui.set_color_presets(strings(
            ColorPreset::ALL.iter().map(|p| p.label().to_string()),
        ));
        ui.set_color_preset_index(
            ColorPreset::ALL
                .iter()
                .position(|p| *p == s.color_preset)
                .unwrap_or(0) as i32,
        );
        ui.set_dust_levels(strings(
            DustLevel::ALL.iter().map(|l| l.label().to_string()),
        ));
        ui.set_dust_index(
            DustLevel::ALL
                .iter()
                .position(|l| *l == s.dust)
                .unwrap_or(0) as i32,
        );
        ui.set_ai_fill(s.ai_fill);
        ui.set_colorize(s.colorize);
        ui.set_deskew(s.deskew);
        ui.set_whiten(s.whiten);
        ui.set_doc_modes(strings(
            DocColorMode::ALL.iter().map(|m| m.label().to_string()),
        ));
        ui.set_doc_mode_index(
            DocColorMode::ALL
                .iter()
                .position(|m| *m == s.doc_mode)
                .unwrap_or(0) as i32,
        );
        ui.set_ocr(s.ocr);
        ui.set_output_dir(s.output_dir.to_string_lossy().to_string().into());
        ui.set_formats(strings(
            ImageFormat::ALL.iter().map(|f| f.label().to_string()),
        ));
        ui.set_photo_format_index(
            ImageFormat::ALL
                .iter()
                .position(|f| *f == s.photo_format)
                .unwrap_or(0) as i32,
        );
        ui.set_doc_format_index(
            ImageFormat::ALL
                .iter()
                .position(|f| *f == s.doc_format)
                .unwrap_or(0) as i32,
        );
        ui.set_jpeg_quality(s.jpeg_quality as f32);
        ui.set_high_bit_depth(s.high_bit_depth);
        ui.set_magnifier(s.magnifier);
        ui.set_loupe_zoom(s.loupe_zoom.clamp(2.0, 16.0) as f32);
        ui.set_name_pattern(s.name_pattern.clone().into());
        self.push_caps();
    }

    fn push_caps(&self) {
        let ui = self.ui();
        let res = &self.caps.resolutions;
        ui.set_resolutions(strings(res.iter().map(|r| format!("{r:.0} dpi"))));
        ui.set_photo_res_index(nearest_index(res, self.settings.photo_dpi));
        ui.set_doc_res_index(nearest_index(res, self.settings.doc_dpi));
        ui.set_sources(strings(self.caps.sources.clone()));
        ui.set_source_index(
            self.caps
                .sources
                .iter()
                .position(|s| *s == self.settings.source)
                .unwrap_or(0) as i32,
        );
    }

    fn pull_settings(&mut self) {
        let ui = self.ui();
        let s = &mut self.settings;
        let old_mode = s.mode;
        s.mode = if ui.get_mode() == 0 {
            Mode::Photos
        } else {
            Mode::Documents
        };
        let res = &self.caps.resolutions;
        if let Some(r) = res.get(ui.get_photo_res_index() as usize) {
            s.photo_dpi = *r;
        }
        if let Some(r) = res.get(ui.get_doc_res_index() as usize) {
            s.doc_dpi = *r;
        }
        s.photo_color = PHOTO_COLORS
            .get(ui.get_color_mode_index() as usize)
            .map(|c| c.1)
            .unwrap_or_default();
        s.background = BACKGROUNDS
            .get(ui.get_background_index() as usize)
            .map(|b| b.1)
            .unwrap_or_default();
        s.auto_detect = ui.get_auto_detect();
        s.color_preset = ColorPreset::ALL
            .get(ui.get_color_preset_index() as usize)
            .copied()
            .unwrap_or_default();
        s.dust = DustLevel::ALL
            .get(ui.get_dust_index() as usize)
            .copied()
            .unwrap_or_default();
        s.ai_fill = ui.get_ai_fill();
        s.colorize = ui.get_colorize();
        s.deskew = ui.get_deskew();
        s.whiten = ui.get_whiten();
        s.doc_mode = DocColorMode::ALL
            .get(ui.get_doc_mode_index() as usize)
            .copied()
            .unwrap_or_default();
        s.ocr = ui.get_ocr();
        s.output_dir = PathBuf::from(ui.get_output_dir().as_str());
        s.photo_format = ImageFormat::ALL
            .get(ui.get_photo_format_index() as usize)
            .copied()
            .unwrap_or_default();
        s.doc_format = ImageFormat::ALL
            .get(ui.get_doc_format_index() as usize)
            .copied()
            .unwrap_or(ImageFormat::Pdf);
        s.jpeg_quality = ui.get_jpeg_quality().round().clamp(1.0, 100.0) as u8;
        s.high_bit_depth = ui.get_high_bit_depth();
        s.magnifier = ui.get_magnifier();
        s.loupe_zoom = ui.get_loupe_zoom() as f64;
        s.name_pattern = ui.get_name_pattern().to_string();
        if let Some(src) = self.caps.sources.get(ui.get_source_index() as usize) {
            s.source = src.clone();
        }
        s.save();
        if old_mode != s.mode {
            self.update_hint();
            self.push_boxes();
        }
    }

    fn update_hint(&self) {
        let hint = match (self.mode(), self.preview.is_some()) {
            (_, false) => "",
            (Mode::Photos, true) => "Drag to add · R rotates output · Del removes",
            (Mode::Documents, true) => "Drag to limit the scan area",
        };
        self.ui().set_hint(hint.into());
    }

    // ---------------------------------------------------------------- devices --
    fn on_devices(&mut self, list: Vec<DeviceInfo>, warning: Option<String>) {
        self.devices = list;
        let ui = self.ui();
        ui.set_devices(strings(self.devices.iter().map(|d| d.label())));
        if let Some(w) = warning {
            self.set_status(format!("SANE: {w}"));
        }
        if self.devices.is_empty() {
            self.set_status("No scanners found. Check the cable and press ⟳.");
            return;
        }
        let idx = self
            .devices
            .iter()
            .position(|d| d.name == self.settings.device)
            .unwrap_or(0);
        ui.set_device_index(idx as i32);
        self.open_device(idx);
    }

    fn open_device(&mut self, idx: usize) {
        let Some(d) = self.devices.get(idx).cloned() else {
            return;
        };
        self.set_status(format!("Opening {}…", d.label()));
        self.set_busy(true);
        self.worker.send(Command::Open(d.name.clone()));
    }

    fn on_opened(&mut self, name: String, caps: Capabilities) {
        self.set_busy(false);
        self.settings.device = name.clone();
        self.settings.save();
        if self.settings.source.is_empty() || !caps.sources.contains(&self.settings.source) {
            self.settings.source = caps.sources.first().cloned().unwrap_or_default();
        }
        self.caps = caps;
        self.push_caps();
        let label = self
            .devices
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.label())
            .unwrap_or(name);
        self.set_status(format!("{label} ready"));
    }

    // ---------------------------------------------------------------- boxes --
    fn mm_to_display(&self, p: Point) -> Point {
        let Some(pv) = &self.preview else { return p };
        let ppm = pv.geo.px_per_mm() * pv.display_scale;
        Point::new(
            (p.x - pv.geo.origin_mm.0) * ppm,
            (p.y - pv.geo.origin_mm.1) * ppm,
        )
    }

    fn display_to_mm(&self, x: f64, y: f64) -> Point {
        let Some(pv) = &self.preview else {
            return Point::new(x, y);
        };
        let ppm = pv.geo.px_per_mm() * pv.display_scale;
        Point::new(x / ppm + pv.geo.origin_mm.0, y / ppm + pv.geo.origin_mm.1)
    }

    fn display_ppm(&self) -> f64 {
        self.preview
            .as_ref()
            .map(|pv| pv.geo.px_per_mm() * pv.display_scale)
            .unwrap_or(1.0)
    }

    fn push_boxes(&self) {
        let ui = self.ui();
        let doc = self.mode() == Mode::Documents;
        let items: Vec<CropBoxUi> = self
            .editor
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let c = b.rect.corners().map(|p| self.mm_to_display(p));
                let h = self.mm_to_display(rotate_handle(&b.rect, 8.0));
                // Edge that ends up on top of the output after quarter turns:
                // 0 → top (c0-c1), 1 → left (c3-c0), 2 → bottom (c2-c3), 3 → right (c1-c2).
                let (a, z) = match b.quarter_turns.rem_euclid(4) {
                    0 => (c[0], c[1]),
                    1 => (c[3], c[0]),
                    2 => (c[2], c[3]),
                    _ => (c[1], c[2]),
                };
                let centre = self.mm_to_display(Point::new(b.rect.cx, b.rect.cy));
                CropBoxUi {
                    x1: c[0].x as f32,
                    y1: c[0].y as f32,
                    x2: c[1].x as f32,
                    y2: c[1].y as f32,
                    x3: c[2].x as f32,
                    y3: c[2].y as f32,
                    x4: c[3].x as f32,
                    y4: c[3].y as f32,
                    hx: h.x as f32,
                    hy: h.y as f32,
                    tx: centre.x as f32,
                    ty: centre.y as f32,
                    ux: ((a.x + z.x) / 2.0) as f32,
                    uy: ((a.y + z.y) / 2.0) as f32,
                    selected: self.editor.selected == Some(b.id),
                    weak: b.confidence < 0.75,
                    label: if doc {
                        "".into()
                    } else {
                        (i + 1).to_string().into()
                    },
                }
            })
            .collect();
        ui.set_boxes(ModelRc::new(VecModel::from(items)));
        ui.set_has_selection(self.editor.selected.is_some());
    }

    fn detect_on_preview(&mut self) {
        let Some(pv) = &self.preview else { return };
        let params = DetectParams {
            background: self.settings.background,
            ..Default::default()
        };
        let rep = detect_photos(&pv.image, pv.geo, &params);
        let n = rep.photos.len();
        self.editor
            .set_detections(rep.photos.into_iter().map(|d| (d.rect, d.confidence)));
        self.push_boxes();
        let bg = if rep.dark_background { "dark" } else { "light" };
        self.set_status(match n {
            0 => "No photos found — drag on the preview to mark them by hand".to_string(),
            1 => format!("Found 1 photo ({bg} background)"),
            n => format!("Found {n} photos ({bg} background)"),
        });
    }

    // ---------------------------------------------------------------- scanning --
    fn scan_settings(
        &self,
        dpi: f64,
        mode: ColorMode,
        area: Option<(f64, f64, f64, f64)>,
    ) -> ScanSettings {
        let fmt = if self.mode() == Mode::Photos {
            self.settings.photo_format
        } else {
            self.settings.doc_format
        };
        let depth = (self.settings.high_bit_depth
            && fmt.supports_16bit()
            && self.caps.depths.contains(&16))
        .then_some(16);
        let source = (!self.settings.source.is_empty()).then(|| self.settings.source.clone());
        ScanSettings {
            resolution: dpi,
            mode,
            depth,
            area_mm: area,
            source,
        }
    }

    fn start_preview(&mut self) {
        if self.busy {
            return;
        }
        self.pull_settings();
        // Previews always use the flatbed at low resolution.
        let mut s = self.scan_settings(self.settings.preview_dpi, ColorMode::Color, None);
        s.depth = None;
        if s.source.as_deref().is_some_and(is_feeder_source) {
            s.source = self
                .caps
                .sources
                .iter()
                .find(|x| !is_feeder_source(x))
                .cloned();
        }
        self.set_busy(true);
        self.set_status("Scanning preview…");
        self.worker.send(Command::Scan {
            settings: s,
            purpose: Purpose::Preview,
            pages: false,
        });
    }

    fn start_final(&mut self) {
        if self.busy {
            return;
        }
        self.pull_settings();
        let bed = self.caps.bed_mm;
        let clamp_area = |x0: f64, y0: f64, x1: f64, y1: f64| {
            let (bw, bh) = bed.unwrap_or((1e4, 1e4));
            let (x0, y0) = (x0.max(0.0), y0.max(0.0));
            let (x1, y1) = (x1.min(bw), y1.min(bh));
            (x0, y0, (x1 - x0).max(1.0), (y1 - y0).max(1.0))
        };
        let bounds = |boxes: &[CropBox], margin: f64| {
            let pts: Vec<Point> = boxes.iter().flat_map(|b| b.rect.corners()).collect();
            let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for p in pts {
                x0 = x0.min(p.x);
                y0 = y0.min(p.y);
                x1 = x1.max(p.x);
                y1 = y1.max(p.y);
            }
            clamp_area(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        };
        let (settings, pages) = match self.mode() {
            Mode::Photos => {
                let area = (!self.editor.boxes.is_empty())
                    .then(|| bounds(&self.editor.boxes, AREA_MARGIN_MM));
                let mode = if self.settings.photo_color == PhotoColor::Gray {
                    ColorMode::Gray
                } else {
                    ColorMode::Color
                };
                self.scan_boxes = self.editor.boxes.clone();
                (
                    self.scan_settings(self.settings.photo_dpi, mode, area),
                    false,
                )
            }
            Mode::Documents => {
                let area = (!self.editor.boxes.is_empty()).then(|| bounds(&self.editor.boxes, 0.0));
                let mode = if self.settings.doc_mode == DocColorMode::Color {
                    ColorMode::Color
                } else {
                    ColorMode::Gray
                };
                let s = self.scan_settings(self.settings.doc_dpi, mode, area);
                let feeder = s.source.as_deref().is_some_and(is_feeder_source);
                (s, feeder)
            }
        };
        self.set_busy(true);
        self.set_status(format!("Scanning at {:.0} dpi…", settings.resolution));
        self.worker.send(Command::Scan {
            settings,
            purpose: Purpose::Final,
            pages,
        });
    }

    fn on_progress(&mut self, fraction: f32, frame: Option<(SharedPixelBuffer<Rgb8Pixel>, f64)>) {
        let ui = self.ui();
        if fraction >= 0.0 {
            ui.set_progress(fraction);
        }
        if let Some((buf, _)) = frame {
            ui.set_preview_width(buf.width() as f32);
            ui.set_preview_height(buf.height() as f32);
            ui.set_preview(Image::from_rgb8(buf));
            ui.set_boxes(ModelRc::default());
        }
    }

    fn on_page(&mut self, r: ScanResult, purpose: Purpose) {
        match purpose {
            Purpose::Preview => {
                let geo = ScanGeometry {
                    dpi: r.dpi,
                    origin_mm: r.origin_mm,
                };
                let (buf, scale) = to_buffer(&r.image, 1800);
                let ui = self.ui();
                ui.set_preview_width(buf.width() as f32);
                ui.set_preview_height(buf.height() as f32);
                ui.set_preview(Image::from_rgb8(buf));
                self.preview = Some(Preview {
                    image: r.image,
                    geo,
                    display_scale: scale,
                });
                self.editor.clear();
                if self.mode() == Mode::Photos && self.settings.auto_detect {
                    self.detect_on_preview();
                } else {
                    self.push_boxes();
                    self.set_status("Preview ready — drag to choose an area, or press Scan");
                }
                self.update_hint();
            }
            Purpose::Final => match self.mode() {
                Mode::Photos => self.process_photos(r),
                Mode::Documents => self.process_page(r),
            },
        }
    }

    fn process_photos(&mut self, r: ScanResult) {
        let geo = ScanGeometry {
            dpi: r.dpi,
            origin_mm: r.origin_mm,
        };
        let boxes = self.scan_boxes.clone();
        let auto = self.settings.auto_detect;
        let params = DetectParams {
            background: self.settings.background,
            ..Default::default()
        };
        let out = PhotoOutput {
            color: self.settings.color_preset,
            inset_mm: self.settings.inset_mm,
            dust: self.settings.dust,
            colorize: self.settings.colorize,
        };
        let ml = self.ensure_ml();
        let dir = self.settings.output_dir.clone();
        let naming = Naming {
            pattern: self.settings.name_pattern.clone(),
            label: "photo".into(),
        };
        let opts = ExportOptions {
            format: self.settings.photo_format,
            jpeg_quality: self.settings.jpeg_quality,
            dpi: r.dpi,
        };
        self.set_status("Finding edges and saving photos…");
        self.busy = true;
        std::thread::spawn(move || {
            // Re-detect on the full-resolution scan to refine edges of untouched boxes.
            let regions: Vec<PhotoRegion> = if !boxes.is_empty() {
                let rep = detect_photos(&r.image, geo, &params);
                let det: Vec<RotatedRect> = rep.photos.iter().map(|d| d.rect).collect();
                editor::refine_boxes(&boxes, &det)
                    .into_iter()
                    .map(|b| PhotoRegion {
                        rect: b.rect,
                        quarter_turns: b.quarter_turns,
                    })
                    .collect()
            } else if auto {
                detect_photos(&r.image, geo, &params)
                    .photos
                    .into_iter()
                    .map(|d| PhotoRegion {
                        rect: d.rect,
                        quarter_turns: 0,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let photos = if regions.is_empty() {
                vec![r.image.clone()]
            } else {
                let enh = ml.as_deref().map(|m| m as &dyn Enhancer);
                extract_photos_with(&r.image, geo, &regions, &out, enh)
            };
            let saved = save_all(&photos, &dir, &naming, &stamp(), &opts);
            let thumbs: Vec<_> = photos.iter().map(thumb).collect();
            post(move |app| {
                app.set_busy(false);
                match saved {
                    Ok(paths) => {
                        let n = paths.len();
                        let batch: Vec<ResultEntry> = paths
                            .into_iter()
                            .zip(thumbs)
                            .map(|(path, thumb)| ResultEntry { path, thumb })
                            .collect();
                        app.results.splice(0..0, batch);
                        app.push_results();
                        app.set_status(format!(
                            "Saved {n} {} to {}",
                            if n == 1 { "photo" } else { "photos" },
                            dir.display()
                        ));
                    }
                    Err(e) => app.set_status(format!("Saving failed: {e}")),
                }
            });
        });
    }

    /// The AI models the current settings ask for, loaded once and kept.
    fn ensure_ml(&mut self) -> Option<Arc<hp_ml::MlEnhancer>> {
        let fill = self.settings.ai_fill && self.settings.dust != DustLevel::Off;
        let col = self.settings.colorize;
        if !fill && !col {
            return None;
        }
        let have = self
            .ml
            .as_ref()
            .map(|m| (m.inpainter.is_some(), m.colorizer.is_some()));
        if have.is_none_or(|(f, c)| (fill && !f) || (col && !c)) {
            self.ml = Some(Arc::new(hp_ml::MlEnhancer::load(fill, col)));
        }
        self.ml.clone()
    }

    fn ensure_ocr(&mut self) -> Option<Arc<hp_ocr::Ocr>> {
        if self.ocr.is_none() {
            match hp_ocr::Ocr::load_default() {
                Ok(o) => self.ocr = Some(Arc::new(o)),
                Err(e) => log::warn!("OCR unavailable: {e}"),
            }
        }
        self.ocr.clone()
    }

    fn process_page(&mut self, r: ScanResult) {
        let params = DocParams {
            deskew: self.settings.deskew,
            whiten: self.settings.whiten,
            mode: self.settings.doc_mode,
        };
        let ocr = if self.settings.ocr {
            self.ensure_ocr()
        } else {
            None
        };
        let fmt = self.settings.doc_format;
        let dir = self.settings.output_dir.clone();
        let naming = Naming {
            pattern: self.settings.name_pattern.clone(),
            label: "document".into(),
        };
        let opts = ExportOptions {
            format: fmt,
            jpeg_quality: self.settings.jpeg_quality,
            dpi: r.dpi,
        };
        self.set_status("Cleaning up page…");
        std::thread::spawn(move || {
            let page = hp_core::document::process(&r.image, r.dpi, &params);
            let words = ocr.as_ref().and_then(|o| o.recognize(&page).ok());
            let t = thumb(&page);
            if fmt == ImageFormat::Pdf {
                post(move |app| {
                    app.pages
                        .push((page, words.map(|w| w.words).unwrap_or_default(), r.dpi));
                    app.ui().set_pending_pages(app.pages.len() as i32);
                    app.results.insert(
                        0,
                        ResultEntry {
                            path: PathBuf::new(),
                            thumb: t,
                        },
                    );
                    app.push_results();
                    app.set_status(format!(
                        "Page {} added — scan more or press Save PDF",
                        app.pages.len()
                    ));
                });
            } else {
                let saved = save_all(std::slice::from_ref(&page), &dir, &naming, &stamp(), &opts);
                if let (Ok(paths), Some(w)) = (&saved, &words)
                    && let Some(p) = paths.first()
                {
                    let _ = std::fs::write(p.with_extension("txt"), &w.text);
                }
                post(move |app| match saved {
                    Ok(paths) => {
                        for p in paths {
                            app.set_status(format!("Saved {}", p.display()));
                            app.results.insert(
                                0,
                                ResultEntry {
                                    path: p,
                                    thumb: t.clone(),
                                },
                            );
                        }
                        app.push_results();
                    }
                    Err(e) => app.set_status(format!("Saving failed: {e}")),
                });
            }
        });
    }

    fn save_document(&mut self) {
        if self.pages.is_empty() {
            return;
        }
        let pages = std::mem::take(&mut self.pages);
        self.ui().set_pending_pages(0);
        let dir = self.settings.output_dir.clone();
        let naming = Naming {
            pattern: self.settings.name_pattern.clone(),
            label: "document".into(),
        };
        let q = self.settings.jpeg_quality.min(90);
        self.set_busy(true);
        self.set_status("Writing PDF…");
        std::thread::spawn(move || {
            let dpi = pages.first().map(|p| p.2).unwrap_or(300.0);
            let list: Vec<(DynamicImage, Vec<TextWord>)> =
                pages.into_iter().map(|(i, w, _)| (i, w)).collect();
            let (path, _) = hp_core::pipeline::next_free_path(&dir, &naming, &stamp(), "pdf", 1);
            let res = hp_core::export::pdf_document(
                &list,
                &ExportOptions {
                    format: ImageFormat::Pdf,
                    jpeg_quality: q,
                    dpi,
                },
                "Scanned document",
            )
            .and_then(|b| hp_core::export::write_atomic(&path, &b));
            post(move |app| {
                app.set_busy(false);
                match res {
                    Ok(()) => {
                        // Point the page thumbnails at the PDF.
                        for r in app
                            .results
                            .iter_mut()
                            .filter(|r| r.path.as_os_str().is_empty())
                        {
                            r.path = path.clone();
                        }
                        app.push_results();
                        app.set_status(format!("Saved {}", path.display()));
                    }
                    Err(e) => app.set_status(format!("Saving PDF failed: {e}")),
                }
            });
        });
    }

    fn push_results(&self) {
        let items: Vec<ResultItem> = self
            .results
            .iter()
            .map(|r| ResultItem {
                image: Image::from_rgb8(r.thumb.clone()),
                label: r
                    .path
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unsaved page".into())
                    .into(),
                path: r.path.to_string_lossy().to_string().into(),
            })
            .collect();
        self.ui().set_results(ModelRc::new(VecModel::from(items)));
    }

    fn rotate_result(&mut self, i: usize) {
        let Some(r) = self.results.get(i) else { return };
        let path = r.path.clone();
        let Some(fmt) = path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(ImageFormat::from_extension)
        else {
            return;
        };
        if fmt == ImageFormat::Pdf {
            return;
        }
        let opts = ExportOptions {
            format: fmt,
            jpeg_quality: self.settings.jpeg_quality,
            dpi: self.settings.photo_dpi,
        };
        std::thread::spawn(move || {
            let res = image::open(&path)
                .map_err(|e| e.to_string())
                .and_then(|img| {
                    let rot = img.rotate90();
                    hp_core::export::save(&rot, &path, &opts).map_err(|e| e.to_string())?;
                    Ok(thumb(&rot))
                });
            post(move |app| match res {
                Ok(t) => {
                    if let Some(r) = app.results.iter_mut().find(|r| r.path == path) {
                        r.thumb = t;
                    }
                    app.push_results();
                }
                Err(e) => app.set_status(format!("Rotate failed: {e}")),
            });
        });
    }

    fn delete_result(&mut self, i: usize) {
        if i >= self.results.len() {
            return;
        }
        let r = self.results.remove(i);
        if !r.path.as_os_str().is_empty() && r.path.extension().is_some_and(|e| e != "pdf") {
            let _ = std::fs::remove_file(&r.path);
            let _ = std::fs::remove_file(r.path.with_extension("txt"));
            self.set_status(format!("Deleted {}", r.path.display()));
        }
        self.push_results();
    }
}

/// Window-frame actions for the app's own title bar (the window has `no-frame`). These use the
/// window directly, not `with_app`: moving or resizing can re-enter the UI synchronously.
fn wire_window_frame(ui: &MainWindow) {
    use slint::winit_030::WinitWindowAccessor;
    use slint::winit_030::winit::window::ResizeDirection;

    // The window manager consumes the button release that ends a move or resize, so Slint
    // never sees it and would keep routing every later click to the title bar. Send it one.
    fn end_grab(w: slint::Weak<MainWindow>) {
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(ui) = w.upgrade() {
                let outside = slint::LogicalPosition::new(-1.0, -1.0);
                ui.window()
                    .dispatch_event(slint::platform::WindowEvent::PointerReleased {
                        position: outside,
                        button: slint::platform::PointerEventButton::Left,
                    });
                ui.window()
                    .dispatch_event(slint::platform::WindowEvent::PointerExited);
            }
        });
    }

    let w = ui.as_weak();
    ui.on_window_drag(move || {
        if let Some(ui) = w.upgrade() {
            ui.window().with_winit_window(|win| {
                if let Err(e) = win.drag_window() {
                    log::debug!("drag_window: {e}");
                }
            });
            end_grab(w.clone());
        }
    });
    let w = ui.as_weak();
    ui.on_window_resize(move |dir| {
        let dir = match dir {
            0 => ResizeDirection::North,
            1 => ResizeDirection::NorthEast,
            2 => ResizeDirection::East,
            3 => ResizeDirection::SouthEast,
            4 => ResizeDirection::South,
            5 => ResizeDirection::SouthWest,
            6 => ResizeDirection::West,
            _ => ResizeDirection::NorthWest,
        };
        if let Some(ui) = w.upgrade() {
            ui.window().with_winit_window(|win| {
                if let Err(e) = win.drag_resize_window(dir) {
                    log::debug!("drag_resize_window: {e}");
                }
            });
            end_grab(w.clone());
        }
    });
    let w = ui.as_weak();
    ui.on_window_minimize(move || {
        if let Some(ui) = w.upgrade() {
            ui.window().set_minimized(true);
        }
    });
    let w = ui.as_weak();
    ui.on_window_toggle_maximize(move || {
        if let Some(ui) = w.upgrade() {
            ui.window().set_maximized(!ui.window().is_maximized());
        }
    });
    let w = ui.as_weak();
    ui.on_window_close(move || {
        if let Some(ui) = w.upgrade() {
            let _ = ui.hide();
        }
    });
}

fn wire(ui: &MainWindow) {
    wire_window_frame(ui);
    ui.on_refresh_devices(|| {
        with_app(|a| {
            a.set_status("Looking for scanners…");
            a.worker.send(Command::ListDevices);
        });
    });
    ui.on_device_selected(|i| {
        with_app(|a| a.open_device(i as usize));
    });
    ui.on_settings_changed(|| {
        with_app(|a| a.pull_settings());
    });
    ui.on_do_preview(|| {
        with_app(|a| a.start_preview());
    });
    ui.on_do_scan(|| {
        with_app(|a| a.start_final());
    });
    ui.on_cancel(|| {
        with_app(|a| {
            a.worker.cancel.store(true, Ordering::Relaxed);
            a.set_status("Cancelling…");
        });
    });
    ui.on_canvas_down(|x, y, scale| {
        with_app(|a| {
            let p = a.display_to_mm(x as f64, y as f64);
            let ppm = a.display_ppm() * scale as f64; // screen px per mm
            a.editor.pointer_down(p, 9.0 / ppm, 8.0);
            a.ui().set_loupe_active(a.editor.is_placing_point());
            a.push_boxes();
        });
    });
    ui.on_canvas_move(|x, y| {
        with_app(|a| {
            let p = a.display_to_mm(x as f64, y as f64);
            if a.editor.pointer_move(p) {
                if a.mode() == Mode::Documents && a.editor.boxes.len() > 1 {
                    // Documents use a single scan area: keep only the newest box.
                    let keep = a.editor.selected;
                    a.editor.boxes.retain(|b| Some(b.id) == keep);
                }
                a.push_boxes();
            }
        });
    });
    ui.on_canvas_hover(|x, y, scale| {
        with_app(|a| {
            if a.preview.is_none() {
                return;
            }
            let p = a.display_to_mm(x as f64, y as f64);
            let ppm = a.display_ppm() * scale as f64;
            let near = a.editor.is_placing_point() || a.editor.corner_near(p, 9.0 / ppm);
            if a.ui().get_loupe_active() != near {
                a.ui().set_loupe_active(near);
            }
        });
    });
    ui.on_canvas_up(|| {
        with_app(|a| {
            a.ui().set_loupe_active(false);
            if a.editor.pointer_up() {
                a.push_boxes();
            }
        });
    });
    ui.on_delete_selected(|| {
        with_app(|a| {
            if a.editor.delete_selected() {
                a.push_boxes();
            }
        });
    });
    ui.on_turn_selected(|q| {
        with_app(|a| {
            if a.editor.turn_selected(q) {
                a.push_boxes();
            }
        });
    });
    ui.on_clear_boxes(|| {
        with_app(|a| {
            a.editor.clear();
            a.push_boxes();
        });
    });
    ui.on_detect_again(|| {
        with_app(|a| {
            a.pull_settings();
            a.detect_on_preview();
        });
    });
    ui.on_browse_output(|| {
        let start = with_app(|a| a.settings.output_dir.clone()).unwrap_or_default();
        std::thread::spawn(move || {
            if let Some(dir) = rfd::FileDialog::new()
                .set_directory(&start)
                .set_title("Output folder")
                .pick_folder()
            {
                post(move |a| {
                    a.ui()
                        .set_output_dir(dir.to_string_lossy().to_string().into());
                    a.pull_settings();
                });
            }
        });
    });
    ui.on_open_output(|| {
        with_app(|a| {
            let _ = std::fs::create_dir_all(&a.settings.output_dir);
            open_path(&a.settings.output_dir);
        });
    });
    ui.on_save_document(|| {
        with_app(|a| a.save_document());
    });
    ui.on_clear_pages(|| {
        with_app(|a| {
            a.pages.clear();
            a.results.retain(|r| !r.path.as_os_str().is_empty());
            a.ui().set_pending_pages(0);
            a.push_results();
        });
    });
    ui.on_result_open(|i| {
        with_app(|a| {
            if let Some(r) = a.results.get(i as usize)
                && !r.path.as_os_str().is_empty()
            {
                open_path(&r.path);
            }
        });
    });
    ui.on_result_rotate(|i| {
        with_app(|a| a.rotate_result(i as usize));
    });
    ui.on_result_delete(|i| {
        with_app(|a| a.delete_result(i as usize));
    });
    ui.on_download_ai_models(|| {
        with_app(|a| a.set_status("Downloading AI models…"));
        std::thread::spawn(|| {
            let specs = [hp_ml::LAMA, hp_ml::DDCOLOR];
            let total: u64 = specs.iter().map(|s| s.download_size()).sum();
            let mut base = 0u64;
            let mut res = Ok(());
            for spec in specs {
                let mut last = 0u64;
                let r = hp_ml::models::download(&spec, &mut |d, _| {
                    if d.saturating_sub(last) > 2_000_000 {
                        last = d;
                        let done = base + d;
                        post(move |a| {
                            a.set_status(format!(
                                "Downloading AI models… {:.0} / {:.0} MB",
                                done as f64 / 1e6,
                                total as f64 / 1e6
                            ))
                        });
                    }
                });
                base += spec.download_size();
                if let Err(e) = r {
                    res = Err(e);
                    break;
                }
            }
            post(move |a| match res {
                Ok(()) => {
                    a.ml = None;
                    a.ui().set_ai_ready(ai_models_ready());
                    a.set_status("AI models installed");
                }
                Err(e) => a.set_status(format!("Download failed: {e}")),
            });
        });
    });
    ui.on_download_models(|| {
        with_app(|a| a.set_status("Downloading OCR models…"));
        std::thread::spawn(|| {
            let mut last = 0u64;
            let res = hp_ocr::download_models(&mut |d, t| {
                if d - last > 500_000 {
                    last = d;
                    post(move |a| {
                        a.set_status(format!(
                            "Downloading OCR models… {:.1} / {:.1} MB",
                            d as f64 / 1e6,
                            t as f64 / 1e6
                        ))
                    });
                }
            });
            post(move |a| match res {
                Ok(_) => {
                    a.ui().set_ocr_ready(true);
                    a.set_status("OCR models installed");
                }
                Err(e) => a.set_status(format!("Download failed: {e}")),
            });
        });
    });
}

fn ai_models_ready() -> bool {
    hp_ml::LAMA.find().is_some() && hp_ml::DDCOLOR.find().is_some()
}

fn main() -> Result<(), slint::PlatformError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let ui = MainWindow::new()?;
    let weak = ui.as_weak();
    let worker = Worker::spawn(move |ev| {
        post(move |app| match ev {
            Event::Devices { list, warning } => app.on_devices(list, warning),
            Event::Opened { name, caps } => app.on_opened(name, caps),
            Event::Progress { fraction, frame } => app.on_progress(fraction, frame),
            Event::Page { result, purpose } => app.on_page(result, purpose),
            Event::Finished { purpose, pages } => {
                // Final photo scans stay busy until saving completes.
                if purpose == Purpose::Preview || app.mode() == Mode::Documents {
                    app.set_busy(false);
                }
                if purpose == Purpose::Final && pages > 1 {
                    app.set_status(format!("Scanned {pages} pages"));
                }
            }
            Event::Error(e) => {
                app.set_busy(false);
                app.set_status(e);
                if app.preview.is_none() {
                    app.ui().set_preview_width(0.0);
                }
            }
        })
    });
    let app = App {
        ui: weak,
        settings: Settings::load(),
        worker,
        devices: Vec::new(),
        caps: Capabilities::default(),
        preview: None,
        editor: Editor::default(),
        results: Vec::new(),
        pages: Vec::new(),
        ocr: None,
        ml: None,
        busy: false,
        scan_boxes: Vec::new(),
    };
    APP.with(|a| *a.borrow_mut() = Some(app));
    with_app(|a| {
        a.push_settings();
        a.ui().set_ocr_ready(hp_ocr::find_models().is_some());
        a.ui().set_ai_ready(ai_models_ready());
        a.set_status("Looking for scanners…");
        a.worker.send(Command::ListDevices);
    });
    wire(&ui);
    let r = ui.run();
    APP.with(|a| *a.borrow_mut() = None);
    r
}
