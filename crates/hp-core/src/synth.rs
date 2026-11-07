//! Synthetic flatbed scenes with known ground truth, for testing and tuning the detector
//! until a corpus of real scans is available (and as a regression guard afterwards).

use crate::geom::{Point, RotatedRect};
use image::{DynamicImage, Rgb, RgbImage};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct SceneParams {
    pub seed: u64,
    pub dpi: f64,
    /// Bed size in mm (width, height). Default: A4 / CanoScan LiDE 400 bed.
    pub bed_mm: (f64, f64),
    pub dark_background: bool,
    /// Number of photos; `None` = random 1..=4.
    pub count: Option<usize>,
    /// Maximum absolute skew in degrees.
    pub max_angle: f64,
    /// Minimum gap between photos in mm (can be 0 for touching photos).
    pub min_gap_mm: f64,
    /// Sensor noise amplitude (0..255 scale).
    pub noise: f64,
}

impl Default for SceneParams {
    fn default() -> Self {
        Self {
            seed: 0,
            dpi: 100.0,
            bed_mm: (216.0, 297.0),
            dark_background: true,
            count: None,
            max_angle: 10.0,
            min_gap_mm: 5.0,
            noise: 3.0,
        }
    }
}

pub struct Scene {
    pub image: DynamicImage,
    pub dpi: f64,
    /// Ground-truth photo rectangles in mm.
    pub truth: Vec<RotatedRect>,
}

/// Small deterministic PRNG (SplitMix64).
#[derive(Clone)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f64()
    }
}

fn hash2(x: i64, y: i64, s: u64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F) ^ s;
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn value_noise(x: f64, y: f64, s: u64) -> f64 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (xi, yi) = (x0 as i64, y0 as i64);
    let a = hash2(xi, yi, s);
    let b = hash2(xi + 1, yi, s);
    let c = hash2(xi, yi + 1, s);
    let d = hash2(xi + 1, yi + 1, s);
    (a + (b - a) * sx) * (1.0 - sy) + (c + (d - c) * sx) * sy
}

fn fbm(x: f64, y: f64, s: u64) -> f64 {
    let mut v = 0.0;
    let mut amp = 0.5;
    let mut f = 1.0;
    for o in 0..5 {
        v += amp * value_noise(x * f, y * f, s.wrapping_add(o));
        amp *= 0.5;
        f *= 2.0;
    }
    v
}

#[derive(Clone)]
struct Photo {
    rect: RotatedRect,
    border_mm: f64,
    corner_r: f64,
    seed: u64,
    sky: [f64; 3],
    ground: [f64; 3],
    brightness: f64,
    horizon: f64,
}

impl Photo {
    /// Colour at local coordinates (mm from top-left), or None if outside (rounded corner).
    fn color(&self, u: f64, v: f64) -> Option<[f64; 3]> {
        let (w, h) = (self.rect.w, self.rect.h);
        let r = self.corner_r;
        if r > 0.0 {
            let cx = if u < r { r } else if u > w - r { w - r } else { u };
            let cy = if v < r { r } else if v > h - r { h - r } else { v };
            if (u - cx).powi(2) + (v - cy).powi(2) > r * r {
                return None;
            }
        }
        let b = self.border_mm;
        if u < b || v < b || u > w - b || v > h - b {
            return Some([246.0, 244.0, 238.0]);
        }
        let s = self.seed;
        let n = fbm(u / 18.0, v / 18.0, s);
        let fine = value_noise(u / 1.5, v / 1.5, s ^ 7) - 0.5;
        let t = v / h;
        let mut c = if t < self.horizon + (n - 0.5) * 0.25 {
            let k = t / self.horizon.max(0.01);
            [
                self.sky[0] * (1.0 - 0.3 * k),
                self.sky[1] * (1.0 - 0.2 * k),
                self.sky[2] * (1.0 - 0.1 * k),
            ]
        } else {
            [self.ground[0] * (0.6 + 0.8 * n), self.ground[1] * (0.6 + 0.8 * n), self.ground[2] * (0.6 + 0.8 * n)]
        };
        for ch in &mut c {
            *ch = (*ch * self.brightness + fine * 30.0).clamp(0.0, 255.0);
        }
        Some(c)
    }
}

pub fn render_scene(p: &SceneParams) -> Scene {
    let mut rng = Rng::new(p.seed);
    let sizes = [(152.0, 102.0), (127.0, 89.0), (178.0, 127.0), (89.0, 89.0), (102.0, 102.0), (89.0, 64.0)];
    let target = p.count.unwrap_or(1 + (rng.f64() * 4.0) as usize);
    let mut photos: Vec<Photo> = Vec::new();
    let mut attempts = 0;
    while photos.len() < target && attempts < 2000 {
        attempts += 1;
        let (mut w, mut h) = sizes[(rng.f64() * sizes.len() as f64) as usize % sizes.len()];
        if rng.f64() < 0.5 {
            std::mem::swap(&mut w, &mut h);
        }
        let angle = rng.range(-p.max_angle, p.max_angle);
        let r = RotatedRect::new(rng.range(0.0, p.bed_mm.0), rng.range(0.0, p.bed_mm.1), w, h, angle);
        let inside = r.corners().iter().all(|c| c.x > 2.0 && c.y > 2.0 && c.x < p.bed_mm.0 - 2.0 && c.y < p.bed_mm.1 - 2.0);
        if !inside {
            continue;
        }
        let grown = r.inset(-p.min_gap_mm / 2.0);
        if photos.iter().any(|o| o.rect.inset(-p.min_gap_mm / 2.0).iou(&grown) > 0.0) {
            continue;
        }
        let dark_scene = rng.f64() < 0.2;
        photos.push(Photo {
            rect: r,
            border_mm: if rng.f64() < 0.35 { rng.range(2.5, 6.0) } else { 0.0 },
            corner_r: if rng.f64() < 0.3 { rng.range(1.0, 4.0) } else { 0.0 },
            seed: rng.next_u64(),
            sky: [rng.range(90.0, 220.0), rng.range(120.0, 230.0), rng.range(150.0, 250.0)],
            ground: [rng.range(30.0, 160.0), rng.range(30.0, 150.0), rng.range(20.0, 120.0)],
            brightness: if dark_scene { rng.range(0.1, 0.3) } else { rng.range(0.7, 1.1) },
            horizon: rng.range(0.2, 0.7),
        });
    }

    let ppm = p.dpi / 25.4;
    let (wpx, hpx) = ((p.bed_mm.0 * ppm).round() as u32, (p.bed_mm.1 * ppm).round() as u32);
    let mut img = RgbImage::new(wpx, hpx);
    let bg_seed = rng.next_u64();
    let dark = p.dark_background;
    let shadow_mm = 0.5;
    let rows: Vec<Vec<[u8; 3]>> = (0..hpx)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(wpx as usize);
            for x in 0..wpx {
                let mut acc = [0.0f64; 3];
                for (sx, sy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    let pm = Point::new((x as f64 + sx) / ppm, (y as f64 + sy) / ppm);
                    let mut col = None;
                    let mut shadow = 0.0f64;
                    for ph in &photos {
                        let (ux, uy) = ph.rect.axes();
                        let d = Point::new(pm.x - ph.rect.cx, pm.y - ph.rect.cy);
                        let u = d.x * ux.x + d.y * ux.y + ph.rect.w / 2.0;
                        let v = d.x * uy.x + d.y * uy.y + ph.rect.h / 2.0;
                        if u >= 0.0 && v >= 0.0 && u <= ph.rect.w && v <= ph.rect.h {
                            if let Some(c) = ph.color(u, v) {
                                col = Some(c);
                                break;
                            }
                        }
                        // Light lid: a thin shadow line around prints.
                        let ou = if u < 0.0 { -u } else if u > ph.rect.w { u - ph.rect.w } else { 0.0 };
                        let ov = if v < 0.0 { -v } else if v > ph.rect.h { v - ph.rect.h } else { 0.0 };
                        let od = (ou * ou + ov * ov).sqrt();
                        if od < shadow_mm {
                            shadow = shadow.max(1.0 - od / shadow_mm);
                        }
                    }
                    let c = col.unwrap_or_else(|| {
                        if dark {
                            let n = value_noise(pm.x / 30.0, pm.y / 30.0, bg_seed) * 6.0;
                            [10.0 + n, 10.0 + n, 13.0 + n]
                        } else {
                            let vig = ((pm.x / p.bed_mm.0 - 0.5).powi(2) + (pm.y / p.bed_mm.1 - 0.5).powi(2)) * 25.0;
                            let s = shadow * 40.0;
                            [240.0 - vig - s, 240.0 - vig - s, 237.0 - vig - s]
                        }
                    });
                    for k in 0..3 {
                        acc[k] += c[k] / 4.0;
                    }
                }
                let n = (hash2(x as i64, y as i64, bg_seed) - 0.5) * 2.0 * p.noise;
                row.push([
                    (acc[0] + n).clamp(0.0, 255.0) as u8,
                    (acc[1] + n).clamp(0.0, 255.0) as u8,
                    (acc[2] + n).clamp(0.0, 255.0) as u8,
                ]);
            }
            row
        })
        .collect();
    for (y, row) in rows.into_iter().enumerate() {
        for (x, c) in row.into_iter().enumerate() {
            img.put_pixel(x as u32, y as u32, Rgb(c));
        }
    }
    Scene { image: DynamicImage::ImageRgb8(img), dpi: p.dpi, truth: photos.iter().map(|ph| ph.rect.normalized()).collect() }
}
