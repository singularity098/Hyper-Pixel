//! Dust and scratch removal for scanned prints.
//!
//! Detection works on luminance at full resolution:
//! - Grey-level top-hats (`Y - open(Y)` for bright defects, `close(Y) - Y` for dark ones)
//!   with a square element a little larger than the biggest speck. Anything narrower than
//!   the element shows up, which covers specks, hairs, fibres and fine scratches alike.
//! - The threshold adapts to the local texture energy, so busy areas (foliage, hair,
//!   fabric) need a much stronger response than smooth sky or skin.
//! - Hysteresis grows each seed to the full extent of the defect, then every candidate is
//!   checked for size, shape and colour: defects are small or thin, and dust is
//!   colourless, so a candidate must not be more saturated than its surroundings.
//!
//! Filling is a healing brush: each defect (long scratches in short chunks) copies texture
//! from the nearby offset whose surroundings match best, then a harmonic correction makes
//! the patch blend seamlessly into its border. Grain and texture survive, unlike a blur.
//! Pixels outside the mask are never changed, and the bit depth is kept.

use crate::raster::Mask;
use image::{DynamicImage, Luma, Rgb, Rgba};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// How hard to look for defects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DustLevel {
    #[default]
    Off,
    /// Only obvious, high-contrast specks and hairs.
    Light,
    /// Good default for typical prints.
    Normal,
    /// Faint and larger defects too; may touch fine image detail.
    Strong,
}

impl DustLevel {
    pub const ALL: [DustLevel; 4] = [
        DustLevel::Off,
        DustLevel::Light,
        DustLevel::Normal,
        DustLevel::Strong,
    ];
    pub fn label(self) -> &'static str {
        match self {
            DustLevel::Off => "Off",
            DustLevel::Light => "Light",
            DustLevel::Normal => "Normal",
            DustLevel::Strong => "Strong",
        }
    }
    pub fn params(self) -> Option<DustParams> {
        Some(match self {
            DustLevel::Off => return None,
            DustLevel::Light => DustParams {
                min_contrast: 0.10,
                texture_factor: 6.0,
                max_speck_mm: 0.5,
                max_width_mm: 0.3,
                min_sharpness: 0.55,
                ..Default::default()
            },
            DustLevel::Normal => DustParams::default(),
            DustLevel::Strong => DustParams {
                min_contrast: 0.05,
                texture_factor: 4.5,
                max_speck_mm: 1.2,
                max_width_mm: 0.6,
                min_sharpness: 0.45,
                ..Default::default()
            },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustParams {
    /// Minimum top-hat response (gamma-encoded 0..1) for a seed pixel.
    pub min_contrast: f32,
    /// A seed must also exceed this multiple of the local texture energy.
    pub texture_factor: f32,
    /// Largest diameter of a compact speck, mm.
    pub max_speck_mm: f64,
    /// Largest width of a hair, fibre or scratch, mm.
    pub max_width_mm: f64,
    /// Accept long thin defects (scratches, hairs) of any length.
    pub scratches: bool,
    /// Maximum chroma a candidate may add over its surroundings (RGB max-min, 0..1).
    pub max_extra_chroma: f32,
    /// Minimum edge sharpness (steepest step / contrast, noise-corrected). Defects lie on
    /// the print surface and are crisp; print detail is softened by the print itself.
    /// Only applied at 300 dpi and above, where prints are visibly softer than the scan.
    pub min_sharpness: f32,
    /// Radius of the window that measures local texture energy, mm.
    pub texture_radius_mm: f64,
}

impl Default for DustParams {
    fn default() -> Self {
        Self {
            min_contrast: 0.06,
            texture_factor: 4.5,
            max_speck_mm: 0.8,
            max_width_mm: 0.4,
            scratches: true,
            max_extra_chroma: 0.08,
            min_sharpness: 0.5,
            texture_radius_mm: 0.5,
        }
    }
}

/// One detected defect.
#[derive(Debug, Clone)]
pub struct Defect {
    /// Pixel indices (row-major) covered by the defect, before the safety dilation.
    pub pixels: Vec<usize>,
    /// Bounding box (x0, y0, x1, y1), inclusive, after dilation.
    pub bbox: (usize, usize, usize, usize),
    /// Brighter than its surroundings (dust on dark areas, scratches) or darker (hairs).
    pub bright: bool,
    pub length_px: f32,
    pub width_px: f32,
}

/// Result of [`find_defects`].
#[derive(Debug, Clone)]
pub struct Defects {
    /// Pixels to replace (defects plus a small safety margin).
    pub mask: Mask,
    pub items: Vec<Defect>,
}

impl Defects {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    /// Fraction of the image covered by the mask.
    pub fn coverage(&self) -> f64 {
        self.mask.count() as f64 / (self.mask.w * self.mask.h).max(1) as f64
    }
}

/// Float channels (gamma-encoded 0..1), 1 (grey) or 3 (RGB) planes.
#[derive(Clone, Debug)]
pub struct Planes {
    pub w: usize,
    pub h: usize,
    pub ch: Vec<Vec<f32>>,
}

impl Planes {
    pub fn from_image(img: &DynamicImage) -> Self {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let grey = matches!(
            img,
            DynamicImage::ImageLuma8(_)
                | DynamicImage::ImageLuma16(_)
                | DynamicImage::ImageLumaA8(_)
                | DynamicImage::ImageLumaA16(_)
        );
        if grey {
            let g = img.to_luma32f();
            return Self {
                w,
                h,
                ch: vec![g.into_raw()],
            };
        }
        let rgb = img.to_rgb32f();
        let raw = rgb.as_raw();
        let mut ch = vec![vec![0.0f32; w * h]; 3];
        for (c, plane) in ch.iter_mut().enumerate() {
            plane
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = raw[i * 3 + c]);
        }
        Self { w, h, ch }
    }

    /// Luminance (Rec. 601 weights on gamma values).
    pub fn luma(&self) -> Vec<f32> {
        if self.ch.len() == 1 {
            return self.ch[0].clone();
        }
        (0..self.w * self.h)
            .into_par_iter()
            .map(|i| 0.299 * self.ch[0][i] + 0.587 * self.ch[1][i] + 0.114 * self.ch[2][i])
            .collect()
    }

    #[inline]
    fn chroma(&self, i: usize) -> f32 {
        if self.ch.len() < 3 {
            return 0.0;
        }
        let (r, g, b) = (self.ch[0][i], self.ch[1][i], self.ch[2][i]);
        r.max(g).max(b) - r.min(g).min(b)
    }

    /// Copy the pixels under `mask` into `orig`, keeping its pixel type. Everything else
    /// stays bit-identical.
    pub fn write_masked(&self, orig: &DynamicImage, mask: &Mask) -> DynamicImage {
        let idx: Vec<usize> = (0..self.w * self.h).filter(|&i| mask.data[i]).collect();
        let rgb = |i: usize| -> [f32; 3] {
            if self.ch.len() == 1 {
                [self.ch[0][i]; 3]
            } else {
                [self.ch[0][i], self.ch[1][i], self.ch[2][i]]
            }
        };
        let grey = |i: usize| -> f32 {
            if self.ch.len() == 1 {
                self.ch[0][i]
            } else {
                0.299 * self.ch[0][i] + 0.587 * self.ch[1][i] + 0.114 * self.ch[2][i]
            }
        };
        let q8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let q16 = |v: f32| (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
        let xy = |i: usize| ((i % self.w) as u32, (i / self.w) as u32);
        let mut out = orig.clone();
        match &mut out {
            DynamicImage::ImageLuma8(b) => idx.iter().for_each(|&i| {
                let (x, y) = xy(i);
                b.put_pixel(x, y, Luma([q8(grey(i))]))
            }),
            DynamicImage::ImageLuma16(b) => idx.iter().for_each(|&i| {
                let (x, y) = xy(i);
                b.put_pixel(x, y, Luma([q16(grey(i))]))
            }),
            DynamicImage::ImageRgb8(b) => idx.iter().for_each(|&i| {
                let (x, y) = xy(i);
                let c = rgb(i);
                b.put_pixel(x, y, Rgb([q8(c[0]), q8(c[1]), q8(c[2])]))
            }),
            DynamicImage::ImageRgb16(b) => idx.iter().for_each(|&i| {
                let (x, y) = xy(i);
                let c = rgb(i);
                b.put_pixel(x, y, Rgb([q16(c[0]), q16(c[1]), q16(c[2])]))
            }),
            DynamicImage::ImageRgba8(b) => idx.iter().for_each(|&i| {
                let (x, y) = xy(i);
                let c = rgb(i);
                let a = b.get_pixel(x, y)[3];
                b.put_pixel(x, y, Rgba([q8(c[0]), q8(c[1]), q8(c[2]), a]))
            }),
            DynamicImage::ImageRgba16(b) => idx.iter().for_each(|&i| {
                let (x, y) = xy(i);
                let c = rgb(i);
                let a = b.get_pixel(x, y)[3];
                b.put_pixel(x, y, Rgba([q16(c[0]), q16(c[1]), q16(c[2]), a]))
            }),
            DynamicImage::ImageRgb32F(b) => idx.iter().for_each(|&i| {
                let (x, y) = xy(i);
                b.put_pixel(x, y, Rgb(rgb(i)))
            }),
            _ => {
                let mut b = orig.to_rgb8();
                for &i in &idx {
                    let (x, y) = xy(i);
                    let c = rgb(i);
                    b.put_pixel(x, y, Rgb([q8(c[0]), q8(c[1]), q8(c[2])]));
                }
                return DynamicImage::ImageRgb8(b);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------
// Fast separable filters
// ---------------------------------------------------------------------------------------

fn transpose(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    const B: usize = 32;
    out.par_chunks_mut(h * B.min(w).max(1))
        .enumerate()
        .for_each(|(bx, chunk)| {
            let x0 = bx * B;
            for (k, col) in chunk.chunks_mut(h).enumerate() {
                let x = x0 + k;
                for (y, v) in col.iter_mut().enumerate() {
                    *v = src[y * w + x];
                }
            }
        });
    out
}

/// Running min or max over a window of 2r+1 (van Herk / Gil-Werman), edges replicated.
fn minmax_rows(src: &[f32], w: usize, h: usize, r: usize, max: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let k = 2 * r + 1;
    out.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        let row = &src[y * w..(y + 1) * w];
        let n = w + 2 * r;
        let padded = |i: usize| row[i.saturating_sub(r).min(w - 1)];
        let op = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
        let mut g = vec![0.0f32; n];
        let mut hb = vec![0.0f32; n];
        for i in 0..n {
            let v = padded(i);
            g[i] = if i % k == 0 { v } else { op(g[i - 1], v) };
        }
        for i in (0..n).rev() {
            let v = padded(i);
            hb[i] = if i % k == k - 1 || i == n - 1 {
                v
            } else {
                op(hb[i + 1], v)
            };
        }
        for (x, d) in dst.iter_mut().enumerate() {
            *d = op(hb[x], g[x + k - 1]);
        }
    });
    out
}

fn minmax2d(src: &[f32], w: usize, h: usize, r: usize, max: bool) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    let a = minmax_rows(src, w, h, r, max);
    let t = transpose(&a, w, h);
    let b = minmax_rows(&t, h, w, r, max);
    transpose(&b, h, w)
}

fn box_rows(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        let row = &src[y * w..(y + 1) * w];
        let mut cum = vec![0.0f64; w + 1];
        for i in 0..w {
            cum[i + 1] = cum[i] + row[i] as f64;
        }
        for (x, d) in dst.iter_mut().enumerate() {
            let a = x.saturating_sub(r);
            let b = (x + r + 1).min(w);
            *d = ((cum[b] - cum[a]) / (b - a) as f64) as f32;
        }
    });
    out
}

fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return src.to_vec();
    }
    let a = box_rows(src, w, h, r);
    let t = transpose(&a, w, h);
    let b = box_rows(&t, h, w, r);
    transpose(&b, h, w)
}

// ---------------------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------------------

/// Find dust, hairs and scratches. `dpi` is the image resolution.
pub fn find_defects(img: &DynamicImage, dpi: f64, p: &DustParams) -> Defects {
    find_defects_planes(&Planes::from_image(img), dpi, p)
}

pub fn find_defects_planes(pl: &Planes, dpi: f64, p: &DustParams) -> Defects {
    let (w, h) = (pl.w, pl.h);
    let pxmm = dpi / 25.4;
    let mut mask = Mask::new(w, h);
    if w < 8 || h < 8 {
        return Defects {
            mask,
            items: Vec::new(),
        };
    }
    let y = pl.luma();
    let r = ((p.max_speck_mm.max(p.max_width_mm) * pxmm / 2.0).ceil() as usize + 1).max(2);
    // Background by an alternating filter: opening removes small bright things, the
    // closing that follows fills small dark ones. Measuring both polarities against the
    // same background avoids halos of the opposite polarity next to every defect.
    let opened = minmax2d(&minmax2d(&y, w, h, r, false), w, h, r, true);
    let bg = minmax2d(&minmax2d(&opened, w, h, r, true), w, h, r, false);
    let tb: Vec<f32> = y
        .par_iter()
        .zip(&bg)
        .map(|(a, b)| (a - b).max(0.0))
        .collect();
    let td: Vec<f32> = y
        .par_iter()
        .zip(&bg)
        .map(|(a, b)| (b - a).max(0.0))
        .collect();

    // Texture energy: local mean of both top-hats, clipped so a defect does not raise its
    // own threshold much.
    let clip = p.min_contrast;
    let both: Vec<f32> = tb
        .par_iter()
        .zip(&td)
        .map(|(a, b)| (a + b).min(clip))
        .collect();
    let er = ((p.texture_radius_mm * pxmm).round() as usize).max(3);
    let energy = box_blur(&both, w, h, er);

    let noise = noise_sigma(&y, w, h);
    // Around 300 dpi print blur is only about a pixel, so the bar is a little higher;
    // below that, print detail is as crisp as dust and the test means nothing.
    let sharp_min = if dpi >= 400.0 {
        p.min_sharpness
    } else if dpi >= 250.0 {
        p.min_sharpness * 1.15
    } else {
        0.0
    };
    let ctx = Ctx {
        y: &y,
        pl,
        pxmm,
        p,
        noise,
        sharp_min,
    };
    let mut items = Vec::new();
    for (bright, t) in [(true, &tb), (false, &td)] {
        let thr: Vec<f32> = energy
            .par_iter()
            .map(|e| p.min_contrast.max(p.texture_factor * e))
            .collect();
        let low = Mask {
            w,
            h,
            data: t
                .par_iter()
                .zip(&thr)
                .map(|(v, th)| *v > 0.6 * th)
                .collect(),
        };
        let dt = low.distance_transform();
        let (_, comps) = low.components();
        let accepted: Vec<Defect> = comps
            .par_iter()
            .filter_map(|c| classify(&ctx, c, t, &thr, &dt, &low, bright))
            .collect();
        let mut accepted = accepted;
        extend_lines(&mut accepted, t, &thr, w, h, pxmm);
        items.extend(accepted);
    }

    // Safety margin: the soft edge of a speck is darker/brighter than it looks.
    let margin = 1.0 + (0.03 * pxmm) as f32;
    for d in &items {
        for &i in &d.pixels {
            mask.data[i] = true;
        }
    }
    let mask = mask.dilate(margin);
    let m = margin.ceil() as usize;
    for d in &mut items {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for &i in &d.pixels {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        d.bbox = (
            x0.saturating_sub(m),
            y0.saturating_sub(m),
            (x1 + m).min(w - 1),
            (y1 + m).min(h - 1),
        );
    }
    log::debug!(
        "restore: {} defects, {:.3}% of pixels",
        items.len(),
        mask.count() as f64 * 100.0 / (w * h) as f64
    );
    Defects { mask, items }
}

/// Follow hairs and scratches into their fainter stretches: from each end of an elongated
/// defect, grow outwards along its axis through pixels with a weaker response (40% of the
/// local threshold), at most 2 mm, and only while the growth stays as thin as the line.
fn extend_lines(items: &mut [Defect], t: &[f32], thr: &[f32], w: usize, h: usize, pxmm: f64) {
    let max_steps = (2.0 * pxmm).ceil() as usize;
    let mut taken = vec![false; w * h];
    for d in items.iter() {
        for &i in &d.pixels {
            taken[i] = true;
        }
    }
    for d in items.iter_mut() {
        if d.length_px < 3.0 * d.width_px || d.pixels.len() < 4 {
            continue;
        }
        // Principal axis.
        let n = d.pixels.len() as f64;
        let (mx, my) = d.pixels.iter().fold((0.0, 0.0), |a, &i| {
            (a.0 + (i % w) as f64 / n, a.1 + (i / w) as f64 / n)
        });
        let (mut cxx, mut cyy, mut cxy) = (0.0, 0.0, 0.0);
        for &i in &d.pixels {
            let (dx, dy) = ((i % w) as f64 - mx, (i / w) as f64 - my);
            cxx += dx * dx;
            cyy += dy * dy;
            cxy += dx * dy;
        }
        let ang = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
        let (ux, uy) = (ang.cos(), ang.sin());
        let proj = |i: usize| ((i % w) as f64 - mx) * ux + ((i / w) as f64 - my) * uy;
        let (pmin, pmax) = d.pixels.iter().fold((f64::MAX, f64::MIN), |a, &i| {
            (a.0.min(proj(i)), a.1.max(proj(i)))
        });
        let max_front = (2.0 * d.width_px + 2.0).ceil() as usize;
        let mut added = Vec::new();
        for (sign, end) in [(1.0f64, pmax), (-1.0, pmin)] {
            // Start from the pixels within a width of this end.
            let mut front: Vec<usize> = d
                .pixels
                .iter()
                .copied()
                .filter(|&i| sign * (proj(i) - end) > -(d.width_px as f64 + 1.0))
                .collect();
            let mut reach = end;
            for _ in 0..max_steps {
                let mut next = Vec::new();
                for &i in &front {
                    let (x, y) = ((i % w) as isize, (i / w) as isize);
                    for dy in -1..=1isize {
                        for dx in -1..=1isize {
                            let (nx, ny) = (x + dx, y + dy);
                            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                                continue;
                            }
                            let j = ny as usize * w + nx as usize;
                            // Only forwards, past the current reach (minus a little slack).
                            if taken[j] || t[j] < 0.4 * thr[j] || sign * (proj(j) - reach) < -0.5 {
                                continue;
                            }
                            taken[j] = true;
                            next.push(j);
                        }
                    }
                }
                if next.is_empty() || next.len() > max_front {
                    for &j in &next {
                        taken[j] = false;
                    }
                    break;
                }
                reach = next
                    .iter()
                    .map(|&j| sign * proj(j))
                    .fold(f64::MIN, f64::max)
                    * sign;
                added.extend_from_slice(&next);
                front = next;
            }
        }
        d.pixels.extend(added);
    }
}

/// Robust per-pixel noise sigma from horizontal neighbour differences (MAD).
fn noise_sigma(y: &[f32], w: usize, h: usize) -> f32 {
    let step = ((w * h) / 200_000).max(1);
    let mut d: Vec<f32> = (0..w * h)
        .step_by(step)
        .filter(|i| i % w + 1 < w)
        .map(|i| (y[i + 1] - y[i]).abs())
        .collect();
    if d.is_empty() {
        return 0.0;
    }
    let mid = d.len() / 2;
    let m = *d.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
    m / 0.6745 / std::f32::consts::SQRT_2
}

/// Pixels of `c` that stand out from a plane fitted to the surrounding background.
fn refine_core(
    c: &[usize],
    low: &Mask,
    y: &[f32],
    w: usize,
    h: usize,
    bright: bool,
) -> Option<Vec<usize>> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for &i in c {
        let (x, yy) = (i % w, i / w);
        x0 = x0.min(x);
        y0 = y0.min(yy);
        x1 = x1.max(x);
        y1 = y1.max(yy);
    }
    let pad = 3;
    let (rx0, ry0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
    let (rx1, ry1) = ((x1 + pad).min(w - 1), (y1 + pad).min(h - 1));
    let (cx, cy) = ((rx0 + rx1) as f64 / 2.0, (ry0 + ry1) as f64 / 2.0);
    // Least squares for v = a + b*dx + c*dy over ring pixels.
    let mut m = [[0.0f64; 3]; 3];
    let mut r = [0.0f64; 3];
    let mut n = 0usize;
    for yy in ry0..=ry1 {
        for x in rx0..=rx1 {
            let i = yy * w + x;
            if low.data[i] {
                continue;
            }
            // Only the ring close to the candidate's bounding box.
            let edge = x < x0 || x > x1 || yy < y0 || yy > y1;
            if !edge {
                continue;
            }
            let f = [1.0, x as f64 - cx, yy as f64 - cy];
            for a in 0..3 {
                for b in 0..3 {
                    m[a][b] += f[a] * f[b];
                }
                r[a] += f[a] * y[i] as f64;
            }
            n += 1;
        }
    }
    if n < 8 {
        return None;
    }
    let coef = solve3(m, r)?;
    let dev = |i: usize| -> f32 {
        let (x, yy) = ((i % w) as f64 - cx, (i / w) as f64 - cy);
        let bg = coef[0] + coef[1] * x + coef[2] * yy;
        let d = y[i] - bg as f32;
        if bright { d } else { -d }
    };
    let maxdev = c.iter().map(|&i| dev(i)).fold(0.0f32, f32::max);
    if maxdev <= 0.0 {
        return Some(Vec::new());
    }
    Some(
        c.iter()
            .copied()
            .filter(|&i| dev(i) > 0.4 * maxdev)
            .collect(),
    )
}

fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&m);
    if d.abs() < 1e-9 {
        return None;
    }
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let mut mk = m;
        for row in 0..3 {
            mk[row][k] = r[row];
        }
        *o = det(&mk) / d;
    }
    Some(out)
}

struct Ctx<'a> {
    y: &'a [f32],
    pl: &'a Planes,
    pxmm: f64,
    p: &'a DustParams,
    noise: f32,
    sharp_min: f32,
}

fn classify(
    ctx: &Ctx,
    c: &[usize],
    t: &[f32],
    thr: &[f32],
    dt: &[f32],
    low: &Mask,
    bright: bool,
) -> Option<Defect> {
    let (pl, pxmm, p) = (ctx.pl, ctx.pxmm, ctx.p);
    let w = pl.w;
    let rej = |why: &str| -> Option<Defect> {
        if log::log_enabled!(log::Level::Trace) && c.len() < 5000 {
            let (sx, sy) = c
                .iter()
                .fold((0usize, 0usize), |a, &i| (a.0 + i % w, a.1 + i / w));
            let (x, y) = (sx / c.len(), sy / c.len());
            log::trace!("restore: reject ({x},{y}) n={} {why}", c.len());
        }
        None
    };
    let max_area = {
        let d = p.max_speck_mm * pxmm;
        let l = 40.0 * pxmm; // generous cap on scratch area: 40 mm long at max width
        (d * d).max(l * p.max_width_mm * pxmm) as usize + 4
    };
    if c.len() < 2 || c.len() > max_area {
        return rej("area");
    }
    // Needs at least one seed pixel above the full threshold.
    let peak = c.iter().map(|&i| t[i] / thr[i]).fold(0.0f32, f32::max);
    if peak < 1.0 {
        return rej("below threshold");
    }
    // Refine against the local background: fit a plane to the ring of pixels around the
    // candidate and keep pixels that deviate by at least 40% of the peak deviation. This
    // drops the plateaus that square openings leave in nearby texture.
    let tmax = c.iter().map(|&i| t[i]).fold(0.0f32, f32::max);
    let core = refine_core(c, low, ctx.y, w, pl.h, bright);
    let core = core.unwrap_or_else(|| c.iter().copied().filter(|&i| t[i] > 0.35 * tmax).collect());
    let c = &core[..];
    if c.len() < 2 {
        return rej("no core");
    }
    // Sharpness: walk across the boundary (one pixel in, three out). On a crisp edge the
    // largest single step is most of the total change; print blur spreads it over several
    // pixels. The median over the boundary keeps a sharp neighbour (a scratch running
    // beside a soft branch) from vouching for a soft blob.
    if ctx.sharp_min > 0.0 {
        let y = ctx.y;
        let inside: std::collections::HashSet<usize> = c.iter().copied().collect();
        let mut ratios: Vec<f32> = Vec::new();
        let (wi, hi) = (w as isize, pl.h as isize);
        let at = |x: isize, yy: isize| -> Option<f32> {
            (x >= 0 && yy >= 0 && x < wi && yy < hi).then(|| y[(yy * wi + x) as usize])
        };
        let min_total = (4.0 * ctx.noise).max(0.3 * tmax);
        for &i in c {
            let (x, yy) = ((i % w) as isize, (i / w) as isize);
            for (dx, dy) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                let o = (yy + dy) * wi + (x + dx);
                if x + dx < 0 || yy + dy < 0 || x + dx >= wi || yy + dy >= hi {
                    continue;
                }
                if inside.contains(&(o as usize)) {
                    continue;
                }
                // Profile from one pixel inside to three outside the boundary.
                let mut v: Vec<f32> = Vec::with_capacity(5);
                let inner = (x - dx, yy - dy);
                if inner.0 >= 0
                    && inner.1 >= 0
                    && inner.0 < wi
                    && inner.1 < hi
                    && inside.contains(&((inner.1 * wi + inner.0) as usize))
                {
                    v.push(y[(inner.1 * wi + inner.0) as usize]);
                }
                let mut ok = true;
                for k in 0..4 {
                    match at(x + k * dx, yy + k * dy) {
                        Some(val) => v.push(val),
                        None => ok = false,
                    }
                }
                if !ok {
                    continue;
                }
                let total = (v[0] - v[v.len() - 1]).abs();
                if total < min_total {
                    continue;
                }
                let step = v
                    .windows(2)
                    .map(|p| (p[0] - p[1]).abs())
                    .fold(0.0f32, f32::max);
                ratios.push(step / total);
            }
        }
        // No clear edge anywhere: the contrast is all gradual, not a defect.
        if ratios.is_empty() {
            return rej("no edge");
        }
        let k = ratios.len() / 2;
        let r = *ratios.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
        if r < ctx.sharp_min {
            return rej("soft");
        }
    }

    let width = 2.0 * c.iter().map(|&i| dt[i]).fold(0.0f32, f32::max) - 0.5;
    if width as f64 > p.max_width_mm.max(p.max_speck_mm) * pxmm + 1.0 {
        return rej("too wide");
    }
    // Second moments: length of the principal axis.
    let n = c.len() as f64;
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for &i in c {
        let (x, y) = (i % w, i / w);
        sx += x as f64;
        sy += y as f64;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut cxx, mut cyy, mut cxy) = (0.0f64, 0.0f64, 0.0f64);
    for &i in c {
        let dx = (i % w) as f64 - mx;
        let dy = (i / w) as f64 - my;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    let (cxx, cyy, cxy) = (cxx / n, cyy / n, cxy / n);
    let tr = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let l1 = tr / 2.0 + ((tr * tr / 4.0 - det).max(0.0)).sqrt();
    let length = ((12.0 * l1).sqrt() as f32).max(1.0);
    let mean_width = (n as f32 / length).max(1.0);
    let elongated = length > 3.0 * mean_width && length as f64 > 0.5 * pxmm;
    if elongated {
        if !p.scratches || mean_width as f64 > p.max_width_mm * pxmm + 1.0 {
            return rej("scratch too wide");
        }
    } else {
        let diam = 2.0 * (n / std::f64::consts::PI).sqrt();
        if diam > p.max_speck_mm * pxmm + 1.0 {
            return rej("speck too big");
        }
    }
    // Colour: dust and scratches are colourless, so they never add chroma.
    if pl.ch.len() == 3 {
        // Judge the colour on the strongest part of the defect.
        let centre: Vec<usize> = c.iter().copied().filter(|&i| t[i] > 0.6 * tmax).collect();
        let comp_chroma =
            centre.iter().map(|&i| pl.chroma(i)).sum::<f32>() / centre.len().max(1) as f32;
        let pad = (width.ceil() as usize + 3).max(3);
        let (rx0, ry0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
        let (rx1, ry1) = ((x1 + pad).min(pl.w - 1), (y1 + pad).min(pl.h - 1));
        let (mut s, mut k) = (0.0f32, 0usize);
        let step = (((rx1 - rx0 + 1) * (ry1 - ry0 + 1)) / 400).max(1);
        let mut idx = 0usize;
        for yy in ry0..=ry1 {
            for xx in rx0..=rx1 {
                idx += 1;
                if !idx.is_multiple_of(step) {
                    continue;
                }
                let j = yy * w + xx;
                if !low.data[j] {
                    s += pl.chroma(j);
                    k += 1;
                }
            }
        }
        if k > 0 && comp_chroma > s / k as f32 + p.max_extra_chroma {
            return rej("colourful");
        }
    }
    log::trace!(
        "restore: accept {:?} n={} len={length:.1} w={width:.1} bright={bright}",
        (x0, y0, x1, y1),
        c.len()
    );
    Some(Defect {
        pixels: c.to_vec(),
        bbox: (x0, y0, x1, y1),
        bright,
        length_px: length,
        width_px: width.max(1.0),
    })
}

// ---------------------------------------------------------------------------------------
// Filling
// ---------------------------------------------------------------------------------------

/// Fill every masked pixel in place with the healing brush.
pub fn heal_planes(pl: &mut Planes, mask: &Mask) {
    let w = pl.w;
    let mut known: Vec<bool> = mask.data.iter().map(|m| !m).collect();
    let (_, comps) = mask.components();
    for comp in comps {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for &i in &comp {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let span = (x1 - x0 + 1).max(y1 - y0 + 1);
        // Estimated stroke width, for the chunk size of long defects.
        let thick = (comp.len() as f64 / span as f64).max(1.0);
        let chunk = ((6.0 * thick) as usize).clamp(12, 48);
        if span <= chunk {
            heal_region(pl, &mut known, &comp);
            continue;
        }
        // Long defect: fill in cells along its main direction so each source patch is
        // chosen for the local content.
        let horizontal = x1 - x0 >= y1 - y0;
        let mut cells: std::collections::BTreeMap<(usize, usize), Vec<usize>> = Default::default();
        for &i in &comp {
            let (x, y) = (i % w, i / w);
            let (a, b) = if horizontal {
                ((x - x0) / chunk, (y - y0) / chunk)
            } else {
                ((y - y0) / chunk, (x - x0) / chunk)
            };
            cells.entry((a, b)).or_default().push(i);
        }
        for (_, px) in cells {
            heal_region(pl, &mut known, &px);
        }
    }
}

/// Heal one connected region: pick a texture source offset, then add a harmonic correction
/// so the copied patch matches the region's border exactly.
fn heal_region(pl: &mut Planes, known: &mut [bool], region: &[usize]) {
    let (w, h) = (pl.w as isize, pl.h as isize);
    let (mut x0, mut y0, mut x1, mut y1) = (isize::MAX, isize::MAX, 0isize, 0isize);
    for &i in region {
        let (x, y) = ((i % pl.w) as isize, (i / pl.w) as isize);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    // Local grid with a 3-pixel border.
    const B: isize = 3;
    let (gx0, gy0) = ((x0 - B).max(0), (y0 - B).max(0));
    let (gx1, gy1) = ((x1 + B).min(w - 1), (y1 + B).min(h - 1));
    let (gw, gh) = ((gx1 - gx0 + 1) as usize, (gy1 - gy0 + 1) as usize);
    // 0 = outside / ignore, 1 = in region (unknown), 2 = known ring pixel
    let mut kind = vec![0u8; gw * gh];
    let gidx = |x: isize, y: isize| (y - gy0) as usize * gw + (x - gx0) as usize;
    for &i in region {
        let (x, y) = ((i % pl.w) as isize, (i / pl.w) as isize);
        kind[gidx(x, y)] = 1;
    }
    let mut ring: Vec<(isize, isize)> = Vec::new();
    for gy in 0..gh {
        for gx in 0..gw {
            let (x, y) = (gx as isize + gx0, gy as isize + gy0);
            if kind[gy * gw + gx] != 0 || !known[(y * w + x) as usize] {
                continue;
            }
            // Within 2 px (Chebyshev) of the region?
            let mut near = false;
            'n: for dy in -2..=2isize {
                for dx in -2..=2isize {
                    let (nx, ny) = (gx as isize + dx, gy as isize + dy);
                    if nx >= 0
                        && ny >= 0
                        && nx < gw as isize
                        && ny < gh as isize
                        && kind[ny as usize * gw + nx as usize] == 1
                    {
                        near = true;
                        break 'n;
                    }
                }
            }
            if near {
                kind[gy * gw + gx] = 2;
                ring.push((x, y));
            }
        }
    }
    let nch = pl.ch.len();
    let px = |pl: &Planes, x: isize, y: isize, c: usize| pl.ch[c][(y * w + x) as usize];

    // Source offset search: rings of candidate displacements, scored on the border.
    let span = (x1 - x0 + 1).max(y1 - y0 + 1) as f64;
    let mut best: Option<((isize, isize), f64)> = None;
    if !ring.is_empty() && region.len() as f64 / span >= 2.5 {
        for rf in [1.0, 1.5, 2.2, 3.2] {
            let rad = (span + 2.0) * rf + 1.0;
            for k in 0..24 {
                let a = k as f64 * std::f64::consts::TAU / 24.0;
                let d = (
                    (rad * a.cos()).round() as isize,
                    (rad * a.sin()).round() as isize,
                );
                if d == (0, 0) {
                    continue;
                }
                // Bounds of the shifted grid.
                if gx0 + d.0 < 0 || gy0 + d.1 < 0 || gx1 + d.0 >= w || gy1 + d.1 >= h {
                    continue;
                }
                let ok = region.iter().all(|&i| {
                    let (x, y) = ((i % pl.w) as isize + d.0, (i / pl.w) as isize + d.1);
                    known[(y * w + x) as usize]
                });
                if !ok {
                    continue;
                }
                let mut ssd = 0.0f64;
                for &(x, y) in &ring {
                    for c in 0..nch {
                        let e = px(pl, x, y, c) - px(pl, x + d.0, y + d.1, c);
                        ssd += (e * e) as f64;
                    }
                }
                let score = ssd / ring.len() as f64 + 1e-6 * rf;
                if best.is_none_or(|b| score < b.1) {
                    best = Some((d, score));
                }
            }
        }
    }
    // Thin lines (a couple of pixels) get the plain harmonic fill: there is no room for
    // texture to show, and a smooth fill is closer on average.
    let thin_region = region.len() as f64 / span < 2.5;
    let d = if thin_region { None } else { best.map(|b| b.0) };

    // Harmonic correction c on the region with c = I(b) - I(b + d) on the border.
    let n = gw * gh;
    for ch in 0..nch {
        let mut val = vec![0.0f32; n];
        let mut fixed = vec![false; n];
        for gy in 0..gh {
            for gx in 0..gw {
                let gi = gy * gw + gx;
                if kind[gi] == 2 {
                    let (x, y) = (gx as isize + gx0, gy as isize + gy0);
                    let src = d.map_or(0.0, |d| px(pl, x + d.0, y + d.1, ch));
                    val[gi] = px(pl, x, y, ch) - src;
                    fixed[gi] = true;
                }
            }
        }
        // Onion-peel initialisation from the border inwards.
        let mut todo: Vec<usize> = (0..n).filter(|&i| kind[i] == 1).collect();
        let mut have = fixed.clone();
        while !todo.is_empty() {
            let mut next = Vec::new();
            let mut set = Vec::new();
            for &gi in &todo {
                let (gx, gy) = ((gi % gw) as isize, (gi / gw) as isize);
                let (mut s, mut k) = (0.0f32, 0u32);
                for (dx, dy) in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                ] {
                    let (nx, ny) = (gx + dx, gy + dy);
                    if nx >= 0 && ny >= 0 && nx < gw as isize && ny < gh as isize {
                        let j = ny as usize * gw + nx as usize;
                        if have[j] {
                            s += val[j];
                            k += 1;
                        }
                    }
                }
                if k > 0 {
                    set.push((gi, s / k as f32));
                } else {
                    next.push(gi);
                }
            }
            if set.is_empty() {
                break; // isolated (no known border at all)
            }
            for (gi, v) in set {
                val[gi] = v;
                have[gi] = true;
            }
            todo = next;
        }
        // Gauss-Seidel with over-relaxation (Neumann where the neighbour is unknown).
        let thin = (region.len() as f64 / span).max(1.0);
        let iters = (20.0 + 4.0 * thin * thin).min(200.0) as usize;
        let unknown: Vec<usize> = (0..n).filter(|&i| kind[i] == 1).collect();
        for _ in 0..iters {
            for &gi in &unknown {
                let (gx, gy) = ((gi % gw) as isize, (gi / gw) as isize);
                let (mut s, mut k) = (0.0f32, 0u32);
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (gx + dx, gy + dy);
                    if nx >= 0 && ny >= 0 && nx < gw as isize && ny < gh as isize {
                        let j = ny as usize * gw + nx as usize;
                        if kind[j] != 0 {
                            s += val[j];
                            k += 1;
                        }
                    }
                }
                if k > 0 {
                    let target = s / k as f32;
                    val[gi] += 1.8 * (target - val[gi]);
                }
            }
        }
        for &gi in &unknown {
            let (x, y) = ((gi % gw) as isize + gx0, (gi / gw) as isize + gy0);
            let src = d.map_or(0.0, |d| px(pl, x + d.0, y + d.1, ch));
            pl.ch[ch][(y * w + x) as usize] = (src + val[gi]).clamp(0.0, 1.0);
        }
    }
    for &i in region {
        known[i] = true;
    }
}

/// Fill the masked pixels of `img`. Everything outside the mask is left untouched.
pub fn heal(img: &DynamicImage, mask: &Mask) -> DynamicImage {
    if mask.count() == 0 {
        return img.clone();
    }
    let mut pl = Planes::from_image(img);
    heal_planes(&mut pl, mask);
    pl.write_masked(img, mask)
}

/// Detect and remove dust and scratches. Returns the cleaned image and what was found.
pub fn remove_dust(img: &DynamicImage, dpi: f64, p: &DustParams) -> (DynamicImage, Defects) {
    let mut pl = Planes::from_image(img);
    let defects = find_defects_planes(&pl, dpi, p);
    if defects.is_empty() {
        return (img.clone(), defects);
    }
    heal_planes(&mut pl, &defects.mask);
    (pl.write_masked(img, &defects.mask), defects)
}

/// Paint the mask in red over the image (for previews and debugging).
pub fn overlay(img: &DynamicImage, mask: &Mask) -> DynamicImage {
    let mut b = img.to_rgb8();
    for (i, p) in b.pixels_mut().enumerate() {
        if mask.data[i] {
            *p = Rgb([255, (p[1] as u16 / 3) as u8, (p[2] as u16 / 3) as u8]);
        }
    }
    DynamicImage::ImageRgb8(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;

    #[test]
    fn minmax_matches_naive() {
        let (w, h) = (23, 17);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 101) as f32).collect();
        for r in [1usize, 2, 5] {
            let fast = minmax2d(&src, w, h, r, true);
            for y in 0..h {
                for x in 0..w {
                    let mut m = f32::MIN;
                    for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                        for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                            m = m.max(src[yy * w + xx]);
                        }
                    }
                    assert_eq!(fast[y * w + x], m, "r={r} at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn cleans_synthetic_print() {
        let scene = crate::synth_dust::render(5, 600.0, 30.0, 20.0, 25);
        let p = DustLevel::Normal.params().unwrap();
        let (out, defects) = remove_dust(&scene.dirty, 600.0, &p);
        let mut st = crate::synth_dust::DustStats::default();
        crate::synth_dust::score(&mut st, &scene, &defects.mask, &out);
        assert!(st.visible_recall() > 0.75, "{st:?}");
        assert!(st.err_after < st.err_before * 0.3, "{st:?}");
        assert!(st.err_global_after < st.err_global_before, "{st:?}");
        // Pixels outside the mask are bit-identical.
        let (a, b) = (scene.dirty.to_rgb8(), out.to_rgb8());
        for (i, (pa, pb)) in a.pixels().zip(b.pixels()).enumerate() {
            if !defects.mask.data[i] {
                assert_eq!(pa, pb);
            }
        }
    }

    #[test]
    fn keeps_bit_depth_and_clean_images() {
        let img = DynamicImage::ImageRgb16(ImageBuffer::from_fn(200, 150, |x, y| {
            let v = (20000 + x * 100 + y * 50) as u16;
            Rgb([v, v / 2 + 9000, 30000])
        }));
        let p = DustLevel::Normal.params().unwrap();
        let (out, d) = remove_dust(&img, 600.0, &p);
        assert!(d.is_empty());
        assert!(matches!(out, DynamicImage::ImageRgb16(_)));
        // A single sharp white speck is found and removed.
        let mut b = img.to_rgb16();
        for y in 70..74 {
            for x in 100..104 {
                b.put_pixel(x, y, Rgb([65535, 65535, 65535]));
            }
        }
        let dirty = DynamicImage::ImageRgb16(b);
        let (out, d) = remove_dust(&dirty, 600.0, &p);
        assert_eq!(d.items.len(), 1);
        let px = out.to_rgb16().get_pixel(101, 71).0;
        let want = img.to_rgb16().get_pixel(101, 71).0;
        for c in 0..3 {
            assert!(
                (px[c] as i32 - want[c] as i32).abs() < 1500,
                "{px:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn grey_images() {
        let img = DynamicImage::ImageLuma8(ImageBuffer::from_fn(120, 100, |x, y| {
            let mut v = 90 + (x / 3) as u8;
            if (40..43).contains(&x) && (30..33).contains(&y) {
                v = 250;
            }
            Luma([v])
        }));
        let (out, d) = remove_dust(&img, 600.0, &DustLevel::Normal.params().unwrap());
        assert_eq!(d.items.len(), 1);
        assert!(matches!(out, DynamicImage::ImageLuma8(_)));
        assert!(out.to_luma8().get_pixel(41, 31).0[0] < 140);
    }

    #[test]
    fn box_blur_flat() {
        let src = vec![0.5f32; 40 * 30];
        let b = box_blur(&src, 40, 30, 4);
        assert!(b.iter().all(|v| (v - 0.5).abs() < 1e-6));
    }
}
