//! Synthetic prints with dust, hairs and scratches, plus scoring, for tuning
//! [`crate::restore`].
//!
//! The print content is deliberately hostile: fine texture, thin branches, small soft
//! highlights and coloured dots are all things a dust filter must leave alone. Print
//! detail is softened by a print blur (prints resolve far less than a 600 dpi scan), while
//! defects sit on the surface and are rendered sharp, as the scanner sees them.

use crate::raster::Mask;
use crate::synth::Rng;
use image::{DynamicImage, Rgb32FImage};
use rayon::prelude::*;
use serde::Serialize;

pub struct DustScene {
    /// Clean print.
    pub clean: DynamicImage,
    /// Print with defects.
    pub dirty: DynamicImage,
    /// Pixels a defect visibly changed (coverage above ~25%).
    pub truth: Mask,
    /// Pixels any defect touched at all (used to forgive masks that hug a defect).
    pub touched: Mask,
    /// Pixel lists of each defect (coverage above 50%), for per-defect recall.
    pub defects: Vec<Vec<usize>>,
    /// "speck", "hair" or "scratch", per defect.
    pub kinds: Vec<&'static str>,
    pub dpi: f64,
}

fn hash(x: i64, y: i64, s: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ s;
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    ((h >> 40) as f32) / (1u64 << 24) as f32
}

fn vnoise(x: f64, y: f64, s: u64) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (xi, yi) = (x0 as i64, y0 as i64);
    let a = hash(xi, yi, s);
    let b = hash(xi + 1, yi, s);
    let c = hash(xi, yi + 1, s);
    let d = hash(xi + 1, yi + 1, s);
    (a + (b - a) * sx) * (1.0 - sy) + (c + (d - c) * sx) * sy
}

fn fbm(x: f64, y: f64, s: u64, oct: u32) -> f32 {
    let (mut v, mut amp, mut f) = (0.0, 0.5, 1.0);
    for o in 0..oct {
        v += amp * vnoise(x * f, y * f, s.wrapping_add(o as u64));
        amp *= 0.5;
        f *= 2.0;
    }
    v
}

fn seg_dist(px: f64, py: f64, a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((px - a.0) * dx + (py - a.1) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (a.0 + t * dx, a.1 + t * dy);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

/// A shape drawn as the union of thick polylines and discs (mm units).
struct Stroke {
    pts: Vec<(f64, f64)>,
    /// Half width, mm.
    hw: f64,
    color: [f32; 3],
    opacity: f32,
}

impl Stroke {
    fn bbox(&self) -> (f64, f64, f64, f64) {
        let mut b = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in &self.pts {
            b = (b.0.min(p.0), b.1.min(p.1), b.2.max(p.0), b.3.max(p.1));
        }
        (b.0 - self.hw, b.1 - self.hw, b.2 + self.hw, b.3 + self.hw)
    }
    fn dist(&self, x: f64, y: f64) -> f64 {
        if self.pts.len() == 1 {
            let p = self.pts[0];
            return ((x - p.0).powi(2) + (y - p.1).powi(2)).sqrt();
        }
        self.pts
            .windows(2)
            .map(|s| seg_dist(x, y, s[0], s[1]))
            .fold(f64::MAX, f64::min)
    }
    /// Coverage of pixel (x, y) given pixel size `px` (mm), with `soft` mm of edge blur.
    fn coverage(&self, x: f64, y: f64, px: f64, soft: f64) -> f32 {
        let d = self.dist(x, y) - self.hw;
        let s = (px * 0.5).max(soft);
        (0.5 - d / (2.0 * s)).clamp(0.0, 1.0) as f32
    }
}

fn paint(
    img: &mut [f32],
    w: usize,
    h: usize,
    pxmm: f64,
    st: &Stroke,
    soft: f64,
) -> Vec<(usize, f32)> {
    let (bx0, by0, bx1, by1) = st.bbox();
    let pad = soft * 2.0 + 1.0 / pxmm;
    let x0 = (((bx0 - pad) * pxmm).floor().max(0.0)) as usize;
    let y0 = (((by0 - pad) * pxmm).floor().max(0.0)) as usize;
    let x1 = (((bx1 + pad) * pxmm).ceil() as usize).min(w.saturating_sub(1));
    let y1 = (((by1 + pad) * pxmm).ceil() as usize).min(h.saturating_sub(1));
    let mut touched = Vec::new();
    if x0 > x1 || y0 > y1 {
        return touched;
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (mx, my) = ((x as f64 + 0.5) / pxmm, (y as f64 + 0.5) / pxmm);
            let c = st.coverage(mx, my, 1.0 / pxmm, soft) * st.opacity;
            if c <= 0.0 {
                continue;
            }
            let i = y * w + x;
            for ch in 0..3 {
                img[i * 3 + ch] += (st.color[ch] - img[i * 3 + ch]) * c;
            }
            touched.push((i, c));
        }
    }
    touched
}

fn random_walk(
    rng: &mut Rng,
    start: (f64, f64),
    len: f64,
    step: f64,
    curl: f64,
) -> Vec<(f64, f64)> {
    let mut pts = vec![start];
    let mut a = rng.range(0.0, std::f64::consts::TAU);
    let mut p = start;
    let n = (len / step).ceil() as usize;
    for _ in 0..n {
        a += rng.range(-curl, curl);
        p = (p.0 + step * a.cos(), p.1 + step * a.sin());
        pts.push(p);
    }
    pts
}

/// Render a print patch of `w_mm` x `h_mm` at `dpi` and add `n_defects` defects.
pub fn render(seed: u64, dpi: f64, w_mm: f64, h_mm: f64, n_defects: usize) -> DustScene {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(17));
    let pxmm = dpi / 25.4;
    let (w, h) = (
        (w_mm * pxmm).round() as usize,
        (h_mm * pxmm).round() as usize,
    );
    let s = rng.next_u64();
    let sky = [
        rng.range(0.35, 0.85) as f32,
        rng.range(0.45, 0.9) as f32,
        rng.range(0.55, 0.95) as f32,
    ];
    let ground = [
        rng.range(0.1, 0.6) as f32,
        rng.range(0.15, 0.6) as f32,
        rng.range(0.05, 0.45) as f32,
    ];
    let horizon = rng.range(0.3, 0.6);
    let texture = rng.range(0.15, 0.45) as f32;
    let mono = rng.f64() < 0.2;
    // Base: sky gradient with clouds, textured ground (foliage-like fine detail).
    let mut img = vec![0.0f32; w * h * 3];
    img.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let (u, v) = (x as f64 / pxmm, y as f64 / pxmm);
            let t = v / h_mm;
            let n = fbm(u / 12.0, v / 12.0, s, 4);
            let c = if t < horizon + (n as f64 - 0.5) * 0.2 {
                let cloud = (fbm(u / 6.0, v / 4.0, s ^ 3, 5) - 0.45).max(0.0) * 1.6;
                let k = (t / horizon) as f32;
                [
                    sky[0] * (1.0 - 0.25 * k) + cloud,
                    sky[1] * (1.0 - 0.15 * k) + cloud,
                    sky[2] * (1.0 - 0.05 * k) + cloud,
                ]
            } else {
                // Foliage: strong mid and fine scale texture.
                let f = fbm(u / 0.9, v / 0.9, s ^ 5, 4) - 0.5;
                let g = fbm(u / 4.0, v / 4.0, s ^ 9, 3);
                let m = 0.55 + 0.9 * g + texture * 2.0 * f;
                [ground[0] * m, ground[1] * m, ground[2] * m]
            };
            for ch in 0..3 {
                row[x * 3 + ch] = c[ch].clamp(0.0, 1.0);
            }
        }
    });
    // Legitimate details: objects with edges, thin branches, soft highlights, coloured dots.
    let mut legit = Vec::new();
    for _ in 0..rng.range(2.0, 6.0) as usize {
        let c = (rng.range(0.0, w_mm), rng.range(0.0, h_mm));
        let r = rng.range(1.5, 8.0);
        let col = [rng.f64() as f32, rng.f64() as f32, rng.f64() as f32];
        legit.push(Stroke {
            pts: vec![c],
            hw: r,
            color: col,
            opacity: 1.0,
        });
    }
    for _ in 0..rng.range(3.0, 10.0) as usize {
        let start = (rng.range(0.0, w_mm), rng.range(0.0, h_mm));
        let pts = {
            let l = rng.range(5.0, 25.0);
            random_walk(&mut rng, start, l, 0.8, 0.35)
        };
        let v = rng.range(0.02, 0.25) as f32;
        legit.push(Stroke {
            pts,
            hw: rng.range(0.12, 0.35),
            color: [v, v * 0.9, v * 0.8],
            opacity: 1.0,
        });
    }
    for _ in 0..rng.range(3.0, 10.0) as usize {
        // Catchlights / sparkles: small but soft (print resolution).
        let c = (rng.range(0.0, w_mm), rng.range(0.0, h_mm));
        legit.push(Stroke {
            pts: vec![c],
            hw: rng.range(0.15, 0.4),
            color: [0.97, 0.97, 0.95],
            opacity: 0.9,
        });
    }
    for _ in 0..rng.range(5.0, 20.0) as usize {
        // Coloured flowers / fabric dots.
        let c = (rng.range(0.0, w_mm), rng.range(0.0, h_mm));
        let col = match rng.next_u64() % 3 {
            0 => [0.85, 0.15, 0.15],
            1 => [0.95, 0.8, 0.1],
            _ => [0.6, 0.2, 0.75],
        };
        legit.push(Stroke {
            pts: vec![c],
            hw: rng.range(0.15, 0.35),
            color: col,
            opacity: 1.0,
        });
    }
    for st in &legit {
        paint(&mut img, w, h, pxmm, st, 0.0);
    }
    // Print blur: prints resolve far less than a 600 dpi scan.
    let sigma = (0.07 * pxmm) as f32;
    for ch in 0..3 {
        let plane = crate::raster::Plane {
            w,
            h,
            data: img.iter().skip(ch).step_by(3).copied().collect(),
        }
        .blur(sigma);
        for (i, v) in plane.data.into_iter().enumerate() {
            img[i * 3 + ch] = v;
        }
    }
    if mono {
        for p in img.chunks_mut(3) {
            let l = 0.3 * p[0] + 0.59 * p[1] + 0.11 * p[2];
            p.copy_from_slice(&[l, l, l]);
        }
    }
    // Scanner noise.
    let noise = rng.range(0.004, 0.012) as f32;
    let ns = rng.next_u64();
    img.par_chunks_mut(3).enumerate().for_each(|(i, p)| {
        for (ch, v) in p.iter_mut().enumerate() {
            let r = hash(i as i64, ch as i64, ns) + hash(i as i64, ch as i64 + 7, ns) - 1.0;
            *v = (*v + r * noise * 1.7).clamp(0.0, 1.0);
        }
    });
    let clean = img.clone();

    // Defects (sharp: they are on the print surface, in the scanner's focal plane).
    let mut truth = Mask::new(w, h);
    let mut touched = Mask::new(w, h);
    let mut defects = Vec::new();
    let mut kinds = Vec::new();
    for _ in 0..n_defects {
        let kind = rng.f64();
        let bright = rng.f64() < 0.6;
        let v = if bright {
            rng.range(0.85, 1.0) as f32
        } else {
            rng.range(0.0, 0.15) as f32
        };
        let start = (rng.range(0.0, w_mm), rng.range(0.0, h_mm));
        let st = if kind < 0.6 {
            // Speck: 1-3 overlapping discs.
            let r = rng.range(0.03, 0.25);
            let mut pts = vec![start];
            for _ in 0..(rng.next_u64() % 3) {
                pts.push((start.0 + rng.range(-r, r), start.1 + rng.range(-r, r)));
            }
            Stroke {
                pts,
                hw: r,
                color: [v, v, v],
                opacity: rng.range(0.7, 1.0) as f32,
            }
        } else if kind < 0.85 {
            // Hair / fibre.
            Stroke {
                pts: {
                    let l = rng.range(1.0, 6.0);
                    random_walk(&mut rng, start, l, 0.25, 0.4)
                },
                hw: rng.range(0.015, 0.05),
                color: [v, v, v],
                opacity: rng.range(0.6, 1.0) as f32,
            }
        } else {
            // Scratch: long, nearly straight, bright.
            let v = rng.range(0.8, 1.0) as f32;
            Stroke {
                pts: {
                    let l = rng.range(4.0, 20.0);
                    random_walk(&mut rng, start, l, 1.0, 0.05)
                },
                hw: rng.range(0.015, 0.05),
                color: [v, v, v],
                opacity: rng.range(0.5, 0.9) as f32,
            }
        };
        let t = paint(&mut img, w, h, pxmm, &st, 0.0);
        let mut px = Vec::new();
        for (i, c) in t {
            touched.data[i] = true;
            if c > 0.25 {
                truth.data[i] = true;
            }
            if c > 0.5 {
                px.push(i);
            }
        }
        if !px.is_empty() {
            defects.push(px);
            kinds.push(if kind < 0.6 {
                "speck"
            } else if kind < 0.85 {
                "hair"
            } else {
                "scratch"
            });
        }
    }
    let to_img = |v: Vec<f32>| {
        DynamicImage::ImageRgb8(
            DynamicImage::ImageRgb32F(Rgb32FImage::from_raw(w as u32, h as u32, v).unwrap())
                .to_rgb8(),
        )
    };
    DustScene {
        clean: to_img(clean),
        dirty: to_img(img),
        truth,
        touched,
        defects,
        kinds,
        dpi,
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DustStats {
    pub scenes: usize,
    pub defects: usize,
    /// Defects with at least 80% of their core pixels masked.
    pub found: usize,
    /// Defects that change their pixels by at least 24/255 on average (clearly visible).
    pub visible: usize,
    pub visible_found: usize,
    /// Masked pixels farther than 3 px from any defect, per million image pixels.
    pub false_ppm: f64,
    /// Connected false-alarm regions (no defect within 3 px).
    pub false_regions: usize,
    /// Mean absolute error (0..255) over defect pixels, before and after cleaning.
    pub err_before: f64,
    pub err_after: f64,
    /// Mean absolute error over the whole image, after (before is the same metric).
    pub err_global_before: f64,
    pub err_global_after: f64,
}

impl DustStats {
    pub fn recall(&self) -> f64 {
        self.found as f64 / self.defects.max(1) as f64
    }
    pub fn visible_recall(&self) -> f64 {
        self.visible_found as f64 / self.visible.max(1) as f64
    }
}

/// Accumulate the score of one cleaned scene.
pub fn score(st: &mut DustStats, scene: &DustScene, mask: &Mask, cleaned: &DynamicImage) {
    let (w, h) = (scene.truth.w, scene.truth.h);
    st.scenes += 1;
    let (cl, di) = (scene.clean.to_rgb8(), scene.dirty.to_rgb8());
    for d in &scene.defects {
        st.defects += 1;
        let hit = d.iter().filter(|&&i| mask.data[i]).count();
        let found = hit as f64 >= 0.8 * d.len() as f64;
        let change: f64 = d
            .iter()
            .map(|&i| {
                (0..3)
                    .map(|c| (cl.as_raw()[i * 3 + c] as f64 - di.as_raw()[i * 3 + c] as f64).abs())
                    .sum::<f64>()
                    / 3.0
            })
            .sum::<f64>()
            / d.len() as f64;
        if found {
            st.found += 1;
        }
        if change >= 24.0 {
            st.visible += 1;
            if found {
                st.visible_found += 1;
            }
        }
    }
    let near = scene.touched.dilate(3.0);
    let fa = Mask {
        w,
        h,
        data: mask
            .data
            .iter()
            .zip(&near.data)
            .map(|(m, n)| *m && !n)
            .collect(),
    };
    let fa_px = fa.count();
    let (_, comps) = fa.components();
    st.false_regions += comps.len();
    let n_img = (w * h) as f64;
    let prev = st.scenes as f64 - 1.0;
    let avg = |old: f64, new: f64| (old * prev + new) / st.scenes as f64;
    st.false_ppm = avg(st.false_ppm, fa_px as f64 * 1e6 / n_img);
    let (c, d, o) = (
        scene.clean.to_rgb8(),
        scene.dirty.to_rgb8(),
        cleaned.to_rgb8(),
    );
    let (c, d, o) = (c.as_raw(), d.as_raw(), o.as_raw());
    let (mut eb, mut ea, mut n) = (0.0f64, 0.0f64, 0usize);
    let (mut gb, mut ga) = (0.0f64, 0.0f64);
    for i in 0..w * h {
        let mut db = 0.0;
        let mut da = 0.0;
        for ch in 0..3 {
            db += (c[i * 3 + ch] as f64 - d[i * 3 + ch] as f64).abs();
            da += (c[i * 3 + ch] as f64 - o[i * 3 + ch] as f64).abs();
        }
        gb += db / 3.0;
        ga += da / 3.0;
        if scene.truth.data[i] {
            eb += db / 3.0;
            ea += da / 3.0;
            n += 1;
        }
    }
    if n > 0 {
        st.err_before = avg(st.err_before, eb / n as f64);
        st.err_after = avg(st.err_after, ea / n as f64);
    }
    st.err_global_before = avg(st.err_global_before, gb / n_img);
    st.err_global_after = avg(st.err_global_after, ga / n_img);
}
