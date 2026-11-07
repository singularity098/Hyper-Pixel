//! Simple single-channel float planes and binary masks with the handful of operations the
//! detectors need. Kept dependency-free and predictable rather than pulling in a large
//! computer-vision crate.

use image::{DynamicImage, imageops::FilterType};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0.0; w * h] }
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.w + x] = v;
    }
    /// Bilinear sample with edge clamping.
    pub fn sample(&self, x: f64, y: f64) -> f32 {
        let x = x.clamp(0.0, (self.w - 1) as f64);
        let y = y.clamp(0.0, (self.h - 1) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = (x - x0 as f64) as f32;
        let fy = (y - y0 as f64) as f32;
        let a = self.at(x0, y0) * (1.0 - fx) + self.at(x1, y0) * fx;
        let b = self.at(x0, y1) * (1.0 - fx) + self.at(x1, y1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Separable Gaussian blur.
    pub fn blur(&self, sigma: f32) -> Plane {
        if sigma <= 0.0 {
            return self.clone();
        }
        let r = (sigma * 3.0).ceil() as isize;
        let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
        let ks: f32 = k.iter().sum();
        let k: Vec<f32> = k.iter().map(|v| v / ks).collect();
        let (w, h) = (self.w, self.h);
        let mut tmp = vec![0.0f32; w * h];
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let mut s = 0.0;
                for (j, kv) in k.iter().enumerate() {
                    let xx = (x as isize + j as isize - r).clamp(0, w as isize - 1) as usize;
                    s += self.data[y * w + xx] * kv;
                }
                *out = s;
            }
        });
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let mut s = 0.0;
                for (j, kv) in k.iter().enumerate() {
                    let yy = (y as isize + j as isize - r).clamp(0, h as isize - 1) as usize;
                    s += tmp[yy * w + x] * kv;
                }
                *o = s;
            }
        });
        Plane { w, h, data: out }
    }

    /// Sobel gradient magnitude.
    pub fn gradient(&self) -> Plane {
        let (w, h) = (self.w, self.h);
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            for (x, o) in row.iter_mut().enumerate() {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                let g = |xx: usize, yy: usize| self.data[yy * w + xx];
                let gx = (g(xp, ym) + 2.0 * g(xp, y) + g(xp, yp)) - (g(xm, ym) + 2.0 * g(xm, y) + g(xm, yp));
                let gy = (g(xm, yp) + 2.0 * g(x, yp) + g(xp, yp)) - (g(xm, ym) + 2.0 * g(x, ym) + g(xp, ym));
                *o = (gx * gx + gy * gy).sqrt() / 8.0;
            }
        });
        Plane { w, h, data: out }
    }
}

/// CIE L*a*b* planes of an image.
pub struct Lab {
    pub l: Plane,
    pub a: Plane,
    pub b: Plane,
}

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

#[inline]
fn lab_f(t: f32) -> f32 {
    if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 }
}

/// sRGB (0..1) to CIE Lab (D65).
#[inline]
pub fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (r, g, b) = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let (fx, fy, fz) = (lab_f(x), lab_f(y), lab_f(z));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// Downscale `img` by `scale` (<= 1) and convert to Lab planes.
pub fn lab_planes(img: &DynamicImage, scale: f64) -> Lab {
    let w = ((img.width() as f64 * scale).round() as u32).max(1);
    let h = ((img.height() as f64 * scale).round() as u32).max(1);
    let small = if w == img.width() && h == img.height() {
        img.to_rgb32f()
    } else {
        img.resize_exact(w, h, FilterType::Triangle).to_rgb32f()
    };
    let (w, h) = (w as usize, h as usize);
    let mut l = Plane::new(w, h);
    let mut a = Plane::new(w, h);
    let mut b = Plane::new(w, h);
    let px = small.as_raw();
    l.data
        .par_iter_mut()
        .zip(a.data.par_iter_mut())
        .zip(b.data.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((lo, ao), bo))| {
            let (ll, aa, bb) = rgb_to_lab(px[i * 3], px[i * 3 + 1], px[i * 3 + 2]);
            *lo = ll;
            *ao = aa;
            *bo = bb;
        });
    Lab { l, a, b }
}

/// Binary mask.
#[derive(Clone, Debug)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<bool>,
}

impl Mask {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![false; w * h] }
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> bool {
        self.data[y * self.w + x]
    }
    pub fn count(&self) -> usize {
        self.data.iter().filter(|v| **v).count()
    }

    /// Chamfer (3-4) distance, in pixels, from each pixel to the nearest pixel whose value
    /// equals `target`. Pixels outside the image count as `outside`.
    fn distance_to(&self, target: bool, outside: bool) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let inf = f32::MAX / 4.0;
        let mut d: Vec<f32> = self.data.iter().map(|&v| if v == target { 0.0 } else { inf }).collect();
        let border = if outside == target { 1.0 } else { inf };
        let get = |d: &Vec<f32>, x: isize, y: isize| -> f32 {
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                border - 1.0 // distance to a virtual outside pixel adjacent to the edge
            } else {
                d[y as usize * w + x as usize]
            }
        };
        let (a, b) = (1.0f32, std::f32::consts::SQRT_2);
        for y in 0..h as isize {
            for x in 0..w as isize {
                let i = y as usize * w + x as usize;
                let mut v = d[i];
                v = v.min(get(&d, x - 1, y) + a);
                v = v.min(get(&d, x, y - 1) + a);
                v = v.min(get(&d, x - 1, y - 1) + b);
                v = v.min(get(&d, x + 1, y - 1) + b);
                d[i] = v;
            }
        }
        for y in (0..h as isize).rev() {
            for x in (0..w as isize).rev() {
                let i = y as usize * w + x as usize;
                let mut v = d[i];
                v = v.min(get(&d, x + 1, y) + a);
                v = v.min(get(&d, x, y + 1) + a);
                v = v.min(get(&d, x + 1, y + 1) + b);
                v = v.min(get(&d, x - 1, y + 1) + b);
                d[i] = v;
            }
        }
        d
    }

    /// Distance from each set pixel to the nearest unset pixel (0 for unset pixels).
    /// Pixels outside the image are treated as unset.
    pub fn distance_transform(&self) -> Vec<f32> {
        self.distance_to(false, false)
    }

    pub fn dilate(&self, r: f32) -> Mask {
        if r <= 0.0 {
            return self.clone();
        }
        let d = self.distance_to(true, false);
        Mask { w: self.w, h: self.h, data: d.iter().map(|&v| v <= r).collect() }
    }

    pub fn erode(&self, r: f32) -> Mask {
        if r <= 0.0 {
            return self.clone();
        }
        // Outside counts as "set" for erosion so objects touching the border are not eaten.
        let d = self.distance_to(false, true);
        Mask { w: self.w, h: self.h, data: d.iter().map(|&v| v > r).collect() }
    }

    pub fn close(&self, r: f32) -> Mask {
        self.dilate(r).erode(r)
    }
    pub fn open(&self, r: f32) -> Mask {
        self.erode(r).dilate(r)
    }

    /// Fill every unset region not connected to the image border.
    pub fn fill_holes(&self) -> Mask {
        let (w, h) = (self.w, self.h);
        let mut outside = vec![false; w * h];
        let mut stack: Vec<usize> = Vec::new();
        let push = |i: usize, outside: &mut Vec<bool>, stack: &mut Vec<usize>| {
            if !self.data[i] && !outside[i] {
                outside[i] = true;
                stack.push(i);
            }
        };
        for x in 0..w {
            push(x, &mut outside, &mut stack);
            push((h - 1) * w + x, &mut outside, &mut stack);
        }
        for y in 0..h {
            push(y * w, &mut outside, &mut stack);
            push(y * w + w - 1, &mut outside, &mut stack);
        }
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            if x > 0 {
                push(i - 1, &mut outside, &mut stack);
            }
            if x + 1 < w {
                push(i + 1, &mut outside, &mut stack);
            }
            if y > 0 {
                push(i - w, &mut outside, &mut stack);
            }
            if y + 1 < h {
                push(i + w, &mut outside, &mut stack);
            }
        }
        Mask { w, h, data: outside.iter().map(|o| !o).collect() }
    }

    /// 8-connected component labelling. Returns per-pixel labels (0 = background) and the
    /// pixel indices of each component (component `k` has label `k + 1`).
    pub fn components(&self) -> (Vec<u32>, Vec<Vec<usize>>) {
        let (w, h) = (self.w, self.h);
        let mut labels = vec![0u32; w * h];
        let mut comps: Vec<Vec<usize>> = Vec::new();
        let mut stack = Vec::new();
        for start in 0..w * h {
            if !self.data[start] || labels[start] != 0 {
                continue;
            }
            let label = comps.len() as u32 + 1;
            let mut pixels = Vec::new();
            labels[start] = label;
            stack.push(start);
            while let Some(i) = stack.pop() {
                pixels.push(i);
                let (x, y) = ((i % w) as isize, (i / w) as isize);
                for dy in -1..=1isize {
                    for dx in -1..=1isize {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if self.data[j] && labels[j] == 0 {
                            labels[j] = label;
                            stack.push(j);
                        }
                    }
                }
            }
            comps.push(pixels);
        }
        (labels, comps)
    }

    pub fn from_indices(w: usize, h: usize, idx: &[usize]) -> Mask {
        let mut m = Mask::new(w, h);
        for &i in idx {
            m.data[i] = true;
        }
        m
    }
}

/// Percentile of a slice (0..=1). Sorts a copy.
pub fn percentile(values: &[f32], p: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    let k = ((v.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    let (_, m, _) = v.select_nth_unstable_by(k, f32::total_cmp);
    *m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_holes_and_components() {
        let mut m = Mask::new(10, 10);
        for i in 2..8 {
            m.data[2 * 10 + i] = true;
            m.data[7 * 10 + i] = true;
            m.data[i * 10 + 2] = true;
            m.data[i * 10 + 7] = true;
        }
        let f = m.fill_holes();
        assert_eq!(f.count(), 36);
        let (_, comps) = f.components();
        assert_eq!(comps.len(), 1);
    }

    #[test]
    fn erode_dilate() {
        let mut m = Mask::new(20, 20);
        for y in 5..15 {
            for x in 5..15 {
                m.data[y * 20 + x] = true;
            }
        }
        assert_eq!(m.erode(1.0).count(), 64);
        assert_eq!(m.dilate(1.0).count(), 140);
        assert_eq!(m.open(1.0).count(), 96);
    }
}

/// Draw rotated rectangle outlines onto an RGB image (debug/visualisation helper).
pub fn draw_rects(img: &mut image::RgbImage, rects: &[crate::geom::RotatedRect], color: [u8; 3], thickness: u32) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    for r in rects {
        let c = r.corners();
        for k in 0..4 {
            let (a, b) = (c[k], c[(k + 1) % 4]);
            let n = (a.dist(b).ceil() as usize).max(1) * 2;
            for s in 0..=n {
                let t = s as f64 / n as f64;
                let (x, y) = (a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
                let half = thickness as i64 / 2;
                for dy in -half..=half {
                    for dx in -half..=half {
                        let (px, py) = (x as i64 + dx, y as i64 + dy);
                        if px >= 0 && py >= 0 && px < w && py < h {
                            img.put_pixel(px as u32, py as u32, image::Rgb(color));
                        }
                    }
                }
            }
        }
    }
}
