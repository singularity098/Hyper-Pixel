//! Document clean-up: deskew, paper whitening and black & white conversion.

use crate::detect::ScanGeometry;
use crate::geom::RotatedRect;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, imageops::FilterType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DocColorMode {
    #[default]
    Color,
    Gray,
    BlackWhite,
}

impl DocColorMode {
    pub const ALL: [DocColorMode; 3] = [
        DocColorMode::Color,
        DocColorMode::Gray,
        DocColorMode::BlackWhite,
    ];
    pub fn label(self) -> &'static str {
        match self {
            DocColorMode::Color => "Colour",
            DocColorMode::Gray => "Greyscale",
            DocColorMode::BlackWhite => "Black & white",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocParams {
    pub deskew: bool,
    pub whiten: bool,
    pub mode: DocColorMode,
}

impl Default for DocParams {
    fn default() -> Self {
        Self {
            deskew: true,
            whiten: true,
            mode: DocColorMode::Color,
        }
    }
}

/// Full document pipeline.
pub fn process(img: &DynamicImage, dpi: f64, p: &DocParams) -> DynamicImage {
    let mut img = img.clone();
    if p.deskew {
        let a = estimate_skew(&img, dpi);
        if a.abs() > 0.05 {
            img = rotate_keep_size(&img, a);
        }
    }
    if p.whiten {
        img = whiten(&img, dpi);
    }
    match p.mode {
        DocColorMode::Color => img,
        DocColorMode::Gray => DynamicImage::ImageLuma8(img.to_luma8()),
        DocColorMode::BlackWhite => DynamicImage::ImageLuma8(binarize(&img.to_luma8(), dpi)),
    }
}

fn otsu(hist: &[u64; 256]) -> u8 {
    let total: u64 = hist.iter().sum();
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut sum_b, mut w_b, mut best, mut thr) = (0.0, 0u64, 0.0, 128u8);
    for (t, &c) in hist.iter().enumerate() {
        w_b += c;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as f64 * c as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum - sum_b) / w_f as f64;
        let between = w_b as f64 * w_f as f64 * (m_b - m_f).powi(2);
        if between > best {
            best = between;
            thr = t as u8;
        }
    }
    thr
}

/// Estimate text skew in degrees (positive = content rotated clockwise). Range ±8°.
pub fn estimate_skew(img: &DynamicImage, dpi: f64) -> f64 {
    let scale = (100.0 / dpi).min(1.0);
    let g = if scale < 1.0 {
        img.resize(
            (img.width() as f64 * scale) as u32,
            (img.height() as f64 * scale) as u32,
            FilterType::Triangle,
        )
        .to_luma8()
    } else {
        img.to_luma8()
    };
    let mut hist = [0u64; 256];
    for p in g.pixels() {
        hist[p[0] as usize] += 1;
    }
    let t = otsu(&hist);
    let mut pts: Vec<(f64, f64)> = g
        .enumerate_pixels()
        .filter(|(_, _, p)| p[0] < t)
        .map(|(x, y, _)| (x as f64, y as f64))
        .collect();
    if pts.len() < 50 || pts.len() > (g.width() * g.height()) as usize / 2 {
        return 0.0;
    }
    if pts.len() > 200_000 {
        let step = pts.len() / 200_000 + 1;
        pts = pts.into_iter().step_by(step).collect();
    }
    let h = g.height() as f64 + g.width() as f64;
    let score = |deg: f64| -> f64 {
        let (s, c) = deg.to_radians().sin_cos();
        let n = (2.0 * h) as usize + 2;
        let mut bins = vec![0u32; n];
        for &(x, y) in &pts {
            let yy = -x * s + y * c + h;
            bins[(yy as usize).min(n - 1)] += 1;
        }
        bins.iter().map(|&b| (b as f64).powi(2)).sum()
    };
    let search = |lo: f64, hi: f64, step: f64| -> f64 {
        let n = ((hi - lo) / step).round() as usize;
        (0..=n)
            .into_par_iter()
            .map(|i| lo + i as f64 * step)
            .map(|a| (a, score(a)))
            .reduce(|| (0.0, f64::MIN), |a, b| if b.1 > a.1 { b } else { a })
            .0
    };
    let coarse = search(-8.0, 8.0, 0.5);
    search(coarse - 0.5, coarse + 0.5, 0.05)
}

/// Rotate by `-deg` about the centre (undoing a skew of `deg`), keeping the canvas size.
pub fn rotate_keep_size(img: &DynamicImage, deg: f64) -> DynamicImage {
    let geo = ScanGeometry::new(25.4);
    let r = RotatedRect::new(
        img.width() as f64 / 2.0,
        img.height() as f64 / 2.0,
        img.width() as f64,
        img.height() as f64,
        deg,
    );
    crate::crop::extract(img, geo, &r, 0.0)
}

/// Normalise paper to white by dividing by a smooth estimate of the local paper colour.
pub fn whiten(img: &DynamicImage, dpi: f64) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    // Background estimate at ~10 DPI: local max (paper is the brightest thing) then blur.
    let f = (dpi / 10.0).max(1.0);
    let (sw, sh) = (
        ((w as f64 / f) as u32).max(4),
        ((h as f64 / f) as u32).max(4),
    );
    let small = image::imageops::resize(&rgb, sw, sh, FilterType::Triangle);
    let mut bg = small.clone();
    let r = 2i64;
    for y in 0..sh as i64 {
        for x in 0..sw as i64 {
            let mut m = [0u8; 3];
            for dy in -r..=r {
                for dx in -r..=r {
                    let (xx, yy) = (
                        (x + dx).clamp(0, sw as i64 - 1),
                        (y + dy).clamp(0, sh as i64 - 1),
                    );
                    let p = small.get_pixel(xx as u32, yy as u32);
                    for c in 0..3 {
                        m[c] = m[c].max(p[c]);
                    }
                }
            }
            bg.put_pixel(x as u32, y as u32, image::Rgb(m));
        }
    }
    let bg = image::imageops::blur(&bg, 1.5);
    let bg = image::imageops::resize(&bg, w, h, FilterType::Triangle);
    let mut out = rgb;
    out.as_mut()
        .par_chunks_mut(3)
        .zip(bg.as_raw().par_chunks(3))
        .for_each(|(p, b)| {
            for c in 0..3 {
                let paper = (b[c] as f32).max(40.0);
                p[c] = ((p[c] as f32 / paper) * 255.0).min(255.0) as u8;
            }
        });
    DynamicImage::ImageRgb8(out)
}

/// Sauvola adaptive threshold. Output is 0 (ink) / 255 (paper).
pub fn binarize(g: &GrayImage, dpi: f64) -> GrayImage {
    let (w, h) = (g.width() as usize, g.height() as usize);
    let mut ii = vec![0u64; (w + 1) * (h + 1)];
    let mut ii2 = vec![0u64; (w + 1) * (h + 1)];
    for y in 0..h {
        let (mut rs, mut rs2) = (0u64, 0u64);
        for x in 0..w {
            let v = g.as_raw()[y * w + x] as u64;
            rs += v;
            rs2 += v * v;
            ii[(y + 1) * (w + 1) + x + 1] = ii[y * (w + 1) + x + 1] + rs;
            ii2[(y + 1) * (w + 1) + x + 1] = ii2[y * (w + 1) + x + 1] + rs2;
        }
    }
    let r = ((dpi / 8.0) as usize).max(7);
    let k = 0.2f64;
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r + 1).min(h);
        for (x, o) in row.iter_mut().enumerate() {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r + 1).min(w);
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            let s = (ii[y1 * (w + 1) + x1] + ii[y0 * (w + 1) + x0]
                - ii[y0 * (w + 1) + x1]
                - ii[y1 * (w + 1) + x0]) as f64;
            let s2 = (ii2[y1 * (w + 1) + x1] + ii2[y0 * (w + 1) + x0]
                - ii2[y0 * (w + 1) + x1]
                - ii2[y1 * (w + 1) + x0]) as f64;
            let m = s / n;
            let sd = (s2 / n - m * m).max(0.0).sqrt();
            let t = m * (1.0 + k * (sd / 128.0 - 1.0));
            *o = if (g.as_raw()[y * w + x] as f64) > t {
                255
            } else {
                0
            };
        }
    });
    ImageBuffer::<Luma<u8>, _>::from_raw(w as u32, h as u32, out).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn fake_page(angle: f64) -> DynamicImage {
        // Lines of "text" (dark bars) on white paper, rotated by `angle`.
        let mut img = RgbImage::from_pixel(600, 800, image::Rgb([250, 248, 240]));
        for line in 0..20 {
            let y0 = 80 + line * 30;
            for y in y0..y0 + 8 {
                for x in 60..540 {
                    if (x / 13) % 5 != 4 {
                        img.put_pixel(x, y, image::Rgb([20, 20, 20]));
                    }
                }
            }
        }
        rotate_keep_size(&DynamicImage::ImageRgb8(img), -angle)
    }

    #[test]
    fn finds_skew() {
        for a in [-3.0, 0.0, 1.7, 5.0] {
            let est = estimate_skew(&fake_page(a), 100.0);
            assert!((est - a).abs() < 0.2, "angle {a} estimated {est}");
        }
    }

    #[test]
    fn binarize_produces_two_levels() {
        let b = binarize(&fake_page(0.0).to_luma8(), 100.0);
        assert!(b.as_raw().iter().all(|&v| v == 0 || v == 255));
        let ink = b.as_raw().iter().filter(|&&v| v == 0).count();
        assert!(ink > 10_000 && ink < 200_000, "{ink}");
    }
}
