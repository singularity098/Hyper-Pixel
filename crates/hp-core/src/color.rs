//! Automatic colour correction and faded-photo restoration.
//!
//! All operations preserve bit depth (8 or 16 bit). Statistics are gathered on a
//! downscaled copy so the cost is dominated by a single per-pixel pass.

use crate::raster::percentile;
use image::{DynamicImage, ImageBuffer, Rgb, Rgb32FImage, imageops::FilterType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColorPreset {
    /// Leave the scan untouched.
    #[default]
    None,
    /// Gentle levels + white balance; safe default for good prints.
    Auto,
    /// Aggressive per-channel recovery for faded / colour-shifted prints.
    RestoreFaded,
    /// Auto plus extra saturation and contrast.
    Vivid,
    /// Neutral black & white with auto levels.
    Monochrome,
}

impl ColorPreset {
    pub const ALL: [ColorPreset; 5] =
        [ColorPreset::None, ColorPreset::Auto, ColorPreset::RestoreFaded, ColorPreset::Vivid, ColorPreset::Monochrome];
    pub fn label(self) -> &'static str {
        match self {
            ColorPreset::None => "None",
            ColorPreset::Auto => "Auto",
            ColorPreset::RestoreFaded => "Restore faded",
            ColorPreset::Vivid => "Vivid",
            ColorPreset::Monochrome => "Monochrome",
        }
    }
    pub fn params(self) -> Option<ColorParams> {
        Some(match self {
            ColorPreset::None => return None,
            ColorPreset::Auto => ColorParams::default(),
            ColorPreset::RestoreFaded => ColorParams {
                per_channel_levels: true,
                white_balance: 1.0,
                midtone_balance: 0.8,
                saturation: 1.25,
                contrast: 0.15,
                ..Default::default()
            },
            ColorPreset::Vivid => ColorParams { saturation: 1.25, contrast: 0.25, ..Default::default() },
            ColorPreset::Monochrome => ColorParams { saturation: 0.0, white_balance: 0.0, contrast: 0.1, ..Default::default() },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorParams {
    /// Fraction of pixels clipped at each end by the levels stretch.
    pub clip: f64,
    /// Stretch each channel independently (removes casts; can shift colours).
    pub per_channel_levels: bool,
    /// 0..1 strength of grey-world white balance on near-neutral pixels.
    pub white_balance: f32,
    /// 0..1 strength of per-channel gamma that neutralises mid-tone casts (fading).
    pub midtone_balance: f32,
    /// Chroma multiplier in Oklab (1 = unchanged, 0 = greyscale).
    pub saturation: f32,
    /// 0..1 S-curve strength on lightness.
    pub contrast: f32,
    /// Maximum levels gain, guards against blowing up near-blank images.
    pub max_gain: f32,
}

impl Default for ColorParams {
    fn default() -> Self {
        Self {
            clip: 0.002,
            per_channel_levels: false,
            white_balance: 0.6,
            midtone_balance: 0.0,
            saturation: 1.05,
            contrast: 0.08,
            max_gain: 3.0,
        }
    }
}

#[inline]
fn srgb_to_lin(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}
#[inline]
fn lin_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.max(0.0).powf(1.0 / 2.4) - 0.055 }
}

#[inline]
fn to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let (r, g, b) = (srgb_to_lin(r), srgb_to_lin(g), srgb_to_lin(b));
    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

#[inline]
fn from_oklab(c: [f32; 3]) -> [f32; 3] {
    let l = c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2];
    let m = c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2];
    let s = c[0] - 0.0894841775 * c[1] - 1.2914855480 * c[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        lin_to_srgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s).clamp(0.0, 1.0),
        lin_to_srgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s).clamp(0.0, 1.0),
        lin_to_srgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s).clamp(0.0, 1.0),
    ]
}

/// Per-channel transfer: x -> ((x - lo) * gain)^gamma, all in 0..1 gamma-encoded space.
#[derive(Debug, Clone, Copy)]
struct ChannelCurve {
    lo: f32,
    gain: f32,
    gamma: f32,
}

impl ChannelCurve {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        ((x - self.lo) * self.gain).clamp(0.0, 1.0).powf(self.gamma)
    }
}

fn analyse(img: &Rgb32FImage, p: &ColorParams) -> [ChannelCurve; 3] {
    let px = img.as_raw();
    let n = px.len() / 3;
    let chan = |c: usize| -> Vec<f32> { (0..n).map(|i| px[i * 3 + c]).collect() };
    let chans = [chan(0), chan(1), chan(2)];
    let lows: Vec<f32> = chans.iter().map(|v| percentile(v, p.clip)).collect();
    let highs: Vec<f32> = chans.iter().map(|v| percentile(v, 1.0 - p.clip)).collect();
    let mut curves = [ChannelCurve { lo: 0.0, gain: 1.0, gamma: 1.0 }; 3];
    if p.per_channel_levels {
        for c in 0..3 {
            let range = (highs[c] - lows[c]).max(1e-3);
            curves[c] = ChannelCurve { lo: lows[c], gain: (1.0 / range).min(p.max_gain), gamma: 1.0 };
        }
    } else {
        let lo = lows.iter().cloned().fold(f32::MAX, f32::min);
        let hi = highs.iter().cloned().fold(f32::MIN, f32::max);
        let gain = (1.0 / (hi - lo).max(1e-3)).min(p.max_gain);
        for c in curves.iter_mut() {
            *c = ChannelCurve { lo, gain, gamma: 1.0 };
        }
    }

    // White balance on near-neutral mid-tones after levels.
    if p.white_balance > 0.0 {
        let mut sums = [0.0f64; 3];
        let mut cnt = 0usize;
        for i in 0..n {
            let v = [curves[0].apply(px[i * 3]), curves[1].apply(px[i * 3 + 1]), curves[2].apply(px[i * 3 + 2])];
            let mx = v[0].max(v[1]).max(v[2]);
            let mn = v[0].min(v[1]).min(v[2]);
            let lum = (v[0] + v[1] + v[2]) / 3.0;
            if lum > 0.15 && lum < 0.92 && (mx - mn) < 0.25 {
                for c in 0..3 {
                    sums[c] += v[c] as f64;
                }
                cnt += 1;
            }
        }
        if cnt < n / 50 {
            // Too few neutral pixels: plain grey world over everything.
            sums = [0.0; 3];
            cnt = n;
            for i in 0..n {
                for c in 0..3 {
                    sums[c] += curves[c].apply(px[i * 3 + c]) as f64;
                }
            }
        }
        if cnt > 0 {
            let mean = (sums[0] + sums[1] + sums[2]) / 3.0;
            for c in 0..3 {
                let g = (mean / sums[c].max(1e-9)) as f32;
                let g = 1.0 + (g.clamp(0.7, 1.4) - 1.0) * p.white_balance;
                curves[c].gain *= g;
            }
        }
    }

    // Mid-tone balance: per-channel gamma that maps each channel's median to the common median.
    if p.midtone_balance > 0.0 {
        let meds: Vec<f32> = (0..3).map(|c| percentile(&chans[c].iter().map(|&x| curves[c].apply(x)).collect::<Vec<_>>(), 0.5)).collect();
        let target = (meds[0] + meds[1] + meds[2]) / 3.0;
        for c in 0..3 {
            if meds[c] > 0.02 && meds[c] < 0.98 && target > 0.02 && target < 0.98 {
                let g = (target.ln() / meds[c].ln()).clamp(0.6, 1.6);
                curves[c].gamma = 1.0 + (g - 1.0) * p.midtone_balance;
            }
        }
    }
    curves
}

/// Apply colour correction. Returns an image with the same bit depth as the input.
pub fn apply(img: &DynamicImage, p: &ColorParams) -> DynamicImage {
    let is16 = matches!(
        img,
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) | DynamicImage::ImageLuma16(_) | DynamicImage::ImageLumaA16(_)
    );
    let was_gray = matches!(img, DynamicImage::ImageLuma8(_) | DynamicImage::ImageLuma16(_));
    let (w, h) = (img.width(), img.height());
    let scale = (512.0 / w.max(h) as f64).min(1.0);
    let small = if scale < 1.0 {
        img.resize((w as f64 * scale).max(1.0) as u32, (h as f64 * scale).max(1.0) as u32, FilterType::Triangle).to_rgb32f()
    } else {
        img.to_rgb32f()
    };
    let curves = analyse(&small, p);
    let mut buf = img.to_rgb32f();
    let sat = p.saturation;
    let con = p.contrast;
    buf.as_mut().par_chunks_mut(3).for_each(|px| {
        let r = curves[0].apply(px[0]);
        let g = curves[1].apply(px[1]);
        let b = curves[2].apply(px[2]);
        let mut lab = to_oklab(r, g, b);
        if con > 0.0 {
            let x = lab[0].clamp(0.0, 1.0);
            let s = x * x * (3.0 - 2.0 * x);
            lab[0] = x + con * (s - x);
        }
        lab[1] *= sat;
        lab[2] *= sat;
        let o = from_oklab(lab);
        px.copy_from_slice(&o);
    });
    let out = DynamicImage::ImageRgb32F(buf);
    match (is16, was_gray || p.saturation == 0.0) {
        (true, true) => DynamicImage::ImageLuma16(out.to_luma16()),
        (true, false) => DynamicImage::ImageRgb16(out.to_rgb16()),
        (false, true) => DynamicImage::ImageLuma8(out.to_luma8()),
        (false, false) => DynamicImage::ImageRgb8(out.to_rgb8()),
    }
}

/// Simulate fading for tests/demos: lifts blacks, crushes contrast and adds a warm cast.
pub fn simulate_fade(img: &DynamicImage) -> DynamicImage {
    let mut b = img.to_rgb8();
    for p in b.pixels_mut() {
        let f = |v: u8, lift: f32, gain: f32| ((v as f32 / 255.0) * gain + lift).clamp(0.0, 1.0);
        let r = f(p[0], 0.18, 0.75);
        let g = f(p[1], 0.10, 0.62);
        let bl = f(p[2], 0.06, 0.55);
        *p = Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (bl * 255.0) as u8]);
    }
    DynamicImage::ImageRgb8(b)
}

/// Mean absolute difference between two same-size RGB8 images (0..255).
pub fn mean_abs_diff(a: &DynamicImage, b: &DynamicImage) -> f64 {
    let (a, b) = (a.to_rgb8(), b.to_rgb8());
    let s: u64 = a.as_raw().iter().zip(b.as_raw()).map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64).sum();
    s as f64 / a.as_raw().len() as f64
}

#[allow(dead_code)]
fn _assert_types(_: ImageBuffer<Rgb<u8>, Vec<u8>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{SceneParams, render_scene};

    #[test]
    fn restore_reduces_fade_error() {
        let scene = render_scene(&SceneParams { seed: 3, dpi: 50.0, count: Some(1), ..Default::default() });
        let photo = crate::crop::extract(&scene.image, crate::ScanGeometry::new(50.0), &scene.truth[0], 1.0);
        let faded = simulate_fade(&photo);
        let restored = apply(&faded, &ColorPreset::RestoreFaded.params().unwrap());
        let before = mean_abs_diff(&photo, &faded);
        let after = mean_abs_diff(&photo, &restored);
        assert!(after < before * 0.6, "before {before} after {after}");
    }

    #[test]
    fn keeps_16_bit() {
        let img = DynamicImage::ImageRgb16(ImageBuffer::from_fn(64, 64, |x, y| Rgb([(x * 900) as u16, (y * 800) as u16, 30000])));
        assert!(matches!(apply(&img, &ColorParams::default()), DynamicImage::ImageRgb16(_)));
    }
}
