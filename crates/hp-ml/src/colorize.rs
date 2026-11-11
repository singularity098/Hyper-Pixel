//! Colourisation of black-and-white photos with DDColor.
//!
//! The model sees a 256 x 256 greyscale version of the photo and predicts the CIE Lab
//! chroma (a, b) channels. Chroma is low-detail by nature, so it is upsampled smoothly and
//! combined with the photo's own full-resolution lightness: all the detail and tonality of
//! the scan is kept, only colour is added.

use crate::models::{DDCOLOR, ModelError, Result};
use hp_core::raster::{lab_to_rgb, rgb_to_lab};
use image::{DynamicImage, Rgb32FImage, imageops::FilterType};
use rayon::prelude::*;
use rten::Model;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use std::path::Path;

const N: usize = 256;

pub struct Colorizer {
    model: Model,
}

impl Colorizer {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(DDCOLOR.files[0].name);
        if !path.is_file() {
            return Err(ModelError::Missing(DDCOLOR.label));
        }
        let model = Model::load_file(path).map_err(|e| ModelError::Engine(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn load_default() -> Result<Self> {
        Self::load(&DDCOLOR.find().ok_or(ModelError::Missing(DDCOLOR.label))?)
    }

    /// Predict (a, b) chroma planes at the model resolution for an image.
    fn predict_ab(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        let small = img
            .resize_exact(N as u32, N as u32, FilterType::Triangle)
            .to_rgb32f();
        let mut input = NdTensor::<f32, 4>::zeros([1, 3, N, N]);
        for (x, y, p) in small.enumerate_pixels() {
            let (l, _, _) = rgb_to_lab(p[0], p[1], p[2]);
            let (r, g, b) = lab_to_rgb(l, 0.0, 0.0);
            let (x, y) = (x as usize, y as usize);
            input[[0, 0, y, x]] = r.clamp(0.0, 1.0);
            input[[0, 1, y, x]] = g.clamp(0.0, 1.0);
            input[[0, 2, y, x]] = b.clamp(0.0, 1.0);
        }
        let out = self
            .model
            .run_one(input.view().into(), None)
            .map_err(|e| ModelError::Engine(e.to_string()))?;
        let t: NdTensor<f32, 4> = out
            .try_into()
            .map_err(|e| ModelError::Engine(format!("{e:?}")))?;
        if t.shape() != [1, 2, N, N] {
            return Err(ModelError::Engine(format!(
                "unexpected output {:?}",
                t.shape()
            )));
        }
        Ok(t.iter().copied().collect())
    }

    /// Colourise a photo. `strength` scales the predicted chroma (1 = as predicted).
    /// The result is RGB with the input's bit depth (8 or 16 bit).
    pub fn colorize(&self, img: &DynamicImage, strength: f32) -> Result<DynamicImage> {
        let ab = self.predict_ab(img)?;
        let (w, h) = (img.width() as usize, img.height() as usize);
        let src = img.to_rgb32f();
        let raw = src.as_raw();
        let mut out = vec![0.0f32; w * h * 3];
        let sample = |c: usize, fx: f32, fy: f32| -> f32 {
            let fx = fx.clamp(0.0, (N - 1) as f32);
            let fy = fy.clamp(0.0, (N - 1) as f32);
            let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
            let (jx, jy) = ((ix + 1).min(N - 1), (iy + 1).min(N - 1));
            let (tx, ty) = (fx - ix as f32, fy - iy as f32);
            let o = &ab[c * N * N..(c + 1) * N * N];
            let a = o[iy * N + ix] * (1.0 - tx) + o[iy * N + jx] * tx;
            let b = o[jy * N + ix] * (1.0 - tx) + o[jy * N + jx] * tx;
            a * (1.0 - ty) + b * ty
        };
        let (sx, sy) = (N as f32 / w as f32, N as f32 / h as f32);
        out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            for x in 0..w {
                let i = (y * w + x) * 3;
                let (l, _, _) = rgb_to_lab(raw[i], raw[i + 1], raw[i + 2]);
                let fx = (x as f32 + 0.5) * sx - 0.5;
                let a = sample(0, fx, fy) * strength;
                let b = sample(1, fx, fy) * strength;
                let (r, g, bl) = lab_to_rgb(l, a, b);
                row[x * 3] = r.clamp(0.0, 1.0);
                row[x * 3 + 1] = g.clamp(0.0, 1.0);
                row[x * 3 + 2] = bl.clamp(0.0, 1.0);
            }
        });
        let buf = DynamicImage::ImageRgb32F(
            Rgb32FImage::from_raw(w as u32, h as u32, out).expect("buffer size"),
        );
        Ok(if is_16bit(img) {
            DynamicImage::ImageRgb16(buf.to_rgb16())
        } else {
            DynamicImage::ImageRgb8(buf.to_rgb8())
        })
    }
}

fn is_16bit(img: &DynamicImage) -> bool {
    matches!(
        img,
        DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA16(_)
            | DynamicImage::ImageRgb16(_)
            | DynamicImage::ImageRgba16(_)
    )
}
