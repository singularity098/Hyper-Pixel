//! Saving images in the supported formats, with resolution metadata where the format allows.

use crate::pdf::{self, Page, PageImage, TextWord};
use image::{DynamicImage, ImageEncoder, codecs::jpeg::JpegEncoder};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("encoding failed: {0}")]
    Encode(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ImageFormat {
    #[default]
    Jpeg,
    Png,
    Tiff,
    /// Lossless JPEG XL (pure Rust encoder).
    JpegXl,
    /// Lossless WebP.
    WebP,
    Bmp,
    Pdf,
}

impl ImageFormat {
    pub const ALL: [ImageFormat; 7] = [
        ImageFormat::Jpeg,
        ImageFormat::Png,
        ImageFormat::Tiff,
        ImageFormat::JpegXl,
        ImageFormat::WebP,
        ImageFormat::Bmp,
        ImageFormat::Pdf,
    ];

    pub fn extension(self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Png => "png",
            ImageFormat::Tiff => "tif",
            ImageFormat::JpegXl => "jxl",
            ImageFormat::WebP => "webp",
            ImageFormat::Bmp => "bmp",
            ImageFormat::Pdf => "pdf",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "JPEG",
            ImageFormat::Png => "PNG (lossless)",
            ImageFormat::Tiff => "TIFF (lossless)",
            ImageFormat::JpegXl => "JPEG XL (lossless)",
            ImageFormat::WebP => "WebP (lossless)",
            ImageFormat::Bmp => "BMP",
            ImageFormat::Pdf => "PDF",
        }
    }
    pub fn supports_16bit(self) -> bool {
        matches!(
            self,
            ImageFormat::Png | ImageFormat::Tiff | ImageFormat::JpegXl
        )
    }
    pub fn from_extension(ext: &str) -> Option<Self> {
        Some(match ext.to_ascii_lowercase().as_str() {
            "jpg" | "jpeg" => ImageFormat::Jpeg,
            "png" => ImageFormat::Png,
            "tif" | "tiff" => ImageFormat::Tiff,
            "jxl" => ImageFormat::JpegXl,
            "webp" => ImageFormat::WebP,
            "bmp" => ImageFormat::Bmp,
            "pdf" => ImageFormat::Pdf,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    pub format: ImageFormat,
    pub jpeg_quality: u8,
    pub dpi: f64,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ImageFormat::Jpeg,
            jpeg_quality: 92,
            dpi: 300.0,
        }
    }
}

fn enc_err<E: std::fmt::Display>(e: E) -> ExportError {
    ExportError::Encode(e.to_string())
}

fn is_gray(img: &DynamicImage) -> bool {
    matches!(
        img,
        DynamicImage::ImageLuma8(_)
            | DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA8(_)
            | DynamicImage::ImageLumaA16(_)
    )
}
fn is_16(img: &DynamicImage) -> bool {
    matches!(
        img,
        DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageRgb16(_)
            | DynamicImage::ImageLumaA16(_)
            | DynamicImage::ImageRgba16(_)
    )
}

pub fn encode_jpeg(img: &DynamicImage, quality: u8, dpi: f64) -> Result<Vec<u8>, ExportError> {
    let mut out = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100));
    enc.set_pixel_density(image::codecs::jpeg::PixelDensity::dpi(dpi.round() as u16));
    if is_gray(img) {
        let g = img.to_luma8();
        enc.write_image(
            g.as_raw(),
            g.width(),
            g.height(),
            image::ExtendedColorType::L8,
        )
        .map_err(enc_err)?;
    } else {
        let c = img.to_rgb8();
        enc.write_image(
            c.as_raw(),
            c.width(),
            c.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(enc_err)?;
    }
    Ok(out)
}

fn encode_png(img: &DynamicImage, dpi: f64) -> Result<Vec<u8>, ExportError> {
    let mut out = Vec::new();
    let (w, h) = (img.width(), img.height());
    let (ct, bd, data): (png::ColorType, png::BitDepth, Vec<u8>) = match (is_gray(img), is_16(img))
    {
        (true, false) => (
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            img.to_luma8().into_raw(),
        ),
        (false, false) => (
            png::ColorType::Rgb,
            png::BitDepth::Eight,
            img.to_rgb8().into_raw(),
        ),
        (true, true) => (
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            img.to_luma16()
                .iter()
                .flat_map(|v| v.to_be_bytes())
                .collect(),
        ),
        (false, true) => (
            png::ColorType::Rgb,
            png::BitDepth::Sixteen,
            img.to_rgb16()
                .iter()
                .flat_map(|v| v.to_be_bytes())
                .collect(),
        ),
    };
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(ct);
        enc.set_depth(bd);
        enc.set_compression(png::Compression::Balanced);
        let ppm = (dpi / 0.0254).round() as u32;
        enc.set_pixel_dims(Some(png::PixelDimensions {
            xppu: ppm,
            yppu: ppm,
            unit: png::Unit::Meter,
        }));
        let mut wr = enc.write_header().map_err(enc_err)?;
        wr.write_image_data(&data).map_err(enc_err)?;
    }
    Ok(out)
}

fn encode_tiff(img: &DynamicImage, dpi: f64) -> Result<Vec<u8>, ExportError> {
    use tiff::encoder::{Compression, Predictor, Rational, TiffEncoder, colortype};
    use tiff::tags::ResolutionUnit;
    let mut cur = Cursor::new(Vec::new());
    let mut enc = TiffEncoder::new(&mut cur)
        .map_err(enc_err)?
        .with_compression(Compression::Lzw)
        .with_predictor(Predictor::Horizontal);
    let (w, h) = (img.width(), img.height());
    let res = Rational {
        n: (dpi * 100.0).round() as u32,
        d: 100,
    };
    macro_rules! write_img {
        ($ct:ty, $data:expr) => {{
            let mut im = enc.new_image::<$ct>(w, h).map_err(enc_err)?;
            im.resolution(ResolutionUnit::Inch, res);
            im.write_data(&$data).map_err(enc_err)?;
        }};
    }
    match (is_gray(img), is_16(img)) {
        (true, false) => write_img!(colortype::Gray8, img.to_luma8().into_raw()),
        (false, false) => write_img!(colortype::RGB8, img.to_rgb8().into_raw()),
        (true, true) => write_img!(colortype::Gray16, img.to_luma16().into_raw()),
        (false, true) => write_img!(colortype::RGB16, img.to_rgb16().into_raw()),
    }
    Ok(cur.into_inner())
}

fn encode_jxl(img: &DynamicImage) -> Result<Vec<u8>, ExportError> {
    use zune_core::{bit_depth::BitDepth, colorspace::ColorSpace, options::EncoderOptions};
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (cs, bd, data): (ColorSpace, BitDepth, Vec<u8>) = match (is_gray(img), is_16(img)) {
        (true, false) => (ColorSpace::Luma, BitDepth::Eight, img.to_luma8().into_raw()),
        (false, false) => (ColorSpace::RGB, BitDepth::Eight, img.to_rgb8().into_raw()),
        (true, true) => (
            ColorSpace::Luma,
            BitDepth::Sixteen,
            img.to_luma16()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
        (false, true) => (
            ColorSpace::RGB,
            BitDepth::Sixteen,
            img.to_rgb16()
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        ),
    };
    let opts = EncoderOptions::new(w, h, cs, bd);
    let enc = zune_jpegxl::JxlSimpleEncoder::new(&data, opts);
    let mut out = Vec::new();
    enc.encode(&mut out)
        .map_err(|e| ExportError::Encode(format!("{e:?}")))?;
    Ok(out)
}

fn encode_with_image(img: &DynamicImage, fmt: image::ImageFormat) -> Result<Vec<u8>, ExportError> {
    let mut cur = Cursor::new(Vec::new());
    let img = match fmt {
        // WebP/BMP encoders in `image` are 8-bit only.
        image::ImageFormat::WebP | image::ImageFormat::Bmp => {
            if is_gray(img) && fmt == image::ImageFormat::Bmp {
                DynamicImage::ImageLuma8(img.to_luma8())
            } else {
                DynamicImage::ImageRgb8(img.to_rgb8())
            }
        }
        _ => img.clone(),
    };
    img.write_to(&mut cur, fmt).map_err(enc_err)?;
    Ok(cur.into_inner())
}

/// Encode a single image to bytes. For PDF this produces a one-page document.
pub fn encode(img: &DynamicImage, opts: &ExportOptions) -> Result<Vec<u8>, ExportError> {
    match opts.format {
        ImageFormat::Jpeg => encode_jpeg(img, opts.jpeg_quality, opts.dpi),
        ImageFormat::Png => encode_png(img, opts.dpi),
        ImageFormat::Tiff => encode_tiff(img, opts.dpi),
        ImageFormat::JpegXl => encode_jxl(img),
        ImageFormat::WebP => encode_with_image(img, image::ImageFormat::WebP),
        ImageFormat::Bmp => encode_with_image(img, image::ImageFormat::Bmp),
        ImageFormat::Pdf => pdf_document(&[(img.clone(), Vec::new())], opts, "Scan"),
    }
}

pub fn save(img: &DynamicImage, path: &Path, opts: &ExportOptions) -> Result<(), ExportError> {
    let bytes = encode(img, opts)?;
    write_atomic(path, &bytes)
}

/// Write via a temporary file + rename so a crash never leaves a truncated image.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), ExportError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
    ));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Is this image effectively black & white (only near-0 and near-max values)?
fn is_bilevel(img: &DynamicImage) -> bool {
    match img {
        DynamicImage::ImageLuma8(g) => g.as_raw().iter().all(|&v| v == 0 || v == 255),
        _ => false,
    }
}

/// Build a multi-page PDF. Each page may carry OCR words (pixel coordinates).
pub fn pdf_document(
    pages: &[(DynamicImage, Vec<TextWord>)],
    opts: &ExportOptions,
    title: &str,
) -> Result<Vec<u8>, ExportError> {
    let mut out = Vec::new();
    for (img, words) in pages {
        let image = if is_bilevel(img) {
            let g = img.to_luma8();
            let (w, h) = g.dimensions();
            let stride = (w as usize).div_ceil(8);
            let mut packed = vec![0u8; stride * h as usize];
            for (x, y, p) in g.enumerate_pixels() {
                if p[0] > 127 {
                    packed[y as usize * stride + x as usize / 8] |= 0x80 >> (x % 8);
                }
            }
            PageImage::Bilevel {
                packed,
                width: w,
                height: h,
            }
        } else {
            PageImage::Jpeg {
                data: encode_jpeg(img, opts.jpeg_quality, opts.dpi)?,
                width: img.width(),
                height: img.height(),
                gray: is_gray(img),
            }
        };
        out.push(Page {
            image,
            dpi: opts.dpi,
            words: words.clone(),
        });
    }
    Ok(pdf::write_pdf(&out, title))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn all_formats_encode_and_decodable_ones_roundtrip() {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_fn(40, 30, |x, y| {
            Rgb([x as u8 * 6, y as u8 * 8, 100])
        }));
        for f in ImageFormat::ALL {
            let bytes = encode(
                &img,
                &ExportOptions {
                    format: f,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(bytes.len() > 20, "{f:?}");
            if matches!(
                f,
                ImageFormat::Png | ImageFormat::Tiff | ImageFormat::WebP | ImageFormat::Bmp
            ) {
                let back = image::load_from_memory(&bytes).unwrap().to_rgb8();
                assert_eq!(back.as_raw(), img.to_rgb8().as_raw(), "{f:?} not lossless");
            }
        }
        let img16 = DynamicImage::ImageRgb16(ImageBuffer::from_fn(20, 20, |x, _| {
            Rgb([x as u16 * 3000, 5, 65535])
        }));
        for f in [ImageFormat::Png, ImageFormat::Tiff, ImageFormat::JpegXl] {
            encode(
                &img16,
                &ExportOptions {
                    format: f,
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let back = image::load_from_memory(
            &encode(
                &img16,
                &ExportOptions {
                    format: ImageFormat::Png,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert!(matches!(back, DynamicImage::ImageRgb16(_)));
    }
}
