//! Extracting (and straightening) rotated rectangles from a scan with high-quality
//! resampling. Works on 8- and 16-bit RGB/grey images without losing bit depth.

use crate::detect::ScanGeometry;
use crate::geom::RotatedRect;
use image::{DynamicImage, ImageBuffer, Luma, Pixel, Primitive, Rgb};
use rayon::prelude::*;

/// Catmull-Rom cubic weights.
#[inline]
fn cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

fn extract_generic<P>(
    src: &ImageBuffer<P, Vec<P::Subpixel>>,
    rect: &RotatedRect,
    out_w: u32,
    out_h: u32,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + Send + Sync,
    P::Subpixel: Primitive + Send + Sync + Into<f32>,
{
    let ch = P::CHANNEL_COUNT as usize;
    let (sw, sh) = (src.width() as isize, src.height() as isize);
    let raw = src.as_raw();
    let maxv: f32 = <P::Subpixel as Primitive>::DEFAULT_MAX_VALUE.into();
    let (ux, uy) = rect.axes();
    // Step in source pixels per output pixel along each local axis.
    let sx = rect.w / out_w as f64;
    let sy = rect.h / out_h as f64;
    let mut out: Vec<P::Subpixel> =
        vec![<P::Subpixel as Primitive>::DEFAULT_MIN_VALUE; out_w as usize * out_h as usize * ch];
    out.par_chunks_mut(out_w as usize * ch)
        .enumerate()
        .for_each(|(oy, row)| {
            let mut acc = [0.0f32; 4];
            for ox in 0..out_w as usize {
                let u = (ox as f64 + 0.5) * sx - rect.w / 2.0;
                let v = (oy as f64 + 0.5) * sy - rect.h / 2.0;
                // Source position in pixel-centre coordinates.
                let px = rect.cx + ux.x * u + uy.x * v - 0.5;
                let py = rect.cy + ux.y * u + uy.y * v - 0.5;
                let (x0, y0) = (px.floor() as isize, py.floor() as isize);
                let wx = cubic_weights((px - x0 as f64) as f32);
                let wy = cubic_weights((py - y0 as f64) as f32);
                acc[..ch].fill(0.0);
                for (j, wyj) in wy.iter().enumerate() {
                    let yy = (y0 - 1 + j as isize).clamp(0, sh - 1) as usize;
                    for (i, wxi) in wx.iter().enumerate() {
                        let xx = (x0 - 1 + i as isize).clamp(0, sw - 1) as usize;
                        let w = wxi * wyj;
                        let base = (yy * sw as usize + xx) * ch;
                        for c in 0..ch {
                            acc[c] += raw[base + c].into() * w;
                        }
                    }
                }
                for c in 0..ch {
                    let v = acc[c].clamp(0.0, maxv).round();
                    row[ox * ch + c] = num_from_f32::<P::Subpixel>(v);
                }
            }
        });
    ImageBuffer::from_raw(out_w, out_h, out).expect("buffer size")
}

#[inline]
fn num_from_f32<T: Primitive>(v: f32) -> T {
    num_traits::NumCast::from(v).unwrap_or(T::DEFAULT_MIN_VALUE)
}

/// Extract `rect` (bed mm) from `img`, straightened, at the image's native resolution.
/// `inset_mm` trims each edge (positive) to avoid slivers of background.
pub fn extract(
    img: &DynamicImage,
    geo: ScanGeometry,
    rect: &RotatedRect,
    inset_mm: f64,
) -> DynamicImage {
    let r = geo.mm_to_px(&rect.inset(inset_mm));
    let out_w = r.w.round().max(1.0) as u32;
    let out_h = r.h.round().max(1.0) as u32;
    match img {
        DynamicImage::ImageRgb8(b) => {
            DynamicImage::ImageRgb8(extract_generic::<Rgb<u8>>(b, &r, out_w, out_h))
        }
        DynamicImage::ImageRgb16(b) => {
            DynamicImage::ImageRgb16(extract_generic::<Rgb<u16>>(b, &r, out_w, out_h))
        }
        DynamicImage::ImageLuma8(b) => {
            DynamicImage::ImageLuma8(extract_generic::<Luma<u8>>(b, &r, out_w, out_h))
        }
        DynamicImage::ImageLuma16(b) => {
            DynamicImage::ImageLuma16(extract_generic::<Luma<u16>>(b, &r, out_w, out_h))
        }
        DynamicImage::ImageRgba16(_) | DynamicImage::ImageLumaA16(_) => DynamicImage::ImageRgb16(
            extract_generic::<Rgb<u16>>(&img.to_rgb16(), &r, out_w, out_h),
        ),
        _ => DynamicImage::ImageRgb8(extract_generic::<Rgb<u8>>(&img.to_rgb8(), &r, out_w, out_h)),
    }
}

/// Rotate an image by a multiple of 90° (clockwise quarter turns).
pub fn rotate_quarter(img: &DynamicImage, quarters: i32) -> DynamicImage {
    match quarters.rem_euclid(4) {
        1 => img.rotate90(),
        2 => img.rotate180(),
        3 => img.rotate270(),
        _ => img.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn axis_aligned_extract_is_exact() {
        let mut img = RgbImage::new(100, 80);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = Rgb([(x * 2) as u8, (y * 3) as u8, 7]);
        }
        let geo = ScanGeometry::new(25.4); // 1 px = 1 mm
        let r = RotatedRect::new(30.0, 25.0, 20.0, 10.0, 0.0);
        let out = extract(&DynamicImage::ImageRgb8(img.clone()), geo, &r, 0.0).to_rgb8();
        assert_eq!(out.dimensions(), (20, 10));
        assert_eq!(out.get_pixel(0, 0), img.get_pixel(20, 20));
        assert_eq!(out.get_pixel(19, 9), img.get_pixel(39, 29));
    }

    #[test]
    fn sixteen_bit_preserved() {
        let img = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(
            50,
            50,
            Rgb([1000u16, 40000, 65535]),
        ));
        let out = extract(
            &img,
            ScanGeometry::new(25.4),
            &RotatedRect::new(25.0, 25.0, 20.0, 20.0, 7.0),
            0.0,
        );
        let DynamicImage::ImageRgb16(b) = out else {
            panic!("bit depth lost")
        };
        assert_eq!(b.get_pixel(10, 10), &Rgb([1000, 40000, 65535]));
    }
}
