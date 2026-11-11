//! `cargo run --release -p hp-ml --example ml_demo -- fetch`
//! `... -- inpaint <image> <mask> <out>` (mask: white = fill)
//! `... -- colorize <image> <out>`
use hp_core::restore::{Defect, Defects, Planes};
use hp_ml::{Colorizer, DDCOLOR, InpaintParams, Inpainter, LAMA, models};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("fetch") => {
            for spec in [LAMA, DDCOLOR] {
                let mut last = 0;
                let dir = models::download(&spec, &mut |d, t| {
                    if d - last > 50_000_000 {
                        last = d;
                        eprintln!("{}: {d}/{t}", spec.label);
                    }
                })
                .unwrap();
                println!("{} ready in {}", spec.label, dir.display());
            }
        }
        Some("inpaint") => {
            let img = image::open(&a[2]).unwrap();
            let m = image::open(&a[3]).unwrap().to_luma8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let m = image::imageops::resize(
                &m,
                w as u32,
                h as u32,
                image::imageops::FilterType::Nearest,
            );
            let mask = hp_core::raster::Mask {
                w,
                h,
                data: m.pixels().map(|p| p[0] > 0).collect(),
            };
            let pixels: Vec<usize> = (0..w * h).filter(|&i| mask.data[i]).collect();
            let defects = Defects {
                mask: mask.clone(),
                items: vec![Defect {
                    pixels,
                    bbox: (0, 0, w - 1, h - 1),
                    bright: true,
                    length_px: 100.0,
                    width_px: 100.0,
                }],
            };
            let mut pl = Planes::from_image(&img);
            hp_core::restore::heal_planes(&mut pl, &mask);
            pl.write_masked(&img, &mask)
                .save(format!("{}.heal.png", a[4]))
                .unwrap();
            let inp = Inpainter::load_default().unwrap();
            let t = std::time::Instant::now();
            let n = inp
                .refill(
                    &mut pl,
                    &defects,
                    300.0,
                    &InpaintParams::default(),
                    &mut |_, _| {},
                )
                .unwrap();
            eprintln!("{n} windows in {:?}", t.elapsed());
            pl.write_masked(&img, &mask).save(&a[4]).unwrap();
        }
        Some("raw") => {
            // Reference pipeline: squash to 512 x 512, run once.
            use rten_tensor::NdTensor;
            let img = image::open(&a[2])
                .unwrap()
                .resize_exact(512, 512, image::imageops::FilterType::Triangle)
                .to_rgb32f();
            let m = image::open(&a[3])
                .unwrap()
                .resize_exact(512, 512, image::imageops::FilterType::Triangle)
                .to_luma8();
            let mut rgb = NdTensor::<f32, 4>::zeros([1, 3, 512, 512]);
            let mut hole = NdTensor::<f32, 4>::zeros([1, 1, 512, 512]);
            for y in 0..512 {
                for x in 0..512 {
                    let h = m.get_pixel(x as u32, y as u32)[0] > 0;
                    hole[[0, 0, y, x]] = h as u8 as f32;
                    for c in 0..3 {
                        rgb[[0, c, y, x]] = if h && a.len() > 5 {
                            0.0
                        } else {
                            img.get_pixel(x as u32, y as u32)[c]
                        };
                    }
                }
            }
            let inp = Inpainter::load_default().unwrap();
            let out = inp.run_window(rgb, hole).unwrap();
            let mut o = image::RgbImage::new(512, 512);
            for y in 0..512 {
                for x in 0..512 {
                    let p = |c: usize| (out[c * 512 * 512 + y * 512 + x] * 255.0) as u8;
                    o.put_pixel(x as u32, y as u32, image::Rgb([p(0), p(1), p(2)]));
                }
            }
            o.save(&a[4]).unwrap();
        }
        Some("colorize") => {
            let img = image::open(&a[2]).unwrap();
            let c = Colorizer::load_default().unwrap();
            let t = std::time::Instant::now();
            let out = c.colorize(&img, 1.0).unwrap();
            eprintln!("colorized in {:?}", t.elapsed());
            out.save(&a[3]).unwrap();
        }
        _ => eprintln!("usage: fetch | inpaint <img> <mask> <out> | colorize <img> <out>"),
    }
}
