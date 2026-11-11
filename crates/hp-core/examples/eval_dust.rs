//! Dust/scratch benchmark: `cargo run --release -p hp-core --example eval_dust -- [n] [dpi] [light|normal|strong]`
//! `HP_DUMP=<dir>` writes dirty / cleaned / mask overlays for every scene, `HP_MISSED=<dir>`
//! crops of missed visible defects, `HP_TRUTH_MASK=1` scores the filling alone on the true
//! defect pixels, `HP_VERBOSE=1` prints false alarms per scene.
use hp_core::restore::{DustLevel, overlay, remove_dust};
use hp_core::synth_dust::{DustStats, render, score};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let dpi: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(600.0);
    let level = match a.get(3).map(|s| s.as_str()) {
        Some("light") => DustLevel::Light,
        Some("strong") => DustLevel::Strong,
        _ => DustLevel::Normal,
    };
    let params = level.params().unwrap();
    let mut st = DustStats::default();
    let t0 = std::time::Instant::now();
    let mut t_clean = 0.0;
    for seed in 0..n {
        let scene = render(seed, dpi, 50.0, 35.0, 40);
        let t = std::time::Instant::now();
        let (cleaned, defects) = if std::env::var_os("HP_TRUTH_MASK").is_some() {
            // Fill quality alone: heal the true defect pixels.
            let mask = scene.touched.dilate(1.0);
            let d = hp_core::restore::Defects {
                mask: mask.clone(),
                items: Vec::new(),
            };
            (hp_core::restore::heal(&scene.dirty, &mask), d)
        } else {
            remove_dust(&scene.dirty, dpi, &params)
        };
        t_clean += t.elapsed().as_secs_f64();
        let before = st.false_regions;
        score(&mut st, &scene, &defects.mask, &cleaned);
        if std::env::var_os("HP_VERBOSE").is_some() {
            eprintln!("scene {seed}: {} false regions", st.false_regions - before);
        }
        if let Ok(dir) = std::env::var("HP_MISSED") {
            dump_missed(&dir, seed, &scene, &defects.mask);
        }
        if let Ok(dir) = std::env::var("HP_DUMP") {
            let _ = scene.dirty.save(format!("{dir}/{seed}_dirty.png"));
            let _ = scene.clean.save(format!("{dir}/{seed}_clean.png"));
            let _ = cleaned.save(format!("{dir}/{seed}_cleaned.png"));
            let _ = overlay(&scene.dirty, &defects.mask).save(format!("{dir}/{seed}_mask.png"));
            let _ = overlay(&scene.dirty, &scene.truth).save(format!("{dir}/{seed}_truth.png"));
        }
    }
    println!("{}", serde_json::to_string_pretty(&st).unwrap());
    println!(
        "recall {:.1}%, visible defects {:.1}%",
        st.recall() * 100.0,
        st.visible_recall() * 100.0
    );
    println!(
        "clean time per 50x35 mm patch: {:.0} ms (total incl. render {:.1} s)",
        t_clean * 1000.0 / n as f64,
        t0.elapsed().as_secs_f64()
    );
}

/// HP_MISSED=<dir>: crops of visible defects that were missed.
#[allow(dead_code)]
pub fn dump_missed(
    dir: &str,
    seed: u64,
    scene: &hp_core::synth_dust::DustScene,
    mask: &hp_core::raster::Mask,
) {
    let (cl, di) = (scene.clean.to_rgb8(), scene.dirty.to_rgb8());
    let w = scene.truth.w;
    for (k, d) in scene.defects.iter().enumerate() {
        let hit = d.iter().filter(|&&i| mask.data[i]).count();
        if hit as f64 >= 0.8 * d.len() as f64 {
            continue;
        }
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
        if change < 24.0 {
            continue;
        }
        let xs = d.iter().map(|i| i % w);
        let ys = d.iter().map(|i| i / w);
        let (x0, x1) = (xs.clone().min().unwrap(), xs.max().unwrap());
        let (y0, y1) = (ys.clone().min().unwrap(), ys.max().unwrap());
        let (cx, cy) = ((x0 + x1) / 2, (y0 + y1) / 2);
        let r = 40usize.max((x1 - x0).max(y1 - y0) / 2 + 10);
        let (bx, by) = (cx.saturating_sub(r) as u32, cy.saturating_sub(r) as u32);
        let crop = |img: &image::DynamicImage| img.crop_imm(bx, by, 2 * r as u32, 2 * r as u32);
        let ov = hp_core::restore::overlay(&scene.dirty, mask);
        let a = crop(&scene.dirty);
        let b = crop(&ov);
        let mut out = image::RgbImage::new(a.width() * 2, a.height());
        image::imageops::overlay(&mut out, &a.to_rgb8(), 0, 0);
        image::imageops::overlay(&mut out, &b.to_rgb8(), a.width() as i64, 0);
        let _ = out.save(format!(
            "{dir}/miss_{seed}_{k}_{}_{}px_c{:.0}_hit{}_at{cx}_{cy}.png",
            scene.kinds[k],
            d.len(),
            change,
            hit
        ));
    }
}
