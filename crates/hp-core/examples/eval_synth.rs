//! Quick synthetic benchmark: `cargo run --release -p hp-core --example eval_synth -- [n] [dpi] [light|dark|mix] [gap_mm]`
use hp_core::eval::Evaluator;
use hp_core::synth::{SceneParams, render_scene};
use hp_core::{DetectParams, ScanGeometry, detect_photos};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let dpi: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(75.0);
    let bg = a.get(3).cloned().unwrap_or("mix".into());
    let gap: f64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let mut ev = Evaluator::new();
    let t0 = std::time::Instant::now();
    for seed in 0..n {
        let dark = match bg.as_str() {
            "dark" => true,
            "light" => false,
            _ => seed % 2 == 0,
        };
        let scene = render_scene(&SceneParams {
            seed: seed + 1000,
            dpi,
            dark_background: dark,
            min_gap_mm: gap,
            ..Default::default()
        });
        let rep = detect_photos(
            &scene.image,
            ScanGeometry::new(dpi),
            &DetectParams::default(),
        );
        let det: Vec<_> = rep.photos.iter().map(|d| d.rect).collect();
        // HP_SCENES=<dir>: also save each raw scene and its truth, for `hyper-pixel-cli eval`.
        if let Ok(dir) = std::env::var("HP_SCENES") {
            let _ = scene.image.save(format!("{dir}/scene_{seed}.png"));
            let truth = serde_json::json!({ "dpi": dpi, "photos": scene.truth });
            let _ = std::fs::write(format!("{dir}/scene_{seed}.json"), truth.to_string());
        }
        let before = ev.stats().perfect_scenes;
        ev.add(&scene.truth, &det);
        if ev.stats().perfect_scenes == before && std::env::var("HP_DUMP").is_ok() {
            let g = ScanGeometry::new(dpi);
            let mut im = scene.image.to_rgb8();
            let t: Vec<_> = scene.truth.iter().map(|r| g.mm_to_px(r)).collect();
            let d: Vec<_> = det.iter().map(|r| g.mm_to_px(r)).collect();
            hp_core::raster::draw_rects(&mut im, &t, [0, 255, 0], 1);
            hp_core::raster::draw_rects(&mut im, &d, [255, 0, 0], 1);
            let dir = std::env::var("HP_DUMP").unwrap();
            let _ = im.save(format!("{dir}/fail_{seed}.png"));
            eprintln!("seed {seed}: truth {} det {}", scene.truth.len(), det.len());
        }
    }
    println!("{}", serde_json::to_string_pretty(ev.stats()).unwrap());
    println!(
        "time per scene (incl. render): {:.1} ms",
        t0.elapsed().as_secs_f64() * 1000.0 / n as f64
    );
}
