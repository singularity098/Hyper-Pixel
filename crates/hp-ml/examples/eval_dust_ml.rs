//! Dust benchmark with LaMa for the large defects:
//! `cargo run --release -p hp-ml --example eval_dust_ml -- [n] [dpi]`
use hp_core::restore::{DustLevel, Planes, find_defects_planes, heal_planes};
use hp_core::synth_dust::{DustStats, render, score};
use hp_ml::{InpaintParams, Inpainter};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let dpi: f64 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(600.0);
    let p = DustLevel::Normal.params().unwrap();
    let inp = Inpainter::load_default().expect("LaMa model (run ml_demo fetch)");
    let (mut classic, mut ml) = (DustStats::default(), DustStats::default());
    let mut windows = 0;
    let t0 = std::time::Instant::now();
    for seed in 0..n {
        let scene = render(seed, dpi, 50.0, 35.0, 40);
        let mut pl = Planes::from_image(&scene.dirty);
        let defects = find_defects_planes(&pl, dpi, &p);
        heal_planes(&mut pl, &defects.mask);
        let healed = pl.write_masked(&scene.dirty, &defects.mask);
        score(&mut classic, &scene, &defects.mask, &healed);
        windows += inp
            .refill(
                &mut pl,
                &defects,
                dpi,
                &InpaintParams::default(),
                &mut |_, _| {},
            )
            .unwrap();
        let out = pl.write_masked(&scene.dirty, &defects.mask);
        score(&mut ml, &scene, &defects.mask, &out);
    }
    println!(
        "classical: defect err {:.2}, global {:.4}",
        classic.err_after, classic.err_global_after
    );
    println!(
        "with LaMa: defect err {:.2}, global {:.4} ({windows} windows, {:.1} s total)",
        ml.err_after,
        ml.err_global_after,
        t0.elapsed().as_secs_f64()
    );
}
