//! List SANE devices and their options; optionally scan the first one.
//! `cargo run -p hp-scan --example sane_probe -- [device] [out.png]`
use hp_scan::{Backend, ScanSettings, sane::SaneBackend};
use std::sync::atomic::AtomicBool;

fn main() {
    let mut b = match SaneBackend::new() {
        Ok(b) => b,
        Err(e) => return eprintln!("{e}"),
    };
    let devs = b.list_devices().unwrap();
    for d in &devs {
        println!("{} — {}", d.name, d.label());
    }
    let a: Vec<String> = std::env::args().collect();
    let Some(name) = a
        .get(1)
        .cloned()
        .or_else(|| devs.first().map(|d| d.name.clone()))
    else {
        return;
    };
    let mut dev = b.open(&name).unwrap();
    for o in dev.options().unwrap() {
        println!(
            "  [{}] {} ({:?}, {:?}) {:?} active={}",
            o.index, o.name, o.kind, o.unit, o.constraint, o.active
        );
    }
    println!("caps: {:?}", dev.capabilities().unwrap());
    if let Some(out) = a.get(2) {
        let r = dev
            .scan(
                &ScanSettings {
                    resolution: 75.0,
                    ..Default::default()
                },
                &mut |p| eprint!("\r{:.0}%", p.fraction * 100.0),
                &AtomicBool::new(false),
            )
            .unwrap();
        println!(
            "\nscanned {}x{} @ {} dpi, origin {:?}",
            r.image.width(),
            r.image.height(),
            r.dpi,
            r.origin_mm
        );
        r.image.save(out).unwrap();
    }
}
