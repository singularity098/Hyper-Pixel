//! Integration test against SANE's virtual "test" scanner. Skips (passes) when libsane or
//! the test backend is unavailable; CI enables it.

use hp_scan::{Backend, ColorMode, ScanSettings, sane::SaneBackend};
use std::sync::atomic::AtomicBool;

#[test]
fn scan_with_sane_test_backend() {
    let Ok(mut b) = SaneBackend::new() else {
        eprintln!("libsane not available; skipping");
        return;
    };
    let devs = b.list_devices().unwrap();
    let Some(d) = devs.iter().find(|d| d.name.starts_with("test")) else {
        eprintln!("SANE test backend not enabled; skipping");
        return;
    };
    let mut dev = b.open(&d.name).unwrap();
    let caps = dev.capabilities().unwrap();
    assert!(!caps.resolutions.is_empty());
    let cancel = AtomicBool::new(false);
    let mut frames = 0;
    for mode in [ColorMode::Color, ColorMode::Gray] {
        let r = dev
            .scan(
                &ScanSettings {
                    resolution: 75.0,
                    mode,
                    area_mm: Some((10.0, 20.0, 50.0, 40.0)),
                    ..Default::default()
                },
                &mut |p| {
                    if p.frame.is_some() {
                        frames += 1;
                    }
                },
                &cancel,
            )
            .unwrap();
        assert_eq!(r.dpi, 75.0);
        assert!((r.origin_mm.0 - 10.0).abs() < 0.5, "{:?}", r.origin_mm);
        let w = (50.0 / 25.4 * 75.0f64).round() as i64;
        assert!(
            (r.image.width() as i64 - w).abs() <= 2,
            "width {}",
            r.image.width()
        );
    }
    assert!(frames > 0);
}
