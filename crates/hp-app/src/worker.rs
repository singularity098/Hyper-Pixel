//! Scanner worker thread. SANE must be driven from a single thread, so the backend and the
//! open device live here; the UI talks to it through a command channel.

use hp_scan::{
    Backend, Capabilities, Device, DeviceInfo, FrameView, ScanError, ScanResult, ScanSettings,
    mock::MockBackend,
};
use slint::{Rgb8Pixel, SharedPixelBuffer};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    Preview,
    Final,
}

pub enum Command {
    ListDevices,
    Open(String),
    Scan {
        settings: ScanSettings,
        purpose: Purpose,
        pages: bool,
    },
}

pub enum Event {
    Devices {
        list: Vec<DeviceInfo>,
        warning: Option<String>,
    },
    Opened {
        name: String,
        caps: Capabilities,
    },
    /// Progressive image (already downscaled for display) and completion fraction.
    Progress {
        fraction: f32,
        frame: Option<(SharedPixelBuffer<Rgb8Pixel>, f64)>,
    },
    Page {
        result: ScanResult,
        purpose: Purpose,
    },
    Finished {
        purpose: Purpose,
        pages: usize,
    },
    Error(String),
}

pub struct Worker {
    tx: Sender<Command>,
    pub cancel: Arc<AtomicBool>,
}

impl Worker {
    pub fn spawn(events: impl Fn(Event) + Send + 'static) -> Self {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let c = cancel.clone();
        std::thread::Builder::new()
            .name("scanner".into())
            .spawn(move || run(rx, c, events))
            .expect("spawn scanner thread");
        Self { tx, cancel }
    }

    pub fn send(&self, cmd: Command) {
        if matches!(cmd, Command::Scan { .. }) {
            self.cancel.store(false, Ordering::Relaxed);
        }
        let _ = self.tx.send(cmd);
    }
}

/// Longest side of progressive preview frames.
const DISPLAY_MAX: u32 = 1600;

/// Nearest-neighbour downsample of the lines received so far into a display buffer.
/// Rows not yet scanned are dark grey. Returns the buffer and its scale (display px per scan px).
fn frame_to_display(f: &FrameView<'_>) -> Option<(SharedPixelBuffer<Rgb8Pixel>, f64)> {
    let total = f.total_lines.unwrap_or(f.lines).max(1);
    let scale = (DISPLAY_MAX as f64 / f.width.max(total) as f64).min(1.0);
    let (dw, dh) = (
        ((f.width as f64 * scale) as u32).max(1),
        ((total as f64 * scale) as u32).max(1),
    );
    let mut buf = SharedPixelBuffer::<Rgb8Pixel>::new(dw, dh);
    let bps = (f.depth as usize / 8).max(1);
    let stride = f.width as usize * f.channels as usize * bps;
    let out = buf.make_mut_bytes();
    for dy in 0..dh as usize {
        let sy = ((dy as f64 + 0.5) / scale) as usize;
        for dx in 0..dw as usize {
            let o = (dy * dw as usize + dx) * 3;
            if sy >= f.lines as usize {
                out[o..o + 3].copy_from_slice(&[24, 26, 31]);
                continue;
            }
            let sx = (((dx as f64 + 0.5) / scale) as usize).min(f.width as usize - 1);
            let base = sy * stride + sx * f.channels as usize * bps;
            // For 16-bit native-endian samples take the high byte.
            let hi = if bps == 2 {
                if cfg!(target_endian = "little") { 1 } else { 0 }
            } else {
                0
            };
            for c in 0..3 {
                let ch = if f.channels == 3 { c } else { 0 };
                out[o + c] = f.data.get(base + ch * bps + hi).copied().unwrap_or(0);
            }
        }
    }
    Some((buf, scale))
}

fn backend_list(
    sane: &mut Option<Box<dyn Backend>>,
    mock: &mut MockBackend,
    warn: &mut Option<String>,
) -> Vec<DeviceInfo> {
    let mut list = Vec::new();
    if let Some(b) = sane.as_mut() {
        match b.list_devices() {
            Ok(l) => list.extend(l),
            Err(e) => *warn = Some(e.to_string()),
        }
    }
    // Offer the demo scanners when there is nothing real, or when explicitly requested.
    if list.is_empty() || std::env::var_os("HYPER_PIXEL_MOCK").is_some() {
        list.extend(mock.list_devices().unwrap_or_default());
    }
    list
}

fn run(rx: Receiver<Command>, cancel: Arc<AtomicBool>, emit: impl Fn(Event)) {
    let mut warn = None;
    let mut sane: Option<Box<dyn Backend>> = if std::env::var_os("HYPER_PIXEL_MOCK").is_some() {
        None
    } else {
        match hp_scan::sane::SaneBackend::new() {
            Ok(b) => Some(Box::new(b)),
            Err(e) => {
                warn = Some(e.to_string());
                None
            }
        }
    };
    let mut mock = MockBackend::new();
    let mut device: Option<Box<dyn Device>> = None;

    while let Ok(cmd) = rx.recv() {
        match cmd {
            Command::ListDevices => {
                device = None; // SANE requires devices be closed before re-listing on some backends
                let list = backend_list(&mut sane, &mut mock, &mut warn);
                emit(Event::Devices {
                    list,
                    warning: warn.take(),
                });
            }
            Command::Open(name) => {
                device = None;
                let res = if name.starts_with("mock:") {
                    mock.open(&name)
                } else if let Some(b) = sane.as_mut() {
                    b.open(&name)
                } else {
                    Err(ScanError::NoDevice(name.clone()))
                };
                match res.and_then(|mut d| d.capabilities().map(|c| (d, c))) {
                    Ok((d, caps)) => {
                        device = Some(d);
                        emit(Event::Opened { name, caps });
                    }
                    Err(e) => emit(Event::Error(format!("Could not open scanner: {e}"))),
                }
            }
            Command::Scan {
                settings,
                purpose,
                pages,
            } => {
                let Some(dev) = device.as_mut() else {
                    emit(Event::Error("No scanner selected".into()));
                    continue;
                };
                let mut last = Instant::now() - Duration::from_secs(1);
                let mut progress = |p: hp_scan::Progress<'_>| {
                    let now = Instant::now();
                    let done = p.fraction >= 0.999;
                    if now.duration_since(last) < Duration::from_millis(120) && !done {
                        return;
                    }
                    last = now;
                    let frame = if purpose == Purpose::Preview {
                        p.frame.as_ref().and_then(frame_to_display)
                    } else {
                        None
                    };
                    emit(Event::Progress {
                        fraction: p.fraction,
                        frame,
                    });
                };
                let mut on_page = |r: ScanResult| emit(Event::Page { result: r, purpose });
                let res = if pages {
                    hp_scan::scan_pages(
                        dev.as_mut(),
                        &settings,
                        &mut progress,
                        &cancel,
                        &mut on_page,
                    )
                } else {
                    dev.scan(&settings, &mut progress, &cancel).map(|r| {
                        on_page(r);
                        1
                    })
                };
                match res {
                    Ok(n) => emit(Event::Finished { purpose, pages: n }),
                    Err(ScanError::Cancelled) => emit(Event::Error("Scan cancelled".into())),
                    Err(e) => emit(Event::Error(e.to_string())),
                }
            }
        }
    }
}
