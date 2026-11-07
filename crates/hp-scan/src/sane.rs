//! SANE backend, loaded dynamically from the host system.

use crate::*;
use libloading::{Library, Symbol};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::Arc;
use std::sync::atomic::Ordering;

type SaneStatus = c_int;
type SaneHandle = *mut c_void;
type SaneWord = c_int;

const STATUS_GOOD: SaneStatus = 0;
const STATUS_CANCELLED: SaneStatus = 2;
const STATUS_EOF: SaneStatus = 5;
const STATUS_NO_DOCS: SaneStatus = 7;

const TYPE_BOOL: c_int = 0;
const TYPE_INT: c_int = 1;
const TYPE_FIXED: c_int = 2;
const TYPE_STRING: c_int = 3;
const TYPE_BUTTON: c_int = 4;

const CAP_SOFT_SELECT: c_int = 1;
const CAP_INACTIVE: c_int = 32;
const CAP_ADVANCED: c_int = 64;

const ACTION_GET: c_int = 0;
const ACTION_SET: c_int = 1;

const FRAME_GRAY: c_int = 0;
const FRAME_RGB: c_int = 1;
const FRAME_RED: c_int = 2;
const FRAME_GREEN: c_int = 3;
const FRAME_BLUE: c_int = 4;

#[repr(C)]
struct RawDevice {
    name: *const c_char,
    vendor: *const c_char,
    model: *const c_char,
    kind: *const c_char,
}

#[repr(C)]
struct RawRange {
    min: SaneWord,
    max: SaneWord,
    quant: SaneWord,
}

#[repr(C)]
struct RawOptionDescriptor {
    name: *const c_char,
    title: *const c_char,
    desc: *const c_char,
    type_: c_int,
    unit: c_int,
    size: c_int,
    cap: c_int,
    constraint_type: c_int,
    constraint: *const c_void,
}

#[repr(C)]
#[derive(Default)]
struct RawParameters {
    format: c_int,
    last_frame: c_int,
    bytes_per_line: c_int,
    pixels_per_line: c_int,
    lines: c_int,
    depth: c_int,
}

type AuthCallback = Option<unsafe extern "C" fn(*const c_char, *mut c_char, *mut c_char)>;

struct Api {
    _lib: Library,
    init: unsafe extern "C" fn(*mut SaneWord, AuthCallback) -> SaneStatus,
    exit: unsafe extern "C" fn(),
    get_devices: unsafe extern "C" fn(*mut *mut *const RawDevice, c_int) -> SaneStatus,
    open: unsafe extern "C" fn(*const c_char, *mut SaneHandle) -> SaneStatus,
    close: unsafe extern "C" fn(SaneHandle),
    get_option_descriptor: unsafe extern "C" fn(SaneHandle, c_int) -> *const RawOptionDescriptor,
    control_option:
        unsafe extern "C" fn(SaneHandle, c_int, c_int, *mut c_void, *mut c_int) -> SaneStatus,
    get_parameters: unsafe extern "C" fn(SaneHandle, *mut RawParameters) -> SaneStatus,
    start: unsafe extern "C" fn(SaneHandle) -> SaneStatus,
    read: unsafe extern "C" fn(SaneHandle, *mut u8, c_int, *mut c_int) -> SaneStatus,
    cancel: unsafe extern "C" fn(SaneHandle),
    strstatus: unsafe extern "C" fn(SaneStatus) -> *const c_char,
}

impl Drop for Api {
    fn drop(&mut self) {
        unsafe { (self.exit)() };
    }
}

impl Api {
    fn load() -> Result<Self> {
        let mut last = String::new();
        for name in ["libsane.so.1", "libsane.so"] {
            match unsafe { Library::new(name) } {
                Ok(lib) => return unsafe { Self::from_lib(lib) },
                Err(e) => last = e.to_string(),
            }
        }
        Err(ScanError::LibraryMissing(last))
    }

    unsafe fn from_lib(lib: Library) -> Result<Self> {
        macro_rules! sym {
            ($n:literal) => {{
                let s: Symbol<_> = unsafe { lib.get(concat!($n, "\0").as_bytes()) }
                    .map_err(|e| ScanError::LibraryMissing(e.to_string()))?;
                *s
            }};
        }
        let api = Api {
            init: sym!("sane_init"),
            exit: sym!("sane_exit"),
            get_devices: sym!("sane_get_devices"),
            open: sym!("sane_open"),
            close: sym!("sane_close"),
            get_option_descriptor: sym!("sane_get_option_descriptor"),
            control_option: sym!("sane_control_option"),
            get_parameters: sym!("sane_get_parameters"),
            start: sym!("sane_start"),
            read: sym!("sane_read"),
            cancel: sym!("sane_cancel"),
            strstatus: sym!("sane_strstatus"),
            _lib: lib,
        };
        let mut version: SaneWord = 0;
        let st = unsafe { (api.init)(&mut version, None) };
        if st != STATUS_GOOD {
            return Err(ScanError::Sane(format!(
                "sane_init failed: {}",
                api.status_str(st)
            )));
        }
        log::info!("SANE initialised, version code {version:#x}");
        Ok(api)
    }

    fn status_str(&self, st: SaneStatus) -> String {
        let p = unsafe { (self.strstatus)(st) };
        if p.is_null() {
            format!("status {st}")
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    fn check(&self, st: SaneStatus, what: &str) -> Result<()> {
        match st {
            STATUS_GOOD => Ok(()),
            STATUS_CANCELLED => Err(ScanError::Cancelled),
            STATUS_NO_DOCS => Err(ScanError::NoDocuments),
            _ => Err(ScanError::Sane(format!("{what}: {}", self.status_str(st)))),
        }
    }
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

#[inline]
fn fixed_to_f64(w: SaneWord) -> f64 {
    w as f64 / 65536.0
}
#[inline]
fn f64_to_fixed(v: f64) -> SaneWord {
    (v * 65536.0).round() as SaneWord
}

pub struct SaneBackend {
    api: Arc<Api>,
}

impl SaneBackend {
    pub fn new() -> Result<Self> {
        Ok(Self {
            api: Arc::new(Api::load()?),
        })
    }
}

impl Backend for SaneBackend {
    fn name(&self) -> &'static str {
        "SANE"
    }

    fn list_devices(&mut self) -> Result<Vec<DeviceInfo>> {
        let mut list: *mut *const RawDevice = std::ptr::null_mut();
        let st = unsafe { (self.api.get_devices)(&mut list, 0) };
        self.api.check(st, "listing devices")?;
        let mut out = Vec::new();
        if list.is_null() {
            return Ok(out);
        }
        let mut i = 0;
        loop {
            let d = unsafe { *list.add(i) };
            if d.is_null() {
                break;
            }
            let d = unsafe { &*d };
            out.push(DeviceInfo {
                name: cstr(d.name),
                vendor: cstr(d.vendor),
                model: cstr(d.model),
                kind: cstr(d.kind),
            });
            i += 1;
        }
        Ok(out)
    }

    fn open(&mut self, name: &str) -> Result<Box<dyn Device>> {
        let c = CString::new(name).map_err(|e| ScanError::Other(e.to_string()))?;
        let mut h: SaneHandle = std::ptr::null_mut();
        let st = unsafe { (self.api.open)(c.as_ptr(), &mut h) };
        self.api.check(st, &format!("opening {name}"))?;
        let info = self
            .list_devices()
            .ok()
            .and_then(|l| l.into_iter().find(|d| d.name == name))
            .unwrap_or_else(|| DeviceInfo {
                name: name.into(),
                vendor: String::new(),
                model: name.into(),
                kind: String::new(),
            });
        let mut dev = SaneDevice {
            api: self.api.clone(),
            handle: h,
            info,
            opts: Vec::new(),
        };
        dev.reload()?;
        Ok(Box::new(dev))
    }
}

pub struct SaneDevice {
    api: Arc<Api>,
    handle: SaneHandle,
    info: DeviceInfo,
    opts: Vec<(OptionDesc, c_int /*type*/, c_int /*size*/)>,
}

impl Drop for SaneDevice {
    fn drop(&mut self) {
        unsafe { (self.api.close)(self.handle) };
    }
}

impl SaneDevice {
    fn reload(&mut self) -> Result<()> {
        self.opts.clear();
        // Option 0 holds the number of options.
        let mut count: SaneWord = 0;
        let st = unsafe {
            (self.api.control_option)(
                self.handle,
                0,
                ACTION_GET,
                &mut count as *mut _ as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        self.api.check(st, "reading option count")?;
        for i in 1..count.max(1) {
            let d = unsafe { (self.api.get_option_descriptor)(self.handle, i) };
            if d.is_null() {
                continue;
            }
            let d = unsafe { &*d };
            let kind = match d.type_ {
                TYPE_BOOL => OptKind::Bool,
                TYPE_INT => OptKind::Int,
                TYPE_FIXED => OptKind::Fixed,
                TYPE_STRING => OptKind::Str,
                TYPE_BUTTON => OptKind::Button,
                _ => OptKind::Group,
            };
            let unit = match d.unit {
                1 => Unit::Pixel,
                2 => Unit::Bit,
                3 => Unit::Mm,
                4 => Unit::Dpi,
                5 => Unit::Percent,
                6 => Unit::Microsecond,
                _ => Unit::None,
            };
            let conv = |w: SaneWord| {
                if d.type_ == TYPE_FIXED {
                    fixed_to_f64(w)
                } else {
                    w as f64
                }
            };
            let constraint = unsafe {
                match d.constraint_type {
                    1 if !d.constraint.is_null() => {
                        let r = &*(d.constraint as *const RawRange);
                        Constraint::Range {
                            min: conv(r.min),
                            max: conv(r.max),
                            quant: conv(r.quant),
                        }
                    }
                    2 if !d.constraint.is_null() => {
                        let p = d.constraint as *const SaneWord;
                        let n = *p as usize;
                        Constraint::List((1..=n).map(|k| conv(*p.add(k))).collect())
                    }
                    3 if !d.constraint.is_null() => {
                        let p = d.constraint as *const *const c_char;
                        let mut v = Vec::new();
                        let mut k = 0;
                        while !(*p.add(k)).is_null() {
                            v.push(cstr(*p.add(k)));
                            k += 1;
                        }
                        Constraint::Strings(v)
                    }
                    _ => Constraint::None,
                }
            };
            self.opts.push((
                OptionDesc {
                    index: i as usize,
                    name: cstr(d.name),
                    title: cstr(d.title),
                    desc: cstr(d.desc),
                    kind,
                    unit,
                    constraint,
                    active: d.cap & CAP_INACTIVE == 0,
                    settable: d.cap & CAP_SOFT_SELECT != 0,
                    advanced: d.cap & CAP_ADVANCED != 0,
                },
                d.type_,
                d.size,
            ));
        }
        Ok(())
    }

    fn find(&self, name: &str) -> Option<&(OptionDesc, c_int, c_int)> {
        self.opts.iter().find(|(o, _, _)| o.name == name)
    }

    fn has_active(&self, name: &str) -> bool {
        self.find(name).is_some_and(|o| o.0.active && o.0.settable)
    }

    fn opt_unit(&self, name: &str) -> Option<Unit> {
        self.find(name).map(|o| o.0.unit)
    }

    /// Set an option if present & settable, ignoring failures (logged).
    fn try_set(&mut self, name: &str, v: OptValue) {
        if self.has_active(name)
            && let Err(e) = self.set_option(name, v.clone())
        {
            log::warn!("could not set {name} = {v:?}: {e}");
        }
    }

    fn mode_string(&self, mode: ColorMode) -> Option<String> {
        let o = self.find("mode")?;
        let Constraint::Strings(list) = &o.0.constraint else {
            return None;
        };
        let want: &[&str] = match mode {
            ColorMode::Color => &["color", "colour", "24bit color", "rgb"],
            ColorMode::Gray => &["gray", "grey", "grayscale", "8bit gray"],
            ColorMode::Lineart => &["lineart", "binary", "black & white", "halftone"],
        };
        list.iter()
            .find(|s| want.iter().any(|w| s.eq_ignore_ascii_case(w)))
            .or_else(|| {
                list.iter()
                    .find(|s| want.iter().any(|w| s.to_ascii_lowercase().starts_with(w)))
            })
            .cloned()
    }
}

impl Device for SaneDevice {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn options(&mut self) -> Result<Vec<OptionDesc>> {
        Ok(self.opts.iter().map(|o| o.0.clone()).collect())
    }

    fn get_option(&mut self, name: &str) -> Result<OptValue> {
        let (desc, ty, size) = self
            .find(name)
            .cloned()
            .ok_or_else(|| ScanError::Other(format!("no option {name}")))?;
        let mut buf = vec![0u8; (size.max(4) as usize) + 1];
        let st = unsafe {
            (self.api.control_option)(
                self.handle,
                desc.index as c_int,
                ACTION_GET,
                buf.as_mut_ptr() as *mut c_void,
                std::ptr::null_mut(),
            )
        };
        self.api.check(st, &format!("reading {name}"))?;
        let word = SaneWord::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]);
        Ok(match ty {
            TYPE_BOOL => OptValue::Bool(word != 0),
            TYPE_INT => OptValue::Int(word),
            TYPE_FIXED => OptValue::Fixed(fixed_to_f64(word)),
            TYPE_STRING => OptValue::Str(
                CStr::from_bytes_until_nul(&buf)
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ),
            _ => OptValue::Button,
        })
    }

    fn set_option(&mut self, name: &str, value: OptValue) -> Result<()> {
        let (desc, ty, size) = self
            .find(name)
            .cloned()
            .ok_or_else(|| ScanError::Other(format!("no option {name}")))?;
        let mut buf = vec![0u8; (size.max(4) as usize) + 1];
        let num = value.as_f64();
        match (ty, &value) {
            (TYPE_STRING, OptValue::Str(s)) => {
                let b = s.as_bytes();
                let n = b.len().min(buf.len() - 1);
                buf[..n].copy_from_slice(&b[..n]);
            }
            (TYPE_FIXED, _) => {
                buf[..4].copy_from_slice(&f64_to_fixed(num.unwrap_or(0.0)).to_ne_bytes())
            }
            (TYPE_INT | TYPE_BOOL, _) => {
                buf[..4].copy_from_slice(&(num.unwrap_or(0.0).round() as SaneWord).to_ne_bytes())
            }
            (TYPE_BUTTON, _) => {}
            _ => return Err(ScanError::Other(format!("type mismatch for {name}"))),
        }
        let mut info: c_int = 0;
        let ptr = if ty == TYPE_BUTTON {
            std::ptr::null_mut()
        } else {
            buf.as_mut_ptr() as *mut c_void
        };
        let st = unsafe {
            (self.api.control_option)(self.handle, desc.index as c_int, ACTION_SET, ptr, &mut info)
        };
        self.api.check(st, &format!("setting {name}"))?;
        if info & 2 != 0 {
            self.reload()?;
        }
        Ok(())
    }

    fn capabilities(&mut self) -> Result<Capabilities> {
        let mut caps = Capabilities::default();
        if let Some((o, _, _)) = self.find("resolution") {
            caps.resolutions = match &o.constraint {
                Constraint::List(v) => v.clone(),
                Constraint::Range { min, max, .. } => [
                    75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 1200.0, 2400.0, 4800.0,
                ]
                .into_iter()
                .filter(|r| r >= min && r <= max)
                .collect(),
                _ => vec![],
            };
        }
        for m in [ColorMode::Color, ColorMode::Gray, ColorMode::Lineart] {
            if self.mode_string(m).is_some() {
                caps.modes.push(m);
            }
        }
        if let Some((o, _, _)) = self.find("depth")
            && let Constraint::List(v) = &o.constraint
        {
            caps.depths = v.iter().map(|d| *d as u8).collect();
        }
        if caps.depths.is_empty() {
            caps.depths = vec![8];
        }
        if let Some((o, _, _)) = self.find("source")
            && let Constraint::Strings(v) = &o.constraint
        {
            caps.sources = v.clone();
        }
        let range_max = |n: &str| match self.find(n).map(|o| &o.0.constraint) {
            Some(Constraint::Range { max, .. }) => Some(*max),
            Some(Constraint::List(v)) => v.iter().cloned().reduce(f64::max),
            _ => None,
        };
        if self.opt_unit("br-x") == Some(Unit::Mm)
            && let (Some(w), Some(h)) = (range_max("br-x"), range_max("br-y"))
        {
            caps.bed_mm = Some((w, h));
        }
        Ok(caps)
    }

    fn scan(
        &mut self,
        s: &ScanSettings,
        progress: &mut ProgressFn<'_>,
        cancel: &AtomicBool,
    ) -> Result<ScanResult> {
        // Order matters: source and mode can change the valid resolutions / area.
        if let Some(src) = &s.source {
            self.try_set("source", OptValue::Str(src.clone()));
        }
        if let Some(m) = self.mode_string(s.mode) {
            self.try_set("mode", OptValue::Str(m));
        }
        if let Some(d) = s.depth {
            self.try_set("depth", OptValue::Int(d as i32));
        }
        let caps = self.capabilities()?;
        let res = if caps.resolutions.is_empty() {
            s.resolution
        } else {
            caps.nearest_resolution(s.resolution)
        };
        self.try_set("resolution", OptValue::Fixed(res));
        // Some backends use x-/y-resolution instead.
        self.try_set("x-resolution", OptValue::Fixed(res));
        self.try_set("y-resolution", OptValue::Fixed(res));
        let actual_res = self
            .get_option("resolution")
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(res);

        let bed = caps.bed_mm;
        let (x, y, w, h) =
            s.area_mm
                .unwrap_or((0.0, 0.0, bed.map_or(1e4, |b| b.0), bed.map_or(1e4, |b| b.1)));
        let to_unit = |mm: f64, n: &str| -> f64 {
            match self.opt_unit(n) {
                Some(Unit::Pixel) => mm / 25.4 * actual_res,
                _ => mm,
            }
        };
        let (tlx, tly, brx, bry) = (
            to_unit(x, "tl-x"),
            to_unit(y, "tl-y"),
            to_unit(x + w, "br-x"),
            to_unit(y + h, "br-y"),
        );
        let clamp = |n: &str, v: f64| match self.find(n).map(|o| &o.0.constraint) {
            Some(Constraint::Range { min, max, .. }) => v.clamp(*min, *max),
            _ => v,
        };
        let (brx, bry) = (clamp("br-x", brx), clamp("br-y", bry));
        let (tlx, tly) = (clamp("tl-x", tlx), clamp("tl-y", tly));
        // Set bottom-right first so top-left values are never beyond it.
        self.try_set("br-x", OptValue::Fixed(brx));
        self.try_set("br-y", OptValue::Fixed(bry));
        self.try_set("tl-x", OptValue::Fixed(tlx));
        self.try_set("tl-y", OptValue::Fixed(tly));
        let tl_x = self.get_option("tl-x").ok().and_then(|v| v.as_f64());
        let tl_y = self.get_option("tl-y").ok().and_then(|v| v.as_f64());
        let unit_mm = |v: f64, n: &str| {
            if self.opt_unit(n) == Some(Unit::Pixel) {
                v / actual_res * 25.4
            } else {
                v
            }
        };
        let origin_mm = (
            tl_x.map(|v| unit_mm(v, "tl-x")).unwrap_or(0.0),
            tl_y.map(|v| unit_mm(v, "tl-y")).unwrap_or(0.0),
        );

        // Scan, handling single-pass and three-pass colour.
        let api = self.api.clone();
        let h = self.handle;
        let mut planes: [Option<Vec<u8>>; 3] = [None, None, None];
        let mut gray_or_rgb: Option<(Vec<u8>, RawParameters)> = None;
        let mut final_params;
        loop {
            let st = unsafe { (api.start)(h) };
            api.check(st, "starting scan")?;
            let mut p = RawParameters::default();
            let st = unsafe { (api.get_parameters)(h, &mut p) };
            api.check(st, "reading parameters")?;
            let channels: u8 = if p.format == FRAME_RGB { 3 } else { 1 };
            let total = if p.lines > 0 {
                Some(p.lines as u32)
            } else {
                None
            };
            let bpl = p.bytes_per_line.max(1) as usize;
            let mut data: Vec<u8> = Vec::with_capacity(bpl * total.unwrap_or(1000) as usize);
            let mut buf = vec![0u8; 256 * 1024];
            let mut last_report = 0usize;
            loop {
                if cancel.load(Ordering::Relaxed) {
                    unsafe { (api.cancel)(h) };
                    return Err(ScanError::Cancelled);
                }
                let mut n: c_int = 0;
                let st = unsafe { (api.read)(h, buf.as_mut_ptr(), buf.len() as c_int, &mut n) };
                if st == STATUS_EOF {
                    break;
                }
                if let Err(e) = api.check(st, "reading scan data") {
                    unsafe { (api.cancel)(h) };
                    return Err(e);
                }
                data.extend_from_slice(&buf[..n.max(0) as usize]);
                let lines = data.len() / bpl;
                if data.len() - last_report >= bpl * 16 || lines as i32 == p.lines {
                    last_report = data.len();
                    let frac = total.map_or(-1.0, |t| lines as f32 / t as f32);
                    let show = p.depth != 1 && (p.format == FRAME_GRAY || p.format == FRAME_RGB);
                    progress(Progress {
                        fraction: frac,
                        frame: show.then(|| FrameView {
                            width: p.pixels_per_line as u32,
                            lines: lines as u32,
                            total_lines: total,
                            channels,
                            depth: p.depth as u8,
                            data: &data[..lines * bpl],
                        }),
                    });
                }
            }
            // Strip any row padding.
            let row_bytes =
                (p.pixels_per_line as usize * channels as usize * p.depth as usize).div_ceil(8);
            let lines = data.len() / bpl;
            let mut packed = Vec::with_capacity(row_bytes * lines);
            for l in 0..lines {
                packed.extend_from_slice(&data[l * bpl..l * bpl + row_bytes.min(bpl)]);
            }
            final_params = RawParameters {
                lines: lines as c_int,
                ..p
            };
            match p.format {
                FRAME_RED => planes[0] = Some(packed),
                FRAME_GREEN => planes[1] = Some(packed),
                FRAME_BLUE => planes[2] = Some(packed),
                _ => {
                    gray_or_rgb = Some((
                        packed,
                        RawParameters {
                            lines: lines as c_int,
                            ..p
                        },
                    ))
                }
            }
            if p.last_frame != 0 {
                break;
            }
        }
        unsafe { (api.cancel)(h) }; // required to finish the scan cycle
        let image = if let Some((data, p)) = gray_or_rgb {
            let ch = if p.format == FRAME_RGB { 3 } else { 1 };
            raw_to_image(
                p.pixels_per_line as u32,
                p.lines as u32,
                ch,
                p.depth as u8,
                &data,
            )?
        } else if let [Some(r), Some(g), Some(b)] = &planes {
            let p = &final_params;
            let bps = (p.depth as usize).div_ceil(8);
            let n = r.len().min(g.len()).min(b.len()) / bps;
            let mut inter = Vec::with_capacity(n * 3 * bps);
            for i in 0..n {
                for plane in [r, g, b] {
                    inter.extend_from_slice(&plane[i * bps..(i + 1) * bps]);
                }
            }
            raw_to_image(
                p.pixels_per_line as u32,
                p.lines as u32,
                3,
                p.depth as u8,
                &inter,
            )?
        } else {
            return Err(ScanError::Sane("scanner returned no image data".into()));
        };
        Ok(ScanResult {
            image,
            dpi: actual_res,
            origin_mm,
        })
    }
}
