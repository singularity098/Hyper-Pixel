//! Multi-photo detection on a flatbed scan.
//!
//! Pipeline:
//!
//! 1. At a low "working" resolution (default 100 DPI): convert to CIE Lab and lightly blur.
//! 2. Pick the background colour: of the colour modes in a ring just inside the scan border,
//!    the one that turns up most widely around the perimeter. Photos that run off the bed can
//!    cover much of the border, so a plain median is not safe. Then refine it into a smooth
//!    background *surface*, which handles lid vignetting and uneven lighting.
//! 3. Foreground = colour distance from background above a noise-adaptive threshold, OR
//!    strong luminance gradient (catches the faint outline of light prints on a white lid).
//!    A thin band along the scan border is ignored (scanner frame, lid-edge shadow).
//! 4. Morphological close → fill holes → open, then connected components.
//! 5. Per component: look for a straight *seam* that cuts it into two photos: a run of
//!    background, a thin ridge towards the background colour (the white paper edge of a
//!    print), or a long straight edge. This uses a finer resolution (default 300 DPI), where
//!    gaps of a few tenths of a millimetre show. The cut is re-centred on the gap, and each
//!    half remembers the seam so its edges cannot be placed on the neighbour. Very
//!    non-rectangular blobs are also tried with erosion-based splitting.
//! 6. Sub-pixel edge refinement at the fine resolution, in a wide pass and then a narrow one:
//!    sample many profiles across each side of the rectangle, take the strongest rise of
//!    background distance that has background just outside it (so a touching neighbour's
//!    edge is not taken), robustly fit a line per side and solve for the best rectangle.
//!
//! Results are reported in millimetres in scanner-bed coordinates so they can be mapped
//! between a low-resolution preview and the high-resolution final scan.
//!
//! Set `HP_DETECT_DEBUG=<dir>` to write the intermediate masks and seam evidence as PNGs.

use crate::geom::{Point, RotatedRect, fit_line_robust, min_area_rect};
use crate::raster::{Lab, Mask, Plane, lab_planes, lab_to_rgb, percentile, rgb_to_lab};
use image::DynamicImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BackgroundHint {
    /// Estimate from the scan border.
    #[default]
    Auto,
    /// Dark backing sheet (recommended).
    Dark,
    /// Scanner lid / white sheet.
    Light,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DetectParams {
    /// Resolution detection runs at. Higher is slower but slightly more precise.
    pub work_dpi: f64,
    /// Resolution for seam finding and edge refinement (capped at the scan resolution).
    pub fine_dpi: f64,
    pub background: BackgroundHint,
    /// Objects whose shorter side is below this are ignored (mm).
    pub min_size_mm: f64,
    /// 1.0 = default. Higher finds fainter photos but may pick up more junk.
    pub sensitivity: f64,
    /// Try to separate photos that touch each other.
    pub split_touching: bool,
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            work_dpi: 100.0,
            fine_dpi: 300.0,
            background: BackgroundHint::Auto,
            min_size_mm: 20.0,
            sensitivity: 1.0,
            split_touching: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// Photo rectangle in bed millimetres.
    pub rect: RotatedRect,
    /// 0..1 rough confidence.
    pub confidence: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectReport {
    pub photos: Vec<Detection>,
    /// Estimated background lightness (L*, 0..100).
    pub background_l: f64,
    /// True if the background looked dark.
    pub dark_background: bool,
}

/// Geometry of a scan relative to the scanner bed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScanGeometry {
    pub dpi: f64,
    /// Bed position of the image's top-left pixel corner, mm.
    pub origin_mm: (f64, f64),
}

impl ScanGeometry {
    pub fn new(dpi: f64) -> Self {
        Self {
            dpi,
            origin_mm: (0.0, 0.0),
        }
    }
    pub fn px_per_mm(&self) -> f64 {
        self.dpi / 25.4
    }
    /// Convert a rect in this image's pixel coordinates to bed mm.
    pub fn px_to_mm(&self, r: &RotatedRect) -> RotatedRect {
        r.scaled(1.0 / self.px_per_mm())
            .translated(self.origin_mm.0, self.origin_mm.1)
    }
    /// Convert a rect in bed mm to this image's pixel coordinates.
    pub fn mm_to_px(&self, r: &RotatedRect) -> RotatedRect {
        r.translated(-self.origin_mm.0, -self.origin_mm.1)
            .scaled(self.px_per_mm())
    }
}

/// Background distance and lightness at the fine resolution.
struct Fine {
    dist: Plane,
    l: Plane,
    /// Fine pixels per working pixel.
    k: f64,
}

struct Ctx<'a> {
    w: usize,
    h: usize,
    /// Background-distance threshold at the fine resolution (texture such as foam grain is
    /// stronger there).
    fine_threshold: f32,
    /// Working pixels per mm.
    px_per_mm: f64,
    min_side_px: f64,
    split: bool,
    fine: &'a Fine,
}

/// Detect individual photos in a flatbed scan.
pub fn detect_photos(img: &DynamicImage, geo: ScanGeometry, params: &DetectParams) -> DetectReport {
    let scale = (params.work_dpi / geo.dpi).min(1.0);
    let work_dpi = geo.dpi * scale;
    let px_per_mm = work_dpi / 25.4;
    let lab = lab_planes(img, scale);
    let lab = Lab {
        l: lab.l.blur(0.7),
        a: lab.a.blur(0.7),
        b: lab.b.blur(0.7),
    };
    let (w, h) = (lab.l.w, lab.l.h);
    if w < 16 || h < 16 {
        return DetectReport::default();
    }
    let grad = lab.l.gradient();

    // --- Background colour and noise from the border ring ------------------------------
    let bg = estimate_background(&lab, params.background);
    let (bl, ba, bb) = (bg.l, bg.a, bg.b);
    let sens = params.sensitivity.clamp(0.25, 4.0) as f32;
    let threshold = ((bg.noise * 3.0).max(4.0)) / sens;
    let ring_g: Vec<f32> = bg.ring_bg.iter().map(|&i| grad.data[i]).collect();
    let gthr = (percentile(&ring_g, 0.98) * 2.5).max(1.2) / sens;

    // --- Pass 1: coarse mask with global background -----------------------------------
    let de = |i: usize, l: f32, a: f32, b: f32| {
        let dl = lab.l.data[i] - l;
        let da = lab.a.data[i] - a;
        let db = lab.b.data[i] - b;
        (dl * dl + da * da + db * db).sqrt()
    };
    let mut dist = Plane::new(w, h);
    for i in 0..w * h {
        dist.data[i] = de(i, bl, ba, bb);
    }
    let mp = MaskParams {
        threshold,
        grad_threshold: gthr,
        close_r: (CLOSE_MM * px_per_mm) as f32,
        open_r: (1.2 * px_per_mm).max(1.5) as f32,
        border: (BORDER_MM * px_per_mm).ceil() as usize,
    };
    let fg1 = build_mask(&dist, &grad, &mp);

    // --- Background surface from pixels well away from the foreground -----------------
    let excl = fg1.dilate((3.0 * px_per_mm) as f32);
    let surface = background_surface(
        &lab,
        &excl,
        (bl, ba, bb),
        (12.0 * px_per_mm).max(8.0) as usize,
    );
    for i in 0..w * h {
        let dl = lab.l.data[i] - surface.l.data[i];
        let da = lab.a.data[i] - surface.a.data[i];
        let db = lab.b.data[i] - surface.b.data[i];
        dist.data[i] = (dl * dl + da * da + db * db).sqrt();
    }
    let fg = build_mask(&dist, &grad, &mp);
    debug_mask("fg1", &fg1);
    debug_mask("fg", &fg);
    debug_plane("dist", &dist, 4.0 * threshold);
    log::debug!(
        "bg L {bl:.1} a {ba:.1} b {bb:.1}, noise {:.2}, thr {threshold:.2}, gthr {gthr:.2}",
        bg.noise
    );

    // --- Fine level for seams and edges --------------------------------------------------
    let fine_scale = (params.fine_dpi.max(work_dpi) / geo.dpi).min(1.0);
    let fine = build_fine(img, fine_scale, geo.px_per_mm() * fine_scale, &surface);

    let fine_noise: Vec<f32> = bg
        .ring_bg
        .iter()
        .map(|&i| {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            fine.dist.sample(x * fine.k - 0.5, y * fine.k - 0.5)
        })
        .collect();
    let fine_threshold = threshold.max(percentile(&fine_noise, 0.9) * 2.5 / sens);
    log::debug!("fine threshold {fine_threshold:.2}");

    // --- Components → rectangles --------------------------------------------------------
    let ctx = Ctx {
        w,
        h,
        fine_threshold,
        px_per_mm,
        min_side_px: params.min_size_mm * px_per_mm,
        split: params.split_touching,
        fine: &fine,
    };
    let (_, comps) = fg.components();
    let mut found: Vec<(RotatedRect, f64)> = Vec::new();
    for comp in comps {
        if (comp.len() as f64) < ctx.min_side_px * ctx.min_side_px * 0.5 {
            continue;
        }
        process_component(&ctx, &comp, 0, &[], &mut found);
    }

    // Remove detections largely contained in a bigger one.
    found.sort_by(|a, b| b.0.area().total_cmp(&a.0.area()));
    let mut kept: Vec<(RotatedRect, f64)> = Vec::new();
    for (r, c) in found {
        let dup = kept.iter().any(|(k, _)| {
            let inter =
                crate::geom::polygon_area(&crate::geom::clip_convex(&r.corners(), &k.corners()));
            inter > 0.6 * r.area()
        });
        if !dup {
            kept.push((r, c));
        }
    }
    // Reading order: top-to-bottom rows, then left-to-right.
    let row_tol = 15.0 * px_per_mm;
    kept.sort_by(|a, b| {
        if (a.0.cy - b.0.cy).abs() < row_tol {
            a.0.cx.total_cmp(&b.0.cx)
        } else {
            a.0.cy.total_cmp(&b.0.cy)
        }
    });

    let work_geo = ScanGeometry {
        dpi: work_dpi,
        origin_mm: geo.origin_mm,
    };
    // The working image may be very slightly rescaled due to rounding; correct for it.
    let sx = img.width() as f64 * scale / w as f64;
    let photos = kept
        .into_iter()
        .map(|(r, c)| Detection {
            rect: work_geo.px_to_mm(&r.scaled(sx)),
            confidence: c,
        })
        .collect();
    DetectReport {
        photos,
        background_l: bl as f64,
        dark_background: bl < 50.0,
    }
}

/// Width of the band along the scan border that is ignored (see `replicate_border`).
const BORDER_MM: f64 = 1.2;
/// The same for edge refinement, where photo edges very close to the border must survive.
const FINE_BORDER_MM: f64 = 0.8;
/// Gaussian sigma (fine pixels) applied to the fine background distance when the fine level
/// is 3× the working resolution or more (scaled down below that).
const FINE_SMOOTH_PX: f32 = 1.0;
/// Edge search range of the first refinement pass, either side of the coarse edge.
const FIRST_SEARCH_MM: f64 = 3.0;
/// See `robust_extents`.
const EXTENT_FRACTION: f64 = 0.5;
/// Closing radius for the coarse mask.
const CLOSE_MM: f64 = 0.8;

struct Background {
    l: f32,
    a: f32,
    b: f32,
    noise: f32,
    /// Ring pixels that clearly match the background.
    ring_bg: Vec<usize>,
}

/// Estimate the background colour from a ring just inside the border.
///
/// Candidates are the main colour modes of the ring. The winner is the one that shows up in
/// the most places around the perimeter: a photo running off the bed covers one stretch of
/// the border, while the lid or backing sheet shows between photos all the way round.
fn estimate_background(lab: &Lab, hint: BackgroundHint) -> Background {
    const SEGS: usize = 32;
    let (w, h) = (lab.l.w, lab.l.h);
    let m = w.min(h) as f64;
    let i0 = (m * 0.01).max(1.0) as usize;
    let i1 = i0 + (m * 0.03).max(3.0) as usize;
    let per = 2 * (w + h);
    let mut ring: Vec<usize> = Vec::new();
    let mut seg: Vec<u8> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let e = x.min(y).min(w - 1 - x).min(h - 1 - y);
            if e < i0 || e >= i1 {
                continue;
            }
            // Position along the perimeter, measured at the nearest edge.
            let pos = if e == y {
                x
            } else if e == w - 1 - x {
                w + y
            } else if e == h - 1 - y {
                w + h + (w - 1 - x)
            } else {
                2 * w + h + (h - 1 - y)
            };
            ring.push(y * w + x);
            seg.push((pos * SEGS / per).min(SEGS - 1) as u8);
        }
    }
    let px = |i: usize| (lab.l.data[i], lab.a.data[i], lab.b.data[i]);
    let de = |p: (f32, f32, f32), q: (f32, f32, f32)| {
        ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2) + (p.2 - q.2).powi(2)).sqrt()
    };
    // Mode finding happens in gamma-encoded RGB (0..255), where scanner noise is roughly
    // the same from black to white; in Lab, chroma noise blows up in the darks.
    let rgb: Vec<[f32; 3]> = ring
        .iter()
        .map(|&i| {
            let (r, g, b) = lab_to_rgb(lab.l.data[i], lab.a.data[i], lab.b.data[i]);
            [r * 255.0, g * 255.0, b * 255.0]
        })
        .collect();
    let drgb = |p: &[f32; 3], q: &[f32; 3]| {
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    const RADIUS: f32 = 20.0;

    // Colour histogram of the ring (8-unit bins), smoothed over neighbouring bins.
    let q = |v: f32| (v / 8.0).round() as i32;
    let mut hist: std::collections::HashMap<(i32, i32, i32), u32> = Default::default();
    for c in &rgb {
        *hist.entry((q(c[0]), q(c[1]), q(c[2]))).or_default() += 1;
    }
    let mut smooth: Vec<((i32, i32, i32), u32)> = hist
        .keys()
        .map(|&(r, g, b)| {
            let mut s = 0;
            for dr in -1..=1 {
                for dg in -1..=1 {
                    for db in -1..=1 {
                        s += hist.get(&(r + dr, g + dg, b + db)).copied().unwrap_or(0);
                    }
                }
            }
            ((r, g, b), s)
        })
        .collect();
    smooth.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    let lightness = |c: &[f32; 3]| rgb_to_lab(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0).0;
    let allowed = |c: &[f32; 3]| match hint {
        BackgroundHint::Auto => true,
        BackgroundHint::Dark => lightness(c) < 50.0,
        BackgroundHint::Light => lightness(c) >= 50.0,
    };
    let mut candidates: Vec<[f32; 3]> = Vec::new();
    for &((r, g, b), _) in &smooth {
        if candidates.len() >= 8 {
            break;
        }
        let c = [r as f32 * 8.0, g as f32 * 8.0, b as f32 * 8.0];
        if !allowed(&c) || candidates.iter().any(|o| drgb(o, &c) < RADIUS) {
            continue;
        }
        candidates.push(c);
    }

    let mut seg_total = [0u32; SEGS];
    for &s in &seg {
        seg_total[s as usize] += 1;
    }
    let median_of = |idx: &[usize], ch: usize| {
        percentile(&idx.iter().map(|&k| rgb[k][ch]).collect::<Vec<_>>(), 0.5)
    };
    let mut best: Option<(f64, Vec<usize>)> = None;
    for mut c in candidates {
        // A few mean-shift steps (with medians) to centre on the mode.
        let mut members: Vec<usize> = Vec::new();
        for _ in 0..3 {
            members = (0..ring.len())
                .filter(|&k| drgb(&rgb[k], &c) < RADIUS)
                .collect();
            if members.is_empty() {
                break;
            }
            c = [
                median_of(&members, 0),
                median_of(&members, 1),
                median_of(&members, 2),
            ];
        }
        if members.is_empty() {
            continue;
        }
        let mut seg_hits = [0u32; SEGS];
        for &k in &members {
            seg_hits[seg[k] as usize] += 1;
        }
        // Saturating per-segment coverage: wide presence beats a big solid patch.
        let score: f64 = (0..SEGS)
            .filter(|&s| seg_total[s] > 0)
            .map(|s| (seg_hits[s] as f64 / seg_total[s] as f64).min(0.4))
            .sum();
        log::trace!("background candidate {c:?}: score {score:.2}");
        if best.as_ref().is_none_or(|(b, _)| score > *b) {
            best = Some((score, members));
        }
    }
    let (bl, ba, bb, mut ring_bg) = match best {
        Some((_, members)) => {
            let idx: Vec<usize> = members.iter().map(|&k| ring[k]).collect();
            let med =
                |p: &Plane| percentile(&idx.iter().map(|&i| p.data[i]).collect::<Vec<_>>(), 0.5);
            (med(&lab.l), med(&lab.a), med(&lab.b), idx)
        }
        None => {
            // No ring colour fits the hint: fall back to the darkest/lightest common colour.
            let l = percentile(
                &lab.l.data,
                if hint == BackgroundHint::Dark {
                    0.05
                } else {
                    0.95
                },
            );
            (l, 0.0, 0.0, ring.clone())
        }
    };
    if ring_bg.len() < 20 {
        ring_bg = ring.clone();
    }
    // Noise: spread of the clearly-background ring pixels. Light photo areas near the
    // background colour would inflate a plain percentile, so cut off at a multiple of the
    // median first.
    let near: Vec<f32> = ring_bg.iter().map(|&i| de(px(i), (bl, ba, bb))).collect();
    let cut = (percentile(&near, 0.5) * 2.5).max(3.0);
    let core: Vec<usize> = ring_bg
        .iter()
        .zip(&near)
        .filter(|(_, d)| **d < cut)
        .map(|(i, _)| *i)
        .collect();
    if core.len() >= 20 {
        ring_bg = core;
    }
    let noise = percentile(
        &ring_bg
            .iter()
            .map(|&i| de(px(i), (bl, ba, bb)))
            .collect::<Vec<_>>(),
        0.9,
    )
    .max(0.8);
    // The tightest background pixels (within the noise level) for texture statistics.
    let tight: Vec<usize> = ring_bg
        .iter()
        .cloned()
        .filter(|&i| de(px(i), (bl, ba, bb)) <= noise)
        .collect();
    if tight.len() >= 20 {
        ring_bg = tight;
    }
    Background {
        l: bl,
        a: ba,
        b: bb,
        noise,
        ring_bg,
    }
}

/// Lightness and background distance at the fine resolution (`px_per_mm` there). The
/// background surface comes from the working resolution and is interpolated.
fn build_fine(img: &DynamicImage, scale: f64, px_per_mm: f64, surface: &Lab) -> Fine {
    let lab = lab_planes(img, scale);
    let (fw, fh) = (lab.l.w, lab.l.h);
    let k = fw as f64 / surface.l.w as f64;
    let ky = fh as f64 / surface.l.h as f64;
    let mut dist = Plane::new(fw, fh);
    dist.data
        .par_chunks_mut(fw)
        .enumerate()
        .for_each(|(y, row)| {
            let sy = (y as f64 + 0.5) / ky - 0.5;
            for (x, o) in row.iter_mut().enumerate() {
                let sx = (x as f64 + 0.5) / k - 0.5;
                let i = y * fw + x;
                let dl = lab.l.data[i] - surface.l.sample(sx, sy);
                let da = lab.a.data[i] - surface.a.sample(sx, sy);
                let db = lab.b.data[i] - surface.b.sample(sx, sy);
                *o = (dl * dl + da * da + db * db).sqrt();
            }
        });
    let border = (FINE_BORDER_MM * px_per_mm).ceil() as usize;
    replicate_border(&mut dist.data, fw, fh, border);
    // Light smoothing against grain (foam backing sheets are rough at 300 DPI) when the fine
    // level is much sharper than the working one; thin gaps stay a couple of pixels wide.
    let sigma = (FINE_SMOOTH_PX * (k as f32 - 1.0) / 2.0).clamp(0.0, FINE_SMOOTH_PX);
    Fine {
        dist: dist.blur(sigma),
        l: lab.l,
        k,
    }
}

/// Write an intermediate mask to `$HP_DETECT_DEBUG/<name>.png` when that variable is set.
fn debug_mask(name: &str, m: &Mask) {
    debug_image(name, m.w, m.h, |i| if m.data[i] { 255 } else { 0 });
}

fn debug_plane(name: &str, p: &Plane, max: f32) {
    debug_image(name, p.w, p.h, |i| {
        (p.data[i] / max * 255.0).clamp(0.0, 255.0) as u8
    });
}

fn debug_image(name: &str, w: usize, h: usize, f: impl Fn(usize) -> u8) {
    let Some(dir) = std::env::var_os("HP_DETECT_DEBUG") else {
        return;
    };
    let buf: Vec<u8> = (0..w * h).map(f).collect();
    if let Some(img) = image::GrayImage::from_raw(w as u32, h as u32, buf) {
        let _ = img.save(std::path::Path::new(&dir).join(format!("{name}.png")));
    }
}

struct MaskParams {
    threshold: f32,
    grad_threshold: f32,
    close_r: f32,
    open_r: f32,
    border: usize,
}

fn build_mask(dist: &Plane, grad: &Plane, p: &MaskParams) -> Mask {
    let (w, h) = (dist.w, dist.h);
    let mut m = Mask::new(w, h);
    for i in 0..w * h {
        m.data[i] = dist.data[i] > p.threshold || grad.data[i] > p.grad_threshold;
    }
    replicate_border(&mut m.data, w, h, p.border);
    m.close(p.close_r).fill_holes().open(p.open_r)
}

/// Overwrite the outermost `b` pixels with the nearest pixel just inside that band.
///
/// The edges of a full-bed scan often show a strip of the scanner frame or the shadow of the
/// lid edge. Copying the inside outwards hides it, while photos that run off the bed still
/// reach the border.
fn replicate_border<T: Copy>(data: &mut [T], w: usize, h: usize, b: usize) {
    if b == 0 || w <= 2 * b || h <= 2 * b {
        return;
    }
    for y in 0..h {
        let sy = y.clamp(b, h - 1 - b);
        for x in 0..w {
            if x >= b && y >= b && x < w - b && y < h - b {
                continue;
            }
            let sx = x.clamp(b, w - 1 - b);
            data[y * w + x] = data[sy * w + sx];
        }
    }
}

/// Smooth background estimate: average Lab of non-excluded pixels in coarse cells,
/// holes filled by diffusion, bilinearly interpolated back to full size.
fn background_surface(lab: &Lab, excl: &Mask, fallback: (f32, f32, f32), cell: usize) -> Lab {
    let (w, h) = (lab.l.w, lab.l.h);
    let gw = w.div_ceil(cell);
    let gh = h.div_ceil(cell);
    let mut sum = vec![[0.0f64; 3]; gw * gh];
    let mut cnt = vec![0usize; gw * gh];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if excl.data[i] {
                continue;
            }
            let g = (y / cell) * gw + x / cell;
            sum[g][0] += lab.l.data[i] as f64;
            sum[g][1] += lab.a.data[i] as f64;
            sum[g][2] += lab.b.data[i] as f64;
            cnt[g] += 1;
        }
    }
    let min_cnt = (cell * cell / 4).max(1);
    let mut val: Vec<Option<[f32; 3]>> = (0..gw * gh)
        .map(|g| {
            (cnt[g] >= min_cnt).then(|| {
                [
                    (sum[g][0] / cnt[g] as f64) as f32,
                    (sum[g][1] / cnt[g] as f64) as f32,
                    (sum[g][2] / cnt[g] as f64) as f32,
                ]
            })
        })
        .collect();
    if val.iter().all(|v| v.is_none()) {
        val.iter_mut()
            .for_each(|v| *v = Some([fallback.0, fallback.1, fallback.2]));
    }
    // Diffuse known values into unknown cells.
    loop {
        let mut changed = false;
        let snapshot = val.clone();
        for gy in 0..gh {
            for gx in 0..gw {
                let g = gy * gw + gx;
                if snapshot[g].is_some() {
                    continue;
                }
                let mut acc = [0.0f32; 3];
                let mut n = 0;
                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (gx as isize + dx, gy as isize + dy);
                    if nx < 0 || ny < 0 || nx >= gw as isize || ny >= gh as isize {
                        continue;
                    }
                    if let Some(v) = snapshot[ny as usize * gw + nx as usize] {
                        for k in 0..3 {
                            acc[k] += v[k];
                        }
                        n += 1;
                    }
                }
                if n > 0 {
                    val[g] = Some([acc[0] / n as f32, acc[1] / n as f32, acc[2] / n as f32]);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let grid: Vec<[f32; 3]> = val
        .into_iter()
        .map(|v| v.unwrap_or([fallback.0, fallback.1, fallback.2]))
        .collect();
    let mut out = Lab {
        l: Plane::new(w, h),
        a: Plane::new(w, h),
        b: Plane::new(w, h),
    };
    for y in 0..h {
        let fy = ((y as f64 + 0.5) / cell as f64 - 0.5).clamp(0.0, (gh - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(gh - 1);
        let ty = (fy - y0 as f64) as f32;
        for x in 0..w {
            let fx = ((x as f64 + 0.5) / cell as f64 - 0.5).clamp(0.0, (gw - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let tx = (fx - x0 as f64) as f32;
            let i = y * w + x;
            let (g00, g10) = (grid[y0 * gw + x0], grid[y0 * gw + x1]);
            let (g01, g11) = (grid[y1 * gw + x0], grid[y1 * gw + x1]);
            let lerp = |k: usize| {
                let a = g00[k] * (1.0 - tx) + g10[k] * tx;
                let b = g01[k] * (1.0 - tx) + g11[k] * tx;
                a * (1.0 - ty) + b * ty
            };
            out.l.data[i] = lerp(0);
            out.a.data[i] = lerp(1);
            out.b.data[i] = lerp(2);
        }
    }
    out
}

fn boundary_points(ctx: &Ctx, comp: &[usize]) -> Vec<Point> {
    let mut inside = std::collections::HashSet::with_capacity(comp.len());
    inside.extend(comp.iter().copied());
    let (w, h) = (ctx.w, ctx.h);
    comp.iter()
        .filter(|&&i| {
            let (x, y) = (i % w, i / w);
            x == 0
                || y == 0
                || x + 1 == w
                || y + 1 == h
                || !inside.contains(&(i - 1))
                || !inside.contains(&(i + 1))
                || !inside.contains(&(i - w))
                || !inside.contains(&(i + w))
        })
        .flat_map(|&i| {
            // Use pixel corners so the hull encloses whole pixels.
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            [
                Point::new(x, y),
                Point::new(x + 1.0, y),
                Point::new(x, y + 1.0),
                Point::new(x + 1.0, y + 1.0),
            ]
        })
        .collect()
}

/// A seam a photo must stay on one side of: points with `(p - p0) · n > 0` are beyond it.
/// Working pixels.
#[derive(Clone, Copy)]
struct Limit {
    p0: Point,
    n: Point,
}

impl Limit {
    fn beyond(&self, p: Point) -> f64 {
        (p.x - self.p0.x) * self.n.x + (p.y - self.p0.y) * self.n.y
    }
}

fn process_component(
    ctx: &Ctx,
    comp: &[usize],
    depth: usize,
    limits: &[Limit],
    out: &mut Vec<(RotatedRect, f64)>,
) {
    let pts = boundary_points(ctx, comp);
    let Some(rect) = min_area_rect(&pts) else {
        return;
    };
    if rect.w.min(rect.h) < ctx.min_side_px {
        return;
    }
    let rectangularity = comp.len() as f64 / rect.area();

    if ctx.split && depth < 4 {
        if let Some((parts, seam)) = seam_split(ctx, comp, &rect, rectangularity) {
            // Part 0 lies on the negative side of the seam normal, part 1 on the positive.
            for (k, p) in parts.iter().enumerate() {
                let n = if k == 0 {
                    seam.n
                } else {
                    Point::new(-seam.n.x, -seam.n.y)
                };
                let mut lim = limits.to_vec();
                lim.push(Limit { p0: seam.p0, n });
                process_component(ctx, p, depth + 1, &lim, out);
            }
            return;
        }
        if rectangularity < 0.88
            && let Some(parts) = try_split(ctx, comp)
        {
            let part_rects: Vec<f64> = parts
                .iter()
                .filter_map(|p| {
                    let pr = min_area_rect(&boundary_points(ctx, p))?;
                    (pr.w.min(pr.h) >= ctx.min_side_px).then(|| p.len() as f64 / pr.area())
                })
                .collect();
            let big_parts = part_rects.len();
            let mean_rect = part_rects.iter().sum::<f64>() / big_parts.max(1) as f64;
            if big_parts >= 2 && mean_rect > rectangularity + 0.04 {
                for p in parts {
                    process_component(ctx, &p, depth + 1, limits, out);
                }
                return;
            }
        }
    }

    // Start from the appendage-free extents, but if they differ from the plain minimum-area
    // rectangle (which keeps faint, patchy parts such as a dark photo's dark half), refine
    // from both.
    let coarse = robust_extents(ctx, comp, &rect);
    log::trace!(
        "component: min-area rect {:?}, robust {:?} (mm), rectangularity {rectangularity:.3}",
        rect.scaled(1.0 / ctx.px_per_mm),
        coarse.scaled(1.0 / ctx.px_per_mm)
    );
    let mut refined = refine_rect(ctx, &coarse, limits);
    if coarse.iou(&rect) < 0.99
        && let Some(alt) = refine_rect(ctx, &rect, limits)
    {
        log::trace!(
            "support: from robust extents {:?}, from min-area rect {:.3}",
            refined.map(|r| r.1),
            alt.1
        );
        // The larger box wins unless its edges are clearly worse supported (an appendage
        // leaves a side with no real edge).
        if refined.is_none_or(|r| alt.1 >= r.1 - 0.03) {
            refined = Some(alt);
        }
    }
    let conf = (rectangularity.clamp(0.0, 1.0) * if refined.is_some() { 1.0 } else { 0.8 })
        .clamp(0.0, 1.0);
    out.push((refined.map_or(coarse, |r| r.0), conf));
}

/// Re-measure the extents of `rect` (same angle) from projection histograms of the
/// component, ignoring thin appendages: a bit of uneven background stuck to one side, or a
/// wedge of background left over from a seam split.
fn robust_extents(ctx: &Ctx, comp: &[usize], rect: &RotatedRect) -> RotatedRect {
    let (ux, uy) = rect.axes();
    let (cx, cy) = (rect.cx, rect.cy);
    let hu = (rect.w / 2.0).ceil() as isize + 2;
    let hv = (rect.h / 2.0).ceil() as isize + 2;
    let mut pu = vec![0u32; (2 * hu + 1) as usize];
    let mut pv = vec![0u32; (2 * hv + 1) as usize];
    for &i in comp {
        let (x, y) = ((i % ctx.w) as f64 + 0.5 - cx, (i / ctx.w) as f64 + 0.5 - cy);
        let u = (x * ux.x + y * ux.y).floor() as isize + hu;
        let v = (x * uy.x + y * uy.y).floor() as isize + hv;
        if let Some(c) = usize::try_from(u).ok().and_then(|u| pu.get_mut(u)) {
            *c += 1;
        }
        if let Some(c) = usize::try_from(v).ok().and_then(|v| pv.get_mut(v)) {
            *c += 1;
        }
    }
    // Extent = where the slice length is at least a fraction of the typical (median) one.
    let extent = |p: &[u32], half: isize| -> Option<(f64, f64)> {
        let mut nz: Vec<u32> = p.iter().cloned().filter(|&c| c > 0).collect();
        if nz.is_empty() {
            return None;
        }
        nz.sort_unstable();
        let lim = (nz[nz.len() / 2] as f64 * EXTENT_FRACTION) as u32;
        let lo = p.iter().position(|&c| c >= lim)? as isize;
        let hi = p.iter().rposition(|&c| c >= lim)? as isize;
        Some(((lo - half) as f64, (hi + 1 - half) as f64))
    };
    let (Some((u0, u1)), Some((v0, v1))) = (extent(&pu, hu), extent(&pv, hv)) else {
        return *rect;
    };
    if u1 - u0 < 0.5 * rect.w || v1 - v0 < 0.5 * rect.h {
        return *rect;
    }
    let (cu, cv) = ((u0 + u1) / 2.0, (v0 + v1) / 2.0);
    RotatedRect {
        cx: cx + ux.x * cu + uy.x * cv,
        cy: cy + ux.y * cu + uy.y * cv,
        w: u1 - u0,
        h: v1 - v0,
        angle: rect.angle,
    }
}

/// Bounding-box copy of a component, for fast inside tests.
struct LocalMask {
    x0: isize,
    y0: isize,
    w: isize,
    h: isize,
    data: Vec<bool>,
}

impl LocalMask {
    fn new(img_w: usize, comp: &[usize]) -> Self {
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for &i in comp {
            let (x, y) = (i % img_w, i / img_w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let (w, h) = (x1 + 1 - x0, y1 + 1 - y0);
        let mut data = vec![false; w * h];
        for &i in comp {
            data[(i / img_w - y0) * w + (i % img_w - x0)] = true;
        }
        Self {
            x0: x0 as isize,
            y0: y0 as isize,
            w: w as isize,
            h: h as isize,
            data,
        }
    }
    fn contains(&self, x: f64, y: f64) -> bool {
        let (ix, iy) = (x.floor() as isize - self.x0, y.floor() as isize - self.y0);
        ix >= 0 && iy >= 0 && ix < self.w && iy < self.h && self.data[(iy * self.w + ix) as usize]
    }
}

struct Seam {
    /// Fraction of the line with gap or edge evidence.
    coverage: f64,
    /// Fraction of the line with gap evidence.
    gap: f64,
    /// A point on the seam and its unit normal, in working pixels.
    p0: Point,
    n: Point,
}

/// Look for a straight seam that cuts the component into two photos: a thin line of
/// background between them, or a long straight edge where one photo lies on another.
fn seam_split(
    ctx: &Ctx,
    comp: &[usize],
    rect: &RotatedRect,
    rectangularity: f64,
) -> Option<(Vec<Vec<usize>>, Seam)> {
    let lm = LocalMask::new(ctx.w, comp);
    let (ux, uy) = rect.axes();
    let c = Point::new(rect.cx, rect.cy);
    let mut best: Option<Seam> = None;
    for (n, d, ext_n, ext_d) in [(ux, uy, rect.w, rect.h), (uy, ux, rect.h, rect.w)] {
        if ext_n < 2.0 * ctx.min_side_px {
            continue;
        }
        if let Some(s) = best_seam(ctx, &lm, c, n, d, ext_n, ext_d)
            && best.as_ref().is_none_or(|b| s.coverage > b.coverage)
        {
            best = Some(s);
        }
    }
    let s = best?;
    // A thin gap is strong evidence. Without one, only split blobs that are clearly not a
    // single clean rectangle, so long straight lines in the picture (a horizon, a door
    // frame) do not cut a photo in two.
    let accept =
        (s.coverage >= 0.85 && s.gap >= 0.5) || (s.coverage >= 0.9 && rectangularity < 0.97);
    log::debug!(
        "seam: coverage {:.2}, gap {:.2}, rectangularity {rectangularity:.3} -> {accept}; \
         through ({:.1}, {:.1}) mm, normal ({:.4}, {:.4})",
        s.coverage,
        s.gap,
        s.p0.x / ctx.px_per_mm,
        s.p0.y / ctx.px_per_mm,
        s.n.x,
        s.n.y
    );
    if !accept {
        return None;
    }
    let w = ctx.w;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &i in comp {
        let p = Point::new((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
        if (p.x - s.p0.x) * s.n.x + (p.y - s.p0.y) * s.n.y < 0.0 {
            a.push(i);
        } else {
            b.push(i);
        }
    }
    for part in [&a, &b] {
        let r = min_area_rect(&boundary_points(ctx, part))?;
        if r.w.min(r.h) < ctx.min_side_px {
            return None;
        }
    }
    Some((vec![a, b], s))
}

/// Best seam parallel (within a few degrees) to `d`, scanning offsets along `n`.
fn best_seam(
    ctx: &Ctx,
    lm: &LocalMask,
    c: Point,
    n: Point,
    d: Point,
    ext_n: f64,
    ext_d: f64,
) -> Option<Seam> {
    let fine = ctx.fine;
    let k = fine.k;
    let mm = ctx.px_per_mm;
    // Rows every 0.5 mm along the seam direction; cells one fine pixel apart across it.
    let du = 1.0 / k;
    let dv = 0.5 * mm;
    let nu = (ext_n / du).ceil() as usize + 1;
    let nv = ((ext_d - 2.0 * mm) / dv).floor().max(0.0) as usize;
    if nv < 20 || nu < 8 {
        return None;
    }
    let u0 = -ext_n / 2.0;
    let v0 = -(nv as f64 - 1.0) * dv / 2.0;
    let cells = |len_mm: f64| ((len_mm * mm / du).round() as usize).max(1);
    let win = cells(0.4);
    let side = cells(0.5);
    let ridge = cells(1.5);
    let narrow = cells(0.6);
    let max_run = cells(4.0);
    let gap_thr = ctx.fine_threshold;
    let edge_thr = 3.0f32;

    // Per row: inside-component flags, evidence bits, and gap intervals [start, end) with
    // their mean background distance.
    type Row = (Vec<bool>, Vec<u8>, Vec<(usize, usize, f32)>, Vec<bool>);
    let rows: Vec<Row> = (0..nv)
        .into_par_iter()
        .map(|j| {
            let v = v0 + j as f64 * dv;
            let mut dist = vec![0f32; nu];
            let mut l = vec![0f32; nu];
            let mut along = vec![0f32; nu];
            let mut inside = vec![false; nu];
            for i in 0..nu {
                let u = u0 + i as f64 * du;
                let p = Point::new(c.x + n.x * u + d.x * v, c.y + n.y * u + d.y * v);
                inside[i] = lm.contains(p.x, p.y);
                let (fx, fy) = (p.x * k - 0.5, p.y * k - 0.5);
                dist[i] = fine.dist.sample(fx, fy);
                l[i] = fine.l.sample(fx, fy);
                along[i] = (fine.l.sample(fx + d.x, fy + d.y) - fine.l.sample(fx - d.x, fy - d.y))
                    .abs()
                    * 0.5;
            }
            let edge: Vec<bool> = (0..nu)
                .map(|i| {
                    if i == 0 || i + 1 >= nu {
                        return false;
                    }
                    let gn = (l[i + 1] - l[i - 1]).abs() * 0.5;
                    gn > edge_thr && gn > 2.5 * along[i]
                })
                .collect();
            // Evidence bits: 1 = edge, 2 = gap.
            let mut ev = vec![0u8; nu];
            let mut gaps: Vec<(usize, usize, f32)> = Vec::new();
            let mean = |a: usize, b: usize| dist[a..b].iter().sum::<f32>() / (b - a) as f32;
            for (i, e) in ev.iter_mut().enumerate() {
                let lo = i.saturating_sub(win);
                let hi = (i + win).min(nu - 1);
                if edge[lo..=hi].iter().any(|&x| x) {
                    *e |= 1;
                }
            }
            // A gap is a run of background-coloured cells, at most a few mm wide, with
            // photo on both sides.
            let fg_thr = 1.5 * gap_thr;
            let mut i = 0;
            while i < nu {
                if dist[i] >= gap_thr {
                    i += 1;
                    continue;
                }
                let s = i;
                while i < nu && dist[i] < gap_thr {
                    i += 1;
                }
                let left = (s.saturating_sub(side)..s).any(|q| dist[q] > fg_thr);
                let right = (i..(i + side).min(nu)).any(|q| dist[q] > fg_thr);
                if s > 0 && i < nu && i - s <= max_run && left && right {
                    for e in &mut ev[s.saturating_sub(win)..(i + win).min(nu)] {
                        *e |= 2;
                    }
                    gaps.push((s, i, mean(s, i)));
                }
            }
            // Or a narrow ridge towards the background colour (the white paper edge of a
            // print, or a gap thinner than a pixel): a local minimum of background distance
            // well below the photos on both sides.
            for i in 1..nu.saturating_sub(1) {
                if dist[i] > dist[i - 1] || dist[i] > dist[i + 1] {
                    continue;
                }
                let lmax = dist[i.saturating_sub(ridge)..i]
                    .iter()
                    .cloned()
                    .fold(0.0f32, f32::max);
                let rmax = dist[i + 1..(i + 1 + ridge).min(nu)]
                    .iter()
                    .cloned()
                    .fold(0.0f32, f32::max);
                let base = lmax.min(rmax);
                if base - dist[i] <= (2.0 * gap_thr).max(0.25 * base) {
                    continue;
                }
                // Narrow: below the half-way level for at most 0.6 mm.
                let half = (dist[i] + base) / 2.0;
                let lo = (0..i).rev().find(|&q| dist[q] >= half).unwrap_or(0);
                let hi = (i + 1..nu).find(|&q| dist[q] >= half).unwrap_or(nu - 1);
                if hi - lo <= narrow {
                    for e in &mut ev[i.saturating_sub(win)..(i + win + 1).min(nu)] {
                        *e |= 2;
                    }
                    gaps.push((i, i + 1, dist[i]));
                }
            }
            let solid: Vec<bool> = dist.iter().map(|&d| d > gap_thr).collect();
            (inside, ev, gaps, solid)
        })
        .collect();

    if std::env::var_os("HP_DETECT_DEBUG").is_some() {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        debug_image(&format!("seam{id}"), nu, nv, |q| {
            let (inside, ev, _, _) = &rows[q / nu];
            let i = q % nu;
            (inside[i] as u8) * 60 + (ev[i] & 1) * 80 + ((ev[i] & 2) >> 1) * 115
        });
    }

    // Score straight lines through the grid: offset t along n, small tilt.
    let t_lo = u0 + ctx.min_side_px;
    let t_hi = -u0 - ctx.min_side_px;
    if t_hi <= t_lo {
        return None;
    }
    let nt = ((t_hi - t_lo) / du) as usize + 1;
    let angles: Vec<f64> = (-6..=6).map(|a| a as f64 * 0.5).collect();
    let min_in = (nv as f64 * 0.3).max(20.0) as u32;
    let scores: Vec<Vec<(f32, f32)>> = angles
        .par_iter()
        .map(|&deg| {
            let tan = deg.to_radians().tan();
            (0..nt)
                .map(|ti| {
                    let t = t_lo + ti as f64 * du;
                    let (mut nin, mut nany, mut ngap) = (0u32, 0u32, 0u32);
                    for (j, (inside, ev, _, _)) in rows.iter().enumerate() {
                        let v = v0 + j as f64 * dv;
                        let i = ((t + v * tan - u0) / du).round();
                        if i < 0.0 || i >= nu as f64 {
                            continue;
                        }
                        let i = i as usize;
                        if inside[i] {
                            nin += 1;
                            nany += (ev[i] != 0) as u32;
                            ngap += (ev[i] & 2 != 0) as u32;
                        }
                    }
                    if nin < min_in {
                        (0.0, 0.0)
                    } else {
                        (nany as f32 / nin as f32, ngap as f32 / nin as f32)
                    }
                })
                .collect()
        })
        .collect();

    let mut best: Option<(usize, usize, f32)> = None;
    for (a, row) in scores.iter().enumerate() {
        for (ti, &(cov, _)) in row.iter().enumerate() {
            if best.is_none_or(|b| cov > b.2) {
                best = Some((a, ti, cov));
            }
        }
    }
    let (a, ti, cov) = best?;
    // A real seam is a ridge: find the plateau of near-best lines (a wide gap gives several),
    // which must be narrow, and lines 1.5 mm beyond it must score much lower.
    let row = &scores[a];
    let (mut l, mut r) = (ti, ti);
    while l > 0 && row[l - 1].0 >= cov - 0.1 {
        l -= 1;
    }
    while r + 1 < row.len() && row[r + 1].0 >= cov - 0.1 {
        r += 1;
    }
    if (r - l) as f64 * du > 5.0 * mm {
        return None;
    }
    let off = cells(1.5);
    let side_cov = |t: Option<usize>| t.and_then(|t| row.get(t)).map_or(0.0, |s| s.0);
    let neighbour = side_cov(l.checked_sub(off)).max(side_cov(Some(r + off)));
    if cov - neighbour < 0.3 {
        return None;
    }
    let ti = (l + r) / 2;
    let cov = row[ti].0;
    let mut tan = angles[a].to_radians().tan();
    let mut t = t_lo + ti as f64 * du;
    // Re-centre on the middle of the gap. A wedge-shaped gap between two photos at different
    // angles scores equally well along either photo's edge; its centre line is the fair cut.
    let mut mids: Vec<(f64, f64)> = Vec::new();
    for (j, (inside, _, gaps, _)) in rows.iter().enumerate() {
        let v = v0 + j as f64 * dv;
        let at = (t + v * tan - u0) / du;
        if at < 0.0 || at >= nu as f64 || !inside[at as usize] {
            continue;
        }
        // Nearby gap with the cleanest background (a dark photo next to a white paper edge
        // can look like a gap too, but less background-like than the real one).
        let reach = cells(1.5) as f64;
        let near = gaps
            .iter()
            .filter(|&&(s, e, _)| at >= s as f64 - reach && at <= e as f64 + reach)
            .min_by(|x, y| x.2.total_cmp(&y.2));
        if let Some(&(s, e, _)) = near {
            mids.push((v, u0 + (s + e - 1) as f64 / 2.0 * du));
        }
    }
    if let Some((new_t, new_tan)) = ransac_line(&mids, 0.3 * mm, 4f64.to_radians().tan()) {
        log::trace!("seam re-centred: t {:.2} mm, tan {new_tan:.4}", new_t / mm);
        t = new_t;
        tan = new_tan;
    }
    // There must be photo on both sides along the seam. A photo's own straight outer edge,
    // with a strip of background swallowed by the mask beyond it, is not a seam.
    let (near, far) = (cells(1.0), cells(3.0));
    let (mut nin, mut both) = (0u32, 0u32);
    for (j, (inside, _, _, solid)) in rows.iter().enumerate() {
        let v = v0 + j as f64 * dv;
        let at = ((t + v * tan - u0) / du).round();
        if at < 0.0 || at >= nu as f64 || !inside[at as usize] {
            continue;
        }
        let i = at as usize;
        nin += 1;
        let left = (i.saturating_sub(far)..i.saturating_sub(near)).any(|q| solid[q]);
        let right = ((i + near).min(nu)..(i + far).min(nu)).any(|q| solid[q]);
        both += (left && right) as u32;
    }
    if nin == 0 || (both as f64) < 0.35 * nin as f64 {
        log::debug!("seam candidate has background on one side ({both}/{nin})");
        return None;
    }
    let nn = Point::new(n.x - d.x * tan, n.y - d.y * tan);
    let len = (nn.x * nn.x + nn.y * nn.y).sqrt();
    Some(Seam {
        coverage: cov as f64,
        gap: row[ti].1 as f64,
        p0: Point::new(c.x + n.x * t, c.y + n.y * t),
        n: Point::new(nn.x / len, nn.y / len),
    })
}

/// Robust fit of `u = t + v * tan` to (v, u) points: the line through a pair of points with
/// the most inliers (within `tol`), then least squares on those inliers. Needs at least 10
/// inliers, and `|tan| <= max_tan`.
fn ransac_line(pts: &[(f64, f64)], tol: f64, max_tan: f64) -> Option<(f64, f64)> {
    if pts.len() < 10 {
        return None;
    }
    let stride = pts.len().div_ceil(150);
    let sample: Vec<(f64, f64)> = pts.iter().step_by(stride).cloned().collect();
    let mut best: Option<(usize, f64, f64)> = None;
    for (i, a) in sample.iter().enumerate() {
        for b in &sample[i + 1..] {
            if (b.0 - a.0).abs() < 1e-9 {
                continue;
            }
            let tan = (b.1 - a.1) / (b.0 - a.0);
            if tan.abs() > max_tan {
                continue;
            }
            let t = a.1 - a.0 * tan;
            let n = pts
                .iter()
                .filter(|p| (p.1 - t - p.0 * tan).abs() <= tol)
                .count();
            if best.is_none_or(|bb| n > bb.0) {
                best = Some((n, t, tan));
            }
        }
    }
    let (n, t, tan) = best?;
    if n < 10 {
        return None;
    }
    let inl: Vec<&(f64, f64)> = pts
        .iter()
        .filter(|p| (p.1 - t - p.0 * tan).abs() <= tol)
        .collect();
    let k = inl.len() as f64;
    let mv = inl.iter().map(|p| p.0).sum::<f64>() / k;
    let mu = inl.iter().map(|p| p.1).sum::<f64>() / k;
    let svv: f64 = inl.iter().map(|p| (p.0 - mv).powi(2)).sum();
    let svu: f64 = inl.iter().map(|p| (p.0 - mv) * (p.1 - mu)).sum();
    if svv < 1e-9 {
        return Some((t, tan));
    }
    let tan = (svu / svv).clamp(-max_tan, max_tan);
    Some((mu - mv * tan, tan))
}

/// Split a blob into pieces by eroding until it falls apart, then growing the cores back
/// geodesically within the original blob.
fn try_split(ctx: &Ctx, comp: &[usize]) -> Option<Vec<Vec<usize>>> {
    let (w, h) = (ctx.w, ctx.h);
    // Work in the component's bounding box for speed.
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for &i in comp {
        let (x, y) = (i % w, i / w);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (bw, bh) = (x1 - x0 + 3, y1 - y0 + 3);
    let local = |i: usize| (i / w - y0 + 1) * bw + (i % w - x0 + 1);
    let global = |j: usize| (j / bw + y0 - 1) * w + (j % bw + x0 - 1);
    let mut m = Mask::new(bw, bh);
    for &i in comp {
        m.data[local(i)] = true;
    }
    let dt = m.distance_transform();
    let maxd = dt.iter().cloned().fold(0.0f32, f32::max);
    let min_core = (ctx.min_side_px * ctx.min_side_px * 0.05).max(16.0) as usize;
    let mut r = 1.5f32;
    while r < maxd * 0.8 {
        let core = Mask {
            w: bw,
            h: bh,
            data: dt.iter().map(|&d| d > r).collect(),
        };
        let (labels, cores) = core.components();
        let big: Vec<usize> = (0..cores.len())
            .filter(|&k| cores[k].len() >= min_core)
            .collect();
        if big.len() >= 2 {
            // Multi-source BFS from the big cores, restricted to the blob.
            let mut owner = vec![u32::MAX; bw * bh];
            let mut queue = std::collections::VecDeque::new();
            for (n, &k) in big.iter().enumerate() {
                for &j in &cores[k] {
                    owner[j] = n as u32;
                    queue.push_back(j);
                }
            }
            let _ = labels;
            while let Some(j) = queue.pop_front() {
                let (x, y) = (j % bw, j / bw);
                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= bw as isize || ny >= bh as isize {
                        continue;
                    }
                    let k = ny as usize * bw + nx as usize;
                    if m.data[k] && owner[k] == u32::MAX {
                        owner[k] = owner[j];
                        queue.push_back(k);
                    }
                }
            }
            let mut parts = vec![Vec::new(); big.len()];
            for (j, &o) in owner.iter().enumerate() {
                if o != u32::MAX {
                    parts[o as usize].push(global(j));
                }
            }
            return Some(parts);
        }
        r += (0.5 * ctx.px_per_mm as f32).max(1.0);
    }
    None
}

/// Sub-pixel refinement of a rectangle's edges using profiles of background distance at the
/// fine resolution. `rect` is in working pixels, and so is the result.
///
/// A wide first pass recovers from a coarse rectangle that is a few mm off (for example a
/// wedge of background left on after a seam split); a narrow second pass then settles.
///
/// Returns the rectangle and a support score: the mean, over the four sides, of the fraction
/// of edge profiles that agree with the fitted side.
fn refine_rect(ctx: &Ctx, rect_w: &RotatedRect, limits: &[Limit]) -> Option<(RotatedRect, f64)> {
    let first = refine_pass(ctx, rect_w, limits, FIRST_SEARCH_MM)?;
    Some(refine_pass(ctx, &first.0, limits, 1.0).unwrap_or(first))
}

fn refine_pass(
    ctx: &Ctx,
    rect_w: &RotatedRect,
    limits: &[Limit],
    search_mm: f64,
) -> Option<(RotatedRect, f64)> {
    let fine = ctx.fine;
    let k = fine.k;
    let rect = rect_w.scaled(k);
    // Sides without usable edge profiles (e.g. running off the scan) keep the input's.
    let corners = rect.corners();
    let fpx_mm = ctx.px_per_mm * k;
    let (fw, fh) = (fine.dist.w, fine.dist.h);
    let (ux, uy) = rect.axes();
    let c = Point::new(rect.cx, rect.cy);
    let search = (search_mm * fpx_mm).max(3.0);
    let step = 0.25;
    // Background must show just outside a real edge (within 0.04–0.4 mm; gaps between
    // touching photos can be very thin).
    let out0 = ((0.04 * fpx_mm) / step).round() as usize;
    let out1 = ((0.4 * fpx_mm) / step).round().max(out0 as f64 + 2.0) as usize;
    // Sides: (outward normal, along-direction, half-length, distance from centre)
    let sides = [
        (Point::new(-uy.x, -uy.y), ux, rect.w / 2.0, rect.h / 2.0), // top
        (ux, uy, rect.h / 2.0, rect.w / 2.0),                       // right
        (uy, ux, rect.w / 2.0, rect.h / 2.0),                       // bottom
        (Point::new(-ux.x, -ux.y), uy, rect.h / 2.0, rect.w / 2.0), // left
    ];
    let mut side_pts: Vec<Vec<Point>> = Vec::new();
    let mut side_samples: Vec<usize> = Vec::new();
    for &(n, along, half, off) in &sides {
        let samples = ((half * 2.0 * 0.8) / fpx_mm).clamp(8.0, 200.0) as usize;
        side_samples.push(samples);
        let pts: Vec<Point> = (0..samples)
            .into_par_iter()
            .filter_map(|j| {
                let t = -half * 0.8 + (half * 1.6) * (j as f64 + 0.5) / samples as f64;
                let base = Point::new(c.x + along.x * t + n.x * off, c.y + along.y * t + n.y * off);
                // Profile from outside (s = +search) to inside (s = -search). Near the scan
                // border the outer end may be cut off; then only an edge with background
                // visibly outside it is accepted.
                let nsteps = (2.0 * search / step) as usize;
                let inside_img = |s: usize| {
                    let d = search - s as f64 * step;
                    let p = Point::new(base.x + n.x * d, base.y + n.y * d);
                    (p.x >= 0.0 && p.y >= 0.0 && p.x <= fw as f64 && p.y <= fh as f64).then_some(p)
                };
                let start = (0..=nsteps).find(|&s| inside_img(s).is_some())?;
                let truncated = start > 0;
                let mut prof = Vec::with_capacity(nsteps + 1 - start);
                for s in start..=nsteps {
                    let p = inside_img(s)?;
                    prof.push(fine.dist.sample(p.x - 0.5, p.y - 0.5));
                }
                if prof.len() < 8 || (truncated && (prof.len() as f64) * step < fpx_mm) {
                    return None;
                }
                // Derivative (rise when moving inward).
                let deriv: Vec<f32> = (1..prof.len() - 1)
                    .map(|i| (prof[i + 1] - prof[i - 1]) / (2.0 * step as f32))
                    .collect();
                let maxd = deriv.iter().cloned().fold(0.0f32, f32::max);
                if maxd < ctx.fine_threshold * 0.05 {
                    return None;
                }
                // Local maxima of the rise. Prefer the strongest one with background just
                // outside it; otherwise (photos overlapping) the strongest overall.
                // Candidate edges: local maxima of the rise. "Clean" ones have background just
                // outside. Take the strongest clean edge; without any (photos overlapping),
                // the strongest edge.
                // An edge may not lie beyond a seam this photo was split off along.
                let tol = 0.3 * ctx.px_per_mm;
                let past_seam = |i: usize| {
                    let d = search - (start + i + 1) as f64 * step;
                    let p = Point::new((base.x + n.x * d) / k, (base.y + n.y * d) / k);
                    limits.iter().any(|l| l.beyond(p) > tol)
                };
                let mut clean: Vec<usize> = Vec::new();
                let mut any: Option<usize> = None;
                for i in 1..deriv.len() - 1 {
                    let v = deriv[i];
                    if v < 0.25 * maxd || v < deriv[i - 1] || v < deriv[i + 1] || past_seam(i) {
                        continue;
                    }
                    if any.is_none_or(|a| v > deriv[a]) {
                        any = Some(i);
                    }
                    // deriv index i is prof index i + 1; outside = smaller prof indices.
                    let pi = i + 1;
                    if pi >= out0 + 2 {
                        let lo = pi.saturating_sub(out1);
                        let outside = prof[lo..=pi - out0]
                            .iter()
                            .cloned()
                            .fold(f32::MAX, f32::min);
                        if outside < ctx.fine_threshold {
                            clean.push(i);
                        }
                    }
                }
                let best_clean = clean
                    .into_iter()
                    .max_by(|&a, &b| deriv[a].total_cmp(&deriv[b]));
                let i = if truncated {
                    best_clean?
                } else {
                    best_clean.or(any)?
                };
                let (a, b, cc) = (deriv[i - 1] as f64, deriv[i] as f64, deriv[i + 1] as f64);
                let den = a - 2.0 * b + cc;
                let frac = if den.abs() > 1e-9 {
                    (0.5 * (a - cc) / den).clamp(-0.5, 0.5)
                } else {
                    0.0
                };
                // deriv index i corresponds to prof index i+1, i.e. step start+i+1.
                let d = search - ((start + i + 1) as f64 + frac) * step;
                Some(Point::new(base.x + n.x * d, base.y + n.y * d))
            })
            .collect();
        side_pts.push(pts);
    }

    // Fit lines and estimate a common angle.
    let mut angle_acc = 0.0;
    let mut angle_w = 0.0;
    let mut support = 0.0;
    let mut fits: Vec<Option<Vec<Point>>> = Vec::new();
    for (si, pts) in side_pts.iter().enumerate() {
        if pts.len() < 8 {
            fits.push(None);
            continue;
        }
        let Some((line, inl)) = fit_line_robust(pts, 4, 0.5) else {
            fits.push(None);
            continue;
        };
        // Line direction is perpendicular to its normal.
        let dir_angle = (-line.nx).atan2(line.ny).to_degrees();
        let base = if si % 2 == 0 {
            rect.angle
        } else {
            rect.angle + 90.0
        };
        let mut dlt = (dir_angle - base).rem_euclid(180.0);
        if dlt > 90.0 {
            dlt -= 180.0;
        }
        if dlt.abs() > 5.0 {
            fits.push(None);
            continue;
        }
        let wgt = inl.len() as f64;
        support += wgt / side_samples[si] as f64 / 4.0;
        log::trace!(
            "side {si}: {} of {} points, angle offset {dlt:.2}",
            inl.len(),
            pts.len()
        );
        angle_acc += dlt * wgt;
        angle_w += wgt;
        fits.push(Some(inl.iter().map(|&i| pts[i]).collect()));
    }
    log::debug!(
        "refine: side points {:?}, fitted {:?}",
        side_pts.iter().map(|p| p.len()).collect::<Vec<_>>(),
        fits.iter()
            .map(|f| f.as_ref().map_or(0, |p| p.len()))
            .collect::<Vec<_>>()
    );
    if angle_w == 0.0 {
        return None;
    }
    let angle = rect.angle + angle_acc / angle_w;
    let probe = RotatedRect { angle, ..rect };
    let (nux, nuy) = probe.axes();
    let proj_u = |p: &Point| p.x * nux.x + p.y * nux.y;
    let proj_v = |p: &Point| p.x * nuy.x + p.y * nuy.y;
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    // Midpoints of the input's sides: top, right, bottom, left.
    let mid = |a: usize, b: usize| {
        Point::new(
            (corners[a].x + corners[b].x) / 2.0,
            (corners[a].y + corners[b].y) / 2.0,
        )
    };
    let fallback = [mid(0, 1), mid(1, 2), mid(2, 3), mid(3, 0)];
    let side_pos = |si: usize, proj: &dyn Fn(&Point) -> f64| {
        fits[si]
            .as_ref()
            .map(|p| med(p.iter().map(proj).collect()))
            .unwrap_or_else(|| proj(&fallback[si]))
    };
    let v_min = side_pos(0, &proj_v);
    let u_max = side_pos(1, &proj_u);
    let v_max = side_pos(2, &proj_v);
    let u_min = side_pos(3, &proj_u);
    log::trace!(
        "sides moved by top {:.2} right {:.2} bottom {:.2} left {:.2} fine px",
        proj_v(&fallback[0]) - v_min,
        u_max - proj_u(&fallback[1]),
        v_max - proj_v(&fallback[2]),
        proj_u(&fallback[3]) - u_min
    );
    if u_max <= u_min || v_max <= v_min {
        return None;
    }
    let (cu, cv) = ((u_min + u_max) / 2.0, (v_min + v_max) / 2.0);
    let refined = RotatedRect {
        cx: nux.x * cu + nuy.x * cv,
        cy: nux.y * cu + nuy.y * cv,
        w: u_max - u_min,
        h: v_max - v_min,
        angle,
    }
    .normalized();
    log::debug!("refine: iou with input {:.3}", refined.iou(&rect));
    (refined.iou(&rect) > 0.85).then_some((refined.scaled(1.0 / k), support))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{SceneParams, render_scene};

    #[test]
    fn detects_synthetic_photos_dark_background() {
        let mut total_iou = 0.0;
        let mut n = 0;
        for seed in 0..6u64 {
            let scene = render_scene(&SceneParams {
                seed,
                dark_background: true,
                dpi: 100.0,
                ..Default::default()
            });
            let rep = detect_photos(
                &scene.image,
                ScanGeometry::new(scene.dpi),
                &DetectParams::default(),
            );
            assert_eq!(rep.photos.len(), scene.truth.len(), "seed {seed}");
            for t in &scene.truth {
                let best = rep.photos.iter().map(|d| d.rect.iou(t)).fold(0.0, f64::max);
                total_iou += best;
                n += 1;
            }
        }
        let mean = total_iou / n as f64;
        assert!(mean > 0.97, "mean IoU {mean}");
    }

    #[test]
    fn detects_synthetic_photos_light_background() {
        for seed in 100..104u64 {
            let scene = render_scene(&SceneParams {
                seed,
                dark_background: false,
                dpi: 100.0,
                ..Default::default()
            });
            let rep = detect_photos(
                &scene.image,
                ScanGeometry::new(scene.dpi),
                &DetectParams::default(),
            );
            assert_eq!(rep.photos.len(), scene.truth.len(), "seed {seed}");
        }
    }

    /// A white-lid scene drawn directly: textured axis-aligned "prints" (mm boxes), with an
    /// optional bright paper-edge line between neighbours. Returns the image at `dpi`.
    fn lid_scene(dpi: f64, size_mm: (f64, f64), prints: &[(f64, f64, f64, f64)]) -> DynamicImage {
        let ppm = dpi / 25.4;
        let (w, h) = ((size_mm.0 * ppm) as u32, (size_mm.1 * ppm) as u32);
        let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([236, 235, 232]));
        for (k, &(x0, y0, x1, y1)) in prints.iter().enumerate() {
            let base = [40.0 + 50.0 * k as f64, 70.0, 110.0 - 20.0 * k as f64];
            for y in (y0 * ppm) as u32..((y1 * ppm) as u32).min(h) {
                for x in (x0 * ppm) as u32..((x1 * ppm) as u32).min(w) {
                    let t = ((x as f64 * 0.37).sin() + (y as f64 * 0.23).cos()) * 12.0;
                    let px = base.map(|c| (c + t).clamp(0.0, 255.0) as u8);
                    img.put_pixel(x, y, image::Rgb(px));
                }
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    fn best_iou(rep: &DetectReport, t: &RotatedRect) -> f64 {
        rep.photos.iter().map(|d| d.rect.iou(t)).fold(0.0, f64::max)
    }

    #[test]
    fn photos_running_off_a_white_lid() {
        // Two prints cover most of the top and left border; the lid only shows elsewhere.
        let prints = [
            (0.0, 0.0, 95.0, 125.0),
            (0.0, 140.0, 90.0, 210.0),
            (105.0, 30.0, 150.0, 120.0),
        ];
        let img = lid_scene(150.0, (150.0, 210.0), &prints);
        let rep = detect_photos(&img, ScanGeometry::new(150.0), &DetectParams::default());
        assert!(!rep.dark_background, "background L {}", rep.background_l);
        assert_eq!(rep.photos.len(), 3, "{:?}", rep.photos);
        for &(x0, y0, x1, y1) in &prints {
            let t = RotatedRect::new((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0, 0.0);
            assert!(best_iou(&rep, &t) > 0.97, "{t:?}: {:?}", rep.photos);
        }
    }

    #[test]
    fn touching_photos_split_at_paper_edge() {
        // Two prints butted together; between them only a 0.3 mm bright paper edge.
        let dpi = 300.0;
        let mut img = lid_scene(
            dpi,
            (130.0, 110.0),
            &[(10.0, 15.0, 60.0, 95.0), (60.0, 12.0, 115.0, 92.0)],
        )
        .to_rgb8();
        let ppm = dpi / 25.4;
        for y in (15.0 * ppm) as u32..(92.0 * ppm) as u32 {
            for x in (59.85 * ppm) as u32..(60.15 * ppm) as u32 {
                img.put_pixel(x, y, image::Rgb([205, 203, 198]));
            }
        }
        let img = DynamicImage::ImageRgb8(img);
        let rep = detect_photos(&img, ScanGeometry::new(dpi), &DetectParams::default());
        assert_eq!(rep.photos.len(), 2, "{:?}", rep.photos);
        let left = RotatedRect::new(35.0, 55.0, 50.0, 80.0, 0.0);
        let right = RotatedRect::new(87.5, 52.0, 55.0, 80.0, 0.0);
        assert!(best_iou(&rep, &left) > 0.97, "{:?}", rep.photos);
        assert!(best_iou(&rep, &right) > 0.97, "{:?}", rep.photos);
    }
}
