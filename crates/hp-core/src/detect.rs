//! Multi-photo detection on a flatbed scan.
//!
//! Pipeline (all at a fixed low "working" resolution, by default 100 DPI):
//!
//! 1. Convert to CIE Lab and lightly blur.
//! 2. Estimate the background colour from a ring just inside the scan border, then refine it
//!    into a smooth background *surface* (handles lid vignetting / uneven lighting).
//! 3. Foreground = colour distance from background above a noise-adaptive threshold, OR
//!    strong luminance gradient (catches the faint shadow outline of white-bordered prints
//!    on a white lid).
//! 4. Morphological close → fill holes → open, then connected components.
//! 5. Per component: minimum-area rectangle of the boundary; if the component is not very
//!    rectangular, try splitting it (photos that touch) by progressive erosion + geodesic
//!    region growing.
//! 6. Sub-pixel edge refinement: sample many profiles across each side of the rectangle,
//!    locate the outermost strong rise of background distance, robustly fit a line per side
//!    and solve for the best exactly-rectangular fit.
//!
//! Results are reported in millimetres in scanner-bed coordinates so they can be mapped
//! between a low-resolution preview and the high-resolution final scan.

use crate::geom::{Point, RotatedRect, fit_line_robust, min_area_rect};
use crate::raster::{Lab, Mask, Plane, lab_planes, percentile};
use image::DynamicImage;
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
pub struct DetectParams {
    /// Resolution detection runs at. Higher is slower but slightly more precise.
    pub work_dpi: f64,
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

struct Ctx<'a> {
    w: usize,
    h: usize,
    /// Distance from local background (ΔE-ish), blurred.
    dist: &'a Plane,
    threshold: f32,
    px_per_mm: f64,
    min_side_px: f64,
    split: bool,
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

    // --- Background estimate from the border ring ---------------------------------------
    let m = w.min(h) as f64;
    let i0 = (m * 0.01).max(1.0) as usize;
    let i1 = i0 + (m * 0.03).max(3.0) as usize;
    let mut ring: Vec<usize> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let e = x.min(y).min(w - 1 - x).min(h - 1 - y);
            if e >= i0 && e < i1 {
                ring.push(y * w + x);
            }
        }
    }
    let med = |p: &Plane, idx: &[usize]| {
        percentile(&idx.iter().map(|&i| p.data[i]).collect::<Vec<_>>(), 0.5)
    };
    let (mut bl, mut ba, mut bb) = (med(&lab.l, &ring), med(&lab.a, &ring), med(&lab.b, &ring));
    match params.background {
        // If the hint disagrees strongly with the ring (photos covering most of the
        // border), fall back to the darkest/lightest common colour.
        BackgroundHint::Dark if bl > 50.0 => {
            let all: Vec<f32> = lab.l.data.clone();
            bl = percentile(&all, 0.05);
            ba = 0.0;
            bb = 0.0;
        }
        BackgroundHint::Light if bl < 50.0 => {
            let all: Vec<f32> = lab.l.data.clone();
            bl = percentile(&all, 0.95);
            ba = 0.0;
            bb = 0.0;
        }
        _ => {}
    }
    let de = |i: usize, l: f32, a: f32, b: f32| {
        let dl = lab.l.data[i] - l;
        let da = lab.a.data[i] - a;
        let db = lab.b.data[i] - b;
        (dl * dl + da * da + db * db).sqrt()
    };
    let ring_d: Vec<f32> = ring.iter().map(|&i| de(i, bl, ba, bb)).collect();
    let mad = percentile(&ring_d, 0.5);
    let ring_bg: Vec<usize> = ring
        .iter()
        .zip(&ring_d)
        .filter(|(_, d)| **d <= (mad * 3.0).max(2.0))
        .map(|(i, _)| *i)
        .collect();
    let ring_bg = if ring_bg.len() > 20 {
        ring_bg
    } else {
        ring.clone()
    };
    let noise = percentile(
        &ring_bg
            .iter()
            .map(|&i| de(i, bl, ba, bb))
            .collect::<Vec<_>>(),
        0.9,
    )
    .max(0.8);

    let sens = params.sensitivity.clamp(0.25, 4.0) as f32;
    let threshold = ((noise * 3.0).max(4.0)) / sens;

    // Gradient threshold from background texture.
    let grad = lab.l.gradient();
    let ring_g: Vec<f32> = ring_bg.iter().map(|&i| grad.data[i]).collect();
    let gthr = (percentile(&ring_g, 0.98) * 2.5).max(1.2) / sens;

    // --- Pass 1: coarse mask with global background -----------------------------------
    let mut dist = Plane::new(w, h);
    for i in 0..w * h {
        dist.data[i] = de(i, bl, ba, bb);
    }
    let close_r = (0.8 * px_per_mm) as f32;
    let open_r = (1.2 * px_per_mm).max(1.5) as f32;
    let fg1 = build_mask(&dist, &grad, threshold, gthr, close_r, open_r);

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
    let fg = build_mask(&dist, &grad, threshold, gthr, close_r, open_r);

    // --- Components → rectangles --------------------------------------------------------
    let ctx = Ctx {
        w,
        h,
        dist: &dist,
        threshold,
        px_per_mm,
        min_side_px: params.min_size_mm * px_per_mm,
        split: params.split_touching,
    };
    let (_, comps) = fg.components();
    let mut found: Vec<(RotatedRect, f64)> = Vec::new();
    for comp in comps {
        if (comp.len() as f64) < ctx.min_side_px * ctx.min_side_px * 0.5 {
            continue;
        }
        process_component(&ctx, &comp, 0, &mut found);
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

fn build_mask(dist: &Plane, grad: &Plane, t: f32, gt: f32, close_r: f32, open_r: f32) -> Mask {
    let (w, h) = (dist.w, dist.h);
    let mut m = Mask::new(w, h);
    for i in 0..w * h {
        m.data[i] = dist.data[i] > t || grad.data[i] > gt;
    }
    m.close(close_r).fill_holes().open(open_r)
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

fn process_component(ctx: &Ctx, comp: &[usize], depth: usize, out: &mut Vec<(RotatedRect, f64)>) {
    let pts = boundary_points(ctx, comp);
    let Some(rect) = min_area_rect(&pts) else {
        return;
    };
    if rect.w.min(rect.h) < ctx.min_side_px {
        return;
    }
    let rectangularity = comp.len() as f64 / rect.area();

    if ctx.split
        && rectangularity < 0.88
        && depth < 3
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
                process_component(ctx, &p, depth + 1, out);
            }
            return;
        }
    }

    let refined = refine_rect(ctx, &rect, &pts).unwrap_or(rect);
    let conf =
        (rectangularity.clamp(0.0, 1.0) * if refined == rect { 0.8 } else { 1.0 }).clamp(0.0, 1.0);
    out.push((refined, conf));
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

/// Sub-pixel refinement of a rectangle's edges using profiles of background distance.
fn refine_rect(ctx: &Ctx, rect: &RotatedRect, hull_pts: &[Point]) -> Option<RotatedRect> {
    let (ux, uy) = rect.axes();
    let c = Point::new(rect.cx, rect.cy);
    let search = (1.5 * ctx.px_per_mm).max(3.0);
    let step = 0.25;
    // Sides: (outward normal, along-direction, half-length, distance from centre)
    let sides = [
        (Point::new(-uy.x, -uy.y), ux, rect.w / 2.0, rect.h / 2.0), // top
        (ux, uy, rect.h / 2.0, rect.w / 2.0),                       // right
        (uy, ux, rect.w / 2.0, rect.h / 2.0),                       // bottom
        (Point::new(-ux.x, -ux.y), uy, rect.h / 2.0, rect.w / 2.0), // left
    ];
    let mut side_pts: Vec<Vec<Point>> = Vec::new();
    for &(n, along, half, off) in &sides {
        let samples = ((half * 2.0 * 0.7) / 1.0).clamp(8.0, 400.0) as usize;
        let mut pts = Vec::new();
        for k in 0..samples {
            let t = -half * 0.7 + (half * 1.4) * (k as f64 + 0.5) / samples as f64;
            let base = Point::new(c.x + along.x * t + n.x * off, c.y + along.y * t + n.y * off);
            // Profile from outside (s = +search) to inside (s = -search).
            let nsteps = (2.0 * search / step) as usize;
            let mut prof = Vec::with_capacity(nsteps + 1);
            let mut valid = true;
            for s in 0..=nsteps {
                let d = search - s as f64 * step;
                let p = Point::new(base.x + n.x * d, base.y + n.y * d);
                if p.x < 0.0 || p.y < 0.0 || p.x > (ctx.w - 1) as f64 || p.y > (ctx.h - 1) as f64 {
                    valid = false;
                    break;
                }
                prof.push(ctx.dist.sample(p.x - 0.5, p.y - 0.5));
            }
            if !valid {
                continue;
            }
            // Derivative (rise when moving inward).
            let deriv: Vec<f32> = (1..prof.len() - 1)
                .map(|i| (prof[i + 1] - prof[i - 1]) / (2.0 * step as f32))
                .collect();
            let maxd = deriv.iter().cloned().fold(0.0f32, f32::max);
            if maxd < ctx.threshold * 0.15 {
                continue;
            }
            // First local maximum from the outside that reaches 40% of the strongest rise.
            let mut best = None;
            for i in 1..deriv.len() - 1 {
                if deriv[i] >= 0.4 * maxd && deriv[i] >= deriv[i - 1] && deriv[i] >= deriv[i + 1] {
                    best = Some(i);
                    break;
                }
            }
            let Some(i) = best else { continue };
            let (a, b, cc) = (deriv[i - 1] as f64, deriv[i] as f64, deriv[i + 1] as f64);
            let den = a - 2.0 * b + cc;
            let frac = if den.abs() > 1e-9 {
                (0.5 * (a - cc) / den).clamp(-0.5, 0.5)
            } else {
                0.0
            };
            // deriv index i corresponds to prof index i+1.
            let d = search - ((i + 1) as f64 + frac) * step;
            pts.push(Point::new(base.x + n.x * d, base.y + n.y * d));
        }
        side_pts.push(pts);
    }

    // Fit lines and estimate a common angle.
    let mut angle_acc = 0.0;
    let mut angle_w = 0.0;
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
        angle_acc += dlt * wgt;
        angle_w += wgt;
        fits.push(Some(inl.iter().map(|&i| pts[i]).collect()));
    }
    if angle_w == 0.0 {
        return None;
    }
    let angle = rect.angle + angle_acc / angle_w;
    let probe = RotatedRect { angle, ..*rect };
    let (nux, nuy) = probe.axes();
    let proj_u = |p: &Point| p.x * nux.x + p.y * nux.y;
    let proj_v = |p: &Point| p.x * nuy.x + p.y * nuy.y;
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let hu: Vec<f64> = hull_pts.iter().map(proj_u).collect();
    let hv: Vec<f64> = hull_pts.iter().map(proj_v).collect();
    let fmin = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
    let fmax = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max);
    let v_min = fits[0]
        .as_ref()
        .map(|p| med(p.iter().map(proj_v).collect()))
        .unwrap_or_else(|| fmin(&hv));
    let u_max = fits[1]
        .as_ref()
        .map(|p| med(p.iter().map(proj_u).collect()))
        .unwrap_or_else(|| fmax(&hu));
    let v_max = fits[2]
        .as_ref()
        .map(|p| med(p.iter().map(proj_v).collect()))
        .unwrap_or_else(|| fmax(&hv));
    let u_min = fits[3]
        .as_ref()
        .map(|p| med(p.iter().map(proj_u).collect()))
        .unwrap_or_else(|| fmin(&hu));
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
    (refined.iou(rect) > 0.85).then_some(refined)
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
}
