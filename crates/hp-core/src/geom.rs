//! Geometry primitives: points, rotated rectangles, convex hulls, polygon clipping.
//!
//! Coordinates are plain `f64` in whatever unit the caller uses (pixels or millimetres).
//! Y points down, matching image coordinates. Angles are in degrees, positive = clockwise
//! on screen (because Y points down this is the usual mathematical rotation applied to
//! image coordinates).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn dist(self, o: Point) -> f64 {
        ((self.x - o.x).powi(2) + (self.y - o.y).powi(2)).sqrt()
    }
}

impl std::ops::Sub for Point {
    type Output = Point;
    fn sub(self, o: Point) -> Point {
        Point::new(self.x - o.x, self.y - o.y)
    }
}
impl std::ops::Add for Point {
    type Output = Point;
    fn add(self, o: Point) -> Point {
        Point::new(self.x + o.x, self.y + o.y)
    }
}

fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// A rectangle of size `w`×`h` centred on (`cx`,`cy`), rotated by `angle` degrees.
/// `w` is measured along the rotated X axis.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct RotatedRect {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub angle: f64,
}

impl RotatedRect {
    pub fn new(cx: f64, cy: f64, w: f64, h: f64, angle: f64) -> Self {
        Self {
            cx,
            cy,
            w,
            h,
            angle,
        }
    }

    /// Unit vectors of the rect's local X and Y axes.
    pub fn axes(&self) -> (Point, Point) {
        let (s, c) = self.angle.to_radians().sin_cos();
        (Point::new(c, s), Point::new(-s, c))
    }

    /// Corners in order: top-left, top-right, bottom-right, bottom-left (in local frame).
    pub fn corners(&self) -> [Point; 4] {
        let (ux, uy) = self.axes();
        let (hw, hh) = (self.w / 2.0, self.h / 2.0);
        let c = Point::new(self.cx, self.cy);
        let p = |a: f64, b: f64| Point::new(c.x + ux.x * a + uy.x * b, c.y + ux.y * a + uy.y * b);
        [p(-hw, -hh), p(hw, -hh), p(hw, hh), p(-hw, hh)]
    }

    pub fn area(&self) -> f64 {
        self.w * self.h
    }

    /// Map a point in the rect's local frame (origin at the top-left corner, units as the
    /// rect) to the outer frame.
    pub fn local_to_world(&self, u: f64, v: f64) -> Point {
        let (ux, uy) = self.axes();
        let a = u - self.w / 2.0;
        let b = v - self.h / 2.0;
        Point::new(self.cx + ux.x * a + uy.x * b, self.cy + ux.y * a + uy.y * b)
    }

    /// Canonical form: angle in (-45, 45]. Swaps w/h as needed. Describes the same region.
    pub fn normalized(mut self) -> Self {
        let mut a = self.angle % 180.0;
        if a > 90.0 {
            a -= 180.0;
        } else if a <= -90.0 {
            a += 180.0;
        }
        if a > 45.0 {
            a -= 90.0;
            std::mem::swap(&mut self.w, &mut self.h);
        } else if a <= -45.0 {
            a += 90.0;
            std::mem::swap(&mut self.w, &mut self.h);
        }
        self.angle = a;
        self
    }

    /// Shrink (positive) or grow (negative) every edge by `d`.
    pub fn inset(&self, d: f64) -> Self {
        Self {
            w: (self.w - 2.0 * d).max(1e-6),
            h: (self.h - 2.0 * d).max(1e-6),
            ..*self
        }
    }

    pub fn scaled(&self, s: f64) -> Self {
        Self {
            cx: self.cx * s,
            cy: self.cy * s,
            w: self.w * s,
            h: self.h * s,
            angle: self.angle,
        }
    }

    pub fn translated(&self, dx: f64, dy: f64) -> Self {
        Self {
            cx: self.cx + dx,
            cy: self.cy + dy,
            ..*self
        }
    }

    /// Does the rect contain point `p`?
    pub fn contains(&self, p: Point) -> bool {
        let (ux, uy) = self.axes();
        let d = Point::new(p.x - self.cx, p.y - self.cy);
        let a = d.x * ux.x + d.y * ux.y;
        let b = d.x * uy.x + d.y * uy.y;
        a.abs() <= self.w / 2.0 && b.abs() <= self.h / 2.0
    }

    /// Intersection-over-union with another rect.
    pub fn iou(&self, other: &RotatedRect) -> f64 {
        let inter = polygon_area(&clip_convex(&self.corners(), &other.corners()));
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }

    /// Smallest difference between the two rects' orientations, in degrees, modulo 90.
    pub fn angle_diff(&self, other: &RotatedRect) -> f64 {
        let d = (self.angle - other.angle).rem_euclid(90.0);
        d.min(90.0 - d)
    }
}

/// Andrew's monotone chain. Returns hull in counter-clockwise order (in a Y-up sense),
/// without repeating the first point.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut pts: Vec<Point> = points.to_vec();
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut lower: Vec<Point> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Point> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum-area enclosing rectangle of a point set (rotating calipers over the hull edges).
pub fn min_area_rect(points: &[Point]) -> Option<RotatedRect> {
    let hull = convex_hull(points);
    if hull.len() < 3 {
        return None;
    }
    let mut best: Option<(f64, RotatedRect)> = None;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let e = b - a;
        let len = (e.x * e.x + e.y * e.y).sqrt();
        if len < 1e-12 {
            continue;
        }
        let ux = Point::new(e.x / len, e.y / len);
        let uy = Point::new(-ux.y, ux.x);
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &p in &hull {
            let u = p.x * ux.x + p.y * ux.y;
            let v = p.x * uy.x + p.y * uy.y;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if best.as_ref().is_none_or(|(a, _)| area < *a) {
            let cu = (min_u + max_u) / 2.0;
            let cv = (min_v + max_v) / 2.0;
            let rect = RotatedRect {
                cx: ux.x * cu + uy.x * cv,
                cy: ux.y * cu + uy.y * cv,
                w: max_u - min_u,
                h: max_v - min_v,
                angle: ux.y.atan2(ux.x).to_degrees(),
            };
            best = Some((area, rect.normalized()));
        }
    }
    best.map(|(_, r)| r)
}

/// Shoelace area (absolute).
pub fn polygon_area(poly: &[Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        s += a.x * b.y - b.x * a.y;
    }
    s.abs() / 2.0
}

/// Sutherland–Hodgman clip of convex `subject` by convex `clip`. Either winding works.
pub fn clip_convex(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    let mut out = subject.to_vec();
    if clip.len() < 3 {
        return Vec::new();
    }
    // Determine clip polygon orientation so "inside" is consistent.
    let mut signed = 0.0;
    for i in 0..clip.len() {
        let a = clip[i];
        let b = clip[(i + 1) % clip.len()];
        signed += a.x * b.y - b.x * a.y;
    }
    let orient = if signed >= 0.0 { 1.0 } else { -1.0 };
    for i in 0..clip.len() {
        if out.is_empty() {
            break;
        }
        let a = clip[i];
        let b = clip[(i + 1) % clip.len()];
        let inside = |p: Point| cross(a, b, p) * orient >= 0.0;
        let input = std::mem::take(&mut out);
        for j in 0..input.len() {
            let cur = input[j];
            let prev = input[(j + input.len() - 1) % input.len()];
            let (ci, pi) = (inside(cur), inside(prev));
            if ci {
                if !pi {
                    out.push(intersect(prev, cur, a, b));
                }
                out.push(cur);
            } else if pi {
                out.push(intersect(prev, cur, a, b));
            }
        }
    }
    out
}

fn intersect(p1: Point, p2: Point, a: Point, b: Point) -> Point {
    let d1 = p2 - p1;
    let d2 = b - a;
    let den = d1.x * d2.y - d1.y * d2.x;
    if den.abs() < 1e-12 {
        return p2;
    }
    let t = ((a.x - p1.x) * d2.y - (a.y - p1.y) * d2.x) / den;
    Point::new(p1.x + d1.x * t, p1.y + d1.y * t)
}

/// A 2D line in normal form: `nx*x + ny*y = d` with unit normal.
#[derive(Debug, Clone, Copy)]
pub struct Line {
    pub nx: f64,
    pub ny: f64,
    pub d: f64,
}

/// Total-least-squares line fit with iterative outlier rejection.
/// Returns the line and the indices of inliers.
pub fn fit_line_robust(
    points: &[Point],
    iterations: usize,
    min_keep: f64,
) -> Option<(Line, Vec<usize>)> {
    if points.len() < 2 {
        return None;
    }
    let mut idx: Vec<usize> = (0..points.len()).collect();
    let mut line = fit_line_tls(points, &idx)?;
    for _ in 0..iterations {
        let mut res: Vec<(usize, f64)> = (0..points.len())
            .map(|i| {
                (
                    i,
                    (line.nx * points[i].x + line.ny * points[i].y - line.d).abs(),
                )
            })
            .collect();
        let mut sorted: Vec<f64> = res.iter().map(|r| r.1).collect();
        sorted.sort_by(f64::total_cmp);
        let med = sorted[sorted.len() / 2];
        let thr = (med * 2.5 * 1.4826).max(0.35);
        res.retain(|r| r.1 <= thr);
        if (res.len() as f64) < min_keep * points.len() as f64 {
            // Too aggressive: keep the best `min_keep` fraction instead.
            let k = ((min_keep * points.len() as f64).ceil() as usize)
                .max(2)
                .min(points.len());
            let mut all: Vec<(usize, f64)> = (0..points.len())
                .map(|i| {
                    (
                        i,
                        (line.nx * points[i].x + line.ny * points[i].y - line.d).abs(),
                    )
                })
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            res = all.into_iter().take(k).collect();
        }
        let new_idx: Vec<usize> = res.into_iter().map(|r| r.0).collect();
        if new_idx.len() < 2 {
            break;
        }
        line = fit_line_tls(points, &new_idx)?;
        idx = new_idx;
    }
    Some((line, idx))
}

fn fit_line_tls(points: &[Point], idx: &[usize]) -> Option<Line> {
    let n = idx.len() as f64;
    if n < 2.0 {
        return None;
    }
    let (mut mx, mut my) = (0.0, 0.0);
    for &i in idx {
        mx += points[i].x;
        my += points[i].y;
    }
    mx /= n;
    my /= n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for &i in idx {
        let dx = points[i].x - mx;
        let dy = points[i].y - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    // Direction = principal eigenvector; normal is perpendicular.
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (s, c) = theta.sin_cos();
    let (nx, ny) = (-s, c);
    Some(Line {
        nx,
        ny,
        d: nx * mx + ny * my,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_rect_recovers_rotated_rect() {
        let r = RotatedRect::new(50.0, 40.0, 30.0, 20.0, 12.0);
        let pts: Vec<Point> = r.corners().to_vec();
        let m = min_area_rect(&pts).unwrap();
        assert!((m.cx - 50.0).abs() < 1e-6 && (m.cy - 40.0).abs() < 1e-6);
        assert!((m.area() - 600.0).abs() < 1e-6);
        assert!(m.angle_diff(&r) < 1e-6);
    }

    #[test]
    fn iou_identity_and_disjoint() {
        let r = RotatedRect::new(0.0, 0.0, 10.0, 5.0, 30.0);
        assert!((r.iou(&r) - 1.0).abs() < 1e-9);
        let o = r.translated(100.0, 0.0);
        assert_eq!(o.iou(&r), 0.0);
        let half = RotatedRect::new(2.5, 0.0, 5.0, 5.0, 0.0);
        let full = RotatedRect::new(0.0, 0.0, 10.0, 5.0, 0.0);
        assert!((half.iou(&full) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn normalize_keeps_region() {
        let r = RotatedRect::new(1.0, 2.0, 10.0, 4.0, 80.0);
        let n = r.normalized();
        assert!(n.angle > -45.0 && n.angle <= 45.0);
        assert!((n.iou(&r) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn robust_line_ignores_outliers() {
        let mut pts: Vec<Point> = (0..50)
            .map(|i| Point::new(i as f64, 0.5 * i as f64 + 3.0))
            .collect();
        pts.push(Point::new(10.0, 40.0));
        pts.push(Point::new(20.0, -30.0));
        let (l, inl) = fit_line_robust(&pts, 4, 0.5).unwrap();
        assert_eq!(inl.len(), 50);
        // y = 0.5x + 3 -> check a point.
        let y = (l.d - l.nx * 30.0) / l.ny;
        assert!((y - 18.0).abs() < 1e-6);
    }
}
