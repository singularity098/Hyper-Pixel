//! Scoring detections against ground truth.

use crate::geom::RotatedRect;
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct EvalStats {
    pub scenes: usize,
    pub truth: usize,
    pub detected: usize,
    /// Truth rects matched with IoU >= 0.9.
    pub matched: usize,
    pub missed: usize,
    pub extra: usize,
    pub mean_iou: f64,
    pub min_iou: f64,
    pub mean_angle_err: f64,
    pub max_angle_err: f64,
    /// Mean of the worst edge displacement per matched photo, mm.
    pub mean_edge_err_mm: f64,
    pub max_edge_err_mm: f64,
    /// Scenes where everything matched and nothing extra was found.
    pub perfect_scenes: usize,
}

/// Greedy one-to-one matching by IoU. Returns pairs (truth index, detection index, iou).
pub fn match_rects(truth: &[RotatedRect], det: &[RotatedRect]) -> Vec<(usize, usize, f64)> {
    let mut cand = Vec::new();
    for (i, t) in truth.iter().enumerate() {
        for (j, d) in det.iter().enumerate() {
            let iou = t.iou(d);
            if iou > 0.0 {
                cand.push((i, j, iou));
            }
        }
    }
    cand.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut ut = vec![false; truth.len()];
    let mut ud = vec![false; det.len()];
    let mut out = Vec::new();
    for (i, j, iou) in cand {
        if !ut[i] && !ud[j] {
            ut[i] = true;
            ud[j] = true;
            out.push((i, j, iou));
        }
    }
    out
}

/// Largest distance between corresponding corners of two rects (orientation-agnostic).
pub fn corner_error(a: &RotatedRect, b: &RotatedRect) -> f64 {
    let ca = a.corners();
    let cb = b.corners();
    (0..4)
        .map(|shift| {
            (0..4)
                .map(|k| ca[k].dist(cb[(k + shift) % 4]))
                .fold(0.0, f64::max)
        })
        .fold(f64::MAX, f64::min)
}

#[derive(Default)]
pub struct Evaluator {
    stats: EvalStats,
    iou_sum: f64,
    ang_sum: f64,
    edge_sum: f64,
}

impl Evaluator {
    pub fn new() -> Self {
        Self {
            stats: EvalStats {
                min_iou: 1.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn add(&mut self, truth: &[RotatedRect], det: &[RotatedRect]) {
        let s = &mut self.stats;
        s.scenes += 1;
        s.truth += truth.len();
        s.detected += det.len();
        let m = match_rects(truth, det);
        let good: Vec<_> = m.iter().filter(|x| x.2 >= 0.9).collect();
        s.matched += good.len();
        s.missed += truth.len() - good.len();
        s.extra += det.len() - good.len();
        if good.len() == truth.len() && det.len() == truth.len() {
            s.perfect_scenes += 1;
        }
        for &&(i, j, iou) in &good {
            self.iou_sum += iou;
            s.min_iou = s.min_iou.min(iou);
            let ae = truth[i].angle_diff(&det[j]);
            self.ang_sum += ae;
            s.max_angle_err = s.max_angle_err.max(ae);
            let ce = corner_error(&truth[i], &det[j]);
            self.edge_sum += ce;
            s.max_edge_err_mm = s.max_edge_err_mm.max(ce);
        }
        let n = s.matched.max(1) as f64;
        s.mean_iou = self.iou_sum / n;
        s.mean_angle_err = self.ang_sum / n;
        s.mean_edge_err_mm = self.edge_sum / n;
    }

    pub fn stats(&self) -> &EvalStats {
        &self.stats
    }
}
