//! LaMa inpainting for the larger defects the healing brush handles less convincingly
//! (big specks, wide scratches, small tears).
//!
//! The model takes a fixed 512 x 512 window. Each window is centred on a large defect and
//! covers enough context around it; a defect too big for a native-resolution window is
//! inpainted on a downscaled window, and the fine grain of the classical fill is added back
//! so the patch does not look smoother than the photo around it.

use crate::models::{LAMA, ModelError, Result};
use hp_core::raster::Mask;
use hp_core::restore::{Defect, Defects, Planes};
use rten::Model;
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;
use std::path::Path;

const N: usize = 512;

pub struct Inpainter {
    model: Model,
}

/// Which defects LaMa should redo, and how many windows at most (each takes a few seconds
/// on a CPU). On dust-sized defects the healing brush is as good or better (see
/// `examples/eval_dust_ml.rs`), so only genuinely large damage goes to the model.
#[derive(Debug, Clone)]
pub struct InpaintParams {
    /// Defects at least this wide (mm) qualify...
    pub min_width_mm: f64,
    /// ...or at least this large in area (mm²).
    pub min_area_mm2: f64,
    pub max_windows: usize,
}

impl Default for InpaintParams {
    fn default() -> Self {
        Self {
            min_width_mm: 0.4,
            min_area_mm2: 0.4,
            max_windows: 16,
        }
    }
}

impl Inpainter {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(LAMA.files[0].name);
        if !path.is_file() {
            return Err(ModelError::Missing(LAMA.label));
        }
        let model = Model::load_file(path).map_err(|e| ModelError::Engine(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn load_default() -> Result<Self> {
        Self::load(&LAMA.find().ok_or(ModelError::Missing(LAMA.label))?)
    }

    /// Inpaint one 512 x 512 window. `rgb` is CHW in 0..1, `hole` is 1 where to fill.
    /// Returns CHW in 0..1.
    pub fn run_window(&self, rgb: NdTensor<f32, 4>, hole: NdTensor<f32, 4>) -> Result<Vec<f32>> {
        let ids = self.model.input_ids();
        let inputs = vec![(ids[0], rgb.view().into()), (ids[1], hole.view().into())];
        let out = self
            .model
            .run(inputs, self.model.output_ids(), None)
            .map_err(|e| ModelError::Engine(e.to_string()))?;
        let t: NdTensor<f32, 4> = out
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::Engine("no output".into()))?
            .try_into()
            .map_err(|e| ModelError::Engine(format!("{e:?}")))?;
        // The export outputs 0..255.
        Ok(t.iter().map(|v| (v / 255.0).clamp(0.0, 1.0)).collect())
    }

    /// Redo the large defects in `pl` (already filled by the healing brush) with LaMa.
    /// Returns the number of windows run.
    pub fn refill(
        &self,
        pl: &mut Planes,
        defects: &Defects,
        dpi: f64,
        p: &InpaintParams,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<usize> {
        let pxmm = dpi / 25.4;
        let large: Vec<&Defect> = defects
            .items
            .iter()
            .filter(|d| {
                d.width_px as f64 >= p.min_width_mm * pxmm
                    || d.pixels.len() as f64 >= p.min_area_mm2 * pxmm * pxmm
            })
            .collect();
        if large.is_empty() {
            return Ok(0);
        }
        let windows = plan_windows(&large, pl.w, pl.h, p.max_windows);
        for (k, win) in windows.iter().enumerate() {
            progress(k, windows.len());
            self.fill_window(pl, &defects.mask, win)?;
        }
        progress(windows.len(), windows.len());
        Ok(windows.len())
    }

    fn fill_window(&self, pl: &mut Planes, mask: &Mask, win: &Window) -> Result<()> {
        let (x0, y0, side) = (win.x0, win.y0, win.side);
        let scale = side as f64 / N as f64; // source pixels per model pixel (>= 1)
        let nch = pl.ch.len();
        let get = |c: usize, x: usize, y: usize| pl.ch[c.min(nch - 1)][y * pl.w + x];
        // Area-average downsampling (or a plain copy at scale 1).
        let mut rgb = NdTensor::<f32, 4>::zeros([1, 3, N, N]);
        let mut hole = NdTensor::<f32, 4>::zeros([1, 1, N, N]);
        for my in 0..N {
            let sy0 = y0 + (my as f64 * scale) as usize;
            let sy1 = (y0 + ((my + 1) as f64 * scale) as usize).max(sy0 + 1);
            for mx in 0..N {
                let sx0 = x0 + (mx as f64 * scale) as usize;
                let sx1 = (x0 + ((mx + 1) as f64 * scale) as usize).max(sx0 + 1);
                let mut acc = [0.0f32; 3];
                let mut any = false;
                let mut n = 0.0f32;
                for sy in sy0..sy1.min(pl.h) {
                    for sx in sx0..sx1.min(pl.w) {
                        for (c, a) in acc.iter_mut().enumerate() {
                            *a += get(c, sx, sy);
                        }
                        any |= mask.data[sy * pl.w + sx] && win.owns(sx, sy);
                        n += 1.0;
                    }
                }
                // The export does not blank the hole itself: masked pixels must be zero.
                if any {
                    hole[[0, 0, my, mx]] = 1.0;
                } else {
                    for c in 0..3 {
                        rgb[[0, c, my, mx]] = acc[c] / n.max(1.0);
                    }
                }
            }
        }
        // Grow the hole by a model pixel: LaMa continues whatever borders the hole, so a
        // defect's faint fringe must be inside it, not around it.
        let mut grown = hole.clone();
        for my in 0..N {
            for mx in 0..N {
                if hole[[0, 0, my, mx]] > 0.0 {
                    for yy in my.saturating_sub(1)..=(my + 1).min(N - 1) {
                        for xx in mx.saturating_sub(1)..=(mx + 1).min(N - 1) {
                            grown[[0, 0, yy, xx]] = 1.0;
                            for c in 0..3 {
                                rgb[[0, c, yy, xx]] = 0.0;
                            }
                        }
                    }
                }
            }
        }
        let out = self.run_window(rgb, grown)?;
        let plane = N * N;
        // Bilinear upsampling of the result; at scale > 1 keep the healing brush's fine
        // detail (its difference from a local mean) so grain continues through the patch.
        let sample = |c: usize, fx: f64, fy: f64| -> f32 {
            let fx = (fx - 0.5).clamp(0.0, (N - 1) as f64);
            let fy = (fy - 0.5).clamp(0.0, (N - 1) as f64);
            let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
            let (jx, jy) = ((ix + 1).min(N - 1), (iy + 1).min(N - 1));
            let (tx, ty) = ((fx - ix as f64) as f32, (fy - iy as f64) as f32);
            let o = &out[c * plane..(c + 1) * plane];
            let a = o[iy * N + ix] * (1.0 - tx) + o[iy * N + jx] * tx;
            let b = o[jy * N + ix] * (1.0 - tx) + o[jy * N + jx] * tx;
            a * (1.0 - ty) + b * ty
        };
        let r = scale.round().max(1.0) as usize;
        let mut updates: Vec<(usize, [f32; 3])> = Vec::new();
        for sy in y0..(y0 + side).min(pl.h) {
            for sx in x0..(x0 + side).min(pl.w) {
                let i = sy * pl.w + sx;
                if !mask.data[i] || !win.owns(sx, sy) {
                    continue;
                }
                let (fx, fy) = (
                    (sx - x0) as f64 / scale + 0.5 / scale,
                    (sy - y0) as f64 / scale + 0.5 / scale,
                );
                let mut v = [0.0f32; 3];
                for (c, val) in v.iter_mut().enumerate() {
                    *val = sample(c, fx, fy);
                    if r > 1 {
                        let cc = c.min(nch - 1);
                        let (mut s, mut n) = (0.0f32, 0.0f32);
                        for yy in sy.saturating_sub(r)..=(sy + r).min(pl.h - 1) {
                            for xx in sx.saturating_sub(r)..=(sx + r).min(pl.w - 1) {
                                s += pl.ch[cc][yy * pl.w + xx];
                                n += 1.0;
                            }
                        }
                        *val += pl.ch[cc][i] - s / n;
                    }
                }
                updates.push((i, v));
            }
        }
        for (i, v) in updates {
            if nch == 1 {
                pl.ch[0][i] = (0.299 * v[0] + 0.587 * v[1] + 0.114 * v[2]).clamp(0.0, 1.0);
            } else {
                for (plane, val) in pl.ch.iter_mut().zip(v) {
                    plane[i] = val.clamp(0.0, 1.0);
                }
            }
        }
        Ok(())
    }
}

/// A square window and the defect boxes it is responsible for.
#[derive(Debug, Clone)]
struct Window {
    x0: usize,
    y0: usize,
    side: usize,
    boxes: Vec<(usize, usize, usize, usize)>,
}

impl Window {
    fn owns(&self, x: usize, y: usize) -> bool {
        self.boxes
            .iter()
            .any(|b| x >= b.0 && x <= b.2 && y >= b.1 && y <= b.3)
    }
}

/// Greedy windows: each starts at the largest uncovered defect, is big enough for it plus
/// context on all sides (at least 512 source pixels), and takes every other large defect
/// that fits comfortably inside.
fn plan_windows(large: &[&Defect], w: usize, h: usize, max: usize) -> Vec<Window> {
    let mut order: Vec<&Defect> = large.to_vec();
    order.sort_by_key(|d| std::cmp::Reverse(d.pixels.len()));
    let mut done = vec![false; order.len()];
    let mut out = Vec::new();
    for k in 0..order.len() {
        if done[k] || out.len() >= max {
            continue;
        }
        let b = order[k].bbox;
        let (bw, bh) = (b.2 - b.0 + 1, b.3 - b.1 + 1);
        // The hole should be at most about a third of the window.
        let side = (bw.max(bh) * 3).max(N).min(w.max(h));
        let side = side.min(w).min(h).max(1);
        let cx = (b.0 + b.2) / 2;
        let cy = (b.1 + b.3) / 2;
        let x0 = cx.saturating_sub(side / 2).min(w.saturating_sub(side));
        let y0 = cy.saturating_sub(side / 2).min(h.saturating_sub(side));
        let margin = side / 8;
        let mut win = Window {
            x0,
            y0,
            side,
            boxes: Vec::new(),
        };
        for (j, d) in order.iter().enumerate() {
            if done[j] {
                continue;
            }
            let c = d.bbox;
            let inside = c.0 >= x0 + margin.min(c.0)
                && c.1 >= y0 + margin.min(c.1)
                && c.2 + margin.min(w - 1 - c.2) < x0 + side
                && c.3 + margin.min(h - 1 - c.3) < y0 + side;
            if inside || j == k {
                win.boxes.push(c);
                done[j] = true;
            }
        }
        out.push(win);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defect(x0: usize, y0: usize, x1: usize, y1: usize) -> Defect {
        Defect {
            pixels: vec![0; (x1 - x0 + 1) * (y1 - y0 + 1)],
            bbox: (x0, y0, x1, y1),
            bright: true,
            length_px: 10.0,
            width_px: 10.0,
        }
    }

    #[test]
    fn windows_group_nearby_defects() {
        let a = defect(100, 100, 120, 120);
        let b = defect(200, 180, 210, 190);
        let c = defect(1500, 1500, 1510, 1510);
        let ws = plan_windows(&[&a, &b, &c], 2000, 2000, 10);
        assert_eq!(ws.len(), 2);
        assert_eq!(ws[0].boxes.len() + ws[1].boxes.len(), 3);
        for w in &ws {
            assert!(w.side >= N);
            assert!(w.x0 + w.side <= 2000 && w.y0 + w.side <= 2000);
        }
        // Small images: the window shrinks to the image.
        let ws = plan_windows(&[&a], 300, 200, 10);
        assert_eq!(ws[0].side, 200);
    }
}
