//! Interactive crop-box editing (move / resize / rotate / draw), independent of the UI
//! toolkit so it can be unit tested. All coordinates are bed millimetres.

use hp_core::geom::{clip_convex, polygon_area};
use hp_core::{Point, RotatedRect};

const MIN_SIZE_MM: f64 = 5.0;

#[derive(Debug, Clone, PartialEq)]
pub struct CropBox {
    pub id: u64,
    pub rect: RotatedRect,
    /// Extra clockwise quarter turns for the output image.
    pub quarter_turns: i32,
    /// True once the user has changed it by hand (then we never replace it with a
    /// re-detected rectangle from the final scan).
    pub edited: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
enum Drag {
    None,
    Move {
        id: u64,
        start: Point,
        orig: RotatedRect,
    },
    Corner {
        id: u64,
        opposite: Point,
        orig: RotatedRect,
    },
    Rotate {
        id: u64,
        start_angle: f64,
        orig: RotatedRect,
    },
    Create {
        start: Point,
        id: u64,
    },
}

#[derive(Debug, Clone)]
pub struct Editor {
    pub boxes: Vec<CropBox>,
    pub selected: Option<u64>,
    drag: Drag,
    next_id: u64,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            boxes: Vec::new(),
            selected: None,
            drag: Drag::None,
            next_id: 1,
        }
    }
}

/// Where the rotation handle sits for a rect, `offset` mm beyond the top edge.
pub fn rotate_handle(r: &RotatedRect, offset: f64) -> Point {
    let (_, uy) = r.axes();
    Point::new(
        r.cx - uy.x * (r.h / 2.0 + offset),
        r.cy - uy.y * (r.h / 2.0 + offset),
    )
}

impl Editor {
    pub fn set_detections(&mut self, rects: impl IntoIterator<Item = (RotatedRect, f64)>) {
        self.boxes.clear();
        self.selected = None;
        for (rect, confidence) in rects {
            let id = self.alloc_id();
            self.boxes.push(CropBox {
                id,
                rect,
                quarter_turns: 0,
                edited: false,
                confidence,
            });
        }
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut CropBox> {
        self.boxes.iter_mut().find(|b| b.id == id)
    }

    /// True while a drag places a point precisely (a corner, or a new box's far corner):
    /// the UI then shows its magnifier.
    pub fn is_placing_point(&self) -> bool {
        matches!(self.drag, Drag::Corner { .. } | Drag::Create { .. })
    }

    /// True if `p` is within `tol` mm of a box corner that `pointer_down` would grab.
    pub fn corner_near(&self, p: Point, tol: f64) -> bool {
        self.boxes
            .iter()
            .any(|b| b.rect.corners().iter().any(|c| c.dist(p) <= tol))
    }

    /// Pointer pressed at `p`. `tol` is the grab tolerance in mm, `handle_offset` the
    /// rotation handle distance in mm.
    pub fn pointer_down(&mut self, p: Point, tol: f64, handle_offset: f64) {
        // Handles of the selected box take priority.
        if let Some(sel) = self
            .selected
            .and_then(|id| self.boxes.iter().find(|b| b.id == id))
        {
            let r = sel.rect;
            if rotate_handle(&r, handle_offset).dist(p) <= tol {
                self.drag = Drag::Rotate {
                    id: sel.id,
                    start_angle: (p.y - r.cy).atan2(p.x - r.cx).to_degrees(),
                    orig: r,
                };
                return;
            }
            let corners = r.corners();
            if let Some(k) = (0..4).find(|&k| corners[k].dist(p) <= tol) {
                self.drag = Drag::Corner {
                    id: sel.id,
                    opposite: corners[(k + 2) % 4],
                    orig: r,
                };
                return;
            }
        }
        // Any box's corner (also selects it).
        for b in self.boxes.iter().rev() {
            let corners = b.rect.corners();
            if let Some(k) = (0..4).find(|&k| corners[k].dist(p) <= tol) {
                self.selected = Some(b.id);
                self.drag = Drag::Corner {
                    id: b.id,
                    opposite: corners[(k + 2) % 4],
                    orig: b.rect,
                };
                return;
            }
        }
        // Inside a box: select + move. Prefer the smallest containing box.
        if let Some(b) = self
            .boxes
            .iter()
            .filter(|b| b.rect.contains(p))
            .min_by(|a, b| a.rect.area().total_cmp(&b.rect.area()))
        {
            self.selected = Some(b.id);
            self.drag = Drag::Move {
                id: b.id,
                start: p,
                orig: b.rect,
            };
            return;
        }
        // Empty space: start drawing a new box.
        let id = self.alloc_id();
        self.selected = None;
        self.drag = Drag::Create { start: p, id };
    }

    /// Returns true if anything changed.
    pub fn pointer_move(&mut self, p: Point) -> bool {
        match self.drag.clone() {
            Drag::None => false,
            Drag::Move { id, start, orig } => {
                if let Some(b) = self.get_mut(id) {
                    b.rect = orig.translated(p.x - start.x, p.y - start.y);
                    b.edited = true;
                }
                true
            }
            Drag::Corner { id, opposite, orig } => {
                let (ux, uy) = orig.axes();
                let d = p - opposite;
                let mut u = d.x * ux.x + d.y * ux.y;
                let mut v = d.x * uy.x + d.y * uy.y;
                if u.abs() < MIN_SIZE_MM {
                    u = MIN_SIZE_MM.copysign(if u == 0.0 { 1.0 } else { u });
                }
                if v.abs() < MIN_SIZE_MM {
                    v = MIN_SIZE_MM.copysign(if v == 0.0 { 1.0 } else { v });
                }
                if let Some(b) = self.get_mut(id) {
                    b.rect = RotatedRect {
                        cx: opposite.x + ux.x * u / 2.0 + uy.x * v / 2.0,
                        cy: opposite.y + ux.y * u / 2.0 + uy.y * v / 2.0,
                        w: u.abs(),
                        h: v.abs(),
                        angle: orig.angle,
                    };
                    b.edited = true;
                }
                true
            }
            Drag::Rotate {
                id,
                start_angle,
                orig,
            } => {
                let a = (p.y - orig.cy).atan2(p.x - orig.cx).to_degrees();
                let mut angle = orig.angle + (a - start_angle);
                // Gentle snap to straight.
                let nearest = (angle / 90.0).round() * 90.0;
                if (angle - nearest).abs() < 0.75 {
                    angle = nearest;
                }
                if let Some(b) = self.get_mut(id) {
                    b.rect.angle = angle;
                    b.edited = true;
                }
                true
            }
            Drag::Create { start, id } => {
                let rect = RotatedRect::new(
                    (start.x + p.x) / 2.0,
                    (start.y + p.y) / 2.0,
                    (p.x - start.x).abs(),
                    (p.y - start.y).abs(),
                    0.0,
                );
                if let Some(b) = self.get_mut(id) {
                    b.rect = rect;
                } else if rect.w >= 1.0 || rect.h >= 1.0 {
                    self.boxes.push(CropBox {
                        id,
                        rect,
                        quarter_turns: 0,
                        edited: true,
                        confidence: 1.0,
                    });
                    self.selected = Some(id);
                }
                true
            }
        }
    }

    pub fn pointer_up(&mut self) -> bool {
        let drag = std::mem::replace(&mut self.drag, Drag::None);
        match drag {
            Drag::Create { id, .. } => {
                // Discard accidental tiny boxes (i.e. a click on empty space).
                let before = self.boxes.len();
                self.boxes
                    .retain(|b| b.id != id || (b.rect.w >= MIN_SIZE_MM && b.rect.h >= MIN_SIZE_MM));
                if self.boxes.len() != before {
                    self.selected = None;
                }
                true
            }
            Drag::None => false,
            _ => {
                // Normalise orientation for clean storage after rotation.
                if let Some(id) = self.selected
                    && let Some(b) = self.get_mut(id)
                {
                    let n = b.rect.normalized();
                    // Normalising may swap w/h, which changes which edge is "up".
                    let turns = ((n.angle - b.rect.angle) / 90.0).round() as i32;
                    b.rect = n;
                    b.quarter_turns = (b.quarter_turns + turns).rem_euclid(4);
                }
                true
            }
        }
    }

    pub fn delete_selected(&mut self) -> bool {
        let Some(id) = self.selected.take() else {
            return false;
        };
        self.boxes.retain(|b| b.id != id);
        true
    }

    pub fn turn_selected(&mut self, quarters: i32) -> bool {
        let Some(id) = self.selected else {
            return false;
        };
        if let Some(b) = self.get_mut(id) {
            b.quarter_turns = (b.quarter_turns + quarters).rem_euclid(4);
            return true;
        }
        false
    }

    pub fn clear(&mut self) {
        self.boxes.clear();
        self.selected = None;
        self.drag = Drag::None;
    }
}

/// Replace un-edited boxes with better matching detections (from the final scan).
///
/// If the final scan finds several photos inside one un-edited box, the box is replaced by
/// all of them: touching prints can merge at preview resolution, where the seam between
/// them is thinner than a pixel.
pub fn refine_boxes(boxes: &[CropBox], detections: &[RotatedRect]) -> Vec<CropBox> {
    let mut out = Vec::with_capacity(boxes.len());
    for b in boxes {
        if b.edited {
            out.push(b.clone());
            continue;
        }
        let inside: Vec<&RotatedRect> = detections
            .iter()
            .filter(|d| {
                let inter = polygon_area(&clip_convex(&d.corners(), &b.rect.corners()));
                inter > 0.85 * d.area() && d.area() > 0.1 * b.rect.area()
            })
            .collect();
        if inside.len() >= 2 {
            out.extend(inside.into_iter().map(|d| CropBox {
                rect: *d,
                ..b.clone()
            }));
            continue;
        }
        let best = detections
            .iter()
            .map(|d| (d, d.iou(&b.rect)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        out.push(match best {
            Some((d, iou)) if iou > 0.8 => CropBox {
                rect: *d,
                ..b.clone()
            },
            _ => b.clone(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ed() -> Editor {
        let mut e = Editor::default();
        e.set_detections([(RotatedRect::new(50.0, 50.0, 40.0, 20.0, 0.0), 1.0)]);
        e
    }

    #[test]
    fn magnifier_state() {
        let mut e = ed();
        let c = e.boxes[0].rect.corners()[0];
        assert!(e.corner_near(Point::new(c.x + 1.0, c.y), 2.0));
        assert!(!e.corner_near(Point::new(c.x + 5.0, c.y + 5.0), 2.0));
        assert!(!e.is_placing_point());
        e.pointer_down(c, 2.0, 8.0);
        assert!(e.is_placing_point());
        e.pointer_up();
        assert!(!e.is_placing_point());
        // Moving a whole box is not a precise placement.
        let r = e.boxes[0].rect;
        e.pointer_down(Point::new(r.cx, r.cy), 2.0, 8.0);
        assert!(!e.is_placing_point());
        e.pointer_up();
        // Drawing a new box is.
        e.pointer_down(Point::new(300.0, 300.0), 2.0, 8.0);
        assert!(e.is_placing_point());
    }

    #[test]
    fn move_box() {
        let mut e = ed();
        e.pointer_down(Point::new(50.0, 50.0), 2.0, 8.0);
        e.pointer_move(Point::new(60.0, 55.0));
        e.pointer_up();
        let r = e.boxes[0].rect;
        assert_eq!((r.cx, r.cy), (60.0, 55.0));
        assert!(e.boxes[0].edited);
    }

    #[test]
    fn resize_from_corner_keeps_opposite() {
        let mut e = ed();
        // bottom-right corner at (70, 60)
        e.pointer_down(Point::new(70.0, 60.0), 2.0, 8.0);
        e.pointer_move(Point::new(80.0, 70.0));
        e.pointer_up();
        let r = e.boxes[0].rect;
        assert!((r.w - 50.0).abs() < 1e-9 && (r.h - 30.0).abs() < 1e-9);
        let c = r.corners();
        assert!(c.iter().any(|p| p.dist(Point::new(30.0, 40.0)) < 1e-9));
    }

    #[test]
    fn rotate_with_handle() {
        let mut e = ed();
        e.selected = Some(e.boxes[0].id);
        let h = rotate_handle(&e.boxes[0].rect, 8.0);
        e.pointer_down(h, 2.0, 8.0);
        // Swing the handle 10 degrees clockwise about the centre.
        let r = h.dist(Point::new(50.0, 50.0));
        let a = (-90.0f64 + 10.0).to_radians();
        e.pointer_move(Point::new(50.0 + r * a.cos(), 50.0 + r * a.sin()));
        e.pointer_up();
        assert!((e.boxes[0].rect.angle - 10.0).abs() < 1e-6);
    }

    #[test]
    fn draw_new_and_ignore_click() {
        let mut e = ed();
        e.pointer_down(Point::new(100.0, 100.0), 2.0, 8.0);
        e.pointer_up();
        assert_eq!(e.boxes.len(), 1, "click must not create a box");
        e.pointer_down(Point::new(100.0, 100.0), 2.0, 8.0);
        e.pointer_move(Point::new(130.0, 140.0));
        e.pointer_up();
        assert_eq!(e.boxes.len(), 2);
        assert_eq!(e.selected, Some(e.boxes[1].id));
        assert!(e.delete_selected());
        assert_eq!(e.boxes.len(), 1);
    }

    #[test]
    fn refine_respects_edits() {
        let mut e = ed();
        let better = RotatedRect::new(50.5, 50.0, 40.0, 20.0, 0.3);
        assert_eq!(refine_boxes(&e.boxes, &[better])[0].rect, better);
        e.boxes[0].edited = true;
        assert_ne!(refine_boxes(&e.boxes, &[better])[0].rect, better);
    }

    #[test]
    fn refine_splits_merged_box() {
        // Preview saw one 40×20 box; the final scan finds two touching 20×20 photos in it.
        let e = ed();
        let left = RotatedRect::new(40.1, 50.0, 19.8, 19.9, 0.0);
        let right = RotatedRect::new(60.0, 50.0, 19.9, 19.9, 0.2);
        let out = refine_boxes(&e.boxes, &[left, right]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].rect, left);
        assert_eq!(out[1].rect, right);
    }
}
