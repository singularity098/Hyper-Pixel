//! Core image processing for Hyper Pixel. No GUI or scanner dependencies, so everything
//! here is testable headless.

pub mod color;
pub mod crop;
pub mod detect;
pub mod document;
pub mod eval;
pub mod export;
pub mod geom;
pub mod pdf;
pub mod raster;
pub mod synth;

pub use detect::{
    BackgroundHint, DetectParams, DetectReport, Detection, ScanGeometry, detect_photos,
};
pub use geom::{Point, RotatedRect};
